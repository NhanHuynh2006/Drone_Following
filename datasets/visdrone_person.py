"""
VisDrone Person Dataset with strong augmentation.

Supports YOLO-format labels:
  Each line: class_id cx cy w h [fx fy]
  All values normalized to [0,1]

Augmentations:
  - Mosaic 4-image composition
  - MixUp (blend two images)
  - Random horizontal flip
  - Random scale jitter
  - Color jitter (brightness, contrast, saturation, hue)
  - Random perspective/affine transform
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VisDronePersonDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=416,
                 augment=True, mosaic_prob=0.5, mixup_prob=0.15,
                 max_labels=100, cache_ram=False):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.max_labels = max_labels

        # Collect valid image-label pairs
        self.samples = []
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

        if os.path.isdir(images_dir):
            for fname in sorted(os.listdir(images_dir)):
                stem, ext = os.path.splitext(fname)
                if ext.lower() not in valid_ext:
                    continue
                img_path = os.path.join(images_dir, fname)
                lbl_path = os.path.join(labels_dir, stem + '.txt')
                if os.path.isfile(lbl_path):
                    self.samples.append((img_path, lbl_path))

        print(f"[Dataset] Found {len(self.samples)} samples in {images_dir}")

        # Cache images in RAM
        self._img_cache = {}
        self._lbl_cache = {}
        if cache_ram and len(self.samples) > 0:
            print(f"[Dataset] Caching {len(self.samples)} images to RAM...", end=" ", flush=True)
            for i in range(len(self.samples)):
                self._img_cache[i] = self._load_image_raw(i)
                self._lbl_cache[i] = self._load_labels_raw(i)
            mb = sum(img.nbytes for img in self._img_cache.values()) / 1e6
            print(f"done ({mb:.0f} MB)")

    def __len__(self):
        return len(self.samples)

    def _load_image_raw(self, idx):
        img_path = self.samples[idx][0]
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_labels_raw(self, idx):
        lbl_path = self.samples[idx][1]
        labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id != 0:
                    continue
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if len(parts) >= 7:
                    fx, fy = float(parts[5]), float(parts[6])
                else:
                    fx = cx
                    fy = min(1.0, cy + h / 2)
                labels.append([0, cx, cy, w, h, fx, fy])
        if len(labels) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return np.array(labels, dtype=np.float32)

    def _load_image(self, idx):
        if idx in self._img_cache:
            return self._img_cache[idx]
        return self._load_image_raw(idx)

    def _load_labels(self, idx):
        if idx in self._lbl_cache:
            return self._lbl_cache[idx].copy()
        return self._load_labels_raw(idx)

    def _letterbox(self, img, target_size):
        """Resize with aspect ratio preserved, pad with gray."""
        h0, w0 = img.shape[:2]
        ratio = min(target_size / h0, target_size / w0)
        nh, nw = int(h0 * ratio), int(w0 * ratio)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        top = (target_size - nh) // 2
        left = (target_size - nw) // 2
        canvas[top:top+nh, left:left+nw] = img_resized

        return canvas, ratio, top, left

    def _resize_with_labels(self, img, labels, target_size):
        """Letterbox resize + adjust labels."""
        h0, w0 = img.shape[:2]
        img_lb, ratio, top, left = self._letterbox(img, target_size)

        if len(labels) > 0:
            # Convert normalized coords to letterboxed coords
            labels = labels.copy()
            # cx, cy in original normalized -> pixel -> letterbox -> normalized
            labels[:, 1] = (labels[:, 1] * w0 * ratio + left) / target_size
            labels[:, 2] = (labels[:, 2] * h0 * ratio + top) / target_size
            labels[:, 3] = labels[:, 3] * w0 * ratio / target_size
            labels[:, 4] = labels[:, 4] * h0 * ratio / target_size
            labels[:, 5] = (labels[:, 5] * w0 * ratio + left) / target_size
            labels[:, 6] = (labels[:, 6] * h0 * ratio + top) / target_size

        return img_lb, labels

    def _mosaic(self, idx):
        """Create mosaic of 4 random images."""
        s = self.img_size
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)

        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        all_labels = []
        result = np.full((s, s, 3), 114, dtype=np.uint8)

        for i, ci in enumerate(indices):
            img = self._load_image(ci)
            labels = self._load_labels(ci).copy()
            h0, w0 = img.shape[:2]

            # Resize image to target size first
            scale = s / max(h0, w0)
            if scale != 1:
                img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR)
                h0, w0 = img.shape[:2]

            if i == 0:    # top-left
                x1a, y1a, x2a, y2a = max(0, xc - w0), max(0, yc - h0), xc, yc
                x1b, y1b, x2b, y2b = w0 - (x2a - x1a), h0 - (y2a - y1a), w0, h0
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(0, yc - h0), min(s, xc + w0), yc
                x1b, y1b, x2b, y2b = 0, h0 - (y2a - y1a), x2a - x1a, h0
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(0, xc - w0), yc, xc, min(s, yc + h0)
                x1b, y1b, x2b, y2b = w0 - (x2a - x1a), 0, w0, y2a - y1a
            else:         # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(s, xc + w0), min(s, yc + h0)
                x1b, y1b, x2b, y2b = 0, 0, x2a - x1a, y2a - y1a

            crop_w = x2b - x1b
            crop_h = y2b - y1b
            if crop_w <= 0 or crop_h <= 0:
                continue

            result[y1a:y1a+crop_h, x1a:x1a+crop_w] = img[y1b:y1b+crop_h, x1b:x1b+crop_w]

            if len(labels) > 0:
                lx = labels[:, 1] * w0
                ly = labels[:, 2] * h0
                lw = labels[:, 3] * w0
                lh = labels[:, 4] * h0
                lfx = labels[:, 5] * w0
                lfy = labels[:, 6] * h0

                offset_x = x1a - x1b
                offset_y = y1a - y1b

                lx = lx + offset_x
                ly = ly + offset_y
                lfx = lfx + offset_x
                lfy = lfy + offset_y

                labels[:, 1] = lx / s
                labels[:, 2] = ly / s
                labels[:, 3] = lw / s
                labels[:, 4] = lh / s
                labels[:, 5] = lfx / s
                labels[:, 6] = lfy / s

                valid = (labels[:, 1] > 0.01) & (labels[:, 1] < 0.99) & \
                        (labels[:, 2] > 0.01) & (labels[:, 2] < 0.99) & \
                        (labels[:, 3] > 0.005) & (labels[:, 4] > 0.005)
                labels = labels[valid]
                all_labels.append(labels)

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
            all_labels[:, 1] = np.clip(all_labels[:, 1], 0.001, 0.999)
            all_labels[:, 2] = np.clip(all_labels[:, 2], 0.001, 0.999)
            all_labels[:, 3] = np.clip(all_labels[:, 3], 0.001, 0.999)
            all_labels[:, 4] = np.clip(all_labels[:, 4], 0.001, 0.999)
            all_labels[:, 5] = np.clip(all_labels[:, 5], 0.0, 1.0)
            all_labels[:, 6] = np.clip(all_labels[:, 6], 0.0, 1.0)
        else:
            all_labels = np.zeros((0, 7), dtype=np.float32)

        return result, all_labels

    def _mixup(self, img1, labels1):
        """MixUp augmentation: blend two images and combine labels."""
        idx2 = random.randint(0, len(self) - 1)
        img2 = self._load_image(idx2)
        labels2 = self._load_labels(idx2)
        img2, labels2 = self._resize_with_labels(img2, labels2, self.img_size)

        # Random blend ratio
        alpha = random.uniform(0.4, 0.6)
        img = (img1.astype(np.float32) * alpha + img2.astype(np.float32) * (1 - alpha)).astype(np.uint8)

        # Combine labels
        if len(labels1) > 0 and len(labels2) > 0:
            labels = np.concatenate([labels1, labels2], axis=0)
        elif len(labels1) > 0:
            labels = labels1
        else:
            labels = labels2

        return img, labels

    def _color_jitter(self, img):
        """Random color augmentation."""
        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.5, 1.5)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img

    def _horizontal_flip(self, img, labels):
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
                labels[:, 5] = 1.0 - labels[:, 5]
        return img, labels

    def _random_affine(self, img, labels, degrees=5, translate=0.1, scale_range=(0.7, 1.3)):
        """Random affine transform: slight rotation + translation + scale."""
        h, w = img.shape[:2]

        # Random parameters
        angle = random.uniform(-degrees, degrees)
        s = random.uniform(*scale_range)
        tx = random.uniform(-translate, translate) * w
        ty = random.uniform(-translate, translate) * h

        # Rotation + scale matrix
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, s)
        M[0, 2] += tx
        M[1, 2] += ty

        img = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))

        if len(labels) > 0:
            # Transform label coordinates
            n = len(labels)
            # Get 4 corners of each box
            cx = labels[:, 1] * w
            cy = labels[:, 2] * h
            bw = labels[:, 3] * w
            bh = labels[:, 4] * h
            fx = labels[:, 5] * w
            fy = labels[:, 6] * h

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            # Transform corners
            corners = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ]).transpose(2, 0, 1).reshape(-1, 2)  # (n*4, 2)

            ones = np.ones((corners.shape[0], 1))
            corners_h = np.hstack([corners, ones])
            corners_t = corners_h @ M.T  # (n*4, 2)
            corners_t = corners_t.reshape(n, 4, 2)

            # New bounding boxes from transformed corners
            new_x1 = corners_t[:, :, 0].min(axis=1)
            new_y1 = corners_t[:, :, 1].min(axis=1)
            new_x2 = corners_t[:, :, 0].max(axis=1)
            new_y2 = corners_t[:, :, 1].max(axis=1)

            # Transform foot point
            foot_h = np.hstack([fx.reshape(-1, 1), fy.reshape(-1, 1), np.ones((n, 1))])
            foot_t = foot_h @ M.T

            # Back to normalized
            new_cx = ((new_x1 + new_x2) / 2) / w
            new_cy = ((new_y1 + new_y2) / 2) / h
            new_w = (new_x2 - new_x1) / w
            new_h = (new_y2 - new_y1) / h
            new_fx = foot_t[:, 0] / w
            new_fy = foot_t[:, 1] / h

            labels[:, 1] = new_cx
            labels[:, 2] = new_cy
            labels[:, 3] = new_w
            labels[:, 4] = new_h
            labels[:, 5] = new_fx
            labels[:, 6] = new_fy

            # Filter out-of-bounds and too small
            valid = (labels[:, 1] > 0.01) & (labels[:, 1] < 0.99) & \
                    (labels[:, 2] > 0.01) & (labels[:, 2] < 0.99) & \
                    (labels[:, 3] > 0.005) & (labels[:, 4] > 0.005)
            labels = labels[valid]

            if len(labels) > 0:
                labels[:, 1] = np.clip(labels[:, 1], 0.001, 0.999)
                labels[:, 2] = np.clip(labels[:, 2], 0.001, 0.999)
                labels[:, 3] = np.clip(labels[:, 3], 0.001, 0.999)
                labels[:, 4] = np.clip(labels[:, 4], 0.001, 0.999)
                labels[:, 5] = np.clip(labels[:, 5], 0.0, 1.0)
                labels[:, 6] = np.clip(labels[:, 6], 0.0, 1.0)

        return img, labels

    def __getitem__(self, idx):
        # Mosaic or standard loading
        if self.augment and random.random() < self.mosaic_prob:
            img, labels = self._mosaic(idx)
            # MixUp on top of mosaic sometimes
            if random.random() < self.mixup_prob:
                img, labels = self._mixup(img, labels)
        else:
            img = self._load_image(idx)
            labels = self._load_labels(idx)
            img, labels = self._resize_with_labels(img, labels, self.img_size)

        # Apply augmentations
        if self.augment:
            img, labels = self._horizontal_flip(img, labels)
            img, labels = self._random_affine(img, labels, degrees=5, translate=0.1, scale_range=(0.7, 1.3))
            img = self._color_jitter(img)

        # Ensure correct size
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Limit labels
        if len(labels) > self.max_labels:
            labels = labels[:self.max_labels]

        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 \
            else torch.zeros((0, 7), dtype=torch.float32)

        return img_tensor, labels_tensor, self.samples[idx][0]


def collate_fn(batch):
    """Custom collate: stack images, keep labels as list."""
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(labels), list(paths)
