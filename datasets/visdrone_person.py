"""
VisDrone Person Dataset with strong augmentation.

Supports YOLO-format labels:
  Each line: class_id cx cy w h [fx fy]
  All values normalized to [0,1]

Augmentations (critical for UAV detection):
  - Mosaic 4-image composition (probability-based)
  - Random horizontal flip
  - Random scale jitter
  - Color jitter (brightness, contrast, saturation, hue)
  - Random crop with person preservation
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VisDronePersonDataset(Dataset):
    """
    Dataset for person detection training on VisDrone-style data.

    Directory structure:
      images_dir/  -> image files (.jpg, .png)
      labels_dir/  -> label files (.txt), same name as image

    Label format per line:
      class_id cx cy w h [fx fy]
      - class_id: 0 for person (other classes ignored)
      - cx, cy, w, h: bounding box center and size (normalized)
      - fx, fy: foot point (normalized), optional
    """
    def __init__(self, images_dir, labels_dir, img_size=416,
                 augment=True, mosaic_prob=0.5, max_labels=100):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
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

    def __len__(self):
        return len(self.samples)

    def _load_image(self, idx):
        """Load image as RGB numpy array."""
        img_path = self.samples[idx][0]
        img = cv2.imread(img_path)
        if img is None:
            # Fallback: create blank image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_labels(self, idx):
        """
        Load labels from txt file.
        Returns: (N, 7) array [cls, cx, cy, w, h, fx, fy]
        """
        lbl_path = self.samples[idx][1]
        labels = []
        with open(lbl_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                if cls_id != 0:  # only person class
                    continue
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Foot point (default: bottom center of bbox)
                if len(parts) >= 7:
                    fx, fy = float(parts[5]), float(parts[6])
                else:
                    fx = cx
                    fy = min(1.0, cy + h / 2)

                labels.append([0, cx, cy, w, h, fx, fy])

        if len(labels) == 0:
            return np.zeros((0, 7), dtype=np.float32)
        return np.array(labels, dtype=np.float32)

    def _resize_with_labels(self, img, labels, target_size):
        """Resize image and adjust labels (labels are already normalized)."""
        h0, w0 = img.shape[:2]
        img_resized = cv2.resize(img, (target_size, target_size),
                                  interpolation=cv2.INTER_LINEAR)
        return img_resized, labels  # labels stay normalized

    def _mosaic(self, idx):
        """
        Create mosaic of 4 random images.
        This is THE most effective augmentation for object detection.
        """
        s = self.img_size
        # Random center point for the mosaic
        yc = random.randint(s // 4, 3 * s // 4)
        xc = random.randint(s // 4, 3 * s // 4)

        indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        all_labels = []
        result = np.full((s, s, 3), 114, dtype=np.uint8)

        for i, ci in enumerate(indices):
            img = self._load_image(ci)
            labels = self._load_labels(ci).copy()
            h0, w0 = img.shape[:2]

            # Determine placement
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

            # Copy image region
            crop_w = x2b - x1b
            crop_h = y2b - y1b
            if crop_w <= 0 or crop_h <= 0:
                continue

            result[y1a:y1a+crop_h, x1a:x1a+crop_w] = img[y1b:y1b+crop_h, x1b:x1b+crop_w]

            # Transform labels to mosaic coordinates
            if len(labels) > 0:
                # Convert from normalized to pixel coords in original image
                lx = labels[:, 1] * w0
                ly = labels[:, 2] * h0
                lw = labels[:, 3] * w0
                lh = labels[:, 4] * h0
                lfx = labels[:, 5] * w0
                lfy = labels[:, 6] * h0

                # Offset to mosaic position
                offset_x = x1a - x1b
                offset_y = y1a - y1b

                lx = lx + offset_x
                ly = ly + offset_y
                lfx = lfx + offset_x
                lfy = lfy + offset_y

                # Normalize to mosaic size
                labels[:, 1] = lx / s
                labels[:, 2] = ly / s
                labels[:, 3] = lw / s
                labels[:, 4] = lh / s
                labels[:, 5] = lfx / s
                labels[:, 6] = lfy / s

                # Filter: keep labels whose center is inside the mosaic
                valid = (labels[:, 1] > 0.01) & (labels[:, 1] < 0.99) & \
                        (labels[:, 2] > 0.01) & (labels[:, 2] < 0.99) & \
                        (labels[:, 3] > 0.005) & (labels[:, 4] > 0.005)
                labels = labels[valid]

                all_labels.append(labels)

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
            # Clip
            all_labels[:, 1] = np.clip(all_labels[:, 1], 0.001, 0.999)
            all_labels[:, 2] = np.clip(all_labels[:, 2], 0.001, 0.999)
            all_labels[:, 3] = np.clip(all_labels[:, 3], 0.001, 0.999)
            all_labels[:, 4] = np.clip(all_labels[:, 4], 0.001, 0.999)
            all_labels[:, 5] = np.clip(all_labels[:, 5], 0.0, 1.0)
            all_labels[:, 6] = np.clip(all_labels[:, 6], 0.0, 1.0)
        else:
            all_labels = np.zeros((0, 7), dtype=np.float32)

        return result, all_labels

    def _color_jitter(self, img):
        """Random color augmentation."""
        # Brightness
        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # Contrast
        if random.random() < 0.5:
            factor = random.uniform(0.6, 1.4)
            mean = img.mean()
            img = np.clip((img.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

        # Saturation (convert to HSV)
        if random.random() < 0.3:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= random.uniform(0.5, 1.5)
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return img

    def _horizontal_flip(self, img, labels):
        """Random horizontal flip with label adjustment."""
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]  # cx
                labels[:, 5] = 1.0 - labels[:, 5]  # fx
        return img, labels

    def _random_scale(self, img, labels):
        """Random scale jitter."""
        scale = random.uniform(0.7, 1.3)
        h, w = img.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to original size
        if nh > h:
            y_off = random.randint(0, nh - h)
            x_off = random.randint(0, nw - w)
            img = img[y_off:y_off+h, x_off:x_off+w]

            if len(labels) > 0:
                # Adjust labels
                labels[:, 1] = (labels[:, 1] * nw - x_off) / w
                labels[:, 2] = (labels[:, 2] * nh - y_off) / h
                labels[:, 3] = labels[:, 3] * nw / w
                labels[:, 4] = labels[:, 4] * nh / h
                labels[:, 5] = (labels[:, 5] * nw - x_off) / w
                labels[:, 6] = (labels[:, 6] * nh - y_off) / h

                # Filter out-of-bounds
                valid = (labels[:, 1] > 0.01) & (labels[:, 1] < 0.99) & \
                        (labels[:, 2] > 0.01) & (labels[:, 2] < 0.99)
                labels = labels[valid]
        else:
            result = np.full((h, w, 3), 114, dtype=np.uint8)
            y_off = (h - nh) // 2
            x_off = (w - nw) // 2
            result[y_off:y_off+nh, x_off:x_off+nw] = img
            img = result

            if len(labels) > 0:
                labels[:, 1] = (labels[:, 1] * nw + x_off) / w
                labels[:, 2] = (labels[:, 2] * nh + y_off) / h
                labels[:, 3] = labels[:, 3] * nw / w
                labels[:, 4] = labels[:, 4] * nh / h
                labels[:, 5] = (labels[:, 5] * nw + x_off) / w
                labels[:, 6] = (labels[:, 6] * nh + y_off) / h

        return img, labels

    def __getitem__(self, idx):
        # Mosaic or standard loading
        if self.augment and random.random() < self.mosaic_prob:
            img, labels = self._mosaic(idx)
        else:
            img = self._load_image(idx)
            labels = self._load_labels(idx)
            img, labels = self._resize_with_labels(img, labels, self.img_size)

        # Apply augmentations
        if self.augment:
            img, labels = self._horizontal_flip(img, labels)
            img, labels = self._random_scale(img, labels)
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
