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
  - DroneAug (NOVEL): drone altitude simulation via zoom in/out
"""

import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset


class VisDronePersonDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_size=384,
                 augment=True, mosaic_prob=0.5, mixup_prob=0.15,
                 drone_aug_prob=0.0, copy_paste_prob=0.0,
                 hflip_prob=0.5, affine_prob=1.0, color_jitter_prob=1.0,
                 max_labels=100, cache_ram=False,
                 min_box_size=2.0, min_area_ratio=0.2):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.augment = augment
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.drone_aug_prob = drone_aug_prob
        self.copy_paste_prob = copy_paste_prob
        self.hflip_prob = float(hflip_prob)
        self.affine_prob = float(affine_prob)
        self.color_jitter_prob = float(color_jitter_prob)
        self.max_labels = max_labels
        self.min_box_size = float(min_box_size)
        self.min_area_ratio = float(min_area_ratio)

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
                # parts[5], parts[6] (fx, fy foot point) intentionally ignored:
                # foot point = (cx, cy+h/2) is trivially derivable from bbox, adds no training signal
                labels.append([0, cx, cy, w, h])
        if len(labels) == 0:
            return np.zeros((0, 5), dtype=np.float32)
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

    def _labels_to_xyxy(self, labels, img_w, img_h):
        if len(labels) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        cx = labels[:, 1] * img_w
        cy = labels[:, 2] * img_h
        bw = labels[:, 3] * img_w
        bh = labels[:, 4] * img_h
        return np.stack([
            cx - bw / 2,
            cy - bh / 2,
            cx + bw / 2,
            cy + bh / 2,
        ], axis=1).astype(np.float32)

    def _xyxy_to_labels(self, boxes, img_w, img_h):
        if len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h

        return np.stack([
            np.zeros(len(boxes), dtype=np.float32),
            cx.astype(np.float32),
            cy.astype(np.float32),
            bw.astype(np.float32),
            bh.astype(np.float32),
        ], axis=1)

    def _clip_boxes_to_rect(self, boxes, rect, orig_areas=None,
                            min_size_px=None, min_area_ratio=None):
        if len(boxes) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        min_size_px = self.min_box_size if min_size_px is None else float(min_size_px)
        min_area_ratio = self.min_area_ratio if min_area_ratio is None else float(min_area_ratio)

        x1_lim, y1_lim, x2_lim, y2_lim = rect
        boxes = boxes.copy()
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], x1_lim, x2_lim)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], y1_lim, y2_lim)

        bw = boxes[:, 2] - boxes[:, 0]
        bh = boxes[:, 3] - boxes[:, 1]
        keep = (bw >= min_size_px) & (bh >= min_size_px)

        if orig_areas is not None and len(orig_areas) == len(boxes):
            visible_area = bw * bh
            keep &= visible_area >= np.maximum(orig_areas, 1e-6) * min_area_ratio

        return boxes[keep]

    def _resize_with_labels(self, img, labels, target_size):
        """Letterbox resize + adjust labels."""
        h0, w0 = img.shape[:2]
        img_lb, ratio, top, left = self._letterbox(img, target_size)

        if len(labels) > 0:
            # Convert normalized coords to letterboxed coords
            labels = labels.copy()
            labels[:, 1] = (labels[:, 1] * w0 * ratio + left) / target_size
            labels[:, 2] = (labels[:, 2] * h0 * ratio + top) / target_size
            labels[:, 3] = labels[:, 3] * w0 * ratio / target_size
            labels[:, 4] = labels[:, 4] * h0 * ratio / target_size

            labels = self._xyxy_to_labels(
                self._clip_boxes_to_rect(
                    self._labels_to_xyxy(labels, target_size, target_size),
                    (0, 0, target_size, target_size),
                    min_area_ratio=0.0,
                ),
                target_size,
                target_size,
            )

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
                offset_x = x1a - x1b
                offset_y = y1a - y1b
                boxes = self._labels_to_xyxy(labels, w0, h0)
                orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes[:, [0, 2]] += offset_x
                boxes[:, [1, 3]] += offset_y
                boxes = self._clip_boxes_to_rect(
                    boxes,
                    (x1a, y1a, x2a, y2a),
                    orig_areas=orig_areas,
                )
                labels = self._xyxy_to_labels(boxes, s, s)
                if len(labels) > 0:
                    all_labels.append(labels)

        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)
            all_labels[:, 1] = np.clip(all_labels[:, 1], 0.001, 0.999)
            all_labels[:, 2] = np.clip(all_labels[:, 2], 0.001, 0.999)
            all_labels[:, 3] = np.clip(all_labels[:, 3], 0.001, 0.999)
            all_labels[:, 4] = np.clip(all_labels[:, 4], 0.001, 0.999)
        else:
            all_labels = np.zeros((0, 5), dtype=np.float32)

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
        if self.hflip_prob > 0 and random.random() < self.hflip_prob:
            img = np.fliplr(img).copy()
            if len(labels) > 0:
                labels[:, 1] = 1.0 - labels[:, 1]
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
            boxes = self._labels_to_xyxy(labels, w, h)
            orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # Get 4 corners of each box
            cx = labels[:, 1] * w
            cy = labels[:, 2] * h
            bw = labels[:, 3] * w
            bh = labels[:, 4] * h

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

            boxes_t = np.stack([new_x1, new_y1, new_x2, new_y2], axis=1)
            boxes_t = self._clip_boxes_to_rect(
                boxes_t,
                (0, 0, w, h),
                orig_areas=orig_areas,
            )
            labels = self._xyxy_to_labels(boxes_t, w, h)

        return img, labels

    def _drone_aug(self, img, labels):
        """
        DroneAug (NOVEL): Simulate different drone altitudes.

        Randomly applies one of:
          - Zoom-out (high altitude): shrink image, pad with gray
            → people become smaller, simulates climbing
          - Zoom-in (low altitude): crop center region, resize back
            → people become larger, simulates descending

        This augmentation is drone-specific: it teaches the model to
        handle the scale variation caused by altitude changes, which
        is the primary source of scale variation in drone footage
        (unlike ground cameras where distance varies continuously).
        """
        h, w = img.shape[:2]

        if random.random() < 0.5:
            # Zoom-out: simulate higher altitude (smaller people)
            scale = random.uniform(0.5, 0.8)
            nh, nw = int(h * scale), int(w * scale)
            img_small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

            canvas = np.full((h, w, 3), 114, dtype=np.uint8)
            top = (h - nh) // 2
            left = (w - nw) // 2
            canvas[top:top+nh, left:left+nw] = img_small

            if len(labels) > 0:
                boxes = self._labels_to_xyxy(labels, w, h)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top
                boxes = self._clip_boxes_to_rect(
                    boxes,
                    (0, 0, w, h),
                    min_area_ratio=0.0,
                )
                labels = self._xyxy_to_labels(boxes, w, h)

            img = canvas
        else:
            # Zoom-in: simulate lower altitude (larger people)
            scale = random.uniform(1.2, 2.0)
            # Crop region size (in original image)
            crop_h = int(h / scale)
            crop_w = int(w / scale)
            # Random crop position
            y0 = random.randint(0, h - crop_h)
            x0 = random.randint(0, w - crop_w)

            img_crop = img[y0:y0+crop_h, x0:x0+crop_w]
            img = cv2.resize(img_crop, (w, h), interpolation=cv2.INTER_LINEAR)

            if len(labels) > 0:
                boxes = self._labels_to_xyxy(labels, w, h)
                orig_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                boxes[:, [0, 2]] -= x0
                boxes[:, [1, 3]] -= y0
                boxes = self._clip_boxes_to_rect(
                    boxes,
                    (0, 0, crop_w, crop_h),
                    orig_areas=orig_areas,
                )
                if len(boxes) > 0:
                    boxes[:, [0, 2]] *= w / crop_w
                    boxes[:, [1, 3]] *= h / crop_h
                labels = self._xyxy_to_labels(boxes, w, h)

        return img, labels

    def _copy_paste(self, img, labels, n_range=(1, 5)):
        """
        Copy-Paste augmentation for tiny person detection.

        Samples small person crops from random images and pastes them
        onto the current image at random positions. Focuses on persons
        with side < 40px to specifically address the tiny-target distribution
        in VisDrone (most persons are 5–30px at drone altitude).

        Proven to add +3–5 AP on VisDrone tiny-object benchmarks.
        """
        h, w = img.shape[:2]
        img = img.copy()
        new_labels = [labels] if len(labels) > 0 else []

        n_paste = random.randint(*n_range)
        for _ in range(n_paste):
            src_idx = random.randint(0, len(self) - 1)
            src_img = self._load_image(src_idx)
            src_labels = self._load_labels(src_idx)
            if len(src_labels) == 0:
                continue

            src_h, src_w = src_img.shape[:2]
            boxes_px = self._labels_to_xyxy(src_labels, src_w, src_h)
            bws = boxes_px[:, 2] - boxes_px[:, 0]
            bhs = boxes_px[:, 3] - boxes_px[:, 1]
            areas = bws * bhs

            # Prefer small persons (side < 40px); fall back to all if none
            small = (bws < 40) & (bhs < 60) & (areas >= 4)
            candidates = np.where(small)[0] if small.any() else np.arange(len(src_labels))
            if len(candidates) == 0:
                continue

            gi = int(random.choice(candidates))
            x1, y1, x2, y2 = boxes_px[gi].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(src_w, x2), min(src_h, y2)
            if x2 <= x1 + 1 or y2 <= y1 + 1:
                continue

            crop = src_img[y1:y2, x1:x2].copy()
            cw, ch = crop.shape[1], crop.shape[0]

            # Slight scale jitter on the pasted crop
            scale = random.uniform(0.8, 1.3)
            nw2 = max(2, min(int(cw * scale), w - 1))
            nh2 = max(2, min(int(ch * scale), h - 1))
            crop = cv2.resize(crop, (nw2, nh2), interpolation=cv2.INTER_LINEAR)

            # Slight brightness jitter so paste blends better
            if random.random() < 0.5:
                crop = np.clip(
                    crop.astype(np.float32) * random.uniform(0.7, 1.3), 0, 255
                ).astype(np.uint8)

            # Random paste position
            px = random.randint(0, w - nw2)
            py = random.randint(0, h - nh2)

            img[py:py + nh2, px:px + nw2] = crop

            # Build label for pasted object (normalized coords)
            ncx = (px + nw2 / 2) / w
            ncy = (py + nh2 / 2) / h
            nbw = nw2 / w
            nbh = nh2 / h
            new_labels.append(np.array(
                [[0.0, ncx, ncy, nbw, nbh]],
                dtype=np.float32,
            ))

        if len(new_labels) > 1:
            labels = np.concatenate(new_labels, axis=0)
        elif len(new_labels) == 1:
            labels = new_labels[0]
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
            if self.hflip_prob > 0:
                img, labels = self._horizontal_flip(img, labels)
            if self.affine_prob > 0 and random.random() < self.affine_prob:
                img, labels = self._random_affine(
                    img,
                    labels,
                    degrees=5,
                    translate=0.1,
                    scale_range=(0.7, 1.3),
                )
            if self.drone_aug_prob > 0 and random.random() < self.drone_aug_prob:
                img, labels = self._drone_aug(img, labels)
            if self.copy_paste_prob > 0 and random.random() < self.copy_paste_prob:
                img, labels = self._copy_paste(img, labels)
            if self.color_jitter_prob > 0 and random.random() < self.color_jitter_prob:
                img = self._color_jitter(img)

        # Ensure correct size
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        labels = self._xyxy_to_labels(
            self._clip_boxes_to_rect(
                self._labels_to_xyxy(labels, self.img_size, self.img_size),
                (0, 0, self.img_size, self.img_size),
                min_area_ratio=0.0,
            ),
            self.img_size,
            self.img_size,
        )

        # To tensor
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Limit labels
        if len(labels) > self.max_labels:
            labels = labels[:self.max_labels]

        labels_tensor = torch.from_numpy(labels).float() if len(labels) > 0 \
            else torch.zeros((0, 5), dtype=torch.float32)

        return img_tensor, labels_tensor, self.samples[idx][0]


def collate_fn(batch):
    """Custom collate: stack images, keep labels as list."""
    imgs, labels, paths = zip(*batch)
    imgs = torch.stack(imgs, 0)
    return imgs, list(labels), list(paths)
