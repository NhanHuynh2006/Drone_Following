"""
Convert VisDrone-DET annotations to YOLO format for person detection.

VisDrone annotation format:
  bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion

Object categories we care about:
  1 = pedestrian
  2 = person (people)

Output YOLO format:
  class_id cx cy w h fx fy
  All normalized to [0,1]
  fx, fy = foot point (bottom center of bbox)
"""

import os
import sys
import argparse
from glob import glob
from PIL import Image
from tqdm import tqdm


# VisDrone categories that map to "person"
PERSON_CATEGORIES = {1, 2}  # pedestrian, people


def convert_annotation(ann_path, img_path, out_path, min_size=10):
    """Convert a single VisDrone annotation file to YOLO format."""
    # Get image dimensions
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
    except Exception as e:
        print(f"  [WARN] Cannot read image {img_path}: {e}")
        return 0

    lines = []
    n_person = 0

    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 8:
                continue

            bbox_left = int(parts[0])
            bbox_top = int(parts[1])
            bbox_w = int(parts[2])
            bbox_h = int(parts[3])
            # score = int(parts[4])      # not used
            category = int(parts[5])
            # truncation = int(parts[6]) # not used
            occlusion = int(parts[7])

            # Filter: only person classes, not heavily occluded, reasonable size
            if category not in PERSON_CATEGORIES:
                continue
            if occlusion >= 2:  # 0=no occlusion, 1=partial, 2=heavy
                continue
            if bbox_w < min_size or bbox_h < min_size:
                continue

            # Convert to YOLO format (normalized cxywh)
            cx = (bbox_left + bbox_w / 2) / img_w
            cy = (bbox_top + bbox_h / 2) / img_h
            w = bbox_w / img_w
            h = bbox_h / img_h

            # Foot point: bottom center of bbox
            fx = cx
            fy = min(1.0, (bbox_top + bbox_h) / img_h)

            # Validate
            if cx <= 0 or cx >= 1 or cy <= 0 or cy >= 1:
                continue
            if w <= 0.001 or h <= 0.001 or w >= 1 or h >= 1:
                continue

            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {fx:.6f} {fy:.6f}")
            n_person += 1

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    return n_person


def convert_split(src_images, src_annotations, dst_images, dst_labels, min_size=10):
    """Convert a full split (train/val/test)."""
    os.makedirs(dst_images, exist_ok=True)
    os.makedirs(dst_labels, exist_ok=True)

    ann_files = sorted(glob(os.path.join(src_annotations, '*.txt')))
    print(f"Found {len(ann_files)} annotation files in {src_annotations}")

    total_persons = 0
    total_images = 0

    for ann_path in tqdm(ann_files, desc="Converting"):
        stem = os.path.splitext(os.path.basename(ann_path))[0]

        # Find corresponding image
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            candidate = os.path.join(src_images, stem + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break

        if img_path is None:
            continue

        # Create symlink or copy image
        dst_img = os.path.join(dst_images, os.path.basename(img_path))
        if not os.path.exists(dst_img):
            os.symlink(os.path.abspath(img_path), dst_img)

        # Convert annotation
        out_label = os.path.join(dst_labels, stem + '.txt')
        n = convert_annotation(ann_path, img_path, out_label, min_size=min_size)
        total_persons += n
        total_images += 1

    print(f"  Converted {total_images} images, {total_persons} person annotations")
    return total_images, total_persons


def main():
    parser = argparse.ArgumentParser(description="Convert VisDrone to YOLO format")
    parser.add_argument("--src", required=True, help="VisDrone dataset root (contains VisDrone2019-DET-*)")
    parser.add_argument("--dst", default="./data/visdrone", help="Output directory")
    parser.add_argument("--min-size", type=int, default=10, help="Minimum bbox side in pixels")
    args = parser.parse_args()

    print(f"Converting VisDrone dataset from {args.src} to {args.dst}")
    print(f"Minimum bbox size: {args.min_size}px")
    print()

    # Standard VisDrone directory structure
    splits = {
        'train': ('VisDrone2019-DET-train', 'VisDrone2019-DET-train'),
        'val': ('VisDrone2019-DET-val', 'VisDrone2019-DET-val'),
        'test': ('VisDrone2019-DET-test-dev', 'VisDrone2019-DET-test-dev'),
    }

    for split_name, (img_dir, ann_dir) in splits.items():
        src_images = os.path.join(args.src, img_dir, 'images')
        src_ann = os.path.join(args.src, ann_dir, 'annotations')

        if not os.path.isdir(src_images) or not os.path.isdir(src_ann):
            print(f"[SKIP] {split_name}: {src_images} or {src_ann} not found")
            continue

        print(f"\n--- Converting {split_name} ---")
        convert_split(
            src_images, src_ann,
            os.path.join(args.dst, split_name, 'images'),
            os.path.join(args.dst, split_name, 'labels'),
            min_size=args.min_size,
        )

    print(f"\nDone! Dataset saved to {args.dst}")
    print("You can now train with: python train.py --config configs/train_config.yaml")


if __name__ == "__main__":
    main()
