"""
Prepare a YOLO/RT-DETR-friendly dataset view from PFDet labels.

PFDet labels may store:
  class cx cy w h fx fy

Ultralytics detection training expects:
  class cx cy w h

This script mirrors the image tree and rewrites labels to the 5-column format.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def _iter_label_rows(src_label: Path):
    with src_label.open('r', encoding='utf-8') as f:
        for raw in f:
            parts = raw.strip().split()
            if len(parts) < 5:
                continue
            yield parts[:5]


def _safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src.resolve(), dst)
    except FileExistsError:
        pass


def convert_split(src_root: Path, dst_root: Path, split: str):
    src_images = src_root / split / 'images'
    src_labels = src_root / split / 'labels'
    dst_images = dst_root / split / 'images'
    dst_labels = dst_root / split / 'labels'

    if not src_images.is_dir() or not src_labels.is_dir():
        print(f"[SKIP] Missing split: {split}")
        return 0

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    converted = 0
    for label_path in sorted(src_labels.glob('*.txt')):
        image_path = None
        stem = label_path.stem
        for ext in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'):
            candidate = src_images / f'{stem}{ext}'
            if candidate.exists():
                image_path = candidate
                break

        if image_path is None:
            continue

        _safe_symlink(image_path, dst_images / image_path.name)
        out_label = dst_labels / label_path.name
        rows = [' '.join(parts) for parts in _iter_label_rows(label_path)]
        out_label.write_text('\n'.join(rows), encoding='utf-8')
        converted += 1

    print(f"[OK] {split}: converted {converted} labels")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Prepare an Ultralytics-compatible dataset view")
    parser.add_argument("--src", default="./data/visdrone", help="Source dataset root with PFDet labels")
    parser.add_argument("--dst", default="./data/visdrone_ultralytics", help="Destination dataset root")
    parser.add_argument(
        "--splits",
        nargs='+',
        default=['train', 'val'],
        help="Dataset splits to mirror",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    total = 0
    for split in args.splits:
        total += convert_split(src_root, dst_root, split)

    print(f"Prepared {total} mirrored samples under {dst_root}")


if __name__ == "__main__":
    main()
