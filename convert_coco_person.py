"""
Convert COCO val2017 annotations to YOLO format (person class only).
Output: data/coco_person/images/ and data/coco_person/labels/
"""

import json
import os
import shutil

# Paths
COCO_IMG_DIR = "val2017"
COCO_ANN_FILE = "annotations/instances_val2017.json"
OUT_IMG_DIR = "data/coco_person/images"
OUT_LBL_DIR = "data/coco_person/labels"

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Load COCO annotations
print("Loading COCO annotations...")
with open(COCO_ANN_FILE, 'r') as f:
    coco = json.load(f)

# Find person category ID
person_cat_id = None
for cat in coco['categories']:
    if cat['name'] == 'person':
        person_cat_id = cat['id']
        break
print(f"Person category ID: {person_cat_id}")

# Build image info lookup
img_info = {img['id']: img for img in coco['images']}

# Group person annotations by image
from collections import defaultdict
img_anns = defaultdict(list)
for ann in coco['annotations']:
    if ann['category_id'] == person_cat_id and ann['area'] > 100:  # skip tiny
        img_anns[ann['image_id']].append(ann)

print(f"Images with person annotations: {len(img_anns)}")

# Convert and copy
n_images = 0
n_boxes = 0
for img_id, anns in img_anns.items():
    info = img_info[img_id]
    w_img = info['width']
    h_img = info['height']
    fname = info['file_name']

    src_path = os.path.join(COCO_IMG_DIR, fname)
    if not os.path.isfile(src_path):
        continue

    # Convert annotations to YOLO format
    lines = []
    for ann in anns:
        if ann.get('iscrowd', 0):
            continue
        x, y, bw, bh = ann['bbox']  # COCO format: x_topleft, y_topleft, w, h

        # To YOLO: cx, cy, w, h (normalized)
        cx = (x + bw / 2) / w_img
        cy = (y + bh / 2) / h_img
        nw = bw / w_img
        nh = bh / h_img

        # Skip invalid
        if nw < 0.005 or nh < 0.005 or cx <= 0 or cy <= 0:
            continue

        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        nw = max(0.001, min(0.999, nw))
        nh = max(0.001, min(0.999, nh))

        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        n_boxes += 1

    if len(lines) == 0:
        continue

    # Copy image
    dst_img = os.path.join(OUT_IMG_DIR, fname)
    if not os.path.exists(dst_img):
        shutil.copy2(src_path, dst_img)

    # Write label
    stem = os.path.splitext(fname)[0]
    lbl_path = os.path.join(OUT_LBL_DIR, stem + '.txt')
    with open(lbl_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    n_images += 1

print(f"Done! {n_images} images, {n_boxes} person boxes")
print(f"Images: {OUT_IMG_DIR}")
print(f"Labels: {OUT_LBL_DIR}")
