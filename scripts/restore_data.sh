#!/usr/bin/env bash
# restore_data.sh — Dựng lại data/visdrone + data/coco_person từ bộ gốc công khai.
#
# Vì sao cần: data/ nằm trong .gitignore nên không có trong git. Bị xoá là mất.
# Nhưng cả hai đều sinh lại được, và tham số convert đã được XÁC MINH bằng cách
# đối chiếu với bản val còn sót trên Pi: 548/548 ảnh khớp tuyệt đối, 8165/8165 hộp.
#
#   VisDrone : --include-people --min-size 10   (categories {1,2}, occlusion < 2)
#   COCO     : person, area > 100, bỏ iscrowd
#
# Dùng --copy cho VisDrone: bản gốc dùng symlink, và xoá thư mục nguồn là hỏng cả
# bộ — đúng cách nó đã hỏng lần trước.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=data/_raw
PY="${PY:-$HOME/.venvs/pfdet/bin/python}"
[ -x "$PY" ] || PY=python3

echo "=== 1. Giải nén ==="
for z in VisDrone2019-DET-train VisDrone2019-DET-val; do
  if [ -s "$RAW/$z.zip" ] && [ ! -d "$RAW/${z}/annotations" ]; then
    echo "  $z.zip"; unzip -q -o "$RAW/$z.zip" -d "$RAW/vd_tmp"
    # zip có thể lồng 1 cấp; gom về $RAW/<tên>
    src=$(find "$RAW/vd_tmp" -maxdepth 2 -type d -name "$z" | head -1)
    [ -n "$src" ] && rm -rf "$RAW/$z" && mv "$src" "$RAW/$z"
    rm -rf "$RAW/vd_tmp"
  fi
done
[ -s "$RAW/val2017.zip" ] && [ ! -d "$RAW/val2017" ] && \
  { echo "  val2017.zip"; unzip -q -o "$RAW/val2017.zip" -d "$RAW"; }
[ -s "$RAW/annotations_trainval2017.zip" ] && [ ! -f "$RAW/annotations/instances_val2017.json" ] && \
  { echo "  annotations_trainval2017.zip"; unzip -q -o "$RAW/annotations_trainval2017.zip" -d "$RAW"; }

echo
echo "=== 2. Convert VisDrone (train + val) ==="
mkdir -p "$RAW/vd_root"
for s in VisDrone2019-DET-train VisDrone2019-DET-val; do
  [ -d "$RAW/$s" ] && ln -sfn "../../../$RAW/$s" "$RAW/vd_root/$s" 2>/dev/null || true
  [ -d "$RAW/$s" ] && { rm -rf "$RAW/vd_root/$s"; cp -rl "$RAW/$s" "$RAW/vd_root/$s" 2>/dev/null || ln -sfn "$(realpath "$RAW/$s")" "$RAW/vd_root/$s"; }
done
"$PY" scripts/convert_visdrone.py --src "$RAW/vd_root" --dst data/visdrone \
      --include-people --min-size 10 --copy

echo
echo "=== 3. Convert COCO person ==="
if [ -d "$RAW/val2017" ] && [ -f "$RAW/annotations/instances_val2017.json" ]; then
  ( cd "$RAW" && "$PY" ../../convert_coco_person.py )
  # script ghi ra $RAW/data/coco_person -> chuyển về đúng chỗ
  if [ -d "$RAW/data/coco_person" ]; then
    rm -rf data/coco_person && mkdir -p data && mv "$RAW/data/coco_person" data/coco_person
    rmdir "$RAW/data" 2>/dev/null || true
  fi
else
  echo "  [BỎ QUA] chưa có val2017/ hoặc annotations/instances_val2017.json"
fi

echo
echo "=== 4. Kiểm tra ==="
chk() {  # thư mục  kỳ_vọng  tên
  n=$(ls "$1" 2>/dev/null | wc -l)
  if [ "$n" = "$2" ]; then echo "  OK   $3: $n"; else echo "  !!   $3: $n (kỳ vọng $2)"; fi
}
chk data/visdrone/train/images 6471 "VisDrone train ảnh"
chk data/visdrone/train/labels 6471 "VisDrone train nhãn"
chk data/visdrone/val/images    548 "VisDrone val ảnh"
chk data/visdrone/val/labels    548 "VisDrone val nhãn"
chk data/coco_person/images    2654 "COCO person ảnh"
chk data/coco_person/labels    2654 "COCO person nhãn"
echo
echo "  ảnh symlink còn sót (phải là 0): $(find data/visdrone data/coco_person -type l 2>/dev/null | wc -l)"
du -sh data/visdrone data/coco_person 2>/dev/null | sed 's/^/  /'
