#!/usr/bin/env bash
# Vẽ lại hai ảnh nền của bản PowerPoint từ đúng mã TikZ của bản LaTeX.
# Chỉ cần chạy khi sửa hoạ tiết trong latex/beamerthemeISLab.sty.
set -euo pipefail
goc="$(cd "$(dirname "$0")/.." && pwd)"
tam="$(mktemp -d)"
cd "$goc/latex"
pdflatex -interaction=nonstopmode -output-directory "$tam" xuat_nen.tex >/dev/null
# 16 cm -> 3200 px
pdftoppm -r 508 -png -f 1 -l 1 -singlefile "$tam/xuat_nen.pdf" "$goc/assets/nen_bia"
pdftoppm -r 508 -png -f 2 -l 2 -singlefile "$tam/xuat_nen.pdf" "$goc/assets/nen_muc"
rm -rf "$tam"
echo "Đã ghi: assets/nen_bia.png, assets/nen_muc.png"
