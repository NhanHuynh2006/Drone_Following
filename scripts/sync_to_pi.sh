#!/usr/bin/env bash
# sync_to_pi.sh — Đẩy code bay + HEF từ laptop sang Pi 5. CHẠY TRÊN LAPTOP.
#
#   bash scripts/sync_to_pi.sh          # CHẠY THỬ — chỉ in ra sẽ đẩy gì, KHÔNG ghi gì
#   bash scripts/sync_to_pi.sh --go     # đẩy thật
#   PI_HOST=raspi@192.168.1.50 bash scripts/sync_to_pi.sh --go     # khi mDNS không ra
#
# KHÔNG đụng tới: run.sh, check_pi.sh, setup_pi.sh, param_check.py, venv/, flight_data/
# — mấy thứ đó chỉ sống trên Pi, không có bản trong repo. Ghi đè là hỏng.
set -euo pipefail
cd "$(dirname "$0")/.."

PI="${PI_HOST:-raspi@pi5.local}"
DST="${PI_DST:-NhanHuynh}"
# Đúng bản HEF mà follow_drone/config.yaml đang trỏ tới. Đổi config thì đổi cả dòng này.
HEF="pfdet_v17_640_coco_hailo8l.hef"

DRY="--dry-run"; MODE="CHẠY THỬ (không ghi gì)"
[ "${1:-}" = "--go" ] && { DRY=""; MODE="ĐẨY THẬT"; }

echo "=== sync_to_pi — $MODE ==="
echo "    đích: $PI:~/$DST/"
echo

echo "--- 0. Kiểm tra nối được Pi"
ssh -o ConnectTimeout=8 "$PI" 'hostname; echo -n "  IP: "; hostname -I' \
  || { echo "Không nối được $PI. Cắm Pi, cùng WiFi, hoặc đặt PI_HOST=raspi@<ip>."; exit 1; }

RS="rsync -avh --progress $DRY --exclude __pycache__ --exclude '*.pyc' --exclude .pytest_cache"

echo
echo "--- 1. Code vòng bám (follow_drone/) — KHÔNG kéo logs của Pi về"
eval $RS --exclude logs/ follow_drone/ "$PI:~/$DST/follow_drone/"

echo
echo "--- 2. Công cụ chạy trên Pi (GCS, thu data, mux, camera)"
eval $RS mavmux.py cam_ctl.py mav_telem.py hailo_gcs.py gcs.html \
        flight_record.py flight_sweep.py hailo_demo_cam.py \
        hailo_stream_web.py hailo_bench_latency.py "$PI:~/$DST/"

echo
echo "--- 3. Model HEF ($HEF)"
if [ -f "$HEF" ]; then
  eval $RS "$HEF" "$PI:~/$DST/models/"
else
  echo "  !! Không có $HEF ở laptop. Bỏ qua — Pi chắc đã có sẵn."
fi

echo
if [ -n "$DRY" ]; then
  echo "=== Đây mới là CHẠY THỬ. Đẩy thật:  bash scripts/sync_to_pi.sh --go"
else
  echo "=== Xong. Kiểm tra trên Pi:"
  echo "    ssh $PI 'cd ~/$DST && ./run.sh check'"
fi
