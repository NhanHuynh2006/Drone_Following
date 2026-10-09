#!/usr/bin/env bash
# deploy_to_pi.sh — Đẩy HEF mới sang Pi rồi ĐO để chọn bản tốt nhất. Chạy khi có Pi.
#
# Không đoán bản nào tốt: đo cả bản cũ lẫn các bản mới trên cùng 92 khung bay thật
# (data/flight_val), cùng một đường decode, rồi so F1.
set -euo pipefail
cd "$(dirname "$0")/.."
# pi5.local qua mDNS thay vi IP cung: router cap DHCP nen IP doi (114 -> 115
# ngay 2026-08-28). Ghi de bang:  PI_HOST=raspi@<ip> bash <script>
PI="${PI_HOST:-raspi@pi5.local}"

echo "=== 1. Kiểm tra Pi ==="
ssh -o ConnectTimeout=8 "$PI" 'hostname; hailortcli fw-control identify 2>/dev/null | head -3' \
  || { echo "Không nối được $PI"; exit 1; }

echo
echo "=== 2. Đẩy HEF mới sang ==="
FILES=()
for f in pfdet_v17_640_coco_hailo8l.hef pfdet_v17_640_coco_c512_hailo8l.hef \
         pfdet_v17_640_coco_o4_hailo8l.hef; do
  [ -f "$f" ] && { scp "$f" "$PI:~/NhanHuynh/models/"; FILES+=("$f"); echo "  đẩy $f"; }
done
[ ${#FILES[@]} -gt 0 ] || { echo "Không có HEF nào để đẩy"; exit 1; }

echo
echo "=== 3. Dừng thứ đang giữ NPU ==="
# NPU chỉ một tiến trình dùng được. Lấy PID rồi kill theo PID — pkill -f tự khớp
# chính dòng lệnh của mình (bài học mục 8 PROJECT.md).
PIDS=$(ssh "$PI" 'pgrep -f "follow_drone/main\.py|hailo_gcs\.py" | tr "\n" " "' || true)
[ -n "${PIDS// /}" ] && { echo "  kill $PIDS"; ssh "$PI" "kill $PIDS" || true; sleep 3; }

echo
echo "=== 4. Đo từng HEF trên 92 khung bay thật ==="
for f in pfdet_v17_640_fair_hailo8l.hef "${FILES[@]}"; do
  echo
  echo "--- $f"
  ssh "$PI" "cd ~/NhanHuynh && ./run.sh py flight_sweep.py --hef models/$f" \
    2>&1 | tee "runs/pi_sweep_${f%.hef}.txt" | tail -14
done

echo
echo "=== 5. Bảng so sánh ==="
printf "  %-42s %s\n" "HEF" "F1 tốt nhất"
for t in runs/pi_sweep_*.txt; do
  printf "  %-42s %s\n" "$(basename "$t" .txt | sed 's/^pi_sweep_//')" \
     "$(grep -o 'conf tot nhat theo F1: .*' "$t" | tail -1)"
done
echo
echo "Chọn bản F1 cao nhất rồi sửa follow_drone/config.yaml -> model.hef, và"
echo "đặt model.conf_threshold bằng đúng conf của bản đó."
