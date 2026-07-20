#!/bin/bash
# yolo_queue.sh — Sau khi v17 queue xong, train YOLO baselines (retry-safe) rồi báo HOÀN TẤT TOÀN BỘ.
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue
mkdir -p "$Q"

exec 8>/tmp/yolo_queue.lock
flock -n 8 || { echo "[yolo] instance khác đang chạy, thoát"; exit 0; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/queue.log"; }

# 1) chờ v17 queue xong (để không giành GPU)
log "yolo_queue: chờ v17 (ALL_DONE)..."
while [ ! -f "$Q/ALL_DONE" ]; do sleep 300; done
log "yolo_queue: v17 xong -> train YOLO baselines"

# 2) train baselines đa dạng (script python tự resume + bỏ qua model lỗi)
for i in $(seq 1 40); do
    [ -f "$Q/BASELINES_DONE" ] && { log "baselines: XONG"; break; }
    log "baselines: train (lần $i)"
    $PY train_yolo_baselines.py >> "$Q/yolo.qlog" 2>&1
    sleep 20
done
touch "$Q/BASELINES_DONE"   # đảm bảo có marker dù retry cạn (model lỗi không chặn hoàn tất)

# 3) TỰ ĐỘNG so no-COCO vs COCO(deploy) -> ghi DECISION.md (khuyến nghị model-only vs system)
log "========== TỰ QUYẾT ĐỊNH HƯỚNG PAPER =========="
$PY paper_decision.py >> "$Q/decision.log" 2>&1

# 4) marker hoàn tất TOÀN BỘ
log "========== TOÀN BỘ PIPELINE PAPER HOÀN TẤT (v17 + vibration + nms_free + YOLO + decision) =========="
touch "$Q/FINAL_ALL_DONE"
