#!/bin/bash
# fix_queue.sh — PHASE 3 (sửa lỗi). CHỜ deploy_queue xong (DEPLOY_ALL_DONE) rồi:
#   (1) train lại RT-DETR phase-1 bằng AdamW lr=1e-4 (bản cũ dùng MuSGD lr=0.01 -> sập, 21ep = số rác)
#   (2) tính lại params/GFLOPs/mAP cho MỌI baseline (bản cũ ghi 0.00M do model.info(verbose=False)=None)
# Chỉ chờ, KHÔNG ngắt job đang chạy. flock 1 instance, cron@reboot được.
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue
mkdir -p "$Q"

exec 6>/tmp/fix_queue.lock
flock -n 6 || { echo "[fix_queue] instance khác đang chạy, thoát"; exit 0; }

# GUARD (thêm 2026-07-25): đã xong hết thì THOÁT NGAY.
# Không có dòng này, cron @reboot sẽ chạy lại phần EVAL (ăn GPU) và tranh với
# ablation_queue.sh đang train. Marker đã có nghĩa là không còn gì phải làm.
[ -f "$Q/FIX_ALL_DONE" ] && { echo "[fix_queue] ĐÃ XONG HẾT TỪ TRƯỚC (FIX_ALL_DONE) -> thoát"; exit 0; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/fix_queue.log"; }

log "========== FIX QUEUE: CHỜ deploy_queue xong (DEPLOY_ALL_DONE) =========="
while [ ! -f "$Q/DEPLOY_ALL_DONE" ]; do sleep 300; done
log "========== BẮT ĐẦU PHASE 3 (FIX) =========="

# --- 1) RT-DETR phase-1 (chỉ còn model này thiếu PAPER_DONE -> script tự bỏ qua 4 con kia) ---
for i in $(seq 1 20); do
    [ -f runs/yolo/rtdetr-l/PAPER_DONE ] && { log "rtdetr-l phase-1: XONG"; break; }
    log "rtdetr-l phase-1: train (lần $i) — AdamW lr=1e-4, patience=50"
    $PY train_yolo_baselines.py >> "$Q/yolo.qlog" 2>&1
    sleep 20
done

# --- 2) Tính lại params/GFLOPs/mAP cho mọi baseline (cả 2 nhánh) ---
log "tính lại bảng baseline (params/GFLOPs/mAP)"
$PY summarize_baselines.py >> "$Q/summarize.log" 2>&1
log "bảng sạch -> $Q/BASELINES_TABLE.md"

log "========== PHASE 3 (FIX) HOÀN TẤT =========="
touch "$Q/FIX_ALL_DONE"
