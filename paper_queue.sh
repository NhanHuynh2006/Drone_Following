#!/bin/bash
# paper_queue.sh — Hàng chờ TỰ ĐỘNG train + eval toàn bộ pipeline paper.
# Tự resume khi crash/OOM (retry loop), chỉ 1 instance (flock), chạy được qua reboot (cron @reboot).
# Trạng thái: runs/paper_queue/  (queue.log, SUMMARY.txt, marker *_done, ALL_DONE)

set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue
mkdir -p "$Q"
SUMMARY=$Q/SUMMARY.txt

# chỉ 1 instance chạy cùng lúc (chống double-launch từ nohup + cron)
exec 9>/tmp/paper_queue.lock
flock -n 9 || { echo "[queue] đã có instance khác đang chạy, thoát"; exit 0; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/queue.log"; }

# train 1 stage với auto-resume khi crash (tối đa 80 lần)
train_stage() {
    local cfg=$1 run=$2 lg=$2.qlog
    for i in $(seq 1 80); do
        grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: ĐÃ XONG"; return 0; }
        local resume=""; [ -f "$run/last.pt" ] && resume="--resume $run/last.pt"
        log "$run: train (lần $i) $resume"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_v3.py --config "$cfg" $resume >> "$lg" 2>&1
        grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: ĐÃ XONG"; return 0; }
        log "$run: crash/thoát (code $?) — resume sau 20s"; sleep 20
    done
    log "$run: HẾT 80 retry — bỏ qua, sang stage sau"; return 1
}

# eval 1 model -> ghi vào SUMMARY
eval_stage() {
    local w=$1 tag=$2
    [ -f "$w" ] || { log "eval $tag: thiếu $w, bỏ"; return; }
    log "eval $tag (COCO + robustness)"
    { echo ""; echo "===== $tag ====="; echo "[weights] $w"; } >> "$SUMMARY"
    $PY run_eval_coco.py --weights "$w" --val-images data/visdrone/val/images \
        --val-labels data/visdrone/val/labels --img-size 640 --profile light --device cuda:0 2>&1 \
        | grep -E "\[COCO\] overall" >> "$SUMMARY"
    $PY eval_robustness.py --weights "$w" --val-images data/visdrone/val/images \
        --val-labels data/visdrone/val/labels --img-size 640 --device cuda:0 --batch 8 2>&1 \
        | grep -E "\[robust\]" >> "$SUMMARY"
}

log "========== BẮT ĐẦU HÀNG CHỜ PAPER =========="
[ -f "$SUMMARY" ] || echo "PAPER QUEUE SUMMARY — bắt đầu $(date)" > "$SUMMARY"

# --- 1) BASELINE (no-COCO) ---
train_stage configs/train_config_v17_light.yaml runs/train_v17_light && touch "$Q/1_light_done"

# --- 2) VIBRATION (finetune từ baseline best.pt) ---
if [ -f runs/train_v17_light/best.pt ]; then
    train_stage configs/train_config_v17_vib.yaml runs/train_v17_vib && touch "$Q/2_vib_done"
else
    log "vib: THIẾU runs/train_v17_light/best.pt -> bỏ"
fi

# --- 3) NMS-FREE (ablation) ---
train_stage configs/train_config_v17_nms_free.yaml runs/train_v17_nms_free && touch "$Q/3_nmsfree_done"

# --- 4) EVAL tất cả ---
log "========== EVAL TẤT CẢ =========="
eval_stage runs/train_v17_light/best.pt    "v17_BASELINE"
eval_stage runs/train_v17_vib/best.pt      "v17_VIBRATION"
eval_stage runs/train_v17_nms_free/best.pt "v17_NMSFREE"
# baseline robustness đã có trong eval_stage; thêm mốc baseline musgd_v3 để so
eval_stage runs/train_v15_light_musgd_v3/best.pt "v15_musgd_v3_(mốc so)"

log "========== HÀNG CHỜ HOÀN TẤT =========="
{ echo ""; echo "ALL DONE $(date)"; } >> "$SUMMARY"
touch "$Q/ALL_DONE"
