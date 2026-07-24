#!/bin/bash
# deploy_queue.sh — PHASE 2 (nhánh DEPLOY/pivot). CHỜ paper_queue no-COCO xong HẾT rồi mới chạy.
# KHÔNG ngắt job đang chạy: block tới khi có marker FINAL_ALL_DONE.
# Việc: (1) vib finetune từ deploy(+COCO)  (2) 6 YOLO baseline trên COCO-mix  (3) eval tất cả trên flight_val THẬT.
# Idiom giống paper_queue.sh: retry-safe, flock 1 instance, cron@reboot được.
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue
mkdir -p "$Q"
SUMMARY=$Q/SUMMARY_DEPLOY.txt
FVAL_I=data/flight_val/images
FVAL_L=data/flight_val/labels

exec 7>/tmp/deploy_queue.lock
flock -n 7 || { echo "[deploy_queue] instance khác đang chạy, thoát"; exit 0; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$Q/deploy_queue.log"; }

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
    log "$run: HẾT 80 retry — bỏ qua"; return 1
}

# eval 1 model pfdet trên flight_val THẬT -> SUMMARY_DEPLOY
eval_flight() {
    local w=$1 tag=$2
    [ -f "$w" ] || { log "eval $tag: thiếu $w, bỏ"; return; }
    log "eval $tag trên flight_val"
    { echo ""; echo "===== $tag (flight_val THẬT @640) ====="; echo "[weights] $w"; } >> "$SUMMARY"
    $PY run_eval_coco.py --weights "$w" --val-images "$FVAL_I" --val-labels "$FVAL_L" \
        --img-size 640 --profile light --device cuda:0 2>&1 | grep -E "\[COCO\] overall" >> "$SUMMARY"
}

log "========== DEPLOY QUEUE: CHỜ paper_queue no-COCO xong (FINAL_ALL_DONE) =========="
while [ ! -f "$Q/FINAL_ALL_DONE" ]; do sleep 300; done
log "========== no-COCO XONG -> BẮT ĐẦU PHASE 2 (DEPLOY) =========="
[ -f "$SUMMARY" ] || echo "SUMMARY DEPLOY (phase 2) — bắt đầu $(date)" > "$SUMMARY"

# --- 1) VIBRATION nhánh deploy (finetune từ deploy best.pt + COCO) ---
if [ -f runs/train_v17_deploy/best.pt ]; then
    train_stage configs/train_config_v17_vib_deploy.yaml runs/train_v17_vib_deploy && touch "$Q/D2_vib_deploy_done"
else
    log "vib_deploy: THIẾU runs/train_v17_deploy/best.pt -> bỏ"
fi

# --- 2) YOLO baselines trên COCO-mix (retry-safe, model lỗi không chặn) ---
for i in $(seq 1 40); do
    [ -f "$Q/BASELINES_DEPLOY_DONE" ] && { log "baselines_deploy: XONG"; break; }
    log "baselines_deploy: train (lần $i)"
    $PY train_yolo_baselines_deploy.py >> "$Q/yolo_deploy.qlog" 2>&1
    sleep 20
done
touch "$Q/BASELINES_DEPLOY_DONE"

# --- 3) EVAL tất cả model pfdet trên flight_val THẬT (baselines đã tự eval trong python) ---
log "========== EVAL pfdet trên flight_val =========="
eval_flight runs/train_v17_deploy/best.pt      "v17_DEPLOY(+COCO)"
eval_flight runs/train_v17_light/best.pt       "v17_LIGHT(no-COCO)"
eval_flight runs/train_v17_vib_deploy/best.pt  "v17_VIB_DEPLOY(+COCO+vib)"
eval_flight runs/train_v17_vib/best.pt         "v17_VIB(no-COCO)"

log "========== PHASE 2 (DEPLOY) HOÀN TẤT =========="
{ echo ""; echo "DEPLOY ALL DONE $(date)"; } >> "$SUMMARY"
touch "$Q/DEPLOY_ALL_DONE"
