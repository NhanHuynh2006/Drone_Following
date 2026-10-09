#!/bin/bash
# followup2_queue.sh — thay followup_queue.sh (2026-07-26, sau reboot).
#
# Khac biet: 2 job pfdet chay SONG SONG (GPU chi dung 2.2/8.2 GB voi 1 job, util 56%).
# Script nay KHONG tu khoi dong job nao dang chay; no CHO ca hai xong roi moi do.
# Moi job co resume rieng neu crash. Chong trung: dung flock rieng cho tung job.
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue; mkdir -p "$Q"; LOG=$Q/followup2.log
exec 9>/tmp/followup2.lock; flock -n 9 || { echo "[fu2] dang chay, thoat"; exit 0; }
[ -f "$Q/FOLLOWUP_ALL_DONE" ] && { echo "[fu2] DA XONG TU TRUOC -> thoat"; exit 0; }
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# Dam bao 1 job pfdet chay den khi xong. flock rieng -> khong bao gio co 2 tien trinh
# cung ghi vao mot run dir, ke ca khi script nay chay 2 lan.
ensure(){ local cfg=$1 run=$2 lg=$3 lock=$4 wk=$5
  ( flock -n 8 || { log "$run: da co tien trinh khac lo, bo qua"; exit 0; }
    for i in $(seq 1 60); do
      grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: XONG"; exit 0; }
      # neu train dang chay (do minh khoi dong tay) thi CHO, khong khoi dong them
      if pgrep -f "train_v3.py --config $cfg" >/dev/null; then
        log "$run: dang chay san, cho..."
        while pgrep -f "train_v3.py --config $cfg" >/dev/null; do sleep 60; done
        continue
      fi
      local r=""; [ -f "$run/last.pt" ] && r="--resume $run/last.pt"
      log "$run: train (lan $i) $r"
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_v3.py --config "$cfg" $r >> "$lg" 2>&1
      grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: XONG"; exit 0; }
      log "$run: crash - resume sau 20s"; sleep 20
    done ) 8>"$lock"
}

log "===== FOLLOWUP2 (2 job pfdet SONG SONG) ====="
ensure configs/train_config_v17_augonly_deploy.yaml runs/train_v17_augonly_deploy \
       runs/train_v17_augonly_deploy.qlog /tmp/fu2_deploy.lock 8 &
ensure configs/train_config_v15_augonly.yaml runs/train_v15_augonly \
       runs/train_v15_augonly.qlog /tmp/fu2_wide.lock 4 &
wait
touch "$Q/F1_deploy_augonly_done" "$Q/F3_wide_augonly_done"
[ -f runs/yolo/yolo11n_augonly/weights/best.pt ] && touch "$Q/F2_yolo_augonly_done"

log "eval hop nhat (chi do model MOI)"
$PY paper_eval_unified.py --seeds 0 1 2 --out "$Q/UNIFIED.json" >> "$Q/unified_eval.log" 2>&1
log "eval tren flight_val THAT"
{ echo ""; echo "===== FLIGHT_VAL: augment-only vs consistency (2026-07-26) ====="; } >> "$Q/SUMMARY_DEPLOY.txt"
for w in runs/train_v17_augonly_deploy/best.pt runs/train_v17_vib_deploy/best.pt runs/train_v17_deploy/best.pt; do
  [ -f "$w" ] || continue
  { echo ""; echo "--- $w ---"; } >> "$Q/SUMMARY_DEPLOY.txt"
  $PY run_eval_coco.py --weights "$w" --val-images data/flight_val/images \
      --val-labels data/flight_val/labels --img-size 640 --profile light --device cuda:0 2>&1 \
      | grep -E "\[COCO\] overall" >> "$Q/SUMMARY_DEPLOY.txt"
done
$PY paper/make_tables.py > "$Q/TABLES.tex" 2>&1
touch "$Q/FOLLOWUP_ALL_DONE"; log "===== XONG HET ====="
