#!/bin/bash
# followup_queue.sh — VONG 2 sau khi ablation lat nguoc claim (2026-07-26).
#
# Ket qua vong 1: augment-only 36.2% > vibration-consistency 30.0% retain (baseline 16.8%).
# => phuong phap CHINH cua bai doi thanh AUGMENTATION. Nhung generality va bang chung
#    BAY THAT hien dang do tren CONSISTENCY -> phai do lai cho phuong phap moi.
#
#   1) v17 deploy(+COCO) + augment-only  -> de bang chung BAY THAT khop phuong phap chinh (~5h)
#   2) YOLO11n + augment-only            -> generality NGOAI ho model (~3h)
#   3) SteadyDet-W (v15) + augment-only   -> generality TRONG ho (~7h)
#   4) eval hop nhat + eval flight_val + sinh bang
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue; mkdir -p "$Q"; LOG=$Q/followup_queue.log
exec 9>/tmp/followup_queue.lock; flock -n 9 || { echo "[fu] dang chay, thoat"; exit 0; }
[ -f "$Q/FOLLOWUP_ALL_DONE" ] && { echo "[fu] DA XONG TU TRUOC -> thoat"; exit 0; }
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

train_pfdet(){ local cfg=$1 run=$2 lg=$3
  for i in $(seq 1 60); do
    grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: XONG"; return 0; }
    local r=""; [ -f "$run/last.pt" ] && r="--resume $run/last.pt"
    log "$run: train (lan $i) $r"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_v3.py --config "$cfg" $r >> "$lg" 2>&1
    grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: XONG"; return 0; }
    log "$run: crash - resume sau 20s"; sleep 20
  done; return 1; }

log "===== FOLLOWUP QUEUE (doi khung sang augmentation) ====="
train_pfdet configs/train_config_v17_augonly_deploy.yaml runs/train_v17_augonly_deploy \
            runs/train_v17_augonly_deploy.qlog && touch "$Q/F1_deploy_augonly_done"

YA=runs/yolo/yolo11n_augonly; YL=runs/yolo_augonly.qlog
for i in $(seq 1 40); do
  [ -f "$YA/weights/best.pt" ] && grep -q "VIB\] XONG" "$YL" 2>/dev/null && { log "yolo11n_augonly: XONG"; break; }
  R=""; [ -f "$YA/weights/last.pt" ] && R="--resume"
  log "yolo11n_augonly: train (lan $i) $R"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_yolo_vibcons.py --mode aug \
      --weights runs/yolo/yolo11n/weights/best.pt --name yolo11n_augonly \
      --epochs 120 --batch 8 --workers 10 $R >> "$YL" 2>&1
  grep -q "VIB\] XONG" "$YL" 2>/dev/null && break
  log "yolo11n_augonly: crash - resume sau 20s"; sleep 20
done
if [ -f "$YA/weights/best.pt" ]; then
  if grep -q "AUGMENTATION đã kích hoạt" "$YL" 2>/dev/null; then touch "$Q/F2_yolo_augonly_done"
    log "OK: xac nhan augmentation thuc su kich hoat"
  else log "!!! CANH BAO: khong thay 'AUGMENTATION da kich hoat' -> DUNG dung so nay"
    touch "$Q/F2_yolo_augonly_SUSPECT"; fi
fi

train_pfdet configs/train_config_v15_augonly.yaml runs/train_v15_augonly \
            runs/train_v15_augonly.qlog && touch "$Q/F3_wide_augonly_done"

log "4) eval hop nhat (chi do model MOI)"
$PY paper_eval_unified.py --seeds 0 1 2 --out "$Q/UNIFIED.json" >> "$Q/unified_eval.log" 2>&1
log "4) eval tren flight_val THAT"
{ echo ""; echo "===== FLIGHT_VAL: augment-only vs consistency (doi khung 2026-07-26) ====="; } >> "$Q/SUMMARY_DEPLOY.txt"
for w in runs/train_v17_augonly_deploy/best.pt runs/train_v17_vib_deploy/best.pt runs/train_v17_deploy/best.pt; do
  [ -f "$w" ] || continue
  { echo ""; echo "--- $w ---"; } >> "$Q/SUMMARY_DEPLOY.txt"
  $PY run_eval_coco.py --weights "$w" --val-images data/flight_val/images \
      --val-labels data/flight_val/labels --img-size 640 --profile light --device cuda:0 2>&1 \
      | grep -E "\[COCO\] overall" >> "$Q/SUMMARY_DEPLOY.txt"
done
$PY paper/make_tables.py > "$Q/TABLES.tex" 2>&1
touch "$Q/FOLLOWUP_ALL_DONE"; log "===== XONG HET ====="
