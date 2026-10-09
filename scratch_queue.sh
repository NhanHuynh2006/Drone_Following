#!/bin/bash
# scratch_queue.sh — 2026-07-27. Tra loi MOT cau hoi:
#   Vi sao vibration-aware training KHONG co tac dung tren YOLO11n
#   (20.1% -> 20.1%) trong khi tren ca 3 bien the SteadyDet deu +17..+20 diem?
#
# Nghi ngo: YOLO11n khoi tao tu COCO-pretrained, ma COCO co san anh nhoe chuyen dong
# -> no CO THE da co dac trung khang nhoe, khong con gi de them. SteadyDet train from scratch.
#
#   A) yolo11n_scratch      : from scratch (yolo11n.yaml), KHONG rung  -> baseline moi
#   B) yolo11n_scratch_aug  : finetune tu A + augmentation             -> xu ly
#   C) do ca hai
#
# Doc ket qua:
#   - Neu B > A ro ret  -> nguyen nhan la COCO-PRETRAIN, khong phai kien truc.
#     => paper noi duoc: ky thuat co tac dung khi model chua co dac trung khang nhoe.
#   - Neu B ~ A         -> nguyen nhan la KIEN TRUC (yolo11n mien nhiem).
#     => phai thu hep claim ve dung ho detector cua minh, va noi thang.
#   - Bonus: A co retain THAP hon yolo11n pretrained (20.1%) khong? Neu co, do la
#     bang chung truc tiep rang COCO-pretrain mang lai robustness san.
set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue; mkdir -p "$Q"; LOG=$Q/scratch_queue.log
exec 9>/tmp/scratch_queue.lock; flock -n 9 || { echo "[sc] dang chay, thoat"; exit 0; }
[ -f "$Q/SCRATCH_ALL_DONE" ] && { echo "[sc] DA XONG TU TRUOC -> thoat"; exit 0; }
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run_yolo(){ local name=$1 lg=$2; shift 2
  for i in $(seq 1 40); do
    [ -f "runs/yolo/$name/weights/best.pt" ] && grep -q "VIB\] XONG" "$lg" 2>/dev/null \
      && { log "$name: XONG"; return 0; }
    local r=""; [ -f "runs/yolo/$name/weights/last.pt" ] && r="--resume"
    log "$name: train (lan $i) $r"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY train_yolo_vibcons.py \
        --name "$name" $r "$@" >> "$lg" 2>&1
    grep -q "VIB\] XONG" "$lg" 2>/dev/null && { log "$name: XONG"; return 0; }
    log "$name: crash - resume sau 20s"; sleep 20
  done; return 1; }

log "===== SCRATCH QUEUE: COCO-pretrain hay kien truc? ====="

# A) baseline from scratch. patience=0 trong script -> chay du 150 epoch (from scratch
#    hoi tu cham hon pretrained nen khong duoc early-stop som).
run_yolo yolo11n_scratch runs/yolo_scratch.qlog \
    --mode none --weights yolo11n.yaml --epochs 150 --batch 16 --workers 8 --lr0 0.01
touch "$Q/S1_scratch_base_done"

# B va C deu chi phu thuoc A -> chay SONG SONG (mien phi ve thoi gian).
#   B = + augmentation        (xu ly)
#   C = + 120ep KHONG rung    (doi chung "train them", giong ft_control cua SteadyDet)
# Neu B > A ma khong co C thi khong biet gain do rung hay do train them 120 epoch.
# workers=2 moi job: 2 job YOLO song song tren may 14 GB RAM, 4 gay thrashing.
if [ -f runs/yolo/yolo11n_scratch/weights/best.pt ]; then
  run_yolo yolo11n_scratch_aug runs/yolo_scratch_aug.qlog \
      --mode aug --weights runs/yolo/yolo11n_scratch/weights/best.pt \
      --epochs 120 --batch 8 --workers 2 --lr0 0.002 &
  BPID=$!
  sleep 90                      # lech nhau cho khoi tranh luc nap dataset
  run_yolo yolo11n_scratch_ft runs/yolo_scratch_ft.qlog \
      --mode none --weights runs/yolo/yolo11n_scratch/weights/best.pt \
      --epochs 120 --batch 8 --workers 2 --lr0 0.002 &
  CPID=$!
  wait $BPID $CPID
  if grep -q "AUGMENTATION đã kích hoạt" runs/yolo_scratch_aug.qlog 2>/dev/null; then
    touch "$Q/S2_scratch_aug_done"; log "OK: xac nhan augmentation thuc su kich hoat"
  else
    log "!!! CANH BAO: khong thay 'AUGMENTATION da kich hoat' -> DUNG dung so nay"
    touch "$Q/S2_scratch_aug_SUSPECT"
  fi
  touch "$Q/S3_scratch_ft_done"
else
  log "A that bai -> bo qua B va C"
fi

log "do ca hai (chi model MOI)"
$PY paper_eval_unified.py --seeds 0 1 2 --out "$Q/UNIFIED.json" >> "$Q/unified_eval.log" 2>&1
$PY paper/make_tables.py > "$Q/TABLES.tex" 2>&1
touch "$Q/SCRATCH_ALL_DONE"; log "===== XONG ====="
