#!/bin/bash
# ablation_queue.sh — Hàng chờ TỰ ĐỘNG cho 3 thí nghiệm còn thiếu của paper SOICT 2026.
#
# Chạy TUẦN TỰ (không song song) vì GPU chỉ 8GB — 2 job cùng lúc là OOM.
# Tự resume khi crash, chỉ 1 instance (flock), chạy được qua reboot (cron @reboot).
#
#   1) [ĐANG CHẠY, chờ]  v17 + rung-làm-augmentation   -> tách "thấy ảnh rung" khỏi "consistency"
#   2) v17 finetune KHÔNG rung, cùng schedule           -> tách tác dụng của "train thêm 120 epoch"
#   3) YOLO11n + vibration-consistency                  -> generality NGOÀI họ model của mình
#   4) eval hợp nhất lại (1 giao thức, 3 seed) + sinh dòng LaTeX cho bảng
#
# Chạy:  nohup ./ablation_queue.sh >> runs/paper_queue/ablation_queue.nohup 2>&1 &
# Theo dõi:  tail -f runs/paper_queue/ablation_queue.log
# Dừng:      pkill -f "[a]blation_queue.sh"

set -u
cd /home/nolan/Documents/pfdet_drone_follow || exit 1
PY=/home/nolan/Documents/Reinforcement_Learning/train/bin/python
Q=runs/paper_queue
mkdir -p "$Q"
LOG=$Q/ablation_queue.log

exec 9>/tmp/ablation_queue.lock
flock -n 9 || { echo "[abl] đã có instance khác đang chạy, thoát"; exit 0; }

# GUARD (them 2026-07-26): da xong het thi THOAT NGAY.
# Khong co dong nay, cron @reboot se bo qua phan train (da co "Training Complete")
# nhung VAN chay buoc 4 (eval hop nhat ~20 phut GPU) va tranh voi followup_queue.
[ -f "$Q/ABLATION_ALL_DONE" ] && { echo "[abl] ĐÃ XONG HẾT TỪ TRƯỚC -> thoát"; exit 0; }

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# train 1 stage pfdet, auto-resume khi crash (tối đa 60 lần)
train_pfdet() {
    local cfg=$1 run=$2 lg=$3
    for i in $(seq 1 60); do
        grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: ĐÃ XONG"; return 0; }
        local resume=""; [ -f "$run/last.pt" ] && resume="--resume $run/last.pt"
        log "$run: train (lần $i) $resume"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            $PY train_v3.py --config "$cfg" $resume >> "$lg" 2>&1
        grep -q "Training Complete" "$lg" 2>/dev/null && { log "$run: ĐÃ XONG"; return 0; }
        log "$run: crash/thoát — resume sau 20s"; sleep 20
    done
    log "$run: HẾT retry — bỏ qua, sang stage sau"; return 1
}

log "========== HÀNG CHỜ ABLATION (paper SOICT 2026) =========="

# ---------------------------------------------------------------------------
# 1) CHỜ job augonly mà user đã tự chạy tay (ghi vào runs/train_v17_vib_augonly.log)
# ---------------------------------------------------------------------------
AUG_LOG=runs/train_v17_vib_augonly.log
if pgrep -f "[t]rain_v3.py.*vib_augonly" > /dev/null; then
    log "1) v17_vib_augonly đang chạy (user tự khởi động) — CHỜ nó xong..."
    while pgrep -f "[t]rain_v3.py.*vib_augonly" > /dev/null; do sleep 60; done
    log "1) tiến trình augonly đã kết thúc"
fi
# nếu chưa xong (crash giữa đường) thì tiếp tục train nốt
train_pfdet configs/train_config_v17_vib_augonly.yaml \
            runs/train_v17_vib_augonly "$AUG_LOG" && touch "$Q/A1_augonly_done"

# ---------------------------------------------------------------------------
# 2) ĐỐI CHỨNG: finetune 120 epoch KHÔNG rung (cô lập "train thêm")
# ---------------------------------------------------------------------------
if [ -f runs/train_v17_light/best.pt ]; then
    train_pfdet configs/train_config_v17_ft_control.yaml \
                runs/train_v17_ft_control runs/train_v17_ft_control.qlog \
        && touch "$Q/A2_ft_control_done"
else
    log "2) THIẾU runs/train_v17_light/best.pt -> bỏ"
fi

# ---------------------------------------------------------------------------
# 3) GENERALITY: vibration-consistency trên YOLO11n (ngoài họ model của mình)
# ---------------------------------------------------------------------------
YV=runs/yolo/yolo11n_vibcons
YLOG=runs/yolo_vibcons.qlog
if [ -f runs/yolo/yolo11n/weights/best.pt ]; then
    for i in $(seq 1 40); do
        [ -f "$YV/weights/best.pt" ] && grep -q "VIB\] XONG" "$YLOG" 2>/dev/null \
            && { log "3) yolo11n_vibcons: ĐÃ XONG"; break; }
        RES=""; [ -f "$YV/weights/last.pt" ] && RES="--resume"
        log "3) yolo11n_vibcons: train (lần $i) $RES"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            $PY train_yolo_vibcons.py --weights runs/yolo/yolo11n/weights/best.pt \
                --name yolo11n_vibcons --epochs 120 --batch 8 --workers 10 $RES >> "$YLOG" 2>&1
        grep -q "VIB\] XONG" "$YLOG" 2>/dev/null && { log "3) yolo11n_vibcons: ĐÃ XONG"; break; }
        log "3) yolo11n_vibcons: crash/thoát — resume sau 20s"; sleep 20
    done
    # KIỂM TRA AN TOÀN: nếu thiếu dòng "ĐÃ KÍCH HOẠT" thì nhánh rung bị guard bỏ qua âm thầm
    # -> lần train đó VÔ NGHĨA (chỉ là finetune thường). Phải phát hiện, không được im lặng.
    if [ -f "$YV/weights/best.pt" ]; then
        if grep -q "consistency ĐÃ KÍCH HOẠT" "$YLOG" 2>/dev/null; then
            touch "$Q/A3_yolo_vibcons_done"
            log "3) OK: đã xác nhận consistency thực sự kích hoạt ($(grep -c 'ĐÃ KÍCH HOẠT' "$YLOG") lần)"
        else
            log "3) !!! CẢNH BÁO: KHÔNG thấy 'consistency ĐÃ KÍCH HOẠT' trong $YLOG"
            log "3) !!! => nhánh rung KHÔNG chạy, đây chỉ là finetune thường. ĐỪNG dùng số này."
            touch "$Q/A3_yolo_vibcons_SUSPECT"
        fi
    fi
else
    log "3) THIẾU runs/yolo/yolo11n/weights/best.pt -> bỏ"
fi

# ---------------------------------------------------------------------------
# 4) ĐO LẠI HỢP NHẤT (1 giao thức, 3 seed) + sinh dòng LaTeX
#    paper_eval_unified.py tự bỏ qua model đã có trong UNIFIED.json -> chỉ đo model mới.
# ---------------------------------------------------------------------------
log "4) eval hợp nhất (chỉ đo model MỚI, model cũ đã có trong UNIFIED.json)"
$PY paper_eval_unified.py --seeds 0 1 2 --out "$Q/UNIFIED.json" >> "$Q/unified_eval.log" 2>&1
log "4) sinh dòng LaTeX -> $Q/TABLES.tex"
$PY paper/make_tables.py > "$Q/TABLES.tex" 2>&1

touch "$Q/ABLATION_ALL_DONE"
log "========== XONG HẾT =========="
log "Bảng tổng: sed -n '/^====/,\$p' $Q/unified_eval.log | tail -25"
log "Dòng LaTeX để dán vào paper/sections/: $Q/TABLES.tex"
log "CÒN PHẢI LÀM TAY: cập nhật prose trong paper/sections/09_exp_main.tex (bảng ablation),"
log "  10_exp_generality.tex (thêm YOLO11n), 14_limitations.tex (BỎ mục 'augmentation control"
log "  is missing' vì giờ đã có), 02_abstract.tex + 03_introduction.tex (thêm 1 câu)."
