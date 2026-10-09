#!/usr/bin/env bash
# retrain_hailo.sh — Sửa lỗi extra_enable_epoch: train lại checkpoint Hailo CÓ COCO.
#
# BỐI CẢNH: HEF đang bay sinh từ runs/train_v17_hailo/best.pt. Checkpoint đó finetune
# 20 epoch từ train_v17_deploy, nhưng config để extra_enable_epoch=100 > epochs=20 nên
# COCO CHƯA BAO GIỜ được bật (train_v3.py:819). 20 epoch đó toàn VisDrone, kéo model
# ngược khỏi khả năng bắt người gần. Đo trên 92 khung bay thật:
#     train_v17_deploy (có COCO)  F1 0.882
#     train_v17_hailo  (mất COCO) F1 0.768   <- HEF sinh từ đây
# Xem runs/flight_val_hef_2026-08-27.md
#
# Chạy từng bước, không phải một phát ăn ngay: BƯỚC 2 trả lời "sửa có ăn không"
# mà chưa cần đụng tới Hailo.
set -euo pipefail
cd "$(dirname "$0")/.."

STEP="${1:-all}"
CKPT=runs/train_v17_hailo_coco/best.pt
ONNX=pfdet_v17_hailo_coco_640.onnx
# ĐƯỜNG DẪN TUYỆT ĐỐI, có chủ đích: hai venv nắm hai nửa khác nhau và không venv nào
# có cả hai. Nếu để "python3" thì kết quả phụ thuộc venv nào đang bật — chạy trong
# (hailo_env) sẽ chết vì "No module named 'torch'".
#   ~/.venvs/pfdet : torch + CUDA + onnx + onnxsim  -> train / eval / export
#   ./hailo_env    : Hailo Dataflow Compiler        -> chỉ bước compile HEF
PY_TRAIN="${PY_TRAIN:-$HOME/.venvs/pfdet/bin/python}"
PY_DFC="${PY_DFC:-$(pwd)/hailo_env/bin/python}"

for _p in "$PY_TRAIN"; do
  "$_p" -c "import torch" 2>/dev/null || { echo "LỖI: $_p không có torch"; exit 1; }
done

step1_train() {
  echo "=== BƯỚC 1: train 20 epoch, BẬT COCO từ epoch 0 ==="
  for d in data/visdrone/train/images data/coco_person/images; do
    n=$(ls "$d" 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "LỖI: $d rỗng. Chạy scripts/restore_data.sh trước."; exit 1; }
    echo "  $d: $n ảnh"
  done
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY_TRAIN" train_v3.py --config configs/train_config_v17_hailo_coco.yaml \
    2>&1 | tee -a runs/train_v17_hailo_coco.log
}

step2_eval() {
  echo "=== BƯỚC 2: đo FP32 trên flight_val — ĐÂY là bước quyết định ==="
  echo "    (so với: deploy 0.882 / hailo-cũ 0.768 / HEF đang bay 0.745)"
  "$PY_TRAIN" fp32_sweep.py --weights "$CKPT" | tee runs/flight_val_hailo_coco_fp32.txt
}

step3_export() {
  echo "=== BƯỚC 3: export ONNX ==="
  "$PY_TRAIN" export_onnx_hailo.py --weights "$CKPT" --img-size 640 --output "$ONNX"
}

step4_calib() {
  echo "=== BƯỚC 4: dựng tập calibration ĐÚNG PHÂN PHỐI TRAIN ==="
  # Calib phải giống phân phối đầu vào lúc chạy thật. Model train trên VisDrone + 25%
  # COCO người-gần, nên calib cũng trộn theo tỉ lệ đó. Lấy từ tập TRAIN, KHÔNG lấy
  # flight_val — đó là tập dùng để chấm điểm, calib bằng nó là rò rỉ dữ liệu.
  rm -rf data/_calib && mkdir -p data/_calib
  ls data/visdrone/train/images/*.jpg | shuf -n 48 --random-source=<(yes) | \
      xargs -I{} cp {} data/_calib/
  ls data/coco_person/images/*.jpg   | shuf -n 16 --random-source=<(yes) | \
      xargs -I{} cp {} data/_calib/
  echo "  $(ls data/_calib | wc -l) ảnh (48 VisDrone + 16 COCO, đúng tỉ lệ 0.25)"
}

step5_compile() {
  echo "=== BƯỚC 5: compile HEF (opt-level 4) ==="
  # opt-level 4 thay vì mặc định 2: lần đo trước INT8 làm AP tổng rớt 12.6%, và
  # README_HAILO.md mục 6 đặt ngưỡng "rớt > 5% -> tăng opt-level".
  "$PY_DFC" compile_hef.py --onnx "$ONNX" --hw-arch hailo8l \
      --calib-dir data/_calib --img-size 640 --name pfdet_v17_640_coco --opt-level 4
}

step6_deploy() {
  echo "=== BƯỚC 6: đẩy sang Pi và đo lại ==="
  scp pfdet_v17_640_coco_hailo8l.hef raspi@192.168.196.114:~/NhanHuynh/models/
  echo "  Trên Pi, đổi follow_drone/config.yaml -> model.hef sang file mới, rồi:"
  echo "    ./run.sh py flight_sweep.py --hef models/pfdet_v17_640_coco_hailo8l.hef"
}

case "$STEP" in
  1|train)   step1_train ;;
  2|eval)    step2_eval ;;
  3|export)  step3_export ;;
  4|calib)   step4_calib ;;
  5|compile) step5_compile ;;
  6|deploy)  step6_deploy ;;
  all)       step1_train; step2_eval; step3_export; step4_calib; step5_compile
             echo; echo "Xong bước 1-5. Chạy 'bash scripts/retrain_hailo.sh 6' để đẩy sang Pi." ;;
  *) echo "Dùng: $0 {1|2|3|4|5|6|all}"; exit 1 ;;
esac
