# PFDet Drone Follow

Phát hiện người tí hon (tiny person detection) từ camera drone, kèm hệ thống điều khiển
drone tự động bay theo người. Target: deploy lên Raspberry Pi 5 + Pixhawk 6C.

## Kết quả

| | Giá trị |
|---|---|
| **Model** | PFDet-Nano v15 light + MuSGD |
| **AP@0.5** (VisDrone person) | **0.5931** |
| **Params** | ~1.0M |
| **Tốc độ** | ~30 FPS trên Pi 5 @ 320px |
| Baseline so sánh (v14) | 0.385 |

## Cấu trúc repo

```
.
├── models/
│   ├── pfdet_nano_v15.py        # Model chính (CSPUIB + LSK + dual AreaAttention)
│   ├── pfdet_nano_v14.py        # Baseline so sánh
│   └── pfdet_nano_v16.py        # Biến thể stability-first
├── utils/
│   ├── losses_v15.py            # Loss v15 (QFL + NWD + CIoU + STAL + ProgLoss)
│   ├── musgd.py                 # MuSGD optimizer (YOLO26-inspired)
│   └── box_ops.py
├── configs/
│   ├── train_config_v15_light_musgd.yaml   # ⭐ Config bản chốt (AP 0.5931)
│   ├── train_config_v15_light.yaml         # AdamW baseline
│   ├── train_config_v14_light.yaml         # v14 baseline
│   └── quick_eval_v15.yaml
├── train_v3.py                  # Training script
├── infer.py                     # Inference (ảnh / video / webcam)
├── infer_pi5_sim.py             # Mô phỏng tốc độ trên Pi 5
├── export.py                    # Export ONNX / TensorRT
├── benchmark.py                 # Benchmark FPS + AP
├── follow_drone/                # ⭐ Hệ thống điều khiển follow person
│   └── README.md                #    (xem hướng dẫn riêng)
├── ARCHITECTURE.md              # Phân tích chi tiết model (vì sao chọn kiến trúc)
└── FOLLOW_CONTROL.md            # Phân tích chi tiết thuật toán điều khiển
```

## Model v15 — kiến trúc

- **Backbone:** `EdgeContextStem → P2 (UIB+LSK) → P3/P4/P5 (CSPUIBStage)`
- **Attention:** `AreaAttention` ở P4 và P5, `LSKBlock` ở P2 + trước mỗi head
- **Neck:** `BiFPN`
- **Head:** `RepConv` decoupled head, anchor-free
- **Pyramid:** `P2/P3/P4/P5` strides `[4, 8, 16, 32]`
- **Loss:** QFL + NWD + CIoU + STAL + ProgLoss + ASL
- **Optimizer:** MuSGD (Muon + SGD blend, YOLO26-style)

Chi tiết đầy đủ + lý do chọn từng thành phần: xem [ARCHITECTURE.md](ARCHITECTURE.md).

## Training

```bash
# Bản chốt — v15 light + MuSGD (AP 0.5931)
python train_v3.py --config configs/train_config_v15_light_musgd.yaml

# AdamW baseline để so sánh
python train_v3.py --config configs/train_config_v15_light.yaml

# v14 baseline (paper comparison)
python train_v3.py --config configs/train_config_v14_light.yaml --model v14
```

## Inference

```bash
# Webcam (camera USB rời thường là --source 1)
python infer.py --weights runs/train_v15_light_musgd_v3/best.pt \
                --source 0 --conf 0.5 --show --device cuda:0

# Ảnh / thư mục / video
python infer.py --weights runs/train_v15_light_musgd_v3/best.pt \
                --source path/to/image.jpg --conf 0.5 --save out.jpg
```

## Mô phỏng tốc độ Pi 5 (trên desktop)

```bash
# PyTorch CPU + INT8 quantization @ 320px (giống điều kiện Pi 5)
python infer_pi5_sim.py --weights runs/train_v15_light_musgd_v3/best.pt --source 1

# ONNX Runtime backend (gần NCNN hơn)
python infer_pi5_sim.py --weights runs/train_v15_light_musgd_v3/best.pt --source 1 --onnx
```

## Export

```bash
# Export ONNX (fused deploy graph mặc định) để deploy lên Pi 5
python export.py --weights runs/train_v15_light_musgd_v3/best.pt --format onnx
```

## Benchmark

```bash
python benchmark.py \
  --weights runs/train_v15_light_musgd_v3/best.pt \
  --profile desktop_4060 \
  --img-size 640
```

## Follow Drone — điều khiển tự động

Hệ thống điều khiển drone bay theo người, dùng detection của model trên.

Pipeline: `PFDet-Nano v15 → OC-SORT tracking → pinhole distance → hybrid 2.5D visual servoing → cascade PID → pymavlink → Pixhawk 6C (ArduPilot GUIDED)`.

```bash
cd follow_drone
# Đọc README.md trong follow_drone/ — có quy trình test 7 bước
```

Chi tiết thiết kế thuật toán + lý do chọn từng module: xem [FOLLOW_CONTROL.md](FOLLOW_CONTROL.md).

⚠️ **An toàn:** test SITL trước, tether (cột dây) khi outdoor đầu tiên, geofence luôn bật.
Xem [follow_drone/README.md](follow_drone/README.md) cho quy trình test đầy đủ.

## Requirements

```bash
# Core (training + inference)
pip install torch torchvision pyyaml tqdm opencv-python matplotlib numpy

# Follow drone control
pip install pymavlink scipy

# Export / Pi 5 deploy
pip install onnx onnxruntime onnxsim
```

## Dataset

Dataset (VisDrone + COCO person) không được commit lên git vì nặng.
Đặt trong `data/visdrone/` theo cấu trúc:

```
data/visdrone/
├── train/images/   train/labels/
└── val/images/     val/labels/
```

Convert VisDrone gốc sang format YOLO:
```bash
python scripts/convert_visdrone.py
```

## Tài liệu

- [ARCHITECTURE.md](ARCHITECTURE.md) — phân tích chi tiết model v15 (kiến trúc, loss, optimizer, vì sao chọn vs alternatives)
- [FOLLOW_CONTROL.md](FOLLOW_CONTROL.md) — phân tích chi tiết hệ thống điều khiển follow
- [follow_drone/README.md](follow_drone/README.md) — hướng dẫn setup + test bay thật
