# PFDet-Nano: Person Following Drone Detection System

## 🎯 Hệ thống Drone Follow-Me | Jetson Nano B01 + Pixhawk 6C

Hệ thống hoàn chỉnh cho drone tự động follow người, tối ưu cho Jetson Nano B01.

### Phần cứng yêu cầu
- **Máy tính bay**: Jetson Nano B01 (4GB RAM)
- **Flight controller**: Pixhawk 6C (ArduCopter firmware)
- **Camera**: CSI camera hoặc USB camera
- **Kết nối**: Serial UART (Jetson UART → Pixhawk TELEM2)

---

## 📁 Cấu trúc Project

```
pfdet_drone_follow/
├── models/
│   ├── __init__.py
│   └── pfdet_nano.py          # Model architecture (tự viết)
├── datasets/
│   ├── __init__.py
│   └── visdrone_person.py     # Dataset + augmentation
├── utils/
│   ├── __init__.py
│   ├── box_ops.py             # CIoU, NMS, decode
│   └── losses.py              # Focal Loss + SimOTA + CIoU loss
├── configs/
│   └── train_config.yaml      # Training configuration
├── scripts/
│   └── convert_visdrone.py    # VisDrone → YOLO format
├── train.py                   # Training script
├── infer.py                   # Inference (image/video)
├── export.py                  # ONNX/TensorRT export
├── drone_follow.py            # Main follow controller
├── trt_infer_engine.py        # TensorRT inference engine
└── README.md
```

---

## 🔧 So sánh Model cũ vs Model mới

| Feature | Model cũ (PFDetUAVFP) | Model mới (PFDetNano) |
|---------|----------------------|----------------------|
| **Loss objectness** | BCE (dễ bị false positive) | **Focal Loss** (xử lý class imbalance) |
| **Loss box** | L1 Loss (kém chính xác) | **CIoU Loss** (tối ưu IoU trực tiếp) |
| **Target assignment** | 1 cell/object (quá ít) | **SimOTA** (nhiều positive cells) |
| **NMS** | ❌ Không có! | ✅ NMS chuẩn |
| **Box encoding** | Sigmoid (giới hạn 0-1) | **Exp encoding** (linh hoạt hơn) |
| **Augmentation** | ❌ Không có | ✅ Mosaic + Color + Flip + Scale |
| **Head** | Coupled (shared) | **Decoupled** (cls/reg riêng) |
| **Attention** | ❌ | ✅ Coordinate Attention |
| **LR schedule** | Fixed | **Cosine + Warmup** |
| **EMA** | ❌ | ✅ Exponential Moving Average |
| **Init bias** | Random | **Prior probability init** (-4.6) |

### Tại sao model cũ sai nhiều?

1. **KHÔNG CÓ NMS** → Hàng chục box chồng lên nhau (ảnh screenshot: 71 detections!)
2. **Target assignment 1 cell** → Model gần như không học được vì quá ít positive samples
3. **BCE loss** → Khi 99.9% cells là negative, model chỉ cần predict toàn 0 là loss thấp
4. **L1 loss cho box** → Không tối ưu trực tiếp IoU, box predict bị lệch
5. **Không augmentation** → Overfit trên training set

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Cài đặt

```bash
# Trên PC training (có GPU NVIDIA)
pip install torch torchvision pyyaml tqdm opencv-python

# Trên Jetson Nano (đã có JetPack)
pip install pymavlink pyyaml opencv-python
```

### Bước 2: Chuẩn bị dataset VisDrone

```bash
# Download VisDrone-DET dataset
# https://github.com/VisDrone/VisDrone-Dataset

# Convert sang YOLO format (chỉ lấy person class)
python scripts/convert_visdrone.py \
    --src /path/to/VisDrone \
    --dst ./data/visdrone \
    --min-size 10
```

Cấu trúc sau khi convert:
```
data/visdrone/
├── train/
│   ├── images/    (6471 images)
│   └── labels/    (YOLO format: 0 cx cy w h fx fy)
└── val/
    ├── images/    (548 images)
    └── labels/
```

### Bước 3: Training

```bash
# Train model (trên PC có GPU)
python train.py --config configs/train_config.yaml

# Theo dõi training:
# - Loss giảm dần
# - AP@0.5 tăng dần (target: >0.5)
# - Kiểm tra val mỗi 5 epochs
```

**Điều chỉnh nếu cần:**
```yaml
# configs/train_config.yaml
model:
  base_c: 24        # 24 cho Jetson Nano (nhanh), 32 cho accuracy cao hơn
  img_size: 416     # 416 cho cân bằng, 320 nếu cần nhanh hơn

train:
  batch_size: 16    # Giảm xuống 8 nếu hết RAM GPU
  epochs: 150       # Tối thiểu 100 epochs
  lr: 0.002         # Tăng nếu loss không giảm, giảm nếu không ổn định
```

### Bước 4: Test inference

```bash
# Test trên ảnh
python infer.py --weights runs/train/best.pt --source test_image.jpg --show

# Test trên video
python infer.py --weights runs/train/best.pt --source video.mp4 --show

# Test webcam
python infer.py --weights runs/train/best.pt --source 0 --show
```

### Bước 5: Export cho Jetson Nano

```bash
# Export ONNX (trên PC)
python export.py --weights runs/train/best.pt --format onnx

# Copy file .onnx sang Jetson Nano, rồi build TensorRT engine:
python3 best_build_trt.py   # (script tự generate)

# Hoặc nếu Jetson Nano có TensorRT:
python export.py --weights runs/train/best.pt --format trt --fp16
```

### Bước 6: Chạy Follow-Me trên Drone

```bash
# Test KHÔNG có Pixhawk (simulation mode)
python drone_follow.py \
    --weights runs/train/best.pt \
    --camera 0

# Kết nối Pixhawk qua UART
python drone_follow.py \
    --weights runs/train/best.pt \
    --camera 0 \
    --pixhawk /dev/ttyTHS1 \
    --baudrate 921600

# Kết nối qua USB
python drone_follow.py \
    --weights runs/train/best.pt \
    --camera 0 \
    --pixhawk /dev/ttyACM0

# Với TensorRT (nhanh hơn nhiều)
python drone_follow.py \
    --weights runs/train/best.pt \
    --camera 0 \
    --pixhawk /dev/ttyTHS1 \
    --device cuda:0
```

---

## 🎛️ Tuning PID cho Drone

### Cách chỉnh PID trong `drone_follow.py`:

```python
# Yaw (xoay trái/phải để giữ người ở giữa khung hình)
self.pid_yaw = PIDController(
    kp=60.0,    # Bắt đầu: 30, tăng dần đến khi phản ứng đủ nhanh
    ki=2.0,     # Thêm nhỏ nếu có steady-state error
    kd=15.0,    # ~30% của Kp, giúp giảm dao động
)

# Forward/backward (giữ khoảng cách)
self.pid_forward = PIDController(
    kp=3.0,     # Bắt đầu: 1.5, tăng dần
    ki=0.1,     
    kd=1.0,     
)

# Altitude (giữ độ cao)
self.pid_altitude = PIDController(
    kp=2.0,     # Bắt đầu: 1.0
    ki=0.05,    
    kd=0.8,     
)
```

### Quy trình tuning:
1. **Bắt đầu chỉ với P** (I=0, D=0)
2. **Tăng P** cho đến khi drone bắt đầu dao động
3. **Giảm P xuống 60-70%** của giá trị dao động
4. **Thêm D = 0.3 * P** để giảm dao động
5. **Thêm I rất nhỏ** (I = 0.05 * P) nếu cần

### An toàn:
- ⚠️ **LUÔN test không có cánh quạt trước!**
- ⚠️ **Giữ remote control sẵn sàng override**
- ⚠️ **Bắt đầu với tốc độ max thấp** (1 m/s)
- ⚠️ **Set failsafe trong Mission Planner**

---

## 🔌 Kết nối phần cứng Jetson Nano ↔ Pixhawk 6C

```
Jetson Nano (UART)          Pixhawk 6C (TELEM2)
─────────────────           ────────────────────
Pin 8  (TXD) ─────────────→ Pin 3 (RX)
Pin 10 (RXD) ←───────────── Pin 2 (TX)
Pin 6  (GND) ─────────────→ Pin 6 (GND)

Lưu ý: KHÔNG nối VCC! Jetson và Pixhawk dùng nguồn riêng.
```

### Cấu hình trong Mission Planner:
```
SERIAL2_PROTOCOL = 2    (MAVLink2)
SERIAL2_BAUD = 921      (921600 baud)
```

---

## 📊 Performance dự kiến

| Cấu hình | Params | FPS (Jetson Nano) | AP@0.5 (VisDrone) |
|-----------|--------|-------------------|-------------------|
| base_c=24, img=320 | ~0.8M | 25-30 FPS (TRT FP16) | ~0.40 |
| base_c=24, img=416 | ~0.8M | 18-22 FPS (TRT FP16) | ~0.48 |
| base_c=32, img=416 | ~1.4M | 14-18 FPS (TRT FP16) | ~0.52 |
| base_c=32, img=512 | ~1.4M | 10-14 FPS (TRT FP16) | ~0.55 |

---

## ⌨️ Phím tắt khi chạy

| Phím | Chức năng |
|------|-----------|
| `q` | Thoát |
| `r` | Chọn lại target |
| `s` | Emergency STOP |
| `a` | Arm lại |
| `+` | Tăng confidence threshold |
| `-` | Giảm confidence threshold |

---

## 🧠 Chi tiết kiến trúc Model

### Backbone: CSP-Lite with Coordinate Attention
- **ShuffleBottleneck**: DWConv + Channel Shuffle (thay vì standard Conv)
- **CSPLite**: Split → Bottlenecks → Concat → Fuse (hiệu quả hơn C2f)
- **CoordAttention**: Capture spatial dependencies với overhead rất thấp

### Neck: 2x BiFPN
- **Weighted Feature Fusion**: Learnable weights cho mỗi input
- **Top-down + Bottom-up**: Trao đổi thông tin giữa các scale

### Head: Decoupled
- **Classification branch**: DWConv → DWConv → 1x1 (objectness)
- **Regression branch**: DWConv → DWConv → 1x1 (box + foot)
- **Prior bias init**: cls bias = -4.6 (sigmoid = 0.01) → giảm false positive

### Loss: Focal + CIoU + SimOTA
- **Focal Loss (α=0.25, γ=2)**: Focus vào hard examples
- **CIoU Loss**: Tối ưu trực tiếp IoU + center distance + aspect ratio
- **SimOTA Assignment**: Gán nhiều positive cells/object dựa trên cost matrix
# Drone_Following
