# PFDet-Nano: Lightweight Person Detector for Drone Applications

Hệ thống phát hiện người từ góc nhìn drone, tối ưu cho thiết bị edge (Raspberry Pi 5, Jetson Nano).

---

## Kiến trúc Model

PFDet-Nano v5 - Anchor-free single-class detector (~900K params)

- **Backbone**: MobileNetV2-style InvertedResidual blocks, SiLU activation
- **Neck**: FPN (top-down) + PAN (bottom-up) - bidirectional feature fusion
- **Head**: Per-scale detection head (3 scales: stride 8/16/32)
- **Output**: 5 channels per cell (objectness, dx, dy, log_w, log_h)

Chi tiet kien truc: xem `runs/train_v5/pfdet_nano_v5_architecture.png`

---

## Cau truc Project

```
pfdet_drone_follow/
├── models/
│   └── pfdet_nano.py          # Model architecture
├── datasets/
│   └── visdrone_person.py     # VisDrone dataset + augmentation
├── utils/
│   ├── box_ops.py             # CIoU, NMS, decode predictions
│   └── losses.py              # BCE + CIoU + L1 loss
├── configs/
│   └── train_config.yaml      # Training configuration
├── train_v3.py                # Training script
├── infer.py                   # Inference (image/video/webcam)
├── drone_follow.py            # Drone follow-me controller
├── trt_infer_engine.py        # TensorRT inference engine
├── make_demo_video.py         # Generate demo video from image sequence
└── plot_architecture.py       # Generate architecture diagram
```

---

## Training

### Dataset: VisDrone-DET (person class only)
- Train: 6,471 images | Val: 548 images
- Augmentation: Mosaic, Color Jitter, Horizontal Flip, Random Affine, MixUp

### Config
```yaml
model:
  base_c: 32
  img_size: 416
train:
  epochs: 200
  batch_size: 64
  lr: 0.003
  warmup_epochs: 10
  ema_decay: 0.9998
loss:
  obj_weight: 0.5
  box_weight: 5.0
```

### Run training
```bash
python train_v3.py --config configs/train_config.yaml
```

---

## Results

| Metric | Value |
|--------|-------|
| AP@0.5 | 0.390 |
| Recall | 61.1% |
| Parameters | ~900K |
| Model FPS (GPU) | ~200 FPS |

Training curves: xem `runs/train_v5/training_curves.png`

---

## Inference

```bash
# Image
python infer.py --weights runs/train_v5/best.pt --source image.jpg --show

# Video
python infer.py --weights runs/train_v5/best.pt --source video.mp4 --show

# Webcam
python infer.py --weights runs/train_v5/best.pt --source 0 --show
```

---

## Requirements

```bash
pip install torch torchvision pyyaml tqdm opencv-python matplotlib
```
