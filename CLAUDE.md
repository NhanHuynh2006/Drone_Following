# CLAUDE.md — Bối cảnh dự án cho Claude Code

File này giúp Claude Code nắm nhanh bối cảnh dự án ở mỗi phiên làm việc.

## Tổng quan

Dự án **PFDet Drone Follow**: phát hiện người tí hon (tiny person detection) từ camera
drone bằng model tự thiết kế **PFDet-Nano v15**, kèm hệ thống điều khiển drone tự động
bay theo người. Target deploy: **Raspberry Pi 5 + Pixhawk 6C (ArduPilot GUIDED)**.

Kết quả bản chốt: **AP@0.5 = 0.5931** trên VisDrone person, ~1.0M params, ~30 FPS trên Pi 5 @ 320px.
Config bản chốt: `configs/train_config_v15_light_musgd.yaml`.

Hướng phát triển tiếp theo: chuyển/mở rộng hệ thống follow-person sang **robot Unitree Go2**
(các dự án "com robot go2" — xem mục Ghi chú phía dưới).

## Ngôn ngữ & quy ước

- Tài liệu, comment, commit message viết bằng **tiếng Việt** (thuật ngữ kỹ thuật giữ tiếng Anh).
- Tài liệu kiến trúc rất chi tiết, có giải thích cho người mới — khi sửa model/control,
  cập nhật luôn doc tương ứng theo cùng phong cách ("Vấn đề → Module đã chọn → Cách hoạt động
  → Các phương án đã cân nhắc → Tại sao chọn").

## Cấu trúc chính

| Đường dẫn | Nội dung |
|---|---|
| `models/pfdet_nano_v15.py` | Model chính (CSPUIB + LSK + dual AreaAttention, BiFPN, RepConv head, anchor-free, P2-P5) |
| `models/pfdet_nano_v14.py` | Baseline so sánh (AP 0.385) |
| `utils/losses_v15.py` | Loss: QFL + NWD + CIoU + STAL + ProgLoss + ASL |
| `utils/musgd.py` | Optimizer MuSGD (Muon + SGD blend, YOLO26-style) |
| `train_v3.py` | Training script |
| `infer.py`, `infer_pi5_sim.py` | Inference / mô phỏng tốc độ Pi 5 |
| `export.py` | Export ONNX (fused deploy graph) / TensorRT |
| `follow_drone/` | Hệ thống điều khiển follow person (xem dưới) |
| `ARCHITECTURE.md` | Phân tích chi tiết model v15 — đọc khi sửa kiến trúc/loss/optimizer |
| `FOLLOW_CONTROL.md` | Phân tích chi tiết thuật toán điều khiển — đọc khi sửa follow_drone |

## Pipeline điều khiển follow (`follow_drone/`)

```
PFDet-Nano v15 → OC-SORT tracking → pinhole distance (+ self-calibration, altitude fusion)
→ 2-stage Kalman Filter (target world position) → hybrid 2.5D visual servoing
→ cascade PID 3 trục → pymavlink → Pixhawk 6C (ArduPilot GUIDED)
```

Module chính trong `follow_drone/follow/`: `detector.py`, `ocsort.py`, `target_selector.py`,
`distance.py`, `visual_servo.py`, `pid.py`, `mavlink_client.py`, `safety.py`, `camera.py`.
Config runtime: `follow_drone/config.yaml`. Entry point: `follow_drone/main.py`.

## Lệnh thường dùng

```bash
# Training bản chốt
python train_v3.py --config configs/train_config_v15_light_musgd.yaml

# Inference webcam / file
python infer.py --weights runs/train_v15_light_musgd_v3/best.pt --source 0 --conf 0.5 --show

# Export ONNX cho Pi 5
python export.py --weights runs/train_v15_light_musgd_v3/best.pt --format onnx

# Benchmark
python benchmark.py --weights runs/train_v15_light_musgd_v3/best.pt --profile desktop_4060 --img-size 640
```

Dataset (VisDrone + COCO person) **không commit lên git** — đặt ở `data/visdrone/`
(train/val × images/labels, format YOLO). Convert: `scripts/convert_visdrone.py`.

## An toàn (khi đụng vào follow_drone)

- Luôn test SITL trước khi bay thật; tether khi outdoor lần đầu; geofence luôn bật.
- Không nới lỏng các giới hạn trong `follow/safety.py` nếu không được yêu cầu rõ ràng.
- Quy trình test 7 bước: xem `follow_drone/README.md`.

## Ghi chú: các dự án "com robot Go2"

Người dùng có một đoạn chat Claude (claude.ai) bàn về các dự án com robot cho **Unitree Go2**,
dự định dùng làm nền cho công việc tiếp theo trong repo này. Claude Code chưa truy cập được
nội dung đoạn chat đó (link share bị chặn). **Khi người dùng cung cấp nội dung chat,
cập nhật mục này** với: mục tiêu dự án Go2, kiến trúc/SDK dùng (vd. unitree_sdk2),
và các quyết định kỹ thuật đã chốt.
