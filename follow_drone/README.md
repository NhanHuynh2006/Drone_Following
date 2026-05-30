# Follow Drone — Hệ thống điều khiển drone follow person

Code hoàn chỉnh để drone tự động bay theo người dùng PFDet-Nano v15 detection,
chạy trên Raspberry Pi 5 + Pixhawk 6C + ArduPilot Copter.

**ĐỌC KỸ TRƯỚC KHI BAY THẬT.** Test theo thứ tự được liệt kê dưới.

---

## Cấu trúc thư mục

```
follow_drone/
├── config.yaml                # Tất cả hyperparameters
├── main.py                    # Main entry point (chạy follow loop)
├── follow/
│   ├── camera.py              # Camera capture thread
│   ├── detector.py            # PFDet-Nano v15 wrapper
│   ├── ocsort.py              # OC-SORT tracker
│   ├── target_selector.py     # Lock target ID
│   ├── distance.py            # Pinhole distance + Kalman smoother
│   ├── visual_servo.py        # Hybrid 2.5D visual servoing
│   ├── pid.py                 # PID controllers (forward/yaw/vertical)
│   ├── mavlink_client.py      # pymavlink wrapper
│   └── safety.py              # Failsafe ladder + clamps
└── scripts/
    ├── test_mavlink.py        # Test MAVLink connection (no fly)
    ├── test_detection.py      # Test detection pipeline (no MAVLink)
    ├── arm_test.py            # Test arm/disarm (NO PROPELLERS)
    └── calibrate_camera.py    # Calibrate camera intrinsics
```

---

## Yêu cầu

```bash
pip install opencv-python numpy torch pymavlink pyyaml scipy
```

Optional cho INT8 quantization tốt hơn:
```bash
pip install onnx onnxruntime onnxsim
```

---

## Setup ArduPilot (Mission Planner)

Set các parameters sau qua **Mission Planner → CONFIG/TUNING → Full Parameter List**:

| Parameter | Value | Mục đích |
|-----------|-------|----------|
| `SERIAL2_PROTOCOL` | 2 | TELEM2 dùng MAVLink |
| `SERIAL2_BAUD` | 921 | Baud 921600 |
| `BRD_SER2_RTSCTS` | 0 | Disable flow control |
| `FS_GCS_ENABLE` | 1 | Enable GCS failsafe |
| `FENCE_ENABLE` | 1 | Enable geofence |
| `FENCE_TYPE` | 7 | Cylinder + Alt + Polygon |
| `FENCE_ACTION` | 1 | RTL khi vượt fence |
| `FENCE_RADIUS` | 100 | 100m từ home |
| `FENCE_ALT_MAX` | 30 | Trần 30m |
| `BATT_LOW_VOLT` | 22.0 | Cảnh báo (cho 6S) |
| `BATT_CRT_VOLT` | 21.0 | Critical |
| `BATT_FS_LOW_ACT` | 2 | RTL khi pin thấp |
| `BATT_FS_CRT_ACT` | 1 | LAND khi pin cực thấp |
| `WPNAV_SPEED` | 500 | Tốc độ tối đa 5 m/s |

Sau đó **WRITE PARAMS** và **REBOOT** Pixhawk.

---

## Wiring Pi 5 ↔ Pixhawk 6C

Pi 5 GPIO TX (GPIO 14, pin 8) → Pixhawk TELEM2 RX
Pi 5 GPIO RX (GPIO 15, pin 10) → Pixhawk TELEM2 TX
Pi 5 GND → Pixhawk TELEM2 GND
**Cấp nguồn riêng cho Pi 5 (5V 3A USB-C)** — KHÔNG dùng từ Pixhawk.

Bật UART trên Pi 5:
```bash
sudo raspi-config  # Interface → Serial → No login shell, Yes serial port
sudo reboot
```

Kiểm tra:
```bash
ls -la /dev/serial0  # → /dev/ttyAMA0
```

---

## Quy trình test (theo thứ tự, KHÔNG SKIP)

### Bước 1: Calibrate camera (chỉ làm 1 lần)
```bash
cd follow_drone
python scripts/calibrate_camera.py
# Chụp 20-30 ảnh checkerboard
# Copy fx, fy, cx, cy vào config.yaml
```

### Bước 2: Test detection trên ground
```bash
python scripts/test_detection.py --config config.yaml
# Press 'q' to quit, 'u' to unlock target
# Verify: phát hiện được người, tracking ID ổn định
```

### Bước 3: Test MAVLink connection (no fly)
```bash
python scripts/test_mavlink.py --config config.yaml
# Verify: heartbeat, mode, GPS, battery hiển thị
# Pi 5 connect ổn định với Pixhawk
```

### Bước 4: Test SITL (mạnh khuyến nghị)

Cài ArduPilot SITL trên laptop riêng:
```bash
git clone https://github.com/ArduPilot/ardupilot.git
cd ardupilot/ArduCopter
sim_vehicle.py --console --map
```

Trên Pi 5, edit `config.yaml`:
```yaml
mavlink:
  connection: "udpin:0.0.0.0:14550"   # hoặc IP của laptop chạy SITL
```

Chạy follow trong SITL:
```bash
python main.py --config config.yaml --sitl --show
```

Verify pipeline hoạt động end-to-end trong simulator. Test failsafe (kill detection process → drone phải hover → loiter → RTL).

### Bước 5: Arm test thật (NO PROPELLERS)
```bash
# THÁO PROPELLERS!
python scripts/arm_test.py
# Verify: motors arm và spin nhẹ trong 5s rồi disarm
```

### Bước 6: Tethered hover test (CỘT DÂY)
- Mount drone, **buộc dây an toàn** (tether dây 3-5m)
- Take off bằng RC remote, độ cao 1m
- Bật `--no-fly` để test pipeline mà KHÔNG send velocity command:
```bash
python main.py --config config.yaml --show --no-fly
```
- Verify: detection + tracking + log hoạt động mà drone vẫn idle.

### Bước 7: Outdoor follow đầu tiên
- Chỗ trống an toàn, gió < 5 m/s
- Geofence 30m × 30m × 10m altitude (qua Mission Planner)
- Mission Planner mở liên tục để monitor + emergency override
- Bắt đầu:
```bash
python main.py --config config.yaml --show
```
- Người mục tiêu di chuyển CHẬM (đi bộ, không chạy)
- Sẵn sàng emergency: bật RC switch để override mode (vd Stabilize/Loiter)

---

## Flag chạy main.py

| Flag | Tác dụng |
|------|----------|
| `--config FILE` | Đường dẫn config (default `config.yaml`) |
| `--sitl` | Dùng UDP SITL thay UART |
| `--no-fly` | Chạy pipeline nhưng KHÔNG gửi velocity command. Test on ground |
| `--no-takeoff` | Skip takeoff sequence (drone đã trên không) |
| `--show` | Hiển thị live video với overlay |

Phím trong `--show` mode:
- `q`: thoát + RTL
- `u`: unlock target hiện tại

---

## Tuning PID

Bắt đầu với giá trị conservative trong `config.yaml`:

| PID | Param | Default | Cách tune |
|-----|-------|---------|-----------|
| Forward | Kp | 0.5 | Tăng nếu drone phản ứng chậm. Giảm nếu oscillate |
| Forward | Kd | 0.1 | Tăng để giảm overshoot |
| Forward | Ki | 0.05 | Tăng nếu có steady-state error |
| Yaw | Kp | 2.0 | Tăng nếu drone xoay chậm theo người |
| Yaw | Kd | 0.3 | Tăng nếu yaw oscillate |
| Vertical | Kp | 0.8 | Tương tự Forward |

**Rule of thumb:** tune Kp trước (chỉ Kp, để Ki=Kd=0). Tăng dần đến khi vừa phản ứng nhanh nhưng chưa oscillate. Sau đó thêm Kd để dampen, Ki cuối cùng nếu cần.

Log CSV trong `logs/` chứa toàn bộ data — load Excel/Python để plot và tune.

---

## Failsafe behavior

Target lost → ladder tự động:
- 0-3s: coast (continue with Kalman prediction)
- 3-8s: hover (velocity = 0)
- 8-15s: loiter mode
- >15s: RTL

Health check fail (GPS, battery, heartbeat) → loiter ngay.

Pi crash → ArduPilot 3s GUIDED timeout → drone tự stop.

Mission Planner mất kết nối → ArduPilot GCS failsafe → RTL.

Pin yếu → ArduPilot battery failsafe → RTL.

Vượt geofence → ArduPilot fence breach → RTL.

---

## Troubleshooting

### "No heartbeat received"
- Check UART wiring (TX/RX không ngược)
- Check `SERIAL2_PROTOCOL=2`, `SERIAL2_BAUD=921`
- Check baud trong config.yaml = 921600
- Test bằng `mavproxy.py --master=/dev/serial0 --baudrate=921600`

### Drone không arm
- GPS chưa 3D fix? (cần outdoor, sky view)
- Pre-arm checks fail? Check Mission Planner messages
- Battery cell count đúng?

### Drone xoay quá mạnh / oscillate
- Giảm `yaw_pid.Kp` từ 2.0 xuống 1.0
- Tăng `yaw_pid.Kd` từ 0.3 lên 0.5

### Drone không tiến tới gần đủ
- Giảm `follow_distance_m` (nhưng phải > `d_min`)
- Tăng `forward_pid.Kp`
- Check `d_min` trong safety section

### Detection FPS quá thấp trên Pi 5
- Set `device: "cpu"` (không phải cuda)
- Giảm `img_size` từ 640 xuống 320 trong config (cần test lại AP)
- Pi 5 với INT8 quantization nên đạt ~25-30 FPS

---

## Tham khảo

- `ARCHITECTURE.md` — chi tiết PFDet-Nano v15 model
- `FOLLOW_CONTROL.md` — chi tiết thiết kế thuật toán điều khiển
- ArduPilot Copter docs: https://ardupilot.org/copter/
- pymavlink: https://github.com/ArduPilot/pymavlink
- Mission Planner: https://ardupilot.org/planner/

---

**An toàn trên hết. Test SITL trước. Tether khi outdoor đầu tiên. Geofence luôn bật.**
