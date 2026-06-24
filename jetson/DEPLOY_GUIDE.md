# DEPLOY GUIDE — Bay theo người, PX4 + Jetson Nano (làm 1 lần, bay test)

Hệ thống: **Detector (Python/ONNX) → UDP → Control (C++ MAVSDK) → PX4**. Mượt nhờ Kalman + feedforward +
50Hz + PX4 jerk-limited smoothing. Đọc hết trước khi bay. **Test SITL trước, bay thật sau.**

```
[Camera] → detector_pub.py (ONNX, ~15-30Hz) ──UDP:5600──> follow_px4 (C++, 50Hz) ──serial──> Pixhawk (PX4)
                 chọn target + pinhole distance        Kalman + PD + feedforward + jerk-limit
```

## 0. CHIA LUỒNG (vì sao chạy nhanh + mượt)
Hệ chạy nhanh nhờ **3 tầng tách biệt** — mỗi cái có nhịp riêng, không chờ nhau:

```
PROCESS 1: detector_pub.py (Python, GPU)          PROCESS 2: follow_px4 (C++)
 ├─ Luồng A: CameraGrabber  (đọc cam liên tục,      ├─ Luồng A: TargetReceiver (UDP, nhận target,
 │           GIỮ FRAME MỚI NHẤT, bỏ frame cũ)       │           lưu cái mới nhất + timestamp)
 └─ Luồng B (main): infer ONNX + chọn target        └─ Luồng B (main): vòng 50Hz — Kalman + control
            + gửi UDP  (~15-30Hz, tuỳ model)                   + gửi setpoint PX4
                                                    + MAVSDK tự chạy luồng RX/TX riêng nội bộ
```
**Vì sao thiết kế này nhanh nhất:**
1. **2 PROCESS riêng** (không phải thread chung): Python có GIL → nếu chung sẽ nghẽn. Tách process → control C++ chạy **50Hz đều tăm tắp** bất kể detection nhanh chậm.
2. **Luồng camera riêng** (CameraGrabber): inference luôn lấy **frame tươi nhất**, không xử lý frame cũ → latency thấp. Không có nó: capture chờ infer → frame cũ dần → bám trễ.
3. **Detection chậm (15Hz) nhưng control nhanh (50Hz)**: Kalman trong control **nội suy + dự đoán** giữa 2 lần detection → drone vẫn mượt 50Hz dù detector chỉ 15Hz. Đây là mấu chốt chống giật.
4. Bỏ frame trùng (`fid==last_fid`) → không infer lại vô ích.

> Muốn control C++ chạy ưu tiên real-time (mượt hơn nữa): `sudo chrt -f 80 ./follow_px4` (SCHED_FIFO).

---

## 1. Đấu nối phần cứng (Jetson ↔ Pixhawk)
- Jetson UART (`/dev/ttyTHS1`) ↔ Pixhawk **TELEM2** (TX↔RX, RX↔TX, GND↔GND). Baud **921600**.
- Camera USB vào Jetson.
- PX4 param cho TELEM2 nhận MAVLink: `MAV_1_CONFIG = TELEM2`, `MAV_1_MODE = Onboard`, `SER_TEL2_BAUD = 921600`.
- (Test bàn: dùng USB, conn `serial:///dev/ttyACM0:57600`.)

---

## 2. Chỉnh PARAM trong QGroundControl (QGC → Vehicle Setup → Parameters)

### A. Cho phép OFFBOARD (bắt buộc)
| Param | Giá trị | Ý nghĩa |
|---|---|---|
| `COM_RCL_EXCEPT` | **4** | Cho offboard chạy khi KHÔNG có RC (bit 2). Bay test nên giữ RC → có thể để 0 và switch bằng RC. |
| `COM_OF_LOSS_T` | **0.5** | Mất tín hiệu offboard >0.5s → failsafe (an toàn). |
| `COM_OBL_RC_ACT` | **2 (RTL)** hoặc 1 (Land) | Mất offboard → tự về nhà / hạ. |
| `NAV_RCL_ACT` | 2 (RTL) | Mất RC → RTL. |

### B. Làm MƯỢT chuyển động (quan trọng — chống giật)
| Param | Giá trị | Ý nghĩa |
|---|---|---|
| `MPC_XY_VEL_MAX` | **3.0** | Trần vận tốc ngang (khớp `v_max_xy` trong code). |
| `MPC_Z_VEL_MAX_UP` | 1.5 | Trần lên. |
| `MPC_Z_VEL_MAX_DN` | 1.5 | Trần xuống. |
| `MPC_ACC_HOR` | **2.0** | Gia tốc ngang mục tiêu — NHỎ = mượt, lớn = gắt. |
| `MPC_ACC_HOR_MAX` | 3.0 | Trần gia tốc ngang. |
| `MPC_JERK_MAX` | **8.0** | Trần jerk — giảm xuống 4-6 nếu vẫn thấy khựng. |
| `MPC_JERK_AUTO` | 4.0 | Jerk auto-smoothing. |
| `MPC_TILTMAX_AIR` | 25 | Nghiêng tối đa (nhỏ = êm, chậm hơn). |
| `MC_YAWRATE_MAX` | **60** | Trần tốc độ xoay (khớp `yaw_rate_max_deg`). Nhỏ = xoay êm. |
| `MPC_YAW_EXPO` | 0.1 | Làm mượt đáp ứng yaw. |

### C. Tinh chỉnh BÁM vận tốc (nếu drone "trôi" hoặc dao động)
| Param | Giá trị | Ý nghĩa |
|---|---|---|
| `MPC_XY_VEL_P_ACC` | 1.8 (mặc định) | P loop vận tốc ngang. Dao động → giảm; chậm → tăng. |
| `MPC_XY_VEL_I_ACC` | 0.4 | I — khử lệch tĩnh (gió). |
| `MPC_XY_VEL_D_ACC` | 0.2 | D — giảm overshoot. |

### D. Ước lượng vị trí (BẮT BUỘC offboard mới chạy)
- **Bay ngoài trời có GPS:** mặc định OK. Kiểm tra `EKF2_GPS_CTRL` bật, fix GPS tốt.
- **Bay thấp / trong nhà (GPS yếu):** cần **optical flow + distance sensor**:
  | Param | Giá trị |
  |---|---|
  | `EKF2_HGT_REF` | 2 (Range) nếu có distance sensor, else Baro |
  | `EKF2_RNG_CTRL` | 1 (bật range cho độ cao) |
  | `EKF2_OF_CTRL` | 1 (bật optical flow) |
  | `EKF2_GPS_CTRL` | 0 nếu không GPS |
  > Không có position estimate hợp lệ → PX4 TỪ CHỐI offboard velocity. Kiểm tra QGC báo "Ready to Fly".

### E. Failsafe an toàn (đặt TRƯỚC khi bay)
| Param | Giá trị |
|---|---|
| `COM_LOW_BAT_ACT` | 2 (RTL) hoặc 3 (Land) |
| `GF_ACTION` | 2 (RTL) — geofence |
| `GF_MAX_HOR_DIST` | 30 (m) — bán kính tối đa |
| `GF_MAX_VER_DIST` | 15 (m) — trần độ cao |
| `RTL_RETURN_ALT` | 10 |

---

## 3. Cài MAVSDK trên Jetson (1 lần)
```bash
# Cách nhanh: tải .deb từ github.com/mavlink/MAVSDK/releases (chọn arm64/ubuntu khớp JetPack)
wget https://github.com/mavlink/MAVSDK/releases/download/v2.12.2/libmavsdk-dev_2.12.2_ubuntu20.04_arm64.deb
sudo dpkg -i libmavsdk-dev_*.deb
# (hoặc build từ nguồn nếu .deb không khớp — xem mavsdk.mavlink.io)
```

## 4. Build bộ điều khiển C++
```bash
cd jetson/control
mkdir build && cd build
cmake .. && make -j4
# ra file ./follow_px4
```

---

## 5. TEST SITL TRƯỚC (an toàn, KHÔNG cần drone)
Trên máy desktop (hoặc Jetson) cài PX4 SITL + jMAVSim/Gazebo:
```bash
# Terminal 1: PX4 SITL
cd PX4-Autopilot && make px4_sitl jmavsim       # drone ảo bay được trong sim
# Terminal 2: control nối SITL (cổng 14540)
./follow_px4 "udpin://0.0.0.0:14540"
# Terminal 3: detector giả lập (hoặc detector_pub với webcam — đứng trước cam)
python3 ../detector_pub.py --onnx pfdet_v17_512.onnx --img-size 512 --cam 0
```
Trong QGC nối SITL: arm → takeoff → switch Offboard. Xem drone ảo bám "target" mượt không.
**Chỉ bay thật khi SITL chạy ổn, không giật, không bay loạn.**

---

## 6. Chạy trên DRONE THẬT
```bash
# Trên Jetson, sau khi đã arm + bay lên + ổn định:
# Terminal 1: detector
python3 jetson/detector_pub.py --onnx deploy/pfdet_v17_512.onnx --img-size 512 --cam 0 \
    --fy 700 --person-h 1.7 --ctrl-port 5600
# Terminal 2: control (serial Pixhawk)
cd jetson/control/build && ./follow_px4          # mặc định serial:///dev/ttyTHS1:921600
```
Quy trình: cất cánh tay (RC) → ổn định → bật **Offboard** (RC switch). Drone bắt đầu bám. **Tay luôn ở RC để giành lại quyền (Position/Altitude mode) bất cứ lúc nào.**

---

## 6b. KIỂU SETPOINT — TỰ ĐỘNG, KHÔNG CẦN CHỈNH GÌ
Code **luôn tự chọn kiểu MƯỢT NHẤT**: gửi **position + velocity feedforward** ("carrot") để PX4 chạy vòng
position + bộ làm mượt jerk nội bộ. **TỰ ĐỘNG fallback** sang velocity-only **chỉ khi** position estimate
chưa sẵn (EKF chưa ổn) — để không bao giờ bay loạn. Bạn **chỉ chạy 1 lệnh, không đụng cờ nào**.
- Log in `[MODE] position+velocity feedforward (MƯỢT NHẤT)` khi estimate tốt (bình thường ngoài trời có GPS).
- Log `[MODE] velocity-only (đang chờ position estimate)` lúc EKF chưa hội tụ (mới khởi động / GPS chưa fix) → tự lên mượt-nhất khi estimate sẵn.
- Muốn mượt/đáp ứng khác đi: chỉ chỉnh 1 số `lookahead_s` trong `Cfg` (0.3 đáp ứng nhanh ↔ 0.6 mượt hơn). Mặc định 0.4 đã tốt.
> Tóm lại: **chạy là tự mượt nhất.** Outdoor có GPS → luôn ở chế độ mượt nhất ngay.

## 6c. (Tùy chọn) Thu data val 3-10m thật — `record_flight.py` (CÔNG CỤ RIÊNG, độc lập)
Khi muốn thu ảnh thật lúc bay để làm **val set 3-10m** (đo đúng deployment của bạn) hoặc finetune:
```bash
python3 record_flight.py --out flight_data --every 15            # lưu ảnh mỗi 15 frame (~2/giây)
python3 record_flight.py --out flight_data --every 15 --video    # + lưu cả video
python3 record_flight.py --out flight_data --show                # xem live (space=chụp ngay, q=dừng)
```
Độc lập hoàn toàn với code deploy (chỉ camera+opencv, không cần model). Thu xong → label tay (hoặc pre-label
bằng model) → val set riêng. **Đây là cách đo model TỐT NHẤT cho bài toán 3-10m của bạn** (hơn VisDrone-val tí xíu).

## 7. Tinh chỉnh controller (trong `follow_px4.cpp`, struct `Cfg`)
| Triệu chứng | Chỉnh |
|---|---|
| Bám chậm/trễ | tăng `kp_fwd`, `kp_yaw`; tăng `vel_ff_gain` (0.8→1.0) |
| Dao động (lắc qua lại) | giảm `kp_*`; tăng `kd_*`; giảm `a_max` |
| Vẫn giật/khựng | giảm `a_max` (1.5→1.0); giảm PX4 `MPC_JERK_MAX`, `MPC_ACC_HOR` |
| Phản ứng nhiễu detection | tăng Kalman `r` (kf_ex.r, kf_range.r); giảm `q` |
| Bám không kịp người nhanh | giảm Kalman `r`; tăng `q`; tăng `ctrl_hz` |
| Khoảng cách sai | calibrate `fy` (chụp người ở 5m, đo bbox px, fy = bbox_px * 5 / 1.7) |

**Quan trọng:** `fx`, `fy` phải **calibrate đúng** (dùng `follow_drone/scripts/calibrate_camera.py`) — sai tiêu cự → khoảng cách & yaw sai → bám lệch.

---

## 8. CHECKLIST AN TOÀN trước khi bay
- [ ] SITL chạy mượt, không bay loạn.
- [ ] `fx/fy` đã calibrate.
- [ ] Failsafe (mục E) đã đặt: mất offboard/RC/pin → RTL/Land.
- [ ] Geofence bật (`GF_*`).
- [ ] RC luôn sẵn sàng giành quyền (test switch Offboard↔Position trên mặt đất).
- [ ] Bay nơi trống, gió nhẹ, có người spotter.
- [ ] Lần đầu: để `v_max_xy=1.5`, `follow_distance_m=6` (chậm + xa cho an toàn), tăng dần sau.
- [ ] Pin đầy, GPS fix tốt (nếu dùng GPS).

---

## Nguồn (SOTA grounded)
- PX4 Offboard: https://docs.px4.io/main/en/flight_modes/offboard
- MAVSDK Offboard C++: https://mavsdk.mavlink.io/main/en/cpp/guide/offboard.html
- Kalman bù trễ (PVT++): https://arxiv.org/pdf/2211.11629 ; SMART-TRACK KF UAV: https://arxiv.org/html/2410.10409v1
- IBVS + acceleration limiting (TU Delft 2024): https://repository.tudelft.nl/file/File_bfe89460-33ee-4842-9f04-181bf6accde1
- Min-jerk setpoint + feedforward + high-freq controller: https://ieeexplore.ieee.org/document/8550309
