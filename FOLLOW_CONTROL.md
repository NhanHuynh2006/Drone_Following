# Hệ thống điều khiển Drone Follow Person — Phân tích chi tiết

> **Mục tiêu:** drone tự động bay theo người được phát hiện bởi PFDet-Nano v15.
> **Hardware:** Raspberry Pi 5 (companion computer) + Pixhawk 6C (FCU) + Mission Planner (GCS).
> **Output cuối cùng:** drone giữ khoảng cách an toàn 5m phía sau người, theo dõi liên tục, có failsafe đầy đủ.

Tài liệu này thiết kế thuật toán điều khiển theo cùng nguyên tắc với ARCHITECTURE.md: **mỗi quyết định đều có lý do chắc chắn, so sánh với alternatives, dựa trên paper top venue và open-source project hàng đầu.**

---

# Phần 0: Thuật ngữ điều khiển drone

Trước khi vào nội dung, đây là các thuật ngữ tiếng Anh sẽ xuất hiện. Đọc qua một lượt để khi gặp lại không bị bối rối.

## 0.1. Thuật ngữ về phần cứng drone

**FCU (Flight Control Unit)** — bộ điều khiển bay. Pixhawk 6C là một FCU. Chạy firmware như ArduPilot hoặc PX4.

**Companion Computer** — máy tính phụ chạy song song với FCU, dùng để xử lý vision/AI nặng. Pi 5 là companion computer. FCU lo điều khiển low-level (motor, attitude), companion lo high-level (detection, tracking, decision).

**GCS (Ground Control Station)** — phần mềm trên laptop/desktop để giám sát và điều khiển drone từ xa. Mission Planner là một GCS. Kết nối với drone qua telemetry radio hoặc WiFi.

**MAVLink** — giao thức (protocol) chuẩn để các phần (FCU ↔ companion ↔ GCS) giao tiếp với nhau. Là tin nhắn binary, định nghĩa sẵn các loại message (heartbeat, position, command...).

**FCU mode (chế độ bay)** — cách drone phản ứng với input. Quan trọng:
- **MANUAL / STABILIZE:** pilot điều khiển trực tiếp qua remote
- **GUIDED:** drone nhận lệnh từ MAVLink (vị trí, vận tốc) — dùng cho follow
- **AUTO:** drone bay theo mission đã định sẵn
- **LOITER:** giữ vị trí (hover)
- **RTL (Return To Launch):** tự bay về điểm cất cánh
- **LAND:** hạ cánh ngay lập tức tại vị trí hiện tại

**IMU (Inertial Measurement Unit)** — cảm biến đo gia tốc + góc quay. Pixhawk 6C có IMU built-in.

**EKF (Extended Kalman Filter)** — bộ lọc trên Pixhawk fuse GPS + IMU + barometer để ước lượng pose drone. Output: vị trí, vận tốc, attitude.

**Barometer** — cảm biến áp suất không khí để đo độ cao.

**Telemetry radio** — radio không dây (vd 433/915 MHz) để truyền MAVLink giữa drone và GCS.

**UART (serial port)** — kết nối có dây giữa Pi 5 và Pixhawk. Tốc độ baud thường 921600.

## 0.2. Thuật ngữ về tọa độ và chuyển động

**Body frame** — hệ tọa độ gắn với drone. Trục X = mũi drone, Y = bên phải, Z = xuống.

**World frame / Inertial frame** — hệ tọa độ cố định trên mặt đất. Hai chuẩn:
- **NED (North-East-Down):** X bắc, Y đông, Z xuống. ArduPilot dùng NED.
- **ENU (East-North-Up):** X đông, Y bắc, Z lên. ROS dùng ENU.

**Local NED** — gốc tọa độ NED đặt tại điểm cất cánh.

**Roll / Pitch / Yaw** — 3 góc Euler:
- **Roll (lăn):** xoay quanh trục X (mũi drone). Nghiêng trái/phải.
- **Pitch (chúi):** xoay quanh trục Y. Cúi xuống / ngửa lên.
- **Yaw (lái):** xoay quanh trục Z. Quay trái/phải.

**LOS (Line of Sight)** — đường thẳng từ camera đến target.

**FOV (Field of View)** — góc nhìn camera. Vd 90° HFOV (horizontal) nghĩa là camera nhìn được 90° theo chiều ngang.

**Setpoint** — giá trị mong muốn (target value). Vd "muốn drone ở độ cao 10m" thì 10m là altitude setpoint.

**Velocity setpoint** — vận tốc mong muốn (m/s). Vd `(vx=2, vy=0, vz=0)` nghĩa là bay tới với 2 m/s.

**Position setpoint** — vị trí mong muốn (m). Vd `(x=10, y=5, z=-10)` trong NED nghĩa là bắc 10m, đông 5m, cao 10m (z âm vì NED Down).

**Yaw rate** — vận tốc góc yaw (rad/s). Vd 0.5 rad/s ≈ 28.6 độ/giây.

## 0.3. Thuật ngữ về điều khiển

**PID (Proportional-Integral-Derivative)** — bộ điều khiển 3 thành phần:
- **P:** tỉ lệ với sai số hiện tại
- **I:** tích lũy sai số theo thời gian (xử lý sai số dài hạn)
- **D:** đạo hàm sai số (dampen — giảm dao động)
- Output = `Kp × error + Ki × ∫error dt + Kd × derror/dt`

**Cascade PID** — nhiều PID nối tiếp. Vd: position PID → velocity setpoint → velocity PID → attitude setpoint → attitude PID → motor commands.

**LQR (Linear Quadratic Regulator)** — bộ điều khiển tối ưu cho hệ tuyến tính. Cần model toán học của drone.

**MPC (Model Predictive Control)** — bộ điều khiển dự đoán: ở mỗi step, giải bài toán tối ưu cho N step tương lai, áp dụng step đầu, lặp lại.

**Visual Servoing** — điều khiển drone dựa trực tiếp trên thông tin từ camera:
- **PBVS (Position-Based Visual Servoing):** ước lượng 3D position của target rồi điều khiển trong world frame
- **IBVS (Image-Based Visual Servoing):** điều khiển trong image space (pixel coords), không cần ước lượng 3D

**State estimation** — ước lượng trạng thái (vị trí, vận tốc) của target từ measurements ồn.

**Kalman Filter (KF)** — bộ lọc tối ưu cho hệ tuyến tính có noise Gaussian. Output: state ước lượng smooth.

**Process noise** — nhiễu trong model dynamics (target có thể accelerate bất ngờ).

**Measurement noise** — nhiễu trong observation (detection bị lệch vài pixel).

## 0.4. Thuật ngữ về tracking

**MOT (Multi-Object Tracking)** — bài toán theo dõi nhiều object qua nhiều frame, gán ID nhất quán.

**Track / Tracklet** — chuỗi detection của cùng 1 object qua nhiều frame.

**Re-identification (ReID)** — xác định cùng 1 object dù ngắt quãng (vd bị che khuất rồi xuất hiện lại).

**HOTA (Higher Order Tracking Accuracy)** — metric chính đánh giá MOT. HOTA = sqrt(detection_accuracy × association_accuracy).

**MOTA, MOTP, IDF1** — các metric khác đo MOT (cũ hơn HOTA).

**Hungarian algorithm** — thuật toán matching tối ưu (gán detection với track).

**Track lost** — track bị mất (target ra khỏi frame, bị che khuất).

**Track confirmed** — track ổn định (đã thấy nhiều frame liên tiếp).

**ID switch** — track gán nhầm ID, lỗi nghiêm trọng trong MOT.

## 0.5. Thuật ngữ khác

**Geofence** — vùng giới hạn drone không được ra khỏi. Có thể là cylinder (bán kính + độ cao) hoặc polygon.

**Failsafe** — cơ chế tự động khi có sự cố (mất GPS, pin yếu, mất GCS link). Vd auto-RTL.

**Heartbeat** — message MAVLink định kỳ (1 Hz) báo "tôi vẫn sống". Mất heartbeat → trigger failsafe.

**WPNAV_SPEED** — tốc độ navigation tối đa của ArduPilot khi auto bay đến waypoint.

**SET_POSITION_TARGET_LOCAL_NED** — message MAVLink để gửi position/velocity setpoint trong local NED frame.

**HDOP (Horizontal Dilution of Precision)** — chất lượng GPS horizontal. <1 = tốt, >2 = kém.

**3D Fix** — GPS đã có ít nhất 4 vệ tinh, đủ để định vị 3D.

---

# Phần I: Tổng quan hệ thống

## 1.1. Sơ đồ data flow

```
┌────────────────────────────────────────────────────────────────────────┐
│ Raspberry Pi 5 (Companion Computer)                                     │
│                                                                          │
│  Camera ──► PFDet-Nano v15 ──► OC-SORT ──► Target Selection ──► EKF     │
│  (USB/CSI)    (detection)      (tracking)    (chọn 1 ID)      (smooth)  │
│                                                                  │       │
│                                                                  ▼       │
│                                            Hybrid Visual Servoing       │
│                                            ├── IBVS (yaw, gimbal)       │
│                                            └── PBVS (forward range)     │
│                                                                  │       │
│                                                                  ▼       │
│                                                      Cascade PID 3 axis  │
│                                                                  │       │
│                                                                  ▼       │
│                                                      pymavlink @ 10 Hz   │
└─────────────────────────────────────────────────────────────────│───────┘
                                                                  │
                                                          UART/Telemetry
                                                                  │
                                                                  ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Pixhawk 6C (Flight Controller, ArduPilot Copter)                        │
│                                                                          │
│  SET_POSITION_TARGET_LOCAL_NED → GUIDED mode handler                    │
│                                            │                             │
│                                            ▼                             │
│  EKF (GPS + IMU + Baro) ─► Position PID ─► Velocity PID ─► Attitude PID │
│                                                                  │       │
│                                                                  ▼       │
│                                                      ESC commands       │
│                                                                  │       │
│                                                                  ▼       │
│                                                      Motors (×4)         │
└────────────────────────────────────────────────────────────────────────┘

Mission Planner ←── Telemetry Radio ──► Pixhawk
(GCS, calib, geofence, emergency override)
```

## 1.2. Kết nối phần cứng

**Pi 5 ↔ Pixhawk 6C:**
- UART qua TELEM2 port của Pixhawk
- Baud rate: 921600 (cần config trong ArduPilot)
- Pin out: TX/RX/GND (3 dây)
- Power: cấp riêng cho Pi 5 (không dùng power từ Pixhawk vì Pi 5 ăn ~5W)

**Pixhawk 6C ↔ GCS (Mission Planner):**
- Telemetry radio 433/915 MHz qua TELEM1 (đã có sẵn từ build drone của bạn)
- ArduPilot route MAVLink giữa TELEM1 và TELEM2 → GCS có thể thấy Pi command

**Camera ↔ Pi 5:**
- USB camera hoặc CSI camera (Pi camera module)
- 640×480 @ 30 FPS

## 1.3. Hệ thống ArduPilot config cần thiết (làm từ Mission Planner)

Trước khi code, set các parameters sau qua Mission Planner → CONFIG/TUNING → Full Parameter List:

| Parameter | Value | Mục đích |
|-----------|-------|----------|
| `SERIAL2_PROTOCOL` | 2 (MAVLink2) | TELEM2 dùng MAVLink |
| `SERIAL2_BAUD` | 921 | Baud 921600 |
| `BRD_SER2_RTSCTS` | 0 | Disable flow control nếu chỉ 3 dây |
| `SYSID_MYGCS` | 255 | ID GCS (default) |
| `FS_GCS_ENABLE` | 1 | Enable GCS failsafe |
| `FS_OPTIONS` | 0 | RTL khi mất GCS |
| `FENCE_ENABLE` | 1 | Enable geofence |
| `FENCE_TYPE` | 7 | Cylinder + altitude + polygon |
| `FENCE_ACTION` | 1 | RTL khi vượt fence |
| `FENCE_RADIUS` | 100 | Radius 100m từ home |
| `FENCE_ALT_MAX` | 30 | Trần 30m |
| `BATT_LOW_VOLT` | 22.0 | Cảnh báo pin (cho 6S) |
| `BATT_CRT_VOLT` | 21.0 | Pin cực thấp |
| `BATT_FS_LOW_ACT` | 2 | RTL khi pin thấp |
| `BATT_FS_CRT_ACT` | 1 | LAND khi pin cực thấp |
| `WPNAV_SPEED` | 500 | Tốc độ tối đa 5 m/s |
| `WPNAV_SPEED_DN` | 200 | Tốc độ giảm độ cao 2 m/s |
| `RTL_ALT` | 1500 | Độ cao RTL 15m |

---

# Phần II: Lựa chọn từng module

Mỗi module dưới đây có cấu trúc:
1. **Vấn đề cần giải quyết**
2. **Lựa chọn được chọn**
3. **Cách hoạt động chi tiết**
4. **Các phương án thay thế đã cân nhắc**
5. **Tại sao chọn cái này, KHÔNG chọn cái kia**

---

## 2.1. Multi-Object Tracking: OC-SORT

### Vấn đề
Detector (PFDet-Nano v15) chỉ output bboxes của người trong từng frame, **không biết đâu là cùng 1 người qua các frame**. Nếu drone đang follow A mà detection thay đổi ID giữa các frame → drone sẽ "nhảy" qua người khác, mất follow.

Yêu cầu cụ thể:
- **Gán ID nhất quán** cho cùng 1 người qua nhiều frame
- **Robust với occlusion** (che khuất tạm thời 2-3 giây): khi người đi sau cây/cột, ID vẫn giữ
- **Real-time trên Pi 5:** <5ms/frame (vì còn budget cho detection và control)
- **Không cần extra NN** (Pi 5 đã ăn 50ms cho detection, không còn budget cho ReID network)

### Lựa chọn được chọn: OC-SORT (CVPR 2023)

**OC-SORT = Observation-Centric SORT**, cải tiến của SORT bằng 2 cơ chế:
1. **ORU (Observation-centric Re-Update):** khi track bị lost rồi tìm lại, update Kalman filter bằng observations cũ thay vì chỉ predicted state → giảm drift
2. **OCM (Observation-Centric Momentum):** thêm directional consistency vào cost function → tracking robust với non-linear motion (người đổi hướng đột ngột)

### Cách hoạt động chi tiết

**Pipeline mỗi frame:**

```
1. Detection từ PFDet → list bboxes [(x1,y1,x2,y2,score), ...]

2. Predict step:
   - Mỗi track active có Kalman filter với state 8-dim:
     [cx, cy, area, aspect_ratio, ċx, ċy, ȧrea, ȧspect]
   - Predict state từ Kalman → predicted bbox

3. Association (Hungarian algorithm):
   - Cost matrix: IoU giữa detection và predicted bbox
   - Modified IoU = IoU + λ × cos(angle giữa motion vectors)  ← OCM
   - Match: detection ↔ track có IoU cao nhất

4. Update step:
   - Matched track: update Kalman với detection
   - Unmatched detection: tạo track mới (chưa confirmed)
   - Unmatched track: tăng age, nếu age > max_age thì delete

5. ORU (khi track tìm lại):
   - Track bị lost N frame, giờ tìm lại
   - Re-run Kalman update từ time bị lost với observation cũ
   - Smooth lại track history → state mới chính xác hơn
```

**Tham số được khuyến nghị:**

| Param | Value | Ý nghĩa |
|-------|-------|---------|
| `max_age` | 90 frame (=3 giây ở 30 FPS) | Giữ track đến 3s sau khi lost |
| `min_hits` | 3 | Cần 3 detection liên tiếp mới confirm track mới |
| `iou_threshold` | 0.3 | IoU tối thiểu để match |
| `delta_t` | 3 | OCM look-back 3 frame |

**Ví dụ cụ thể với occlusion:**

- Frame 100: track ID=1, người ở (320, 240), Kalman ước lượng vận tốc (5 px/frame, 0)
- Frame 101-130 (1 giây): người bị cây che, không có detection
  - Kalman tự predict: frame 101 → (325, 240), frame 102 → (330, 240), ...
  - Track không bị delete vì age < max_age
- Frame 131: người ra khỏi cây, detection ở (480, 240)
  - Predicted bbox tại (475, 240) — IoU với detection = 0.85
  - Match thành công, track ID=1 vẫn giữ
  - ORU: re-update Kalman với observations cũ → giảm cumulative error

### Các phương án đã cân nhắc

| Tracker | Pros | Cons | Speed CPU |
|---------|------|------|-----------|
| **A. SORT** (2016) | Đơn giản, rất nhanh | Drop ID sau ~0.5s occlusion (chỉ Kalman + Hungarian) | 1000+ FPS |
| **B. DeepSORT** (2017) | ReID embedding giúp robust | Cần thêm CNN ~10ms, không có budget | 100 FPS + CNN |
| **C. ByteTrack** (ECCV 2022) | Recover low-score detections | Vẫn dùng linear KF, kém hơn OC-SORT cho non-linear motion | 800 FPS |
| **D. BoT-SORT** (2022) | Thêm camera motion compensation + ReID | Phức tạp, cần ReID NN | Slower |
| **E. StrongSORT** (2022) | SOTA accuracy on benchmark | Quá chậm cho real-time | <30 FPS |
| **F. OC-SORT (đã chọn)** | ORU + OCM xử lý non-linear motion + occlusion | Hơi phức tạp hơn SORT | 700+ FPS |

### Tại sao chọn OC-SORT
1. **Specifically xử lý use case này:**
   - Drone bay → camera chuyển động → motion target không linear
   - Người đi sau cây 2-3s → cần track survive
   - OC-SORT design cho đúng 2 vấn đề này (CVPR 2023 paper)
2. **Không cần ReID NN:**
   - Pi 5 đã ăn 50ms cho detection
   - Thêm ReID 10ms → <30 FPS, không real-time
   - OC-SORT hoàn toàn dựa vào motion + IoU → free
3. **HOTA cao hơn ByteTrack:**
   - MOT17: OC-SORT 63.2 HOTA, ByteTrack 63.1 (gần ngang)
   - MOT20: OC-SORT 62.4, ByteTrack 61.3 (OC-SORT hơn 1.1)
   - DanceTrack (non-linear motion): OC-SORT 55.1, ByteTrack 47.7 (hơn 7.4!) — đúng use case của ta
4. **Lý do KHÔNG chọn alternatives:**
   - **Không SORT:** chỉ 0.5s occlusion tolerance, không đủ
   - **Không DeepSORT:** ReID NN quá đắt cho Pi 5
   - **Không ByteTrack:** kém hơn OC-SORT cho non-linear motion 7%
   - **Không BoT-SORT/StrongSORT:** quá nặng

### Single-target mode tweak

Vì ta chỉ follow 1 người, có thể đơn giản hóa OC-SORT:
1. Track all candidate detections như bình thường
2. Chỉ "lock" 1 ID làm target (chọn ban đầu hoặc người gần nhất)
3. Locked ID có priority cao hơn trong association (cost matrix giảm 0.1 cho IoU)
4. Nếu locked ID lost > 3s → unlock, switch sang tracking mode bình thường

**Reference implementation:** [`noahcao/OC_SORT`](https://github.com/noahcao/OC_SORT) — copy file `ocsort.py` (200 dòng).

---

## 2.2. Distance Estimation: Pinhole Model

### Vấn đề
Để drone biết "cách người bao xa", cần ước lượng khoảng cách từ camera đến người. Pi 5 không có LiDAR/depth camera → phải ước lượng từ ảnh đơn (monocular).

### Lựa chọn được chọn: Pinhole Model + Known Person Height

**Công thức:**
```
distance = (focal_length × real_height) / pixel_height
       Z = (f_y × H_real) / h_pixels
```

Với:
- `f_y` = focal length theo trục Y của camera (đơn vị pixel)
- `H_real` = chiều cao thật của người (giả định 1.7m = 1700mm)
- `h_pixels` = chiều cao bbox trong ảnh (pixel)

### Cách hoạt động chi tiết

**Bước 1: Calibrate camera (chỉ làm 1 lần)**

Dùng OpenCV calibrateCamera với checkerboard 9×6 ô (mỗi ô 25mm):
```python
import cv2
import numpy as np

# Chụp ~20 ảnh checkerboard ở các góc khác nhau
# Detect corners, dùng cv2.calibrateCamera()
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(...)

# K = [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
f_x, f_y = K[0, 0], K[1, 1]
c_x, c_y = K[0, 2], K[1, 2]
print(f"f_x={f_x}, f_y={f_y}")
```

Thường f_x ≈ f_y ≈ 600-800 pixel cho camera 640×480 với HFOV ~70°.

**Bước 2: Compute distance từ bbox**

```python
def estimate_distance(bbox_height_px, f_y=700, person_height_m=1.7):
    """Pinhole distance estimation."""
    if bbox_height_px <= 0:
        return None
    distance_m = (f_y * person_height_m) / bbox_height_px
    return distance_m
```

**Ví dụ số cụ thể:**

| Bbox height (px) | Distance ước lượng | Sai số expect |
|------------------|---------------------|----------------|
| 240 (người gần, full body) | 4.96 m | ±0.3 m |
| 120 | 9.92 m | ±0.5 m |
| 60 | 19.83 m | ±1.5 m |
| 30 (người xa) | 39.67 m | ±5 m |

Càng xa, sai số càng lớn (do bbox nhỏ, sai 1 pixel = sai nhiều mét).

**Bước 3: Robust check và filtering**

```python
def robust_distance(bbox_height_px, f_y, person_height_m=1.7,
                    min_height_px=20, max_dist_m=30):
    """Reject outliers."""
    if bbox_height_px < min_height_px:
        return None  # Người quá xa, không reliable
    d = (f_y * person_height_m) / bbox_height_px
    if d > max_dist_m:
        return None
    return d
```

**Bước 4: Smooth bằng Kalman filter (Section 2.4)**

Distance estimates raw có thể nhảy ±1m giữa các frame do bbox jitter. Pass qua Kalman để smooth.

### Self-calibration cho người không cao 1.7m

**Vấn đề:** giả định 1.7m không đúng cho mọi người (1.5m - 1.9m thực tế).

**Solution one-shot:**
- Lúc bắt đầu follow, đo bbox height ở 1 distance đã biết
- Distance này có thể lấy từ GPS-to-phone distance nếu user mang phone với GPS chia sẻ
- Back-solve: `H_real = bbox_height × distance / f_y`
- Cache `H_real` cho user này

```python
def self_calibrate_height(bbox_height_px, gps_distance_m, f_y):
    """Compute real person height from GPS distance + bbox."""
    return (bbox_height_px * gps_distance_m) / f_y
```

### Fusion với altitude khi có sẵn

Nếu drone có altitude AGL `h` từ barometer/rangefinder và pitch `θ`:

```
ground_distance = h / tan(θ + α)
```

Với α là góc từ camera ray đến target bottom row (tính từ pixel coords và camera intrinsics).

Average với pinhole distance theo inverse-variance weight:
```python
def fused_distance(d_pinhole, d_ground, var_pinhole, var_ground):
    if d_ground is None:
        return d_pinhole
    w1 = 1 / var_pinhole
    w2 = 1 / var_ground
    return (w1 * d_pinhole + w2 * d_ground) / (w1 + w2)
```

### Các phương án đã cân nhắc

| Phương án | Cost | Accuracy | Notes |
|-----------|------|----------|-------|
| **A. Pinhole + person height (đã chọn)** | ~0 | ±5-10% ở 2-15m | Chuẩn ngành, paper NeoARCADE 2025 báo MAPE 1.94% ở 1.1-7.5m |
| **B. Pinhole + shoulder width** | ~0 | Hơi noisy hơn | Useful khi feet bị clip ra ngoài frame |
| **C. Altitude + ground homography** | ~0 | Tốt khi target trên đất phẳng | Cần barometer/rangefinder + pitch |
| **D. MiDaS (monocular depth NN)** | 100s ms trên Pi 5 | Chỉ relative depth | Quá chậm, không metric |
| **E. ZoeDepth (metric depth NN)** | Không feasible | Metric tốt | Cần GPU, không cho Pi 5 |
| **F. Stereo camera** | Real-time được | Best | Cần thêm hardware |
| **G. RealSense D435 (active depth)** | Real-time | Best đến 3m | Outdoor lighting yếu |

### Tại sao chọn Pinhole
1. **Chuẩn ngành:** mọi paper drone person-following từ 2017+ dùng cách này
2. **Cost ~0:** chỉ 1 phép chia, không tốn compute
3. **Đủ accuracy cho follow:**
   - Sai số 5-10% ở 2-15m → đủ để giữ khoảng cách 5±0.5m
   - Paper NeoARCADE 2025: MAPE 1.94% ở 1.1-7.5m
4. **Lý do KHÔNG chọn alternatives:**
   - **Không MiDaS/ZoeDepth:** quá chậm cho Pi 5 (>100ms), MiDaS chỉ relative không metric
   - **Không stereo/depth camera:** thay đổi hardware, tăng cost
   - **Không altitude+homography one-shot:** chỉ tốt khi đất phẳng, nhưng có thể fuse như backup

---

## 2.3. Visual Servoing: Hybrid 2.5D

### Vấn đề
Có 2 trường phái lớn về drone visual control:
- **PBVS:** ước lượng 3D position của target → control trong world frame (NED)
- **IBVS:** dùng pixel error trực tiếp → control trong image space

PBVS dễ hiểu (control như "đi tới điểm X"), nhưng phụ thuộc vào depth estimation. Khi depth nhảy (do bbox jitter), drone sẽ jerk.

IBVS robust hơn (không cần 3D), nhưng phải define setpoint trong image space (vd "giữ bbox center ở pixel (320, 240) với area 100×200"). Khi người thay đổi pose (cúi xuống → bbox area giảm), drone sẽ tưởng người đi xa và bay tới → wrong.

### Lựa chọn được chọn: Hybrid 2.5D

**Ý tưởng:** dùng IBVS cho axis nào KHÔNG cần 3D, dùng PBVS cho axis nào CẦN 3D.

| Axis | Approach | Lý do |
|------|----------|-------|
| **Yaw (xoay trái/phải)** | IBVS | Chỉ cần giữ bbox horizontal center → không cần 3D |
| **Pitch gimbal (camera up/down)** | IBVS | Tương tự yaw |
| **Forward/backward (range)** | PBVS | Cần biết khoảng cách metric (m) để giữ 5m |
| **Lateral (trái/phải drone)** | Mixed | Có thể IBVS nếu chỉ cần follow, hoặc PBVS nếu muốn orbit |
| **Altitude** | PBVS hoặc fix | Giữ độ cao cố định (vd 5m AGL) hoặc match person altitude |

### Cách hoạt động chi tiết

**IBVS cho yaw (drone xoay đối mặt người):**

```python
def compute_yaw_rate(bbox_cx, image_width, fx, K_yaw=0.5):
    """
    Tính yaw rate để giữ bbox horizontal center ở giữa image.
    
    bbox_cx: x của bbox center (pixel)
    image_width: chiều rộng image
    fx: focal length theo x
    K_yaw: gain proportional
    """
    # Pixel error
    error_px = bbox_cx - image_width / 2
    
    # Convert sang góc (rad)
    error_rad = error_px / fx
    
    # Yaw rate proportional to angle error
    yaw_rate = -K_yaw * error_rad  # negative: turn ngược chiều error
    
    # Saturate
    return clip(yaw_rate, -1.5, 1.5)  # max 1.5 rad/s ≈ 86 deg/s
```

**Ý nghĩa:** nếu người ở phía bên phải image (bbox_cx > image_width/2) → error dương → yaw_rate âm → drone xoay phải để center người.

**IBVS cho gimbal pitch (camera nhìn lên/xuống):**

Tương tự yaw nhưng với trục Y. Nếu drone không có gimbal, có thể skip hoặc dùng altitude PBVS.

**PBVS cho forward range (giữ khoảng cách 5m):**

```python
def compute_forward_velocity(distance_m, distance_setpoint=5.0,
                              Kp=0.5, Ki=0.05, Kd=0.1, dt=0.1):
    """
    PID trên distance error → forward velocity.
    """
    error = distance_m - distance_setpoint  # >0: quá xa, cần tiến
    
    # P term
    P = Kp * error
    
    # I term (integrate)
    integral += error * dt
    integral = clip(integral, -2, 2)  # anti-windup
    I = Ki * integral
    
    # D term
    D = Kd * (error - prev_error) / dt
    prev_error = error
    
    forward_vel = P + I + D
    return clip(forward_vel, -3.0, 3.0)  # cap ±3 m/s
```

**Ví dụ vận hành:**
- Setpoint: 5m
- Distance hiện tại: 8m → error = +3m → forward_vel = 0.5×3 = 1.5 m/s → tiến tới
- Distance hiện tại: 3m → error = -2m → forward_vel = -1.0 m/s → lùi lại

### Các phương án đã cân nhắc

| Phương án | Pros | Cons |
|-----------|------|------|
| **A. PBVS thuần** | Dễ hiểu, control trong meter | Phụ thuộc depth, nhảy khi bbox jitter |
| **B. IBVS thuần** | Không cần 3D, robust với detection noise | Phải define image-space setpoint, không tốt khi pose thay đổi |
| **C. Hybrid 2.5D (đã chọn)** | Best of both | Hơi phức tạp hơn |

### Tại sao chọn Hybrid 2.5D
1. **Yaw error trong image space → control angle:** không cần biết khoảng cách. Nếu người ở bên phải image, drone xoay phải bất kể xa gần
2. **Forward range cần metric:** user nói "giữ 5m" → phải có metric. PBVS với pinhole distance là phù hợp
3. **Robust hơn pure PBVS:** khi depth estimate nhảy, chỉ forward velocity bị ảnh hưởng, yaw vẫn smooth (vì IBVS không dùng depth)
4. **Đã prove:** GPS-Denied IBVS UAV Navigation paper (arXiv 2509.17435, 2025) sử dụng hybrid này

---

## 2.4. State Estimation: 2-stage Kalman Filter

### Vấn đề
Detection có noise (bbox lệch ±2-5 pixel mỗi frame). Tracking có gap (occlusion). Distance ước lượng bị jitter. Nếu feed raw vào PID → output velocity nhảy → drone jerk.

Cần smooth state trước khi pass vào controller.

### Lựa chọn được chọn: 2-stage Kalman Filter

**Stage 1: Image-space KF (đã có trong OC-SORT)**
- State: `[cx, cy, area, aspect, ċx, ċy, ȧrea, ȧspect]` (8-dim)
- Predict bbox khi occlusion
- Smooth bbox jitter

**Stage 2: World-frame EKF (target trong NED)**
- State: `[x, y, z, vx, vy, vz]` (6-dim)
- Measurement: drone NED pose + camera ray + estimated range
- Output: target's smooth position trong world frame

### Cách hoạt động chi tiết

**Stage 1: Image-space KF (built into OC-SORT)**

State vector 8-dim với constant velocity model:
```
state = [cx, cy, a, h, ċx, ċy, ȧ, ḣ]
        |       |   |       |
        position    velocity
```

Trong đó a = area, h = bbox height.

Process model (giả định constant velocity):
```
cx(t+1) = cx(t) + ċx(t) × dt
ċx(t+1) = ċx(t) + noise
```

Predict step (mỗi frame, kể cả không có detection):
```
x_pred = F × x_prev   # transition matrix
P_pred = F × P_prev × F^T + Q   # covariance + process noise
```

Update step (khi có detection):
```
y = z - H × x_pred   # innovation
S = H × P_pred × H^T + R   # innovation covariance
K = P_pred × H^T × S^-1   # Kalman gain
x_new = x_pred + K × y
P_new = (I - K × H) × P_pred
```

**Stage 2: World-frame EKF**

State vector 6-dim:
```
state = [x_target, y_target, z_target, vx_target, vy_target, vz_target]
        in world NED frame
```

Tại mỗi frame:

```python
# Step 1: Get drone pose from MAVLink LOCAL_POSITION_NED
drone_pose = (drone_x, drone_y, drone_z)
drone_attitude = (roll, pitch, yaw)

# Step 2: Bbox center → camera ray (in body frame)
ray_camera = pixel_to_ray(bbox_cx, bbox_cy, fx, fy, cx, cy)
# ray_camera is normalized vector

# Step 3: Camera ray → world ray (apply drone attitude rotation)
R_drone = euler_to_rotation_matrix(roll, pitch, yaw)
ray_world = R_drone @ ray_camera

# Step 4: Estimate target world position
target_world = drone_pose + estimated_range * ray_world

# Step 5: EKF update với measurement = target_world
ekf.predict(dt)
ekf.update(target_world, R_measurement)

# Output smoothed target position + velocity
target_state = ekf.state
```

**Process noise tuning:**
- Variance trên position: nhỏ (~0.1 m²) — assume position changes smoothly
- Variance trên velocity: cao (~1 m²/s²) — người có thể accelerate bất ngờ (đang đi → chạy)

**Measurement noise tuning:**
- Phụ thuộc range ước lượng:
  - Near (2-5m): R = 0.5 m²
  - Mid (5-15m): R = 2.0 m²
  - Far (15-30m): R = 10.0 m²

### Ví dụ cụ thể

Tại frame 100:
- Drone tại NED (10, 5, -8), yaw = 0.5 rad
- Detection bbox center = (340, 250), bbox height = 120 px
- Estimated range = 5.83 m

Camera ray (giả định fx=fy=700, cx=320, cy=240):
- Ray camera frame ≈ (0.029, 0.014, 1) → normalize
- After drone rotation: ray world frame ≈ (sin 0.5, cos 0.5, 0) ≈ (0.48, 0.88, 0)
- Target world = (10, 5, -8) + 5.83 × (0.48, 0.88, 0) = (12.8, 10.13, -8)

EKF input measurement = (12.8, 10.13, -8). Sau update, EKF state có velocity ước lượng.

### Các phương án đã cân nhắc

| Filter | Cost | Best for |
|--------|------|----------|
| **A. Low-pass / EMA** | Trivial | Quick prototype, không xử lý occlusion |
| **B. Linear KF** (constant velocity) | Trivial | Standard cho bbox tracking |
| **C. EKF** (nonlinear measurement) | Low | Cần khi fuse pose drone với camera ray |
| **D. UKF** (Unscented KF) | Medium | Robust hơn EKF cho high-nonlinear |
| **E. Particle Filter** | Heavy | Multimodal posterior, hiếm khi cần cho single target |
| **F. 2-stage KF (đã chọn)** | Low | Best practice paper |

### Tại sao chọn 2-stage KF
1. **Stage 1 đã free từ OC-SORT:** không thêm cost
2. **Stage 2 EKF cần thiết:** vì measurement là camera_ray × range (nonlinear) → linear KF không đủ
3. **EKF run <0.1ms:** rẻ hơn detection 500×
4. **Lý do KHÔNG chọn alternatives:**
   - **Không EMA:** không xử lý occlusion gap
   - **Không linear KF stage 2:** measurement nonlinear (cần Jacobian)
   - **Không UKF:** EKF đủ cho hệ này, UKF tốn 2× compute
   - **Không Particle:** overkill cho single target

---

## 2.5. Controller: Cascade PID

### Vấn đề
Cần biến target state (vị trí, vận tốc trong world frame) thành **velocity setpoints** gửi cho Pixhawk. Pixhawk sẽ lo phần còn lại (attitude control, motor mixing).

### Lựa chọn được chọn: Cascade PID 3 axis

**3 PID độc lập:**
1. **Forward PID:** range error → forward velocity (trong body frame X)
2. **Lateral PID:** image-space yaw error → yaw rate
3. **Vertical PID:** altitude error → vertical velocity (NED Z)

### Cách hoạt động chi tiết

**Forward PID (giữ khoảng cách 5m):**
```python
class ForwardPID:
    def __init__(self, Kp=0.5, Ki=0.05, Kd=0.1, setpoint=5.0,
                 v_max=3.0, integral_max=2.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.setpoint = setpoint
        self.v_max = v_max
        self.integral_max = integral_max
        self.integral = 0
        self.prev_error = 0
        self.prev_time = None
    
    def compute(self, distance, current_time):
        if self.prev_time is None:
            dt = 0.1
        else:
            dt = current_time - self.prev_time
        self.prev_time = current_time
        
        error = distance - self.setpoint  # >0: too far, need forward
        
        # P
        P = self.Kp * error
        
        # I (with anti-windup)
        self.integral += error * dt
        self.integral = clip(self.integral, -self.integral_max, self.integral_max)
        I = self.Ki * self.integral
        
        # D
        D = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error
        
        v = P + I + D
        return clip(v, -self.v_max, self.v_max)
```

**Yaw rate PID (giữ horizontal centered):**
```python
class YawPID:
    def __init__(self, Kp=2.0, Kd=0.3, rate_max=1.5):
        # Pure P + D (no integral cho yaw vì sẽ drift)
        self.Kp, self.Kd = Kp, Kd
        self.rate_max = rate_max
        self.prev_error = 0
    
    def compute(self, bbox_cx, image_width, fx, dt):
        error_px = bbox_cx - image_width / 2
        error_rad = error_px / fx
        
        P = self.Kp * error_rad
        D = self.Kd * (error_rad - self.prev_error) / dt
        self.prev_error = error_rad
        
        yaw_rate = -(P + D)  # negative: counter the error
        return clip(yaw_rate, -self.rate_max, self.rate_max)
```

**Vertical PID (giữ độ cao):**
```python
class VerticalPID:
    def __init__(self, Kp=0.8, Ki=0.1, Kd=0.2, setpoint=5.0,
                 v_max=2.0):
        # Tương tự ForwardPID nhưng cho altitude
        ...
    
    def compute(self, current_alt, dt):
        error = self.setpoint - current_alt  # >0: too low
        ...
        return clip(v, -v_max, v_max)
```

**Output cuối cùng — body frame velocity setpoint:**
```python
forward_vel = forward_pid.compute(distance, t)        # m/s, body X
lateral_vel = 0  # follow straight, không orbit
vertical_vel = vertical_pid.compute(altitude, dt)     # m/s, body Z (negative=up in NED)
yaw_rate = yaw_pid.compute(bbox_cx, w, fx, dt)        # rad/s

# Convert to NED (drone frame → world frame)
yaw = drone_yaw_from_mavlink
vx_ned = forward_vel * cos(yaw) - lateral_vel * sin(yaw)
vy_ned = forward_vel * sin(yaw) + lateral_vel * cos(yaw)
vz_ned = vertical_vel

# Send via MAVLink
send_velocity_setpoint(vx_ned, vy_ned, vz_ned, yaw_rate)
```

### Tuning guide

**Forward PID tuning:**
- Bắt đầu chỉ với Kp:
  - Kp = 0.3 → drone phản ứng chậm, smooth
  - Kp = 1.0 → drone phản ứng nhanh, có thể oscillate
- Thêm Kd để dampen oscillation:
  - Kd = 0.05 × Kp typical
- Thêm Ki nếu có steady-state error (vd drone luôn ở 5.5m thay vì 5m):
  - Ki = 0.1 × Kp typical
- Anti-windup: clip integral để không tích lũy quá lớn khi drone bị stuck

**Yaw PID tuning:**
- Higher Kp than forward (yaw response cần fast):
  - Kp = 2.0 typical
- Skip Ki cho yaw (sẽ drift, không cần)
- Kd để smooth

**Vertical PID:**
- Tương tự forward
- Limit v_max thấp hơn (drone climb/descend an toàn ở 1-2 m/s)

### Các phương án đã cân nhắc

| Controller | Cost | Tuning | Notes |
|-----------|------|--------|-------|
| **A. Cascade PID (đã chọn)** | Trivial | Easy | Match ArduPilot inner loop |
| **B. LQR** | Low (offline) | Cần model | Không thắng PID cho problem này |
| **C. MPC** | Heavy (10-30ms/solve) | Hard | Overkill, duplicate ArduPilot |
| **D. Adaptive / SMC** | Low | Hard | ArduPilot inner loop đã handle wind |
| **E. RL-based** | Heavy + huge train cost | Black box | Không có data |

### Tại sao chọn Cascade PID
1. **ArduPilot Copter GUIDED mode** đã có position+velocity+attitude PID inner loop
   - Nhiệm vụ Pi: chỉ generate smooth velocity vector
   - Cascade PID outer loop là exactly cái cần
2. **Easy tuning:** 3 PID độc lập, mỗi cái 3 gain → tổng 9 gain tune
3. **ECCV 2024 paper** "Autonomous Drone-Person Tracking in Uniform Appearance Scenarios" dùng PID với architecture y hệt
4. **Lý do KHÔNG chọn alternatives:**
   - **Không LQR:** không win significant qua PID cho problem này, cần model toán học
   - **Không MPC:** Pi 5 solving QP at 30 Hz steals CPU từ detection. Duplicate ArduPilot inner loop
   - **Không RL:** không có data train, black box → khó debug khi crash

### Velocity command rate

**Quan trọng:** ArduPilot Copter 3.3+ stops drone nếu không nhận command trong 3 giây (GUIDED timeout).

→ Pi phải send velocity setpoint **at least 2 Hz, recommend 10 Hz**.

Nếu detection bị slow (vd 20 FPS), vẫn phải send command 10 Hz bằng cách interpolate hoặc resend last command.

---

## 2.6. Communication: pymavlink

### Vấn đề
Cần thư viện Python để Pi 5 gửi/nhận MAVLink message với Pixhawk.

### Lựa chọn được chọn: pymavlink

**pymavlink** là thư viện chính thức từ ArduPilot org, low-level nhất, full control.

### Cách hoạt động chi tiết

**Setup connection:**
```python
from pymavlink import mavutil

# Kết nối qua UART (Pi 5 GPIO serial → Pixhawk TELEM2)
master = mavutil.mavlink_connection(
    '/dev/serial0',     # hoặc '/dev/ttyAMA0' tùy Pi 5 config
    baud=921600
)

# Đợi heartbeat đầu tiên để confirm Pixhawk đã sẵn sàng
master.wait_heartbeat()
print(f"Heartbeat from system {master.target_system} component {master.target_component}")
```

**Set GUIDED mode:**
```python
def set_mode(mode_name):
    """Set ArduPilot mode by name."""
    mode_id = master.mode_mapping()[mode_name]
    master.mav.set_mode_send(
        master.target_system,
        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        mode_id
    )
    # Wait for confirmation
    while True:
        ack = master.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if ack and ack.custom_mode == mode_id:
            return True

set_mode('GUIDED')
```

**Arm motors:**
```python
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0,
    1,  # arm
    0, 0, 0, 0, 0, 0
)
```

**Send velocity setpoint @ 10 Hz:**
```python
def send_velocity_ned(vx, vy, vz, yaw_rate):
    """Send velocity setpoint in local NED frame."""
    master.mav.set_position_target_local_ned_send(
        0,  # time_boot_ms
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,
        # type_mask: ignore position, ignore acceleration, use velocity + yaw_rate
        0b0000_1111_1100_0111,
        # NOTE: bit 0 (LSB) = ignore_x, bit 1 = ignore_y, ...
        # 0b1111_1000_0111 = use velocity (bits 3-5), use yaw_rate (bit 11)
        0, 0, 0,            # x, y, z position (ignored)
        vx, vy, vz,         # velocity NED
        0, 0, 0,            # acceleration (ignored)
        0,                  # yaw (ignored)
        yaw_rate            # rad/s
    )
```

**Receive drone state:**
```python
def get_drone_state():
    """Get latest drone position, attitude, velocity."""
    pos = master.recv_match(type='LOCAL_POSITION_NED', blocking=False)
    att = master.recv_match(type='ATTITUDE', blocking=False)
    bat = master.recv_match(type='BATTERY_STATUS', blocking=False)
    return pos, att, bat
```

**Main control loop (10 Hz):**
```python
import time

while True:
    t0 = time.time()
    
    # 1. Get latest detection from camera thread
    detection = camera_thread.get_latest_bbox()
    
    # 2. Track update
    track = oc_sort.update(detection)
    
    # 3. Get drone state
    drone_state = get_drone_state()
    
    # 4. Compute control
    if track is not None:
        distance = pinhole_distance(track.bbox_height)
        target_state = ekf.update(track, drone_state, distance)
        
        vx, vy, vz, yaw_rate = compute_pid(target_state, drone_state)
    else:
        # Target lost: hover
        vx, vy, vz, yaw_rate = 0, 0, 0, 0
    
    # 5. Send command
    send_velocity_ned(vx, vy, vz, yaw_rate)
    
    # 6. Maintain 10 Hz
    elapsed = time.time() - t0
    sleep_time = max(0, 0.1 - elapsed)
    time.sleep(sleep_time)
```

### Các phương án đã cân nhắc

| Library | Status 2026 | Best with | Notes |
|---------|------------|-----------|-------|
| **A. DroneKit-Python** | Đã chết (no commits since 2018) | ArduPilot | Avoid for new projects |
| **B. pymavlink (đã chọn)** | Active, official ArduPilot | Both ArduPilot và PX4 | Low-level, full control |
| **C. MAVSDK-Python** | Active, Dronecode | PX4-first (ArduPilot partial) | Cleaner API nhưng PX4-leaning |
| **D. ROS 2 + MAVROS** | Active | Either | Heavy install, chỉ worth it nếu rest of stack là ROS |

### Tại sao chọn pymavlink
1. **ArduPilot + Pixhawk 6C** là exactly nơi pymavlink supports tốt nhất
2. **DroneKit đã chết:** không update từ 2018, không hỗ trợ MAVLink 2.0 features
3. **MAVSDK PX4-leaning:** ArduPilot path không first-class, thiếu một số Copter messages
4. **ROS 2 quá nặng:** Pi 5 chỉ có 4-8GB RAM, ROS install ~2GB. Wrapping thứ pymavlink làm trong 200 dòng
5. **Reference:** [ArduPilot Discourse](https://discuss.ardupilot.org) và GitHub pymavlink có examples đầy đủ

---

# Phần III: Safety & Failsafe

Đây là phần CỰC KỲ quan trọng. Drone tự bay outdoor → bất kỳ lỗi nào cũng có thể gây tai nạn.

## 3.1. Target-lost behavior (3-tier ladder)

```
Time since track lost:
  0-3s:  Coast mode (Kalman dead-reckoning, drone tiếp tục theo predicted position)
  3-8s:  Hover (gửi velocity = 0, drone giữ vị trí)
  8-15s: LOITER mode (chuyển sang loiter qua MAVLink)
  >15s:  RTL (Return to Launch)
```

```python
class TargetLostHandler:
    def __init__(self):
        self.lost_since = None
    
    def update(self, track_active):
        if track_active:
            self.lost_since = None
            return 'follow'
        else:
            if self.lost_since is None:
                self.lost_since = time.time()
            
            elapsed = time.time() - self.lost_since
            if elapsed < 3:
                return 'coast'   # Continue with Kalman prediction
            elif elapsed < 8:
                return 'hover'   # Velocity = 0
            elif elapsed < 15:
                return 'loiter'  # Switch to LOITER mode
            else:
                return 'rtl'     # Return to launch
```

## 3.2. Geofence (set qua Mission Planner)

| Param | Value | Mục đích |
|-------|-------|----------|
| `FENCE_ENABLE` | 1 | Bật |
| `FENCE_TYPE` | 7 | Cylinder + Altitude + Polygon |
| `FENCE_ACTION` | 1 | RTL khi vượt fence |
| `FENCE_RADIUS` | 100 | 100m từ home |
| `FENCE_ALT_MAX` | 30 | Trần 30m |

Geofence enforce ở Pixhawk level — Pi không cần lo. Khi drone gần biên fence, Pixhawk tự RTL.

## 3.3. Min/max distance to person (enforce ở Pi)

```python
D_MIN = 3.0   # Hard stop, không tiến gần hơn
D_MAX = 15.0  # Beyond này, re-detection fragile

def safe_forward_velocity(distance, raw_velocity):
    if distance < D_MIN:
        # Quá gần, lùi lại
        return min(raw_velocity, -0.5)
    if distance > D_MAX:
        # Quá xa, có thể là detection error
        return clip(raw_velocity, 0, 1.0)  # Tiến chậm
    return raw_velocity
```

## 3.4. Battery failsafe (Mission Planner)

```
BATT_LOW_VOLT = 22.0   # Cảnh báo (cho 6S Li-Po)
BATT_CRT_VOLT = 21.0   # Critical
BATT_FS_LOW_ACT = 2    # RTL khi pin low
BATT_FS_CRT_ACT = 1    # LAND khi pin critical
```

Pixhawk tự handle, không cần Pi.

## 3.5. GCS heartbeat failsafe

```
FS_GCS_ENABLE = 1   # Enable
```

Nếu Mission Planner mất kết nối >5s → Pixhawk auto-RTL.

## 3.6. Pi watchdog

Pi gửi velocity command 10 Hz → ArduPilot timeout sau 3s nếu không nhận.

→ **Nếu Pi crashed/hang, drone tự stop.**

Bonus: thêm hardware watchdog timer trên Pi để reboot nếu hang quá lâu.

## 3.7. GPS quality check

Trước khi enter follow mode, check:
```python
def gps_ready():
    gps = master.recv_match(type='GPS_RAW_INT', blocking=True, timeout=1)
    if gps is None: return False
    if gps.fix_type < 3: return False  # Cần 3D fix
    if gps.eph > 200: return False  # HDOP > 2.0 (cm * 100)
    return True
```

## 3.8. Speed cap (Pi side)

```python
V_MAX = 5.0   # m/s, không bay quá nhanh
YAW_RATE_MAX = 1.5  # rad/s ≈ 86 deg/s

vx = clip(vx, -V_MAX, V_MAX)
vy = clip(vy, -V_MAX, V_MAX)
yaw_rate = clip(yaw_rate, -YAW_RATE_MAX, YAW_RATE_MAX)
```

## 3.9. Distance sanity check (cross-validation)

Nếu có GPS distance đến phone của user (qua Mission Planner follow target broadcast):
```python
def distance_sanity_check(d_pinhole, d_gps):
    if d_gps is None:
        return True  # No GPS reference, trust pinhole
    
    relative_error = abs(d_pinhole - d_gps) / d_gps
    if relative_error > 0.5:  # 50% mismatch
        # Detection có thể đang theo nhầm người
        return False
    return True
```

---

# Phần IV: Implementation Plan

## 4.1. Timeline khuyến nghị (2-4 tuần)

### Tuần 1: Setup & Test communication
- [ ] Wire UART từ Pi 5 → Pixhawk TELEM2
- [ ] Set ArduPilot params trong Mission Planner
- [ ] Test pymavlink: send heartbeat, receive HEARTBEAT
- [ ] Test arm/disarm trên ground (KHÔNG mount propeller)
- [ ] Test set GUIDED mode

### Tuần 2: SITL (Software In The Loop) test
- Chạy ArduPilot SITL trên laptop
- Pi 5 connect tới SITL qua UDP
- Test toàn bộ pipeline detection → control trong simulator
- **CỰC KỲ QUAN TRỌNG:** test failsafe scenarios trong sim trước

### Tuần 3: Real drone, tethered (cột)
- Mount drone trong sân, **buộc dây an toàn**
- Test arm và takeoff đến độ cao 1m
- Test follow tại chỗ (di chuyển trước drone)
- Đo telemetry, tune PID

### Tuần 4: Outdoor flight
- Bắt đầu với fence 30m × 30m × 10m altitude
- Test follow ở tốc độ đi bộ
- Tăng fence dần khi confident

## 4.2. Code structure đề xuất

```
follow_drone/
├── camera/
│   ├── camera_thread.py      # Capture frame
│   └── preprocess.py
├── detection/
│   ├── pfdet_runner.py       # PFDet-Nano v15 inference
│   └── postprocess.py
├── tracking/
│   ├── ocsort.py              # OC-SORT (copy từ noahcao/OC_SORT)
│   └── target_selector.py    # Lock 1 ID
├── estimation/
│   ├── distance.py            # Pinhole distance
│   └── ekf.py                 # World-frame EKF
├── control/
│   ├── pid.py                 # 3 PID
│   └── visual_servo.py        # Hybrid 2.5D logic
├── comm/
│   ├── mavlink_client.py     # pymavlink wrapper
│   └── safety.py              # Failsafe handlers
├── main.py                    # Main loop
├── config.yaml                # Hyperparameters
└── tests/
    ├── test_pid.py
    └── test_sitl.py           # SITL integration test
```

## 4.3. Hyperparameter cheat sheet

```yaml
camera:
  fx: 700        # calibrate trên drone của bạn
  fy: 700
  cx: 320
  cy: 240
  hfov_deg: 70

tracking:
  oc_sort:
    max_age: 90      # 3s @ 30 FPS
    min_hits: 3
    iou_threshold: 0.3
    delta_t: 3

distance:
  person_height_m: 1.7
  min_height_px: 20      # bỏ box quá nhỏ
  max_distance_m: 30

control:
  follow_distance_m: 5.0
  follow_altitude_m: 5.0
  
  forward_pid:
    Kp: 0.5
    Ki: 0.05
    Kd: 0.1
    v_max: 3.0
    integral_max: 2.0
  
  yaw_pid:
    Kp: 2.0
    Kd: 0.3
    rate_max: 1.5
  
  vertical_pid:
    Kp: 0.8
    Ki: 0.1
    Kd: 0.2
    v_max: 2.0

safety:
  d_min: 3.0
  d_max: 15.0
  v_max_horizontal: 5.0
  yaw_rate_max: 1.5
  
  target_lost:
    coast_duration_s: 3
    hover_duration_s: 5
    loiter_duration_s: 7
  
  command_rate_hz: 10
  detection_min_fps: 10
```

---

# Phần V: Tổng kết và References

## 5.1. End-to-end stack

PFDet-Nano v15 detector trên Pi 5 → **OC-SORT** lock single target ID → **pinhole distance** từ bbox height (1.7m assumption) → **2.5D hybrid visual servoing** (IBVS for yaw + gimbal pitch, PBVS for forward range) → **6-D world-frame EKF** smoothing → **3 cascade PID** → (vx, vy, vz, yaw_rate) @ 10 Hz → **pymavlink** sang **Pixhawk 6C / ArduPilot Copter GUIDED mode** → ArduPilot inner loops handle attitude. Mission Planner luôn connected để monitor + emergency override. Failsafe ladder: target-lost → Loiter → RTL, ArduPilot 3s GUIDED timeout, two-tier battery RTL/Land, polygonal geofence.

## 5.2. Reference papers

| Paper | Venue | Năm | Cho phần nào |
|-------|-------|-----|--------------|
| Cao et al., **Observation-Centric SORT** | CVPR | 2023 | Tracking — OC-SORT |
| Zhang et al., **ByteTrack** | ECCV | 2022 | Tracking — alternative baseline |
| Aharon et al., **BoT-SORT** | arXiv | 2022 | Tracking — alternative |
| Wojke et al., **DeepSORT** | ICIP | 2017 | Tracking foundation |
| Bonatti et al., **Autonomous Aerial Cinematography** | J. Field Robotics | 2020 | Person-following architecture (CMU/AirLab) |
| Bonatti et al., **Aerial cinematography platform** | IROS | 2019 | Perception+planning+control |
| Naseer et al., **FollowMe Quadrocopter** | IROS | 2013 | Foundational person-following paper |
| Reddy et al., **NeoARCADE Robust Calibration** | arXiv 2504.01988 | 2025 | Distance estimation |
| Skydio team, **Autonomy Engine** | IEEE 9567400 | 2021 | Industry reference |
| **Vision-Based Learning for Drones: Survey** | arXiv 2312.05019 | 2024 | Related work |
| **Autonomous Drone-Person Tracking** | ECCV W | 2024 | Architecture mirror — tracker + PID |
| **UAV target tracking: a survey** | Artif. Intell. Rev. | 2025 | Taxonomy |
| Xing et al., **Siamese Transformer Pyramid** | WACV | 2022 | Tracker for CPU |
| **GPS-Denied IBVS UAV Navigation** | arXiv 2509.17435 | 2025 | Hybrid 2.5D visual servoing |
| **PID vs LQR vs MPC for Parrot Mambo** | MDPI Aerospace 9:298 | 2022 | Controller justification |

## 5.3. Open-source projects để tham khảo

| Repo | Mục đích |
|------|----------|
| [`noahcao/OC_SORT`](https://github.com/noahcao/OC_SORT) | OC-SORT canonical impl — drop in |
| [`mikel-brostrom/boxmot`](https://github.com/mikel-brostrom/boxmot) | Unified MOT framework |
| [`durner/yolo-autonomous-drone`](https://github.com/durner/yolo-autonomous-drone) | YOLO + person follow, minimal example |
| [`alexseysua/AeroCompanion`](https://github.com/alexseysua/AeroCompanion) | Pi 5 + Pixhawk 6X, modular (gần với hardware bạn) |
| [`rlew631/ObjectTrackingDrone`](https://github.com/rlew631/ObjectTrackingDrone) | Pixhawk + Pi + OpenCV + gimbal IBVS |
| [`AmirPliev/Drone-Follow-Me`](https://github.com/AmirPliev/Drone-Follow-Me) | Utrecht thesis project |
| [ArduPilot Copter Follow mode docs](https://ardupilot.org/copter/docs/follow-mode.html) | Reference for `FOLLOW_TARGET` + PID position controller |
| [PX4 Follow-Me docs](https://docs.px4.io/main/en/flight_modes_mc/follow_me.html) | Reference for filter responsiveness tuning |
| [pymavlink GitHub](https://github.com/ArduPilot/pymavlink) | Official Python MAVLink |

## 5.4. Khuyến nghị nhỏ cuối

1. **Test trong SITL trước** — đừng bay drone thật cho tới khi pipeline chạy hoàn hảo trong simulator
2. **Tether (dây cột)** trong test outdoor đầu tiên
3. **Mission Planner mở liên tục** — telemetry là cứu cánh khi có sự cố
4. **Log everything** — mỗi command + mỗi detection lưu CSV → debug crash
5. **Geofence luôn bật** — không chừa exception
6. **PID tune từ thấp lên cao** — bắt đầu Kp nhỏ, tăng dần đến vừa đủ phản ứng

Bạn đã có model AP=0.5931, hardware sẵn sàng. Phần điều khiển này có thể làm trong 2-4 tuần nếu test cẩn thận. Khi gặp vấn đề cụ thể (PID dao động, MAVLink không kết nối...), tham khảo lại đúng section trong document này.
