# Kiểm tra liên lạc Pi ↔ mạch bay — 2026-08-27

Lần đầu Pi được nối vào mạch bay. Toàn bộ số dưới đây đọc trực tiếp từ máy bay,
không phải SITL.

| | |
|---|---|
| Mạch bay | **Holybro Pixhawk6C**, USB (`ID 3162:0053`) |
| Firmware | **ArduPilot Copter 4.7.0**, git `1511f271` |
| Loại xe | QUADROTOR, MAVLink **v2.0**, system id 1 |
| Cổng | `/dev/ttyACM0` (`if00`) — `if02` là cổng phụ, không dùng cho MAVLink |
| Nhịp gói | **94–100 gói/giây**, `errors_comm = 0` |
| Chế độ | 27 chế độ, **có GUIDED** — đúng thứ `follow_drone/` cần |

## ✅ Đã chạy đúng

- Mọi bản tin dashboard cần đều tự về đủ nhịp: `ATTITUDE` 10 Hz, `VFR_HUD` 10 Hz,
  `GLOBAL_POSITION_INT` 10 Hz, `SYS_STATUS` 5 Hz, `GPS_RAW_INT` 5 Hz,
  `VIBRATION` / `EKF_STATUS_REPORT` / `BATTERY_STATUS` 3,1 Hz.
- Cảm biến: gyro / accel / la bàn / áp suất / GPS / RC / AHRS / pin — **tất cả healthy**.
  CPU mạch bay 26%.
- EKF3: mọi phương sai < 0,01 (ngưỡng an toàn 0,5).
- Rung nền 0,01–0,07 m/s², clipping 0/0/0 — **nền sạch** để so sau khi nổ máy.
- Pin 15,9 V / 98% / board 5,0 V. RC **đang bật thật** (16 kênh, giá trị khác nhau).

## 🐛 Hai lỗi trong CODE CỦA TA đã sửa

1. **`mav_telem.py` tự bóp nhịp telemetry.** Bản đầu gửi `SET_MESSAGE_INTERVAL` cho
   mọi bản tin, với chu kỳ **dài hơn** cái ArduPilot đang tự phát: VFR_HUD 10→5 Hz,
   GLOBAL_POSITION_INT 10→5 Hz, SYS_STATUS 5→2 Hz, GPS_RAW_INT 5→2 Hz.
   → Sửa thành `_request_missing()`: nghe 3 giây, **chỉ xin bản tin nào chưa thấy**.

2. **Bảng bit cảm biến gõ tay bị lệch.** `0x400` bị gán nhãn "thu RC" (thật ra là
   `ANGULAR_RATE_CONTROL`; `RC_RECEIVER` là `0x10000`), và bit prearm dùng
   `0x20000000` (thật ra là `OBSTACLE_AVOIDANCE`; `PREARM_CHECK` là `0x10000000`)
   nên **trạng thái prearm không bao giờ đọc được**.
   → Sửa: sinh bảng từ `mavutil.mavlink.enums["MAV_SYS_STATUS_SENSOR"]`, không gõ tay.

## 🔧 Một cổng, nhiều chương trình → `mavmux.py`

`/dev/ttyACM0` chỉ **một** tiến trình mở được; mở cái thứ hai là
`device reports readiness to read but returned no data`. Nhưng lúc bay cần cả
`follow_drone/main.py` (gửi lệnh GUIDED) **và** `hailo_gcs.py` (đọc telemetry).

`mavmux.py` giữ cổng serial rồi phát lại nguyên byte ra nhiều đầu UDP:

```bash
./run.sh mux                                    # 14550 + 14551
./run.sh gcs --mavlink udpin:127.0.0.1:14551    # dashboard
# follow_drone/ trỏ tới udpin:127.0.0.1:14550
```

Đã kiểm: hai client đồng thời, mỗi bên **~100 gói/giây**, telemetry đầy đủ ở cả hai.

## ⛔ BỐN VẤN ĐỀ CẤU HÌNH — PHẢI SỬA TRƯỚC KHI BAY TỰ ĐỘNG

| # | Tham số | Hiện tại | Vấn đề | Nên đặt |
|---|---|---|---|---|
| 1 | `ARMING_SKIPCHK` | **−1** | **Bỏ qua TOÀN BỘ kiểm tra trước khi arm.** Tài liệu ArduPilot: *"Never disable arming checks, except for bench testing."* | **0** |
| 2 | `BATT_LOW_VOLT` / `BATT_CRT_VOLT` | 10 / **14** | **Ngược nhau**: ngưỡng nguy kịch (14 V) cao hơn ngưỡng yếu (10 V). Pack 4S (`MOT_BAT_VOLT_MIN`=13,2) thì 10 V không bao giờ chạm tới | ≈ **14,0** / **13,5** |
| 3 | `BATT_FS_LOW_ACT` / `BATT_FS_CRT_ACT` | 0 / 0 | Cả hai = "không làm gì" → **không có bảo vệ pin nào cả** | 2 (RTL) / 1 (Land) |
| 4 | `FS_GCS_ENABLE` | 0 | Không có failsafe khi mất máy tính đồng hành. `GUID_TIMEOUT`=3 s chỉ làm máy bay **đứng yên** trong GUIDED, không tự về | bật (RTL) |
| 5 | `FENCE_ENABLE` | 0 | Hàng rào đã cấu hình sẵn (bán kính 300 m, trần 100 m, hành động RTL) nhưng **đang tắt** | 1 |

Cấu hình đã đúng sẵn: `GUID_TIMEOUT`=3 s, `FS_THR_ENABLE`=3, `FS_EKF_ACTION`=1,
`FS_CRASH_CHECK`=1, `BATT_MONITOR`=4, `INS_HNTCH_ENABLE`=1 (lọc notch hài — tốt cho rung).

## ⚠️ Lưu ý khác

- **GPS: NO FIX, 0 vệ tinh, HDOP 99,99** → `PreArm: Need Position Estimate`. Chỉ vì
  đang trong nhà, và chế độ hiện tại là LOITER (cần vị trí). Ra trời là hết.
- **Kênh 3 (ga) đang ở 1943** trong khi `RC3_MIN`=983. Ga đang gần hết cỡ — hạ về
  min trước khi arm.
- `RC_CHANNELS.rssi = 255` = "không rõ". Bộ thu không báo cường độ sóng, nên dashboard
  không cảnh báo được khi sóng yếu. Nếu bộ thu hỗ trợ thì nên cấu hình `RSSI_TYPE`.
- **`INS_LOG_BAT_MASK = 1`** — chỉ ghi phổ rung của IMU thứ nhất. Đề tài này viết về
  rung, nên cân nhắc đặt **7** (cả ba IMU) để phần phân tích rung có đủ dữ liệu.
- `console=serial0,115200` trong `cmdline.txt` đang chiếm UART trên chân GPIO. Không
  ảnh hưởng gì lúc này (đang dùng USB), nhưng nếu sau này chuyển sang nối TELEM2 bằng
  dây thì phải gỡ dòng đó ra.

## ⚠️ ArduPilot 4.7 ĐỔI TÊN VÀ ĐỔI ĐƠN VỊ THAM SỐ

Tải đủ 1038/1038 param và xác nhận: nhiều tên quen thuộc **không còn tồn tại**.

| tên cũ (tài liệu trên mạng) | tên trên 4.7 | đơn vị |
|---|---|---|
| `WPNAV_SPEED` | `WP_SPD` | **m/s** (cũ: cm/s) |
| `WPNAV_SPEED_UP` / `_DN` | `WP_SPD_UP` / `WP_SPD_DN` | **m/s** |
| `WPNAV_ACCEL` | `WP_ACC` | **m/s²** (cũ: cm/s²) |
| `ANGLE_MAX` | `ATC_ANGLE_MAX` | **độ** (cũ: phần trăm độ) |
| `ATC_SLEW_YAW` | `ATC_RATE_WPY_MAX` | **độ/s** |
| `PSC_VELXY_P` | `PSC_NE_VEL_P` | — |
| `PSC_POSXY_P` | `PSC_NE_POS_P` | — |
| `PSC_VELZ_P` | `PSC_D_VEL_P` | — |
| `RTL_ALT` | `RTL_ALT_M` | **m** (cũ: cm) |
| `ARMING_CHECK` | `ARMING_SKIPCHK` (nghĩa **ngược lại**) | bitmask |

**Chép số từ hướng dẫn cũ sang sẽ sai 100 lần.** Bảng tham số trong
`baocao/phuluc/pl_a_cau_hinh.tex` đã cập nhật theo tên và đơn vị của 4.7, và điền
giá trị đọc thật từ máy bay.

Bản sao toàn bộ param: `raspi@192.168.196.114:~/NhanHuynh/params_dump.json`.
