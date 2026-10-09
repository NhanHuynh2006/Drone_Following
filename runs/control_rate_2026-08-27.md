# Nhịp gửi lệnh: 10 Hz hay 50 Hz? — phân tích trên số đo thật (2026-08-27)

## Số đo của chính hệ thống này

| đại lượng | giá trị | nguồn |
|---|---|---|
| Nhịp phát hiện đầu-cuối | **30,0 FPS** (33,3 ms/khung) | `runs/pi_camera_measured_2026-08-27.md` |
| Độ trễ phát hiện | p50 **32,04 ms**, p99 36,15 ms | như trên |
| Khứ hồi lệnh Pi→máy bay→Pi | p50 **4,75 ms**, p95 6,24, p99 7,73 | `MAV_CMD_REQUEST_MESSAGE` → `COMMAND_ACK`, 120 mẫu |
| Một chiều Pi→máy bay | ≈ **2,4 ms** | nửa khứ hồi |
| Mất gói khi gửi 50 Hz | **0,0 %** (200/200 có ACK) | đo 4 giây mỗi nhịp, tới 50 Hz |
| Tỷ lệ bắt được người | 57,6 % ở conf 0,30 | `runs/flight_val_hef_2026-08-27.md` |

## Ngân sách độ trễ theo từng nhịp

Độ trễ từ lúc ánh sáng vào cảm biến tới lúc máy bay nhận lệnh:

| nhịp gửi | phát hiện | chờ tới lượt gửi (t.bình / xấu nhất) | truyền | **tổng t.bình** | **tổng xấu nhất** |
|---|---|---|---|---|---|
| 10 Hz | 32,0 ms | 50,0 / 100,0 ms | 2,4 ms | **84,4 ms** | 138,4 ms |
| 20 Hz | 32,0 ms | 25,0 / 50,0 ms | 2,4 ms | **59,4 ms** | 88,4 ms |
| **30 Hz** | 32,0 ms | **16,7 / 33,3 ms** | 2,4 ms | **51,1 ms** | **71,7 ms** |
| 50 Hz | 32,0 ms | 10,0 / 20,0 ms | 2,4 ms | **44,4 ms** | 58,4 ms |

## Kết luận: **30 Hz**

**Vì sao không phải 10 Hz.** Ở 10 Hz, mỗi phát hiện phải nằm chờ trung bình 50 ms mới
tới lượt được gửi đi. Đó là **thêm 50 ms độ trễ mà không đổi lấy bất cứ thứ gì** —
không tiết kiệm băng thông đáng kể (xem dưới), không giảm tải CPU đáng kể. Với người
đi bộ 1,5 m/s thì 50 ms là 7,5 cm sai lệch; với máy bay đang khép khoảng cách ở 3 m/s
thì là 15 cm. Cộng dồn vào vòng kín, nó biến thành vọt lố.

**Vì sao không phải 50 Hz.** Detector chỉ sinh thông tin mới **30 lần mỗi giây**. Gửi
50 lệnh mỗi giây nghĩa là 20 lệnh trong đó **không mang phát hiện mới nào** — chúng
chỉ lặp lại, hoặc mang giá trị do bộ lọc Kalman nội suy. Đổi lại chỉ được 6,7 ms.
ArduPilot vốn đã làm mượt đầu vào bằng bộ giới hạn độ giật (`PSC_JERK_NE`), nên phần
"làm mượt bậc thang" mà nhịp cao mang lại phần lớn là thừa.

**Vì sao đúng 30 Hz.** Khớp một-một với nhịp detector: **mỗi lệnh gửi đi mang đúng
một phát hiện mới**, không lệnh nào thừa, không phát hiện nào phải chờ. Đây là điểm
mà tăng thêm nhịp không còn thêm thông tin, và giảm bớt nhịp là vứt thông tin đi.

**Đường truyền chịu được.** Đã đo 0 % mất gói tới 50 Hz, khứ hồi p99 dưới 8 ms. Ở
30 Hz, `SET_POSITION_TARGET_LOCAL_NED` (~65 byte đóng khung) tốn khoảng **2 KB/s** —
không đáng kể với USB.

> ⚠️ Nếu sau này chuyển sang bộ đàm telemetry 57 600 baud thay vì USB thì 2 KB/s là
> **28 % dung lượng đường truyền**, chưa kể telemetry chiều về. Lúc đó phải hạ xuống
> 15–20 Hz hoặc dùng đường riêng cho lệnh.

## Đối chiếu tài liệu

Leonard Hall (lập trình viên ArduPilot) trong thread
[Adventures with fast updates for guided mode](https://discuss.ardupilot.org/t/adventures-with-fast-updates-for-guided-mode-10hz/77996):

> *"If you are generating complex pos/vel/accel defined paths with a higher update rate
> (anything above 10 Hz works well) then just feed all three in."*

và:

> *"you may want to set your PSC_JERK_XY and WPNAV_ACCEL parameter based on how often
> you are giving updates and how aggressively it makes adjustments to follow your
> commands."*

Hai ý rút ra:
1. **Trên 10 Hz là vùng chạy tốt** — 30 Hz nằm an toàn trong đó.
2. Nếu gửi được cả **vị trí + vận tốc + gia tốc** cùng lúc thì nên gửi cả ba.
   `follow_drone/` hiện chỉ gửi vận tốc. Thêm trường vị trí và gia tốc vào
   `SET_POSITION_TARGET_LOCAL_NED` là một hướng nâng cấp có cơ sở, không phải suy đoán.

Cũng đáng ghi: thread đó ban đầu tưởng ArduPilot lỗi vì máy bay *"occasionally launch
itself at full speed in one of the cardinal directions"*, hoá ra là **gói tin bị hỏng
trên đường truyền** chứ không phải nhịp cao. Đường truyền của ta là USB với 0 lỗi
truyền (`SYS_STATUS.errors_comm = 0`), nên rủi ro đó không áp dụng — nhưng nếu chuyển
sang bộ đàm không dây thì phải kiểm lại.

## Bảng giá trị đề xuất

### Trong `follow_drone/config.yaml` (đã sửa)

| khoá | cũ | **mới** | lý do |
|---|---|---|---|
| `control.command_rate_hz` | 10 | **30** | khớp nhịp detector; cắt 33 ms độ trễ trung bình |
| `safety.command_timeout_s` | 3,0 | **1,0** | 3 s = 90 lệnh mất ở nhịp 30 Hz mới chịu dừng. Đường xử lý p99 chỉ 36 ms nên 1 s vẫn dư |
| `model.conf_threshold` | 0,40 | **0,30** | đo trên `flight_val`: 0,40 → recall 20 %; 0,30 → recall 58 % / precision 88 % |
| `tracking.max_age` | 90 | **45** | 3 s bám bằng dự đoán là quá dài, người đã đổi hướng |
| `safety.yaw_rate_max` | 1,5 rad/s | **1,0** | 57 °/s, giữ dưới `ATC_RATE_WPY_MAX` = 60 °/s của máy bay |
| `camera.width/height` | 640×480 | **giữ 640×480** | `fx/fy/cx/cy` hiệu chuẩn cho độ phân giải này; detector letterbox về 640 nên 720p không thêm điểm ảnh trên người |
| `mavlink.connection` | `/dev/serial0` | **`udpin:127.0.0.1:14550`** | Pixhawk nối USB; đi qua `mavmux.py` để chạy cùng dashboard |

### Trên máy bay, chỉnh bằng Mission Planner

| tham số | hiện tại | **đề xuất** | lý do |
|---|---|---|---|
| `GUID_TIMEOUT` | 3 | **1,0** | khớp `command_timeout_s`; dừng sau 1 s nếu Pi treo |
| `ARMING_SKIPCHK` | −1 | **0** | đang bỏ qua toàn bộ kiểm tra trước khi arm |
| `BATT_LOW_VOLT` | 10 | **14,0** | pin 4S (`MOT_BAT_VOLT_MIN` = 13,2). Ngưỡng 10 V không bao giờ chạm |
| `BATT_CRT_VOLT` | 14 | **13,5** | phải THẤP hơn ngưỡng cảnh báo — hiện đang ngược |
| `BATT_FS_LOW_ACT` | 0 | **2** (tự về nhà) | hiện không có bảo vệ pin |
| `BATT_FS_CRT_ACT` | 0 | **1** (hạ cánh) | như trên |
| `FS_GCS_ENABLE` | 0 | **bật** (tự về nhà) | mất Pi thì hiện máy bay chỉ treo tại chỗ |
| `FENCE_ENABLE` | 0 | **1** | hàng rào đã cấu hình sẵn 300 m / trần 100 m mà đang tắt |
| `WP_SPD` | 10 m/s | **4,0** | phòng thủ nhiều lớp: PID giới hạn 3 m/s, an toàn 5 m/s, nhưng máy bay vẫn sẵn sàng chạy 10 m/s nếu có lỗi |
| `INS_LOG_BAT_MASK` | 1 | **7** | ghi phổ rung cả ba IMU — đề tài này viết về rung |

Giữ nguyên vì đã hợp lý: `WP_ACC` = 2,5 m/s², `PSC_JERK_NE` = 5, `PSC_NE_VEL_P` = 1,80,
`FS_THR_ENABLE` = 3, `FS_EKF_ACTION` = 1, `FS_CRASH_CHECK` = 1, `INS_HNTCH_ENABLE` = 1.

## Việc còn lại có cơ sở để làm

1. **Gửi cả vị trí + vận tốc + gia tốc** thay vì chỉ vận tốc (theo khuyến nghị của
   Leonard Hall). `follow_drone/follow/mavlink_client.py:272` hiện chỉ điền trường vận tốc.
2. **Backend Hailo cho `follow_drone/follow/detector.py`** — nó đang chạy PyTorch trên
   CPU Pi. Toàn bộ phân tích ngân sách trên giả định 30 FPS, con số đó chỉ có được khi
   dùng HEF trên NPU (17,7 ms/khung). Chạy PyTorch CPU thì nhịp thật sẽ thấp hơn nhiều
   và mọi kết luận ở đây phải tính lại.
