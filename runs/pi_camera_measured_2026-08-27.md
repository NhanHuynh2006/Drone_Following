# Camera trên Pi 5 — đo thật 2026-08-27 (lần đầu Pi có camera)

Trước đây `runs/pi_hailo_measured_2026-08-24.md` ghi "Pi CHƯA cắm camera nào" nên
`capture` và e2e thật chưa đo được. Nay đã cắm.

| | |
|---|---|
| Camera | **Logitech C270 HD**, USB, `/dev/video0` (`usb-xhci-hcd.1-1`) |
| Định dạng | YUYV + **MJPG**, tối đa **1280×720** |
| fps tối đa của phần cứng | 30 |
| `./run.sh check` | **10 OK / 0 FAIL** — Hailo, lib, HEF, camera, NPU, nhiệt, đĩa, NTP đều qua |

---

## ⚠️ HAI LỖI làm cả pipeline kẹt ở 15 FPS — đã sửa

Trước khi sửa, e2e đo được **15.4 FPS** trong khi NPU mới ăn 17.8/33.3 ms. Tưởng
Hailo yếu, hoá ra **cả hai lỗi đều ở đường camera**, và chúng NHÂN nhau:

| cấu hình | fps đo được |
|---|---|
| `BUFFERSIZE=1` + auto-exposure ← **code cũ** | **11.1** |
| `BUFFERSIZE=1` + manual exposure | 15.0 |
| `BUFFERSIZE=2` + auto-exposure | 19.7 |
| **`BUFFERSIZE=2` + manual exposure (+SoftAE)** ← code mới | **30.0** |

### Lỗi 1 — `cv2.CAP_PROP_BUFFERSIZE = 1` rớt ĐÚNG một nửa số khung

Có mặt trong `flight_record.py`, `hailo_bench_latency.py`, `hailo_demo_cam.py`.
Đặt 1 buffer thì trong lúc app đang giữ buffer, driver **không có chỗ ghi khung kế
nên bỏ luôn** → 30 fps thành đúng 15.0. Đo tách bạch bằng `buf_probe.py`:

```
A. KHONG set BUFFERSIZE, doc dong bo    30.0 fps
B. BUFFERSIZE=1,      doc dong bo       15.0 fps   <-- code cu
C. BUFFERSIZE=2,      doc dong bo       30.0 fps
D. BUFFERSIZE=3,      doc dong bo       30.0 fps
E. KHONG set BUFFERSIZE, luong rieng    30.0 fps
F. BUFFERSIZE=1,      luong rieng       15.0 fps
```

`BUFFERSIZE=1` vốn được đặt để **tránh khung cũ**. Không cần: vòng đọc đã luôn lấy
khung mới nhất. Đo độ cũ của khung với `BUFFERSIZE=2` + luồng riêng:
**p50 15.7 ms, p95 31.5 ms, max 33.8 ms** = nửa chu kỳ khung, đúng như kỳ vọng.

→ Sửa: `BUFFERSIZE = 2` ở cả 4 công cụ.

### Lỗi 2 — auto-exposure của firmware tự hạ fps khi thiếu sáng

Ở `auto_exposure=3` (Aperture Priority), thiếu sáng thì firmware kéo dài thời gian
phơi vượt 1/30 s rồi **tự hạ khung hình**. Đo ngưỡng bằng `exp_knee.py`:

| exposure (đơn vị 100 µs) | = shutter | fps |
|---|---|---|
| 40 – 333 | 1/250 – 1/30 | **30.0** |
| 400 | 1/25 | 24.9 |

Ngưỡng đúng bằng 333 = 1/30 s như lý thuyết.

⚠️ Cờ UVC `exposure_dynamic_framerate=0` — sinh ra đúng để cấm hành vi này —
**C270 phớt lờ**. Đã thử, vẫn 15.0 fps. Đừng mất thời gian với nó.

→ Sửa: `cam_ctl.py` + lớp `SoftAE` — khoá `auto_exposure=1` (Manual, giữ đúng 30 fps)
rồi **tự chỉnh sáng bằng phần mềm**: đo luma trung bình, chỉnh
`exposure_time_absolute` + `gain`, với **trần phơi sáng cứng** nên fps không bao giờ
tụt. Đo thực tế: luma bám target 110 (đo được 103–110), fps giữ 30.1.

Tiện thể đặt `power_line_frequency = 1` (**50 Hz**, điện VN) thay mặc định 60 Hz —
tránh sọc nhấp nháy dưới đèn huỳnh quang.

---

## End-to-end SAU khi sửa — `./run.sh bench --minutes 2 --source cam`, @640

| chặng | mean | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| capture | 13.85 | 12.47 | 16.52 | 16.91 | 226.64 |
| preprocess | 0.98 | 0.99 | 1.02 | 1.04 | 1.67 |
| inference | 17.73 | 17.72 | 17.83 | 17.89 | 18.09 |
| decode + NMS | 0.84 | 0.84 | 0.87 | 0.88 | 7.57 |
| **tổng** | **33.39** | **32.04** | **36.08** | **36.15** | 246.76 |

**FPS e2e: p50 31, p99 28.** FPS bền: phút đầu 29.9 → phút cuối 30.0 (+0.3%).
Nhiệt 48.5 °C, `get_throttled = 0x0` suốt bài đo.

**So với trước khi sửa: 15.4 → 30.0 FPS, ×1.95, không đổi một dòng model nào.**

### Đọc bảng: nút cổ chai giờ là CAMERA, không phải NPU

Tính toán chỉ ăn **19.5 ms** trong ngân sách 33.3 ms của một khung (duty cycle 59%),
còn lại 13.8 ms là **ngồi chờ camera**. C270 hết cỡ là 30 fps. Muốn nhanh hơn 30 FPS
thì phải đổi camera, tối ưu phần mềm không giúp được nữa.

---

## Công cụ mới

| file | việc |
|---|---|
| `cam_ctl.py` | `SoftAE` + `apply_locked_fps` + CLI `--show/--apply/--restore`. Đây là chỗ chứa toàn bộ hiểu biết về mấy cái bẫy exposure ở trên |
| `hailo_stream_web.py` | **`./run.sh stream`** — phát camera + box lên **trình duyệt** (MJPEG). Không cần màn hình, không cần X11. Mở `http://<ip-pi>:8080`. Có thanh trượt conf, hiện fps thật của camera / nhiệt / exposure / gain / latency inference |
| `buf_probe.py`, `exp_knee.py`, `ae_probe.py`, `cam_probe.py` | script đo đã dùng để tìm ra 2 lỗi trên — giữ lại để đo camera mới |

`./run.sh demo` (cửa sổ OpenCV) vẫn chạy được qua `ssh -Y` — cv2 trên Pi build với Qt5.
Nhưng X11 đẩy pixel thô nên qua WiFi sẽ giật; `stream` nhẹ hơn nhiều.

## Cấu hình khi RA BÃI BAY

Trong nhà thì để mặc định. Lúc bay, shutter dài là **nhoè vì rung** — đúng thứ paper
đang nói tới:

```bash
./run.sh record --alt 5 --scene grass --persons 3 --mode hover \
    --max-exposure 20 --prefer-gain          # 1/500 s, gain gánh phần sáng
```

`--max-exposure 20` cũng cắt luôn ~25 ms độ trễ glass-to-detection (thời gian phơi
sáng nằm THẲNG trong đường trễ điều khiển bay).
