# FLIGHT_DATA_SPEC.md — phải quay được gì trong chuyến bay đầu tháng 8

> Viết 2026-07-26, trước chuyến bay. **Đọc hết trước khi ra bãi.**
> Bay xong mới phát hiện thiếu thì phải bay lại, mà lịch bay không rẻ.

## Vì sao chuyến này quan trọng

Điểm yếu số 1 của bài, chính tôi đã viết vào Limitations:

> *"VisDrone-Shake được xây từ cơ chế đã biết của suy giảm ảnh trên rotorcraft, nhưng
> tham số của nó được chọn để bao quanh chuyến bay chứ không hiệu chuẩn theo phổ gia
> tốc kế đo trên chính airframe của chúng tôi."*

Nghĩa là **toàn bộ kết quả chính hiện là phát biểu về một MÔ HÌNH của rung**, không phải
về rung thật. Dữ liệu bay hiện có (92 frame) cũng không vá được vì nó thiếu 3 thứ:
quay lại màn hình GCS (có overlay, mất chất lượng), **không có log IMU**, và chỉ 1 người
/ 1 cảnh / 1 độ cao.

Chuyến này nếu quay đúng thì Limitations đó **biến mất**, và claim chuyển từ *"kết quả
mô phỏng có đối chứng thực tế"* thành *"kết quả đo được"*. Đó là bước nhảy lớn nhất còn
lại của bài.

---

## BẮT BUỘC — thiếu một trong 4 cái này thì chuyến bay không vá được Limitations

### 1. Video RAW từ thẻ nhớ camera, KHÔNG phải quay màn hình GCS
Bản hiện tại lấy từ `2026-07-21 09-43-12.mkv` là **screen capture của ground station**,
nên có crosshair "FRAME CENTER" đè lên ảnh và đã qua 2 lần nén. Paper đang phải thú nhận
điều đó.

- Ghi thẳng ra thẻ trong camera, lấy file gốc sau khi hạ cánh.
- Ghi lại **thông số phơi sáng**: shutter speed, ISO, fps, có auto-exposure không.
  Mô hình rung của bài là về **nhoè do thời gian phơi sáng**, nên shutter là biến vật lý
  trực tiếp liên quan. Không có nó thì không hiệu chuẩn được.
- Nếu camera cho khoá shutter: bay **2 lượt**, một lượt shutter nhanh (1/1000) và một
  lượt chậm (1/120). Đó là bằng chứng trực tiếp cho cơ chế bài nêu: phơi sáng dài hơn →
  nhoè nhiều hơn → AP tụt nhiều hơn. Rất mạnh, và gần như không ai có.

### 2. Log bay ArduPilot (`.bin` DataFlash) chạy SUỐT chuyến, đồng bộ thời gian với video
Đây là thứ vá được Limitations #1.

- Bật logging từ trước khi arm tới sau khi disarm.
- Cần các topic: `sensor_accel` (raw, tần số cao), `vehicle_imu_status`,
  `sensor_combined`, `battery_status`, `vehicle_local_position`.
  Trong QGC: `SDLOG_PROFILE` bật thêm profile **"High rate"** và **"Vibration"**.
- **Đồng bộ thời gian**: cách rẻ nhất là ngay sau khi arm, cho drone lắc mạnh 1 phát
  (hoặc che camera 1 giây). Trong video và trong log đều thấy được mốc đó → căn chỉnh sau.
  Ghi lại giờ hệ thống lúc bắt đầu quay.
- Sau chuyến, từ log ArduPilot `.bin` rút phổ gia tốc kế (FFT) → đó chính là con số dùng để **hiệu
  chuẩn** tham số `_PRESETS` trong `vib_consistency.py`, thay vì chọn tay như hiện nay.

### 3. Nhiều độ cao, nhiều người, nhiều cảnh
Hiện tại: 1 người / 1 cảnh / 1 độ cao (~6m) / 92 frame. Quá hẹp để gọi là benchmark.

Tối thiểu nên có:

| trục | hiện có | cần |
|---|---|---|
| độ cao | ~6m | **3, 5, 8, 12 m** (4 mức, mỗi mức ≥2 phút) |
| người | 1 | **≥3 người**, khác quần áo (sáng/tối/tương phản thấp) |
| cảnh | 1 (sân bê tông) | **≥3**: bê tông, cỏ, có bóng đổ mạnh |
| chế độ bay | POSHOLD | **hover + bay tiến + xoay** (rung khác nhau rõ rệt) |
| giờ | 1 | sáng sớm + trưa (bóng đổ và độ tương phản khác hẳn) |

Số frame mục tiêu: **≥1000 frame đã gán nhãn** (hiện 92). Ở 1 fps thì ~17 phút video hữu ích.

### 4. Ít nhất một lượt bay có rung MẠNH hơn bình thường
Bài chia 3 mức sev1/sev2/sev3, nhưng dữ liệu thật hiện chỉ có một mức rung.

- An toàn nhất: bay **tiến nhanh + đổi hướng gấp** (rung tăng do khí động và do motor
  phải bù). Không cần phá cân bằng cánh quạt.
- Ghi rõ trong log lượt nào là lượt "rung mạnh" để tách ra khi phân tích.
- ⚠️ **KHÔNG** cố tình làm lệch cánh quạt để tăng rung. Rủi ro rơi không đáng.

---

## NÊN CÓ — nếu còn thời gian tại bãi

- **Chạy detector trực tiếp trên Pi lúc bay** và ghi lại: timestamp mỗi frame, box dự
  đoán, latency từng chặng. Cái này cho luôn số cho mục Edge Deployment
  (`hailo_bench_latency.py` đã viết sẵn), và cho phép so *"model thấy gì"* với
  *"nhãn thật"* trên đúng frame đó.
- **Nhiệt độ Pi + trạng thái throttle** trong suốt chuyến (`vcgencmd get_throttled`).
  Bài đang claim 186 FPS mà chưa biết giữ được bao lâu khi ở ngoài nắng.
- Quay 30 giây **cảnh không có người** ở mỗi độ cao → dùng đo false-positive rate,
  thứ mà VisDrone-val không đo được tốt.
- Chụp ảnh cái drone + sơ đồ gắn camera → dùng cho hình setup trong paper.

---

## Sau chuyến bay: quy trình đưa số vào paper

1. Copy video RAW + log `.bin` về máy.
2. Tách frame: `jetson/record_flight.py` (hoặc ffmpeg 1 fps), gán nhãn Roboflow như lần
   trước, xuất YOLO format vào `data/flight_val_v2/`.
3. Rút phổ rung từ log `.bin` → cập nhật `_PRESETS` trong `vib_consistency.py` nếu lệch nhiều.
   **Nếu sửa `_PRESETS` thì phải đo lại TOÀN BỘ**, vì mọi số retention trong bài phụ thuộc nó.
   Cách làm đúng:
   ```bash
   mv runs/paper_queue/UNIFIED.json runs/paper_queue/UNIFIED_presets_cu.json
   python paper_eval_unified.py --seeds 0 1 2 --out runs/paper_queue/UNIFIED.json
   ```
   `paper_eval_unified.py` lưu **vân tay của `_PRESETS`** trong JSON và sẽ **từ chối chạy**
   (exit code 2) nếu phát hiện tham số đã đổi mà file cũ vẫn còn, kèm thông báo chỉ rõ phải
   làm gì. Không có guard này thì nó sẽ *bỏ qua* 12 model cũ và bạn được một bảng trộn số đo
   bằng 2 bộ tham số rung khác nhau — đúng cái lỗi mà script sinh ra để chống.
4. Eval: `run_eval_coco.py --val-images data/flight_val_v2/images ...` cho 3 model
   (baseline / vibration-aware / consistency) để bảng bay thật có đủ 3 dòng.
5. Sửa `paper/sections/12_exp_flight.tex` và **xoá đoạn Limitations về IMU**.

---

## Checklist mang ra bãi

```
[ ] the nho camera da format, con >32 GB
[ ] SDLOG_PROFILE da bat High rate + Vibration (kiem trong QGC TRUOC khi di)
[ ] pin drone: du cho >=6 luot bay
[ ] pin du phong cho Pi (neu chay detector on-board)
[ ] ghi giay: gio bat dau moi luot, do cao, ten nguoi, shutter/ISO
[ ] moc dong bo: lac manh ngay sau khi arm, MOI luot
[ ] sau moi luot: kiem video da ghi that (dung tin den bao)
[ ] truoc khi ve: copy log .bin (the nho Pixhawk) ra laptop NGAY tai bai
```

Ghi giấy quan trọng hơn bạn nghĩ. Về nhà mà không nhớ clip nào ở độ cao nào thì coi như
mất trục "độ cao", và đó là trục có giá trị nhất cho use case 3-5m của bạn.
