# Quy trình ra bãi bay — làm theo đúng thứ tự

Mọi thứ đã cài và kiểm thử trên Pi ngày 2026-08-27. Còn **đúng một việc bạn phải làm
bằng tay**: chỉnh 10 tham số trong Mission Planner (Bước 0). Sau đó là chạy lệnh.

---

## Bước 0 — Mission Planner (LÀM TRƯỚC, ở nhà)

Chưa chỉnh xong 5 dòng đầu thì **không bay tự động**.

| tham số | đặt thành | vì sao |
|---|---|---|
| `ARMING_SKIPCHK` | **0** | đang là −1 = bỏ qua TOÀN BỘ kiểm tra trước khi arm |
| `BATT_LOW_VOLT` | **14.0** | đang là 10 — pin 4S không bao giờ tụt tới đó |
| `BATT_CRT_VOLT` | **13.5** | đang là 14, tức CAO hơn ngưỡng cảnh báo — ngược |
| `BATT_FS_LOW_ACT` | **2** (RTL) | đang là 0 = không làm gì |
| `BATT_FS_CRT_ACT` | **1** (Land) | đang là 0 = không làm gì |
| `FS_GCS_ENABLE` | **1** (RTL) | mất Pi thì hiện máy bay chỉ treo tại chỗ |
| `FENCE_ENABLE` | **1** | hàng rào 300 m / trần 100 m đã cấu hình sẵn mà đang tắt |
| `GUID_TIMEOUT` | **1.0** | khớp `command_timeout_s`; dừng sau 1 s nếu Pi treo |
| `WP_SPD` | **4.0** | đang 10 m/s — phòng thủ nhiều lớp |
| `INS_LOG_BAT_MASK` | **7** | ghi phổ rung cả 3 IMU, cần cho phần rung của bài báo |

⚠️ Firmware là **ArduPilot 4.7.0**: tên tham số đã đổi so với hướng dẫn cũ trên mạng, và
**đơn vị đổi từ cm/s sang m/s**. Đừng chép số từ hướng dẫn viết cho bản cũ.

Kiểm lại: `cd ~/NhanHuynh && ./run.sh py param_check.py` (cần dừng những thứ đang giữ cổng).

---

## Bước 1 — Trên Pi, trước khi ra bãi

```bash
cd ~/NhanHuynh
./run.sh check          # phải ra "San sang ra bai." — 0 FAIL
```

Kiểm: Hailo, thư viện, HEF, camera, chạy thử model, nhiệt, đĩa, đồng hồ NTP.
Đồng hồ chưa đồng bộ NTP thì dấu thời gian ảnh **không căn được với log bay `.bin`**,
và toàn bộ dữ liệu mất khả năng đối chiếu — đừng bỏ qua mục này.

---

## Bước 2 — Ra bãi: bật hai tiến trình

**Cửa sổ 1 — chia cổng MAVLink:**
```bash
cd ~/NhanHuynh && ./run.sh mux
```
Giữ `/dev/ttyACM0` rồi phát ra UDP 14550 (vòng bám) và 14551 (dự phòng, cho Mission
Planner trên laptop nếu muốn: thêm `--out <ip-laptop>:14552`).

**Cửa sổ 2 — vòng bám + bảng giám sát:**
```bash
cd ~/NhanHuynh/follow_drone
../venv/bin/python3 -u main.py --no-fly  # thử trên bàn, KHÔNG gửi lệnh nào
../venv/bin/python3 -u main.py           # bay thật
#   -u = tắt buffer stdout. Thiếu nó thì run.log RỖNG cho tới khi chương trình thoát,
#        đúng lúc ngoài bãi cần đọc log nhất thì không có gì để đọc.
```

Rồi mở trên laptop/điện thoại cùng WiFi: **http://&lt;ip-pi&gt;:8080**

> IP do DHCP cấp nên **đổi theo ngày** (114 → 115 hôm 2026-08-28, → 115 hôm 2026-09-04).
> Lấy IP đúng bằng `hostname -I` trên Pi, hoặc dùng thẳng `http://pi5.local:8080`.

> ⚠️ **KHÔNG chạy `./run.sh gcs` cùng lúc.** NPU Hailo chỉ một tiến trình dùng được;
> `main.py` đã tự phục vụ bảng giám sát bằng chính khung hình và kết quả phát hiện của
> nó, nên không tốn thêm lượt suy luận nào.

---

## Bước 3 — Trên bảng giám sát, trước khi arm

Nhìn đủ 5 thứ này rồi mới cất cánh:

| ô trên bảng | phải thấy |
|---|---|
| **Sẵn sàng bay** | `SẴN SÀNG ARM` màu xanh (không phải `CHƯA ARM ĐƯỢC`) |
| **GPS** | `3D` trở lên, **≥ 8 vệ tinh**, HDOP < 2.0 |
| **Rung** | cả ba trục **< 30 m/s²**, clipping **0/0/0** |
| **Pin** | > 15.5 V khi chưa nổ máy |
| **Nhiệt** | < 70 °C, throttle `0x0` |

Cần ga (kênh 3) ở **vị trí thấp nhất** trước khi arm — lần kiểm tra 27/08 thấy nó đang
ở 1943 trong khi min là 983.

---

## Bước 3b — Bay TAY trước để soi model (khuyên làm trước khi giao quyền)

```bash
cd ~/NhanHuynh/follow_drone
../venv/bin/python3 -u main.py --no-fly 2>&1 | tee run_nofly.log
```
Chạy đủ camera → NPU → OC-SORT → chọn mục tiêu → PID, đẩy **hết** lên bảng giám sát
như lúc bay tự động, nhưng **không gửi một lệnh vận tốc nào**: `--no-fly` chặn arm
(main.py:385), chặn cất cánh (main.py:398) và chặn `send_velocity_body` (main.py:501).
Bạn cầm tay điều khiển bay bình thường.

Dùng để trả lời "độ cao nào nhận diện tốt nhất": CSV `logs/flight_*.csv` đặt
`altitude_agl`, `n_detections`, `best_conf` trên **cùng một dòng**.

⚠️ `--no-fly` **bỏ qua preflight** và chỉ *in cảnh báo* khi health check trượt chứ không
dừng. Nên chạy được ở đây **không** chứng minh bay tự động sẽ chạy được.

---

## Bước 4 — Bay

Có hai cách giao quyền. **Lần đầu nên dùng cách B.**

**Cách A — GUIDED từ dưới đất, phần mềm tự cất cánh:**
```bash
../venv/bin/python3 -u main.py
```
Chuỗi trong `main.py`: chuyển GUIDED → arm → cất cánh tới `takeoff_altitude_m` (5 m)
→ vào vòng bám ở **30 Hz**.

**Cách B — cất cánh Loiter bằng tay rồi mới giao:**
1. Cất cánh **Loiter** bằng tay, lên ~5 m, hover cho ổn định.
2. Trên Pi: `../venv/bin/python3 -u main.py --no-takeoff`
3. **Tự gạt công tắc chế độ sang GUIDED** trên tay điều khiển.

> ⚠️ `--no-takeoff` **KHÔNG tự chuyển GUIDED**. Lệnh `set_mode('GUIDED')` nằm bên trong
> khối `if not args.no_fly and not args.no_takeoff:` (main.py:398) nên cờ này bỏ qua
> luôn cả việc đổi mode. Quên gạt thì ArduPilot lặng lẽ bỏ hết setpoint — drone treo
> yên, **không có báo lỗi nào**. Nhìn ô "chế độ bay" trên bảng giám sát để xác nhận.

Cả hai cách đều chạy `preflight_check()` (main.py:385 — chỉ `--no-fly` mới bỏ qua, còn
`--no-takeoff` thì **không**). Trượt là thoát ngay, không cất cánh.

Trong lúc bay, bảng giám sát hiện: chế độ bay, ARM, độ cao, pin, GPS, tư thế, **rung
thật**, trạng thái vòng bám (FOLLOW / COAST / HOVER), mục tiêu đang khoá,
khoảng cách ước lượng, và bốn lệnh vận tốc đang gửi.

**Nó bám ai?** Lần đầu thấy người, nó chọn **hộp to nhất** (≈ người gần nhất) rồi **khoá
vào ID đó** — sau đó bám đúng người đó kể cả khi có người khác gần hơn. Nếu mất người:

| mốc | hành vi |
|---|---|
| 0 → 1,5 s | OC-SORT giữ ID sống bằng dự đoán Kalman |
| 1,5 → 4,5 s | vẫn giữ khoá, máy bay bám theo dự đoán (pha coast) |
| 4,5 → 9,5 s | **chỉ tìm lại đúng người cũ** (so vị trí + cỡ hộp cuối). Người cũ quay lại — kể cả với ID mới — thì bám lại ngay. Người lạ đứng gần **không** bị vồ |
| sau 9,5 s | mới chấp nhận chọn người khác (tắt bằng `auto_relock: false`) |

**Bấm ĐỔI MỤC TIÊU** trên bảng giám sát để chọn lại ngay, không cần chờ. Đây là thứ **duy
nhất** trang web tác động được — không arm, không đổi chế độ bay, không gửi vận tốc.

**Giành lại quyền bất cứ lúc nào**: gạt công tắc chế độ trên tay điều khiển. ArduPilot
rời GUIDED ngay lập tức, mọi lệnh từ Pi bị bỏ qua.

`Ctrl+C` trên Pi → gửi vận tốc 0 rồi buông quyền. **KHÔNG đổi chế độ bay, KHÔNG RTL.**
Máy bay đứng yên tại chỗ, vẫn ở chế độ bạn đang để — **bạn phải tự gạt công tắc và hạ
cánh bằng tay.**

---

## ⚠️ Sự cố 2026-09-04 và luật mới về chế độ bay

**Chuyện gì đã xảy ra.** Bay LOITER, máy bay tự vọt thẳng lên trời.

**Nguyên nhân.** `SafetyManager.update_target_status()` bản cũ leo thang tới LOITER rồi
RTL và **gọi thẳng `mav.set_mode()`**. `main.py` gọi thang đó **vô điều kiện** — không
nằm trong `if not args.no_fly` — nên kể cả chạy `--no-fly` phần mềm vẫn cướp được chế độ
bay. Chuyến đó camera **không thấy người lần nào suốt 249 giây** (`n_detections = 0`),
nên thang chạy tới cùng liên tục. Và **RTL của ArduPilot leo lên `RTL_ALT` (mặc định
15 m) trước khi bay về** — đó chính là cú vọt lên. Đường thứ hai: khối `finally` của
`main.py` cũng gọi `set_mode('RTL')`, nên chỉ cần `Ctrl+C` là đủ để nó leo.

**Luật bây giờ — người lái sở hữu chế độ bay:**

1. Phần mềm **không bao giờ** gọi `set_mode`, trừ đúng một chỗ: chuỗi cất cánh tự động
   (`main.py:401`), và chỗ đó chỉ chạy khi KHÔNG có `--no-fly` và KHÔNG có `--no-takeoff`.
2. **Không còn RTL tự động**, ở bất cứ đâu.
3. Mất mục tiêu: thang dừng ở **HOVER** — gửi vận tốc 0, đứng yên tại chỗ, **mãi mãi**,
   cho tới khi bạn tự gạt công tắc.
4. Lệnh chỉ được gửi khi máy bay **đang ở GUIDED**. Gạt sang chế độ khác là phần mềm câm
   ngay, và in một dòng báo trên terminal.

Chốt bằng test: `tests/test_safety_no_mode_change.py`. Chạy với code cũ thì hỏng 8/8 mục.
**Đừng đưa `set_mode` trở lại `safety.py`.**

---

## Bước 5 — Sau chuyến bay, lấy dữ liệu

| nguồn | ở đâu | dùng cho |
|---|---|---|
| Log vòng bám | `follow_drone/logs/flight_*.csv` | có **`vibe_x/y/z`, `clip`, `n_detections`, `best_conf`** cùng dòng với lệnh điều khiển |
| Log bay ArduPilot | thẻ nhớ trong Pixhawk, `.bin` | phổ rung IMU đầy đủ, đọc bằng Mission Planner hoặc `mavlogdump` |
| Video + ảnh | chạy `./run.sh record` cho lượt riêng | gán nhãn làm val set 3–6 m |

Kéo về máy: `bash pull_flight_data.sh`

### Vì sao cột `vibe_*` trong CSV quan trọng với bài báo

Bài báo đang lập luận rằng huấn luyện có mô phỏng rung làm model bền hơn khi rung.
Nhưng mức rung lúc train là **mô phỏng**, chưa từng đối chiếu với rung thật. File CSV
này đặt **rung đo được** và **chất lượng phát hiện** trên **cùng một dòng, cùng một
dấu thời gian**. Đó là thứ biến một lập luận thành một phép đo:

- vẽ `best_conf` và `n_detections` theo `vibe_z` → đường cong chính xác-theo-rung **thật**
- so đường cong đó với đường cong trên VisDrone-Shake mô phỏng
- nếu hai đường khớp nhau thì mô hình rung mô phỏng được kiểm chứng; nếu lệch thì
  bản thân chỗ lệch đó là một phát hiện đáng viết

Muốn có đủ dải rung: bay **một lượt bình thường** và **một lượt rung mạnh** (bay tiến
nhanh, đổi hướng gấp) trên cùng cảnh, cùng độ cao.

---

## Sự cố thường gặp

| hiện tượng | nguyên nhân | xử lý |
|---|---|---|
| `HAILO_OUT_OF_PHYSICAL_DEVICES` | có tiến trình khác đang giữ NPU | `pkill -f hailo_gcs.py` rồi chạy lại |
| `device reports readiness to read but returned no data` | hai tiến trình cùng mở `/dev/ttyACM0` | dùng `./run.sh mux`, mọi thứ khác trỏ vào UDP |
| `PreArm: Need Position Estimate` | chưa có GPS fix | ra chỗ trống trời, chờ ≥ 8 vệ tinh |
| Camera chỉ 15 fps | `BUFFERSIZE=1` hoặc auto-exposure | đã sửa cả hai; nếu tái diễn xem `runs/pi_camera_measured_2026-08-27.md` |
| Vòng bám không khoá được người | conf quá cao, hoặc người quá xa | conf đang 0.25 (điểm F1 tốt nhất đo được cho HEF `_coco`); dưới 8 m thì mới bắt tốt |
| `--no-fly` thoát ngay vì GPS | (đã sửa) bản cũ dừng vòng lặp | cập nhật `main.py` mới nhất |

---

## Giới hạn cần biết trước khi tin vào số

> ⚠️ **CẬP NHẬT 2026-09-04 — đoạn dưới nói về HEF `_fair` ĐỜI TRƯỚC, không còn dùng.**
> `config.yaml` hiện trỏ `pfdet_v17_640_coco_hailo8l.hef`, conf **0.25**, đo trên chính
> 92 khung đó được **F1 0.880** (recall 0.837, prec 0.928), 56.9 FPS. Tức lỗi COCO nói
> dưới đây **đã sửa**. Giữ đoạn cũ để đối chiếu lịch sử.

**(LỊCH SỬ — HEF `_fair`)** Trên 92 khung bay thật ở ~6 m, HEF đó đạt F1 **0.697**
(bắt được người ở 57.6 % số khung), trong khi checkpoint gốc `train_v17_deploy` đạt
**0.882** (81.5 %). Nguyên nhân là một lỗi cấu hình: COCO chưa bao giờ được bật trong
20 epoch finetune sinh ra HEF này. Xem `runs/flight_val_hef_2026-08-27.md`.

Nghĩa là: **bay được, thu data được**, nhưng nếu thấy tỷ lệ bắt hụt cao thì đó là giới
hạn đã biết của HEF này, không phải lỗi hệ thống. Sửa được sau khi khôi phục
`data/visdrone` + `data/coco_person` và train lại bằng
`configs/train_config_v17_hailo_coco.yaml`.
