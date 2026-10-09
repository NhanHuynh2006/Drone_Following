# PROJECT.md — SteadyDet-N (phát hiện người từ drone)

> File này là **bản đồ dự án**: đọc 1 lần hiểu ngay đang làm gì / đã làm gì, khỏi lội hết source.
> **Quy ước: mỗi lần sửa gì đáng kể → cập nhật file này.** (Cập nhật gần nhất: 2026-09-08)

---

## 0. QUY ƯỚC TÊN (chốt 2026-09-08) — đọc trước khi sửa bất cứ tài liệu nào

Tên cũ `SteadyDet-N` / `SteadyDet-W` đã đổi thành **SteadyDet**. Ánh xạ:

| Tên trình bày | Là model nào | Tham số / FLOPs | Tên trong CODE (giữ nguyên) |
|---|---|---|---|
| **SteadyDet-N** | bản chốt, dùng cho paper + deploy | 1.17 M / 4.6 G | `v17` — `models/pfdet_nano_v17.py`, `PFDetNanoV17`, `configs/train_config_v17_*.yaml`, `runs/train_v17_*`, `pfdet_v17_*.hef` |
| **SteadyDet-W** | bản rộng, dòng "generality" trong bảng chính | 1.00 M / 12.9 G | `v15` — `models/pfdet_nano_v15.py`, `PFDetNanoV15`, `runs/train_v15_*` |

⚠️ **CHỈ đổi tên hiển thị trong tài liệu.** Tên file, tên lớp, tên config, đường dẫn `runs/`
và khoá `model.version` trong checkpoint **vẫn là `v15`/`v17`** — đổi những thứ đó là hỏng
việc nạp checkpoint và mất dấu vết mọi kết quả đã đo.

**Tên nhánh (đổi 2026-09-09):** nhánh `PAPER` → **nhánh CHUẨN** (VisDrone thuần, không COCO,
để so công bằng với prior work). Nhánh `DEPLOY` giữ nguyên tên. Chữ "paper/bài báo" trong tài
liệu giờ chỉ còn nghĩa **bài báo**, không còn là tên nhánh.

Hai thứ **CỐ Ý giữ tên cũ**:
- `runs/yolo/*/PAPER_DONE` — file mốc, được `fix_queue.sh` và `check_progress.sh` đọc; đổi là
  vỡ logic guard mà chẳng được gì (thí nghiệm đã xong hết).
- `backup_v17_baseline/` — bản sao đóng băng của mã cũ; đổi tên trong đó là mất ý nghĩa backup.

---

## 1. Dự án này là gì?
Phát hiện **người** (1 class) từ **camera drone**.
> ⚠️ **DEPLOYMENT THẬT: bay THẤP 3-5m** → người **to/gần** (KHÔNG phải tí xíu). Khác VisDrone-val (bay cao,
> người tí xíu). Vì vậy: (a) **trộn COCO person** (người gần) là CÓ ÍCH cho use case thật; (b) AP đo trên
> VisDrone-val không phản ánh đúng deployment — lý tưởng cần val set quay ở 3-5m.

Train trên **VisDrone** (data drone có nhãn) + **COCO person** (người gần, hợp tầm thấp).
Hai mục tiêu song song:
1. **Paper hội nghị** — đóng góp chính = **vibration-consistency**. Chỉ viết về MODEL (không có phần deploy).
2. **Deploy realtime** — **DỰ ÁN KHÁC**. Phần cứng ĐÃ ĐỔI: **Hailo NPU trên Raspberry Pi 5** (AI Kit/HAT, INT8).
   *(Trước tính Jetson Nano TensorRT FP16 — nay chuyển Hailo. Điều khiển bay vẫn chạy CPU Pi 5; chỉ đổi inference sang HailoRT.)*
> **⚠️ FIRMWARE BAY = ArduPilot Copter** (chốt 2026-08-27, user đã đổi từ PX4). Điều khiển companion = chế độ
> **GUIDED** (ArduPilot KHÔNG có OFFBOARD — đó là chế độ riêng của PX4). Code dùng: **`follow_drone/`**
> (Python + pymavlink). `jetson/control/follow_px4.cpp` là **nhánh cũ PX4/MAVSDK, KHÔNG bay được** — đã dán
> banner cảnh báo. Chỉnh param bằng **Mission Planner**, không phải QGroundControl. Log bay là `.bin`
> DataFlash, không phải `.ulg`.

> **2 nhánh model độc lập:**
> | | CHUẨN (số cho bài báo) | DEPLOY (dự án khác) |
> |---|---|---|
> | model | **v17 NMS-free** (end-to-end, config `train_config_v17_nms_free.yaml`) | v17 thường (`train_config_v17_light.yaml`) |
> | data | **VisDrone CHUẨN, KHÔNG COCO** (so sánh công bằng) | + COCO + footage 3-10m (cảnh thật) |
> | đóng góp | vibration-consistency | — |
> Lưu ý: NMS-free có thể cho AP thấp hơn (head o2o khó train) → phải train thử; nếu tụt thì làm ablation, dùng v17 thường làm headline.

> **⭐ QUYẾT ĐỊNH VỀ BÀI BÁO (2026-07-21): HYBRID — 1 bài, model + system.**
> - **Bảng chính** = VisDrone-CHUẨN (light + vibration + baselines) → so công bằng với prior work, AN TOÀN.
> - **Section "Real-world Deployment"** = model `deploy` (COCO variant) + Hailo (**112.55 FPS @320 / 55.78 FPS @640**, đo lại 2026-08-24; con số 186fps cũ không tái lập được) + ArduPilot follow (định tính; định lượng khi có footage 3-5m).
> - **KHÔNG** lấy deploy làm model chính vì chưa có val set 3-5m → không chứng minh được COCO có ích (VisDrone-val làm COCO trông "hại"). Lý do bỏ full-pivot: buộc retrain 5 baseline trên COCO-mix + refinetune lại vib từ deploy + phải đi quay/gán nhãn val 3-5m trước.
> - Hệ quả: pipeline hiện tại (paper_queue + yolo_queue trên VisDrone-chuẩn) ĐÚNG hướng, không đổi. Nếu sau này quay được footage 3-5m thật → nâng cấp thành full-pivot.

Có sẵn cả hệ thống **bay theo người** (follow drone) — xem `FOLLOW_CONTROL.md`, `follow_drone/`.

---

## 2. Trạng thái hiện tại (TL;DR) — cập nhật 2026-07-27

**MỌI THÍ NGHIỆM ĐÃ XONG.** Không còn job nào train.

**Cron `@reboot`: ĐÃ TẮT HẾT (kiểm lại 2026-07-27, 7/7 dòng).** Backup:
`runs/crontab_backup_2026-07-27.txt`. Kiểm nhanh bất cứ lúc nào:

```bash
crontab -l | grep -vE '^\s*#|^\s*$'    # ra RỖNG = không dòng nào bật
```

⚠️ Lần 2026-07-26 tôi bỏ sót `scratch_queue.sh` (bật cho thí nghiệm scratch rồi quên tắt).
Nó có chạy lúc reboot nhưng guard `[ -f $Q/SCRATCH_ALL_DONE ] && exit 0` chặn được, thoát
ngay không tốn GPU. **Bài học: sau khi bật tạm một queue, tắt ngay khi nó xong**, và kiểm
bằng lệnh trên chứ đừng tin trí nhớ. Đã xác minh không có systemd service / autostart /
rc.local nào khác gọi tới queue.

| Hạng mục | Trạng thái |
|---|---|
| **Model đóng góp** | `runs/train_v17_vib_augonly/best.pt` — **retain 36.2±0.7%** (baseline 16.8±0.7%). SteadyDet-W: `runs/train_v15_augonly/best.pt` **39.4±0.3%** (baseline 22.2±0.9%) |
| Model tốt nhất clean AP | musgd_v3 `runs/train_v15_light_musgd_v3/best.pt`, AP@0.5 0.5932 |
| Model deploy v17 (+COCO) | `runs/train_v17_deploy/best.pt`; bản vib-aware `runs/train_v17_augonly_deploy/best.pt` |
| **Chọn & giữ mục tiêu** | Chọn **hộp to nhất** (≈ gần nhất) rồi **khoá ID**. Mất người: giữ khoá 3 s (coast) → 5 s **chỉ tìm lại đúng người cũ theo vị trí+cỡ hộp** (OC-SORT cấp ID MỚI khi quay lại, không có re-ID ngoại hình) → sau đó mới chấp nhận người khác. Nút **ĐỔI MỤC TIÊU** trên dashboard để chọn lại ngay. Test hồi quy: `follow_drone/tests/test_target_selector.py` (12/12) |
| ✅✅ **HEF MỚI ĐÃ ĐO TRÊN PI (2026-08-28)** | `pfdet_v17_640_coco_hailo8l.hef` — **F1 0.880 @conf 0.25** trên 92 khung bay thật, **recall 0.620 → 0.837** so với bản cũ. 56.9 FPS (17.6 ms), dư xa ngân sách 30 Hz. Đã đặt vào `follow_drone/config.yaml`. INT8 **không mất gì** so với FP32 (0.880 vs 0.836, trong nhiễu). ⚠️ Ngưỡng conf khác nhau theo từng HEF — bản cũ 0.30, bản mới 0.25 |
| ⚠️ **IP Pi đổi theo DHCP** | Dùng `raspi@pi5.local` (mDNS) thay IP cứng. 114 → 115 ngày 28/08. Script đã chuyển hết sang hostname |
| ✅ **ĐÃ SỬA lỗi COCO (2026-08-27 tối)** | Khôi phục data + train lại `runs/train_v17_hailo_coco/best.pt`. Trên flight_val FP32: **F1 0.768 → 0.836** (lấy lại 60% khoảng cách tới 0.882 của `train_v17_deploy`; 0.046 còn lại là do bỏ AreaAttention, muốn lấy nốt thì dùng `ConvAttnBlock`). HEF đã compile: `pfdet_v17_640_coco{,_c512}_hailo8l.hef`. **CHƯA đo INT8** — chạy `scripts/deploy_to_pi.sh` khi có Pi. Chi tiết: `runs/retrain_coco_2026-08-27.md` |
| ⛔ **HEF đang deploy YẾU trên domain bay** (bản cũ, sẽ thay) | Đo trên `data/flight_val` (92 frame bay thật 6m): HEF hiện tại F1 **0.697** @conf 0.30 (recall 57.6%), trong khi `train_v17_deploy` FP32 đạt **0.882** (recall 81.5%). Nguyên nhân: `train_config_v17_hailo.yaml` có `extra_enable_epoch: 100` > `epochs: 20` → **COCO chưa bao giờ bật** trong 20 epoch finetune. Fix sẵn ở `configs/train_config_v17_hailo_coco.yaml` nhưng **chưa chạy được: `data/visdrone` + `data/coco_person` không còn trên đĩa**. Chi tiết: `runs/flight_val_hef_2026-08-27.md` |
| **Ngưỡng conf deploy** | **0.30** (không phải 0.4). Đo được: 0.40 → recall 19.6%; 0.20 → 1.30 box rác/khung; 0.30 → recall 57.6% / precision 88.3% |
| **Liên lạc Pi ↔ mạch bay** | ✅ Nối thật 2026-08-27: **Holybro Pixhawk6C, ArduPilot Copter 4.7.0**, USB `/dev/ttyACM0`, MAVLink v2, 94–100 gói/s, 0 lỗi. Cảm biến + EKF3 + rung nền đều sạch. ⛔ **5 tham số PHẢI sửa trước khi bay tự động** (`ARMING_SKIPCHK=-1`, ngưỡng pin ngược, failsafe pin tắt, `FS_GCS_ENABLE=0`, `FENCE_ENABLE=0`) — xem `runs/mavlink_check_2026-08-27.md` |
| **⚠️ ArduPilot 4.7 đổi tên + ĐƠN VỊ param** | `WPNAV_SPEED`→`WP_SPD` (m/s, không phải cm/s), `ANGLE_MAX`→`ATC_ANGLE_MAX` (độ), `PSC_VELXY_P`→`PSC_NE_VEL_P`, `RTL_ALT`→`RTL_ALT_M`, `ARMING_CHECK`→`ARMING_SKIPCHK` (nghĩa NGƯỢC). Chép số từ hướng dẫn cũ = sai 100 lần |
| **✅ VÒNG BÁM CHẠY ĐƯỢC TRÊN HAILO** | `follow_drone/follow/detector_hailo.py` (mới) — cùng giao diện với bản PyTorch nhưng chạy HEF trên NPU: **19,93 ms/khung → trần 50 Hz**. Chọn bằng `model.backend: hailo` trong config. Chạy thử `main.py --no-fly` cho **29,9 Hz**, đúng nhịp đặt |
| **Dashboard GỘP vào vòng bám** | NPU chỉ 1 tiến trình dùng được → `hailo_gcs.py` và `follow_drone` KHÔNG chạy song song được. `follow/dashboard.py` (mới) nhúng thẳng bảng giám sát vào `main.py`, dùng lại khung hình + phát hiện sẵn có, **không tốn thêm lượt suy luận**. Ngoài bãi chỉ cần 2 lệnh: `./run.sh mux` và `main.py` |
| **Log bay có RUNG THẬT** | `flight_*.csv` thêm cột `vibe_x/y/z`, `clip`, `n_detections`, `best_conf` — rung đo được và chất lượng phát hiện **trên cùng một dòng**. Đây là dữ liệu để đối chiếu mô hình rung mô phỏng với rung thật, thứ bài báo đang thiếu |
| **Quy trình ra bãi** | `follow_drone/RUNBOOK.md` — 5 bước, kèm 10 tham số Mission Planner và cách đọc dữ liệu sau bay |
| **Nhịp gửi lệnh = 30 Hz** | Chọn theo số đo, không đoán: detector sinh thông tin 30 lần/s → mỗi lệnh mang đúng 1 phát hiện mới. 10 Hz thêm 33 ms trễ vô ích; 50 Hz thì 40% lệnh không mang tin mới. Khứ hồi lệnh đo được **p50 4.75 ms / p99 7.73 ms**, 0% mất gói tới 50 Hz. Xem `runs/control_rate_2026-08-27.md` |
| **Chia cổng MAVLink** | `./run.sh mux` (`mavmux.py`) — `/dev/ttyACM0` chỉ 1 tiến trình mở được; mux giữ cổng rồi phát UDP để `follow_drone/` + dashboard cùng chạy. Đã test 2 client × 100 gói/s |
| **Camera trên Pi** | ✅ C270 HD, 2026-08-27. **e2e @640 = 30.0 FPS** sau khi sửa 2 lỗi camera (BUFFERSIZE=1 rớt nửa khung + auto-exposure tự hạ fps) — trước đó kẹt 15.4. Xem `runs/pi_camera_measured_2026-08-27.md`. Xem trực tiếp: `./run.sh stream` → `http://<ip-pi>:8080` |
| **Deploy Hailo** | ✅ Pi5+Hailo-8L, đo 2026-08-24. **@640: 63.66 FPS / 15.26ms** (HEF recompile; bản cũ 15/7 chỉ 55.78 — compile kém). **@320: 112.55 FPS / 8.40ms**. **Nhanh hơn YOLO26n 1.70×** trên cùng chip (37.46 FPS) — biến quyết định là *contexts* (4 vs 6), không phải GFLOPs. 186 FPS cũ = batch=4. Xem `runs/pi_hailo_measured_2026-08-24.md`. Công cụ: `raspi@192.168.196.114:~/NhanHuynh/` |
| Điều khiển bay | ✅ **ArduPilot Copter, chế độ GUIDED** — `follow_drone/` (pymavlink, OC-SORT, PID, safety ladder). `jetson/control/follow_px4.cpp` = nhánh cũ PX4, KHÔNG dùng |
| **BÀI BÁO** | `paper/` (SOICT 2026, LNCS). Zip nộp: `soict_paper_overleaf_v3.zip`. Số liệu → `runs/paper_queue/UNIFIED.json`. Xem `PAPER_INPUTS.md` + `paper/README_OVERLEAF.txt` |
| **Eval hợp nhất** | ✅ `paper_eval_unified.py` (VisDrone-val, 3 seed) + `flight_eval_unified.py` (bay thật, 1 code path, bootstrap trên frame) |
| **BAY THẬT đầu tháng 8** | ⚠️ **ĐỌC `FLIGHT_DATA_SPEC.md` TRƯỚC KHI BAY.** Cần video RAW + log ArduPilot `.bin` có IMU + nhiều độ cao/người/cảnh |
| Còn phải làm (người) | compile Overleaf đếm trang thật (ước ~12.25/12); điền tên+email GVHD + Acknowledgements |

---

## 3. Đóng góp khoa học (ĐỔI KHUNG 2026-07-27, model đứng tên chính)

**Tựa bài:** *SteadyDet-N: A 4.6-GFLOP Person Detector That Leads Every Compact Baseline
Under UAV Vibration*. Đóng góp (i) = **model + chế độ train, tính chung**.

**Lập luận giữ cho kiến trúc là đóng góp — "không nửa nào đủ một mình":**

| | mCP |
|---|---|
| chỉ kiến trúc (SteadyDet-N không xử lý) | 0.0318 — bét bảng |
| chỉ kỹ thuật (YOLO11n + xử lý) | 0.0542 — không hơn YOLO11n thường (0.0557) |
| **cả hai** | **0.0617 — thắng cả 4 YOLO nano** |

⚠️ **KHÔNG được bỏ phần rung để làm bài kiến trúc thuần.** Trên clean AP, SteadyDet-N thua
YOLO26n (baseline yếu nhất) **29%** mà chỉ tiết kiệm 21% FLOPs — tỷ giá đó không ai nhận.
Bài cũng không có module mới nào (UIB=MobileNetV4, attention=Swin, CSP=CSPNet, BiFPN=
EfficientDet). Rung là trục DUY NHẤT model thắng. Bỏ rung = bỏ luôn bằng chứng.

Phương pháp là **vibration-aware training**: lúc train, thay 50% mỗi batch bằng bản
**đã làm rung**, rồi tính **loss detection bình thường** trên đó. Train-time only → 0 chi phí
inference. Code: `vib_consistency.py` (corrupt_batch) + `vib.mode: aug` trong config.

Hai kết quả ÂM, và chúng là phần giá trị nhất của bài:

1. **Consistency THUA augmentation.** Bản forward 2 lượt + ép dự đoán rung khớp dự đoán sạch
   (stop-gradient) chỉ đạt **30.0±0.6%** vs **36.2±0.7%**, mà tốn **1.8×** thời gian train.
   Đối chứng "train thêm 120 epoch KHÔNG rung" = 16.9% ≈ baseline → gain KHÔNG do train thêm.
2. **Không transfer sang YOLO11n.** 21.9 → 21.5% (from scratch), 20.1 → 20.1% (pretrained).
   Đã bác 3 giả thuyết: COCO-pretrain, object scale, headroom. Nguyên nhân còn lại là kiến
   trúc hoặc loss — chưa cô lập được, bài viết thẳng là chưa biết.

**Kiến trúc LÀ đóng góp — nhưng trên trục mCP, không phải clean AP.** Xem mục 4.

---

## 4. Kết quả chính (VisDrone val 548 ảnh, @640, person-only, 1 giao thức + 3 seed)

Nguồn: `runs/paper_queue/UNIFIED.json`. **mCP** = AP trung bình qua sev1-3 = độ chính xác
KHI ĐANG RUNG. Đây mới là trục triển khai; clean AP không phải.

| model | M | GFLOPs | clean AP | sev2 | **mCP** | retain% |
|---|---|---|---|---|---|---|
| **SteadyDet-W + vib-aware** | 1.00 | 12.9 | 0.2219 | 0.0839 | **0.0874** | **39.4±0.3** |
| RT-DETR-l | 32.81 | 108.0 | 0.2502 | 0.0389 | 0.0641 | 25.6±1.1 |
| **SteadyDet-N + vib-aware** | **1.17** | **4.6** | 0.1708 | **0.0518** | **0.0617** | **36.2±0.7** |
| YOLO11-n | 2.59 | 6.4 | **0.2768** | 0.0210 | 0.0557 | 20.1±1.2 |
| YOLOv10-n | 2.71 | 8.4 | 0.2730 | 0.0224 | 0.0538 | 19.7±1.4 |
| YOLOv8-n | 3.01 | 8.2 | 0.2684 | 0.0200 | 0.0522 | 19.5±0.8 |
| YOLO26-n | 2.50 | 5.8 | 0.2675 | 0.0207 | 0.0507 | 19.0±1.0 |
| SteadyDet-N + consistency | 1.17 | 4.6 | 0.1746 | 0.0392 | 0.0524 | 30.0±0.6 |
| SteadyDet-N baseline | 1.17 | 4.6 | 0.1892 | 0.0123 | 0.0318 | 16.8±0.7 |

- Trên **clean AP**: SteadyDet-N thua cả 4 YOLO nano. Đừng claim kiến trúc ở đây.
- Trên **mCP**: SteadyDet-N+vib **thắng cả 4 YOLO nano**, với 45% tham số và 72% compute của
  con mạnh nhất. SteadyDet-W+vib nhất bảng.
- vs RT-DETR-l (28× tham số): thắng ở **sev2** (5.4σ). **THUA** ở mCP trung bình (0.0617 <
  0.0641). Ở **sev3 chỉ 0.64σ = trong nhiễu → KHÔNG được claim.**
- **Phép so sánh này công bằng** vì kỹ thuật KHÔNG nâng được YOLO11n (mCP 0.0557 → 0.0542).
  Nếu nó nâng mọi model thì "ta có xử lý vs họ không" là ăn gian.

---

## 5. Kiến trúc model
**Backbone chung:** stem → P2(s4) → P3(s8) → P4(s16) → P5(s32) → neck → 4 head. **Có P2 (stride 4)** là điểm
mạnh cho người tí xíu (YOLO26n/11n bỏ P2). Loss = **PFDetLossV15** (STAL + ProgLoss + NWD + CIoU, DFL-free —
giống YOLO26). Optimizer tốt nhất = **MuSGD** (Muon+SGD, của YOLO26).

| ver | file | đặc điểm | params |
|---|---|---|---|
| v14 | `models/pfdet_nano_v14.py` | cũ: UIB + BiFPN + AreaAttention | ~2.1M |
| **v15** | `models/pfdet_nano_v15.py` | chính: CSPUIB + LSKBlock + P2. **AP cao nhưng FLOPs cao (12.9G)** | 1.0M |
| v16 | `models/pfdet_nano_v16.py` | variant của v15 | |
| **v17** | `models/pfdet_nano_v17.py` | **MỚI: P2 mỏng (kênh giảm) + LiteHead depthwise-sep → cắt FLOPs 2.8×.** | 1.17M |

**v17 vì sao tốt nhất cho bài toán:** v15 params nhỏ NHƯNG **head P2 ở 160×160 ăn 1.82 GMACs** (28% tổng) →
FLOPs 12.9G. v17 làm P2 path **mỏng** (kênh `p2_ratio`, head `head_ratio`) → **4.6 GFLOPs**, giữ P2.
- **v17 NMS-free (Plan B):** thêm head `o2o` (one-to-one, train k=1) song song head thường (`o2m`). Inference
  dùng o2o → **bỏ NMS**. Bật bằng `model.nms_free: true` trong config. Loss = `DualLossV17` (`utils/losses_v17_e2e.py`).

---

## 6. Bản đồ file quan trọng
```
train_v3.py            # TRAIN chính (mọi version). Có sẵn: DGS, KD, vibration, NMS-free (qua config flag).
models/                # pfdet_nano_v14/15/16/17.py  + __init__(build_model)
utils/losses_v15.py    # loss chính (STAL/ProgLoss/NWD). v17 dùng lại loss này.
utils/losses_v17_e2e.py# DualLossV17 (NMS-free)
utils/musgd.py         # optimizer MuSGD
vib_consistency.py     # ĐÓNG GÓP: mô phỏng rung + consistency loss
eval_robustness.py     # benchmark VisDrone-Shake (đường cong AP theo độ rung)
eval_coco.py / run_eval_coco.py   # eval chuẩn pycocotools (AP@.5:.95, AP75, breakdown mật độ)
build_deploy.py        # đóng gói model → ONNX (FP32) [+ INT8 đã bỏ vì hỏng tiny]
test_accuracy_onnx.py  # 1 FILE test accuracy model ONNX (chạy máy này hoặc Pi)
demo_cam.py            # demo webcam real-time (vẽ box)
README_HAILO.md        # convert v17 -> Hailo HEF (Pi 5 AI Kit) — HD từng bước
export_onnx_hailo.py   # export ONNX sạch cho Hailo (fuse, static, opset11)
compile_hef.py         # ONNX -> HEF (Hailo DFC, chạy trên x86 laptop)
hailo_demo_cam.py      # TEST detect trên Pi5+Hailo (chỉ model+camera, KHÔNG control) — decode/nms nhúng sẵn
hailo_eval_val.py      # ⭐ AP của model HEF INT8 trên VisDrone-val (2 bước: dump trên Pi -> chấm bằng coco_eval)
hailo_bench_latency.py # ⭐ latency end-to-end p50/p95/p99 từng chặng + FPS bền + nhiệt/throttle
cam_ctl.py             # ⭐ CAMERA: SoftAE (auto-sáng mà KHÔNG tụt fps) + khoá manual. Chứa mọi cái bẫy exposure
hailo_gcs.py + gcs.html# ⭐⭐ TRẠM MẶT ĐẤT: ./run.sh gcs -> http://<ip-pi>:8080. Video+box, telemetry ArduPilot
                       #   (ARM/mode/cao/pin/GPS/tư thế), latency từng chặng, đổi model runtime. CHỈ ĐỌC telemetry.
mav_telem.py           # đọc telemetry ArduPilot cho dashboard (pymavlink, KHÔNG gửi lệnh)
mavmux.py              # ⭐ ./run.sh mux — chia 1 cổng MAVLink cho nhiều chương trình (serial -> nhiều UDP)
hailo_stream_web.py    # bản stream gọn hơn (chỉ video+box), tiền thân của hailo_gcs.py
flight_sweep.py/fp32_sweep.py # quét ngưỡng conf trên flight_val -> chọn conf deploy, tách INT8 vs model
flight_record.py       # thu data chuyến bay trên Pi (bản gốc ở ~/NhanHuynh trên Pi)
paper_eval_unified.py  # ⭐ ĐO CHUẨN CHO BÀI BÁO: mọi model 1 giao thức + 3 seed -> runs/paper_queue/UNIFIED.json
train_yolo_vibcons.py  # áp vib-consistency lên YOLO11n (ghi đè model.loss của ultralytics, KL trên bin DFL)
ablation_queue.sh      # HÀNG CHỜ ablation: augonly -> ft_control -> yolo_vibcons -> đo lại + sinh bảng
paper/                 # BẢN VIẾT SOICT 2026 (LNCS). make_tables.py sinh dòng bảng từ UNIFIED.json.
                       #   README_OVERLEAF.txt (cách nộp + thứ tự cắt trang), PENDING_ABLATION.md
paper_queue.sh         # HÀNG CHỜ tự động: train v17 baseline->vib->nms_free->eval (auto-resume crash, cron@reboot)
yolo_queue.sh          # chờ v17 xong -> train baselines so sánh
train_yolo_baselines.py# train YOLOv8n/11n/26n/v10n + RT-DETR (đa dạng họ) trên cùng split, xuất params/FLOPs/mAP
follow_drone/          # ⭐⭐ ĐIỀU KHIỂN BAY THẬT — ArduPilot GUIDED, pymavlink. ĐÂY là nhánh đang dùng.
                       #   main.py (vòng bám 30 Hz + bảng giám sát nhúng) + RUNBOOK.md (quy trình ra bãi)
                       #   follow/detector_hailo.py  — detector chạy NPU, 19.93 ms/khung
                       #   follow/dashboard.py       — web giám sát, dùng lại phát hiện của vòng bám
                       #   follow/{mavlink_client,ocsort,pid,visual_servo,safety,distance}.py
jetson/                # ⚠️ NHÁNH CŨ PX4: build_trt_jetson.py (TensorRT), DEPLOY_GUIDE.md (param PX4 — KHÔNG áp dụng)
jetson/control/        # follow_px4.cpp — PX4/MAVSDK Offboard. GIỮ ĐỂ ĐỐI CHIẾU THUẬT TOÁN, không build
jetson/record_flight.py# CÔNG CỤ RIÊNG (độc lập): thu ảnh+video lúc bay -> làm val set 3-10m thật
jetson/pfdet_v17_*.onnx # model deploy v17 (FP32 ONNX @320/512/640) — TensorRT tự build FP16 trên Jetson
configs/               # train_config_v15_*.yaml, train_config_v17_*.yaml
deploy/                # checkpoint sạch + ONNX để deploy
runs/                  # các lần train (best.pt/last.pt/log/curves)
follow_drone/, FOLLOW_CONTROL.md, drone_follow.py   # hệ thống bay theo người
docs/architecture.html # ⭐ SƠ ĐỒ KIẾN TRÚC toàn hệ thống (train -> HEF -> vòng bám 30 Hz -> Pixhawk).
docs/architecture.json #    Nguồn của sơ đồ trên (archify). Sửa file này rồi render lại, đừng sửa HTML.
```
**Doc khác:** `V17_DESIGN.md` (thiết kế v17 + nghiên cứu SOTA), `RESULTS_*.md` (bảng số), `BEST_PLAN_DATA_DRIVEN.md` (kế hoạch DGS — đã chuyển thành ablation phụ).

---

## 7. Lệnh hay dùng
```bash
# Train (nền, không chết khi đóng terminal)
nohup python train_v3.py --config configs/train_config_v17_light.yaml >> runs/train_v17_light.log 2>&1 &
#   resume:  thêm  --resume runs/train_v17_light/last.pt
#   theo dõi: grep -E "VAL|SAVE" runs/train_v17_light.log | tail
#   dừng:     pkill -f "train_v3.py.*v17"

# Eval accuracy chuẩn (COCO)
python run_eval_coco.py --weights <ckpt.pt> --val-images data/visdrone/val/images \
  --val-labels data/visdrone/val/labels --img-size 640 --profile light --device cuda:0

# Eval robustness (động cơ paper)
python eval_robustness.py --weights <ckpt.pt> --val-images ... --val-labels ... --img-size 640 --device cuda:0

# Đóng gói deploy + test ONNX
python build_deploy.py --weights runs/train_v15_vib_E/last.pt --name pfdet_E --img-size 512
python test_accuracy_onnx.py --onnx deploy/pfdet_E_512.onnx --img-size 512

# Demo webcam
python demo_cam.py --onnx deploy/pfdet_E_320.onnx --img-size 320 --cam 0 --conf 0.3

# Xem sơ đồ kiến trúc hệ thống (HTML tự chứa, không cần server, có sáng/tối + Export PNG/SVG)
xdg-open docs/architecture.html

# Sửa sơ đồ: sửa docs/architecture.json rồi render lại. --repo-root . là bắt buộc vì
# sơ đồ có ghim link mã nguồn theo commit trong meta.repository.revision.
ARCHIFY=~/.claude/skills/archify/bin/archify.mjs
node $ARCHIFY validate architecture docs/architecture.json --quality showcase --repo-root .
node $ARCHIFY deliver  architecture docs/architecture.json docs/architecture.html --quality showcase --repo-root .
```
> Sơ đồ vẽ bằng skill `archify` (cài ở `~/.claude/skills/`, KHÔNG nằm trong repo).
> Đổi commit thì nhớ cập nhật `meta.repository.revision` trong `docs/architecture.json`,
> nếu không `deliver` sẽ báo lỗi file không tồn tại ở revision đã ghim.

---

## 8. Bài học (đừng lặp lại sai lầm)
- **DGS (density head + crowd-loc α=1.0) → HẠI** (AP75 0.093 < baseline 0.131; box loss áp đảo phá định vị).
  → chuyển thành ablation phụ, KHÔNG dùng làm chính.
- **COCO: TÙY DOMAIN.** COCO làm GIẢM AP trên VisDrone-val (người tí xíu bay cao) → tưởng "hại". NHƯNG user
  bay **3-5m** (người to/gần) → COCO **CÓ ÍCH** cho deployment thật. → **BẬT COCO** ở v17 (enable epoch 100,
  ratio 0.25). Bài học: chọn data theo DEPLOYMENT, đừng theo val set không khớp domain.
- **INT8 PTQ phá nát accuracy** detector tiny (AP 0.18→0.009). → Jetson dùng **FP16** (đã verify = FP32); nếu thật cần INT8 phải **QAT**.
- **Jetson Nano B01 = FP16 TensorRT** (Maxwell không hỗ trợ INT8). RepVGG-dense SAI (to gấp 3 vô ích) → dùng depthwise-efficient + P2 mỏng.
- **FLOPs > params** mới là vấn đề: P2 ở độ phân giải cao ăn FLOPs khủng → làm P2 mỏng.
- **Resolution là vật lý:** người tí xíu @320 mất nửa AP (0.235→0.089). Deploy nên 512.
- **Loss (từ v12/v13):** focal gamma=2 → precision≈0; hard target=1 → FP nhiều; **cls bias init -5.5..-6.0**; obj_weight ≫ box_weight lúc đầu.
- **RT-DETR KHÔNG được để `optimizer=auto`.** Ultralytics auto chọn **MuSGD(lr=0.01)** (công thức CNN) → Transformer/DETR
  **sập ngay epoch 2-4** (mAP50 0.47→0.07), bò lại không kịp vượt epoch 1 → early-stop ở ep21 = **số RÁC**
  (RT-DETR-l 32M mà thua yolov8n 3M). Phải ép `optimizer="AdamW", lr0=1e-4`, và `patience` lớn (50) vì DETR hội tụ chậm.
  → Bài học chung: **early-stopping patience phải theo từng họ model**, không dùng chung 1 giá trị.
- **EVAL RUNG KHÔNG SEED = số RÁC.** `corrupt_batch()` random hướng blur + nhiễu + độ sáng, mà
  `eval_robustness.py` không seed → **cùng 1 checkpoint** ra retain 17.0 / 17.1 / 17.6% ở 3 lần chạy
  khác nhau. Số 1 lần đo là 1 lần rút thăm, không phải kết quả. → dùng `paper_eval_unified.py`
  (seed lại trước MỖI severity pass, 3 seed, báo mean ± std). Gain phải lớn hơn std mới là gain.
- **GIAO THỨC ĐO PHẢI ĐỒNG NHẤT, KHÔNG ĐƯỢC TRỘN HARNESS.** pfdet đo pycocotools vs YOLO đo
  `ultralytics.val()` lệch ~0.01 AP (yolov8n 0.2684 vs 0.2787) — cùng cỡ với khoảng cách GIỮA các
  baseline → trộn vào 1 bảng là báo cáo "khác biệt do code eval" thành "khác biệt giữa detector".
  Riêng **RT-DETR lệch tới 0.084** (0.2502 vs 0.3340) vì predictor gốc của nó dùng
  `LetterBox(scale_fill=True)` = **KÉO GIÃN** ảnh, còn loader repo **letterbox đệm xám** → nó nhận
  hình học chưa từng được calibrate. Retain% thì không bị (là tỉ số 2 pass cùng pipeline).
- **QUY TẮC CHỌN CHECKPOINT phải giống nhau cho model mình VÀ baseline.** v15_vib_E: `last.pt` cho
  retain 33.4%, `best.pt` cho 30.5%. Baseline dùng `best.pt` → paper phải dùng `best.pt`. Lấy `last`
  cho mình + `best` cho baseline = cherry-pick, reviewer bắt là mất bài.
- **`model.info(verbose=False)` trả `None`** ở ultralytics 8.4.x (hàm `model_info` return sớm khi không verbose)
  → code `info[1] if info else 0.0` nuốt lỗi, ghi **params=0.00M GFLOPs=0.0** cho mọi baseline. Tính trực tiếp bằng
  `sum(p.numel() for p in m.model.parameters())` + `get_flops(m.model, imgsz=640)`. Xem `summarize_baselines.py`.
- **Ultralytics `project` PHẢI là đường dẫn TUYỆT ĐỐI.** Đưa relative (`"runs/yolo"`) thì ultralytics ghép vào `runs_dir/task`
  → ghi thật vào `runs/detect/runs/yolo/...`, lệch chỗ script đi tìm `best.pt` → train xong mới crash ở bước eval, rồi
  hàng chờ train lại từ đầu (vòng lặp đốt GPU). Đã fix bằng `os.path.join(os.path.dirname(__file__), "runs", ...)`.
- **`pgrep -f` / `pkill -f` TỰ KHỚP command line của chính mình.** Không chỉ `pkill` — `pgrep -f`
  cũng vậy, và nó gây lỗi âm thầm hơn: nó báo "tiến trình đang chạy" trong khi thực ra chỉ khớp
  chính lệnh của bạn, làm bạn bỏ qua một bước cần làm. Bracket trick `[p]attern` KHÔNG cứu được
  nếu chuỗi đó xuất hiện **không có ngoặc** ở chỗ khác trong cùng command line (ví dụ dòng
  `nohup python train_x.py` ngay bên dưới). **Luật: lấy PID trước bằng một lệnh RIÊNG, rồi
  `kill <PID>`.** Đã dính 3 lần trong ngày 26/07.
- **Regex thay thế có `\1` phải dùng raw string ở MỌI phần của chuỗi.** `r'\1abc' '\n\1def'` —
  phần thứ hai không raw nên `\1` thành byte 0x01, ghi vào file YAML làm nó **hỏng không parse
  được**, và vòng lặp retry của hàng chờ quay 20s/lần cho tới khi phát hiện. An toàn hơn: dùng
  `lambda m: m.group(1) + '...'` thay vì chuỗi backreference.
- **`pkill -f <tên script>` tự giết chính shell đang chạy lệnh** (vì `-f` khớp cả command line của mình). Dùng `pkill -f "[t]ên"` hoặc kill theo PID.
- **`cv2.CAP_PROP_BUFFERSIZE=1` RỚT ĐÚNG MỘT NỬA KHUNG.** Đặt 1 buffer để "tránh khung cũ" thì
  driver không có chỗ ghi khung kế trong lúc app đang giữ buffer → 30 fps thành đúng 15.0. Dùng
  **2** (độ cũ của khung vẫn chỉ p50 15.7 ms nếu vòng đọc luôn lấy khung mới nhất). Con số "đúng
  một nửa" là dấu hiệu nhận biết: fps tròn trịa bằng 1/2 giá trị danh nghĩa thì nghi buffer, đừng
  nghi model.
- **Auto-exposure của webcam TỰ HẠ FPS khi thiếu sáng** (C270: 30 → 19.7). Cờ UVC
  `exposure_dynamic_framerate=0` sinh ra đúng để cấm việc này nhưng **firmware Logitech phớt lờ**.
  Cách duy nhất ăn: khoá `auto_exposure=1` (Manual) rồi tự chỉnh sáng bằng phần mềm với **trần
  phơi sáng** (`cam_ctl.SoftAE`). Bài học chung: **đo fps THẬT của camera** (đếm khung mới / giây),
  đừng tin `CAP_PROP_FPS` của driver — nó báo 30 trong khi thực tế 15.
- **Đừng đổ tại model khi e2e chậm.** Kẹt 15 FPS mà `inference` chỉ 17.8 ms trên ngân sách 33.3 ms
  → nút cổ chai chắc chắn ở chỗ khác. Đo TỪNG CHẶNG trước khi tối ưu model.
- **Hạ tầng train (laptop 8GB GPU / 14GB RAM):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` + tắt `cudnn.benchmark` khi multiscale (đã baked vào train_v3) để khỏi OOM tích lũy; `num_workers ≤ 6` để khỏi cạn RAM giết VS Code; batch nhỏ (v15 batch 6-8 @640, v17 nhẹ hơn batch 12 OK).

---

## 9. Roadmap

### A. DEPLOY (Hailo Pi5 + ArduPilot) — CƠ BẢN XONG
1. ✅ **v17 deploy** `runs/train_v17_deploy/best.pt` (AP@0.5 0.5136, COCO 3-10m). ONNX @320/512/640 (== PyTorch).
2. ✅ **Model Hailo** `train_config_v17_hailo.yaml` (bỏ AreaAttention P5 = op Hailo chê → thuần conv). `runs/train_v17_hailo/best.pt` (0.875M).
3. ✅ **HEF chạy trên Pi5+Hailo-8L**: `pfdet_v17_hailo8l.hef`. Convert bằng Hailo DFC 3.34 trên
   chính laptop (`compile_hef.py`). Xem `README_HAILO.md`.

   ⚠️ **Số FPS đã đo lại 2026-08-24 trên đúng con Pi deploy** (`raspi@192.168.196.114`, HailoRT 4.20.0,
   HAILO8L, PCIe Gen2 x1):

   | model | `hailortcli benchmark` hw_only | gọi đồng bộ p50 | e2e (capture→decode) |
   |---|---|---|---|
   | 320 | **112.55 FPS** (8.40 ms) | 8.88 ms | — |
   | 640 | **55.78 FPS** (18.0 ms) | 18.00 ms | **19.96 ms → 50 FPS** |

   **Chỗ lệch với 186 FPS đã giải xong**: 186 là **throughput theo lô**, không phải tốc độ
   từng khung. `--batch-size 4` cho **188.78 FPS** nhưng latency tăng 8.40 → 17.05 ms; batch 16
   cho 231.8 FPS / 50.5 ms. Deploy thật dùng **batch=1** vì gom đủ lô ở camera 30fps tốn thêm
   133 ms — vòng điều khiển bay không chịu được.

   ⚠️ `paper/sections/13_exp_deploy.tex:9` đang ghi trần "sustains 186 FPS" **không nói batch**
   → reviewer chạy `hailortcli benchmark` mặc định (batch=1) ra 112.55 rồi nghi cả mục.
   Cách phát biểu đề xuất: xem `runs/pi_hailo_measured_2026-08-24.md`.

   Bối cảnh: camera 30 fps → 112 FPS là **dư 3.7×**, model không phải nút cổ chai.
   Thứ quyết định bay bám là latency/p99, và p99 = 17.85 ms, không throttle.

5. ✅ **Head-to-head với YOLO26n trên cùng Hailo-8L** (2026-08-24), cả hai compile bằng
   đúng một lệnh, cắt end-node ở conv head, decode+NMS đều trên CPU:

   | model | M | GFLOPs | contexts | FPS @640 | latency |
   |---|---|---|---|---|---|
   | **SteadyDet-N** | **1.17** | **4.6** | **4** | **63.66** | **15.26 ms** |
   | YOLO26n | 2.38 | 5.2 | 6 | 37.46 | 24.79 ms |

   **SteadyDet nhanh hơn 1.70×.** Cơ chế là **số context** (model không vừa bộ nhớ trên chip
   thì phải nạp lại trọng số mỗi context), không phải GFLOPs. Đây là luận điểm deployment
   mạnh, nên đưa vào paper.

   ⚠️ Hai HEF này calibrate bằng **val images** → chỉ dùng so TỐC ĐỘ. Lấy AP từ chúng là
   rò rỉ dữ liệu; muốn AP phải compile lại với `--calib-dir` trỏ tập train.
   ⚠️ Con số "YOLO26n 80 FPS" là đo trên **Pi khác** (nhiều khả năng Hailo-8 26 TOPS).

   **Độ chính xác HEF 640 (INT8) đã đo**, checkpoint gốc `runs/train_v17_hailo/best.pt`,
   548 ảnh VisDrone val @640, cùng decode/NMS với đường eval chuẩn:

   | | FP32 (GPU) | INT8 (Hailo-8L) | rớt |
   |---|---|---|---|
   | AP@[.50:.95] | 0.1734 | **0.1516** | −12.6% |
   | AP@0.50 | 0.4812 | **0.4585** | −4.7% |
   | AP@0.75 | 0.0703 | 0.0493 | −29.9% |

   → INT8 làm **box kém khít** (AP75 rớt mạnh) chứ không làm mất object (AP50 gần như giữ).

   ⚠️ Đã sửa **3 lỗi trong đường đo HEF** (`hailo_eval_val.py`): decode sai encoding
   (`tanh` thay vì `sigmoid−0.5`), NMS khác đường chuẩn (thiếu `pre_nms_topk`, dùng IoU
   thay vì DIoU), và `score()` không đưa GT qua letterbox. Xem file measured để biết chi tiết.

4. ✅ **Bộ test + thu data trên Pi**: `raspi@192.168.196.114:~/NhanHuynh/` (đẩy từ scratchpad, 2026-08-24).
   - `./run.sh check` — checklist trước khi ra bãi (Hailo / lib / HEF / camera / nhiệt / đĩa / NTP)
   - `./run.sh record --alt 5 --scene grass ...` — thu video + ảnh + `detections.jsonl` + `health.csv`
     + `markers.csv` (mốc đồng bộ với log `.bin`) + `manifest.json` (metadata lượt bay)
   - `./run.sh bench` — latency p50/p95/p99 cho mục Edge Deployment
   - `pull_flight_data.sh` (ở root repo này) — kéo data về `data/flight_raw/`
   - Python dùng `venv/` tạo bằng `--system-site-packages`: xài chung `hailo_platform`/`cv2`/`numpy`/
     `picamera2` của hệ thống, chỉ cài thêm vào venv qua `requirements.txt`.
   - ✅ **ĐÃ CẮM CAMERA (2026-08-27)**: Logitech C270 HD, USB `/dev/video0`, MJPG tối đa 1280×720@30.
     `./run.sh check` = **10 OK / 0 FAIL**.
   - 🔥 **Sửa 2 lỗi camera làm cả pipeline kẹt 15 FPS → 30.0 FPS (×1.95), không đụng model.**
     (1) `cv2.CAP_PROP_BUFFERSIZE=1` **rớt đúng một nửa khung** (1 buffer thì driver không có
     chỗ ghi khung kế) — có trong `flight_record.py`, `hailo_bench_latency.py`, `hailo_demo_cam.py`.
     (2) auto-exposure của firmware **tự hạ fps** khi thiếu sáng (30→19.7); cờ UVC
     `exposure_dynamic_framerate=0` **C270 phớt lờ**, vô dụng. → `cam_ctl.py` khoá manual rồi
     làm **auto-exposure bằng phần mềm** (`SoftAE`) có **trần phơi sáng** nên fps không bao giờ tụt.
     Số đo + cách tách bạch 2 lỗi: **`runs/pi_camera_measured_2026-08-27.md`**.
   - **e2e @640 sau khi sửa** (bench 2 phút, camera thật): capture 12.47 / preprocess 0.99 /
     inference 17.72 / decode+NMS 0.84 → **p50 32.04 ms = 31 FPS**, p99 28 FPS, 48.5 °C,
     throttled 0x0, giữ đều 2 phút. **Nút cổ chai giờ là CAMERA** (tính toán chỉ ăn 19.5 ms
     trong ngân sách 33.3 ms; C270 hết cỡ 30 fps).
   - `./run.sh stream` (**mới**, `hailo_stream_web.py`) — phát camera + box lên **trình duyệt**
     qua MJPEG, `http://<ip-pi>:8080`. Không cần màn hình/X11. Có thanh trượt conf, hiện fps
     thật của camera + nhiệt + exposure/gain + latency.
   - ⚠️ **Lúc BAY** dùng `--max-exposure 20 --prefer-gain` (1/500 s): shutter dài là nhoè vì
     rung, và thời gian phơi sáng nằm THẲNG trong độ trễ điều khiển (~25 ms).
4. ✅ **Điều khiển bay ArduPilot GUIDED** — `follow_drone/` (pymavlink, OC-SORT, 3 vòng PID, safety ladder;
   chạy CPU Pi5). Param chỉnh bằng **Mission Planner**, bảng tham số ArduPilot ở `baocao/phuluc/pl_a_cau_hinh.tex`.
   *(Nhánh cũ PX4 `jetson/control/follow_px4.cpp` + `jetson/DEPLOY_GUIDE.md` đã dán banner, không dùng.)*
5. ⏳ **Còn lại (khi bay):** cài MAVSDK Pi, test SITL, calibrate fx/fy, bay. Cân nhắc HEF @640 cho accuracy (Hailo-8L thừa TOPS). Quay footage 3-10m thật (`jetson/record_flight.py`) làm val + finetune.

### B. NHÁNH CHUẨN — hướng CHỐT: efficiency + robustness. **HÀNG CHỜ TỰ ĐỘNG ĐANG CHẠY**
- `paper_queue.sh` (nohup + cron@reboot + auto-resume crash): v17 baseline → +vibration → nms_free → eval (COCO + robustness).
- `yolo_queue.sh` (chờ v17 xong): train **5 baseline đa dạng** cùng split VisDrone-person @640 → `train_yolo_baselines.py`:
  YOLOv8n/11n/26n (CNN) + **YOLOv10n** (NMS-free) + **RT-DETR** (Transformer) = 3 paradigm.
- Kết quả gom ở **`runs/paper_queue/SUMMARY.txt`** (params/FLOPs/mAP + robustness). Marker: `1_light_done...FINAL_ALL_DONE`.
- **PHASE 2 (nhánh DEPLOY, tự động sau no-COCO):** `deploy_queue.sh` (nohup + cron@reboot, flock) CHỜ `FINAL_ALL_DONE` rồi:
  (1) vib finetune từ deploy `train_config_v17_vib_deploy.yaml` → `runs/train_v17_vib_deploy`;
  (2) 6 YOLO baseline trên **COCO-mix** `train_yolo_baselines_deploy.py` (`configs/visdrone_coco_ultralytics.yaml`, val=**VisDrone-val 548 ĐỒNG NHẤT deploy**);
  (3) eval các model pfdet trên **`data/flight_val`** (held-out THẬT 3-6m, 92 frame — chỉ để tham chiếu domain, KHÔNG dùng làm val lúc train vì quá ít). Kết quả → `runs/paper_queue/SUMMARY_DEPLOY.txt`, marker `DEPLOY_ALL_DONE`.
  Bằng chứng mở khoá: deploy(+COCO) AP@0.5 **0.762** vs light(no-COCO) **0.143** trên flight_val (xem `RESULTS_FLIGHT_VAL.md`).
- **TODO sau (cần debug cùng user):** thêm classic non-YOLO qua **mmdetection** (RetinaNet/FCOS/EfficientDet) + NanoDet — mmcv hay vỡ nên làm supervised. Còn RemDet/PicoDet/MobileNet-SSD → **CITE số công bố**.
- Venue nhắm: **Drones (MDPI, IF 4.8)** / IEEE Access / J. Real-Time Image Processing. TRÁNH predatory. **Rủ thầy đồng tác giả** (kinh phí APC + review).

---

## 10. Hạ tầng
- **Dev:** laptop RTX 4060 Laptop (8GB, ~7.3GB dùng được) + 14GB RAM, CUDA.
- **Deploy:** Jetson Nano B01 (GPU Maxwell 128-core, ARM A57, TensorRT FP16).
- **Data:** `data/visdrone/{train,val}` (6471 / 548 ảnh, person-only) + `data/coco_person` (2654 ảnh, trộn cho tầm thấp 3-5m).
- **Memory dự án:** `~/.claude/projects/.../memory/` — fact bền vững giữa các phiên Claude.
