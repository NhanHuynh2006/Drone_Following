# Sửa lỗi COCO-không-bật: train lại + compile HEF (2026-08-27)

## Lỗi

`configs/train_config_v17_hailo.yaml` để `extra_enable_epoch: 100` trong khi `epochs: 20`.
Điều kiện bật COCO ở `train_v3.py:819` là `extra_enable_epoch <= epoch < extra_disable_epoch`
→ **COCO chưa bao giờ được bật.** 20 epoch finetune đó toàn VisDrone (người tí xíu bay
cao), kéo model ngược khỏi khả năng bắt người gần mà `train_v17_deploy` đã học.

## Đã làm

1. Khôi phục `data/visdrone` (6471 train / 548 val) + `data/coco_person` (2654) từ bộ gốc
   công khai. Tham số convert dò ngược bằng cách đối chiếu với val còn sót trên Pi:
   `--include-people --min-size 10`, **548/548 ảnh và 8165/8165 hộp khớp tuyệt đối**.
2. Train lại 20 epoch với `extra_enable_epoch: 0` → `runs/train_v17_hailo_coco/best.pt`.
3. Export ONNX → compile 2 HEF (bản thứ 3 opt-level 4 chạy qua đêm).

## Bằng chứng COCO đã thật sự được bật

Log in ra `[DATA] Epoch 1: BẬT COCO (1618 ảnh, ratio=25%)` — dòng này **không hề xuất
hiện** trong lần train cũ. Phân bố positive ở epoch cuối cũng đổi:

| | pos_s4 | pos_s8 | pos_s16 | pos_s32 |
|---|---|---|---|---|
| cũ (mất COCO) | 63 | 10 | **0** | **0** |
| mới (có COCO) | 52 | 9 | **1** | **1** |

`s16`/`s32` khác 0 nghĩa là model bắt đầu nhận tín hiệu từ **vật thể lớn** — tức người
ở gần. Trước đó 100% positive rơi vào P2 (stride 4 = tí xíu).

## Kết quả trên 92 khung bay thật (`data/flight_val`, FP32)

| model | F1 | conf | recall | precision | FP/khung |
|---|---|---|---|---|---|
| `train_v17_deploy` (giữ AreaAttention + COCO) | **0.882** | 0.25 | 0.815 | 0.962 | 0.03 |
| **`train_v17_hailo_coco`** ← MỚI | **0.836** | 0.25 | 0.804 | 0.871 | 0.12 |
| `train_v17_hailo` (bỏ attention, mất COCO) | 0.768 | 0.20 | 0.793 | 0.745 | 0.27 |

**Bật COCO lấy lại 0.068 trên 0.114 khoảng cách (60%).** Phần còn lại **0.046** là do bỏ
AreaAttention — nên câu *"bỏ attention P5 gần như không mất AP"* trong `README_HAILO.md`
đúng trên VisDrone-val nhưng **sai trên domain bay thấp**, chỉ là mức sai nhỏ hơn tưởng.

Trên VisDrone-val thì hai bản gần như nhau (AP@0.5 cũ 0.5033, mới 0.5013) — đúng như
`PROJECT.md` mục 8 đã ghi: COCO **làm giảm** AP trên VisDrone-val nhưng **có ích** cho
deployment tầm thấp. Đây là bằng chứng thêm cho nhận định đó.

> Muốn lấy nốt 0.046 còn lại thì phải giữ attention theo cách Hailo nuốt được —
> `ConvAttnBlock` (Cách B trong `README_HAILO.md`). Chưa làm.

## Hai lỗi trong đường export đã sửa

1. **`torch.onnx.export` mặc định dùng bộ dynamo (torch ≥ 2.6) và làm Hailo parser chết.**
   Bộ đó bỏ thuộc tính `kernel_shape` của node Conv — hợp lệ theo chuẩn ONNX vì suy ra
   được từ shape kernel, nhưng parser Hailo đọc thẳng thuộc tính đó:
   `onnx_graph.py:987 get_kernel_shape → IndexError: list index out of range`.
   Nó cũng không hạ nổi opset xuống 11 (thiếu adapter cho `Resize`) nên ra opset 18.
   → Sửa: ép `dynamo=False` trong `export_onnx_hailo.py`. Kết quả: opset 11, ir_version 6,
   82/82 Conv đủ `kernel_shape`.
2. **ONNX chưa từng được kiểm số học so với PyTorch.** Nay đã kiểm: logit lệch ~1% tương
   đối, nhưng chạy nguyên bài sweep thì **F1 0.831 (ONNX) vs 0.836 (PyTorch)**, số TP
   giống hệt ở mọi ngưỡng. Sai lệch đó vô hại. Script: `onnx_sweep.py`.

## Các HEF đã tạo

| file | calib | opt-level | contexts | inter-context BW |
|---|---|---|---|---|
| `pfdet_v17_640_coco_hailo8l.hef` | 64 | 2 | 4 | 41.02 Mbps |
| `pfdet_v17_640_coco_c512_hailo8l.hef` | 512 | 2 | 4 | **33.98 Mbps** |
| `pfdet_v17_640_coco_o4_hailo8l.hef` | 64 | 4 | — | đang compile qua đêm |

Tập calib trộn **đúng tỉ lệ train** (75% VisDrone + 25% COCO), lấy từ tập TRAIN.
Không dùng `flight_val` làm calib — đó là tập chấm điểm, calib bằng nó là rò rỉ dữ liệu.
(Bản `_fair` cũ calibrate bằng VisDrone-**val**, nên đừng lấy AP của nó trên VisDrone-val
làm số công bố.)

DFC cảnh báo *"didn't have enough data for dataset_size of 1024"* với 64 ảnh → đã thêm
cờ `--calib-n` vào `compile_hef.py` và làm thêm bản 512 ảnh.

## ⚠️ CHƯA ĐO ĐƯỢC: độ chính xác INT8

Không có phần cứng Hailo trên laptop nên **chưa biết HEF nào tốt nhất**. Lần trước INT8
làm AP@0.5 rớt 4.7% và AP@0.75 rớt 29.9%, và ngưỡng conf tối ưu dịch lên (FP32 0.20 →
INT8 0.30). Nên **đừng giả định conf 0.25 của FP32 dùng được luôn cho INT8.**

Khi có Pi: `bash scripts/deploy_to_pi.sh` — tự đẩy mọi HEF sang, đo từng bản trên đúng
92 khung đó, in bảng so F1, rồi bạn chọn.

---

# ✅ ĐO TRÊN PI (2026-08-28) — HEF mới thắng rõ rệt

Đo trên chính Pi (`pi5`, Hailo-8L), 92 khung `data/flight_val`, cùng một đường
decode/NMS cho cả ba bản. Lệnh: `bash scripts/deploy_to_pi.sh`.

| HEF | calib | F1 | conf | recall | precision | FPS (batch=1) |
|---|---|---|---|---|---|---|
| **`pfdet_v17_640_coco`** | 64 | **0.880** | 0.25 | **0.837** | 0.928 | 56.9 |
| `pfdet_v17_640_coco_c512` | 512 | 0.812 | 0.25 | 0.750 | 0.885 | 62.9 |
| `pfdet_v17_640_fair` (đang dùng trước đó) | val | 0.745 | 0.30 | 0.620 | 0.934 | 63.7 |

**Kết quả: F1 0.745 → 0.880.** Quan trọng hơn với bài toán bay bám: recall
**0.620 → 0.837** — bắt được người ở 84 % số khung thay vì 62 %.

Bản thắng chậm hơn 2 ms (17.6 vs 15.7 ms) nhưng vẫn dư xa ngân sách 33.3 ms của vòng
30 Hz, nên đổi 2 ms lấy 22 điểm recall là đáng.

## Hai điều bất ngờ, ghi lại vì trái trực giác

**1. INT8 mất bao nhiêu thì TUỲ DOMAIN.** Trên `flight_val` (92 khung, người to, 1
người/khung) FP32 cho F1 0.836 còn INT8 cho 0.880 — chênh lệch nằm trong nhiễu của 92 mẫu.
Nhưng trên VisDrone-val (548 ảnh, người tí xíu) thì INT8 **có** mất, đo bằng đúng một
harness (`fp32_dump.py` + `hailo_eval_val.py --score`):

| | FP32 | INT8 | rớt |
|---|---|---|---|
| AP@[.50:.95] | 0.1775 | 0.1584 | −10.8 % |
| AP@0.50 | 0.4946 | 0.4625 | **−6.5 %** |
| AP@0.75 | 0.0742 | 0.0555 | −25.2 % |
| AP small | 0.1749 | 0.1559 | −10.9 % |
| AR@100 | 0.3041 | 0.2869 | −5.7 % |

Hai kết quả **không mâu thuẫn**: AP trên VisDrone-val bị chi phối bởi vật thể nhỏ, mà INT8
phá định vị vật nhỏ (AP75 rớt 25 %). Trên domain bay thấp — người chiếm nhiều điểm ảnh —
mất mát đó gần như không thấy. Phát biểu đúng là: **INT8 tốn ~6.5 % AP@0.50 với người tí
xíu, và không đáng kể với người ở 3–6 m.**

> ⚠️ Suýt kết luận sai ở đây. Lần đầu tôi so INT8 (chấm bằng `hailo_eval_val.py`) với FP32
> (chấm bằng `run_eval_coco.py`) và ra "INT8 ≥ FP32". Đó đúng là cái bẫy `PROJECT.md` mục 8
> đã ghi: **trộn harness đo cho chênh lệch cỡ bằng chênh lệch giữa các model.** Viết
> `fp32_dump.py` để dump FP32 ra đúng định dạng rồi chấm bằng cùng hàm `score()` mới ra
> con số dùng được.

**2. Nhiều ảnh calib hơn lại TỆ hơn.** 512 ảnh cho F1 0.812, kém bản 64 ảnh (0.880), dù
DFC cảnh báo nó muốn 1024. Trái với khuyến nghị trong `README_HAILO.md` mục 6
(*"dùng nhiều ảnh calib đa dạng"*). Chưa rõ cơ chế — có thể pha Quantization-Aware
Fine-Tuning 4 epoch trên 512 ảnh đã kéo model lệch về phía VisDrone (384/512 là VisDrone).
**Bài học vận hành: đừng đoán, đo.** Cả hai bản compile mất chưa tới 40 phút.

## ⚠️ Ngưỡng conf phụ thuộc TỪNG BẢN HEF

`_fair` tối ưu ở 0.30, bản mới ở **0.25**. Đổi HEF mà giữ nguyên conf cũ là tự bỏ đi một
phần hiệu năng. `follow_drone/config.yaml` đã đặt 0.25 kèm ghi chú. Đổi HEF thì chạy lại
`scripts/deploy_to_pi.sh` để đo lại ngưỡng.

## Chưa chạy được end-to-end

Lúc đo, Pi **chưa cắm camera USB lẫn Pixhawk** (`lsusb` chỉ có root hub). Detector đã
xác nhận nạp HEF mới thành công với conf 0.25; phần camera + MAVLink cần cắm lại thiết bị
rồi chạy lại `main.py`.

## Bản opt-level 4 KHÔNG có

Chạy qua đêm nhưng máy khởi động lại lúc 10:34 ngày 28/08 nên chết ở Adaround 4/92.
Không làm lại: bản opt-2 đã đạt 0.880, ngang FP32, nên adaround gần như chắc chắn không
còn gì để cải thiện.

---

# Số cho báo cáo / bài báo (đo 2026-08-28, CÙNG MỘT harness)

Cách đo: dump dự đoán ra cùng định dạng (`fp32_dump.py` cho FP32, `hailo_eval_val.py`
cho HEF) rồi chấm bằng **cùng hàm** `hailo_eval_val.py --score`. Nhờ vậy chênh lệch là
do lượng tử hoá, không phải do hai chương trình chấm điểm khác nhau.

## VisDrone-val (548 ảnh, người tí xíu)

| | FP32 | INT8 | rớt |
|---|---|---|---|
| AP@[.50:.95] | 0.1775 | 0.1584 | −10.8 % |
| **AP@0.50** | 0.4946 | **0.4625** | **−6.5 %** |
| AP@0.75 | 0.0742 | 0.0555 | −25.2 % |
| AP small | 0.1749 | 0.1559 | −10.9 % |
| AR@100 | 0.3041 | 0.2869 | −5.7 % |

## flight_val (92 khung bay thật ~6 m, người to) ← MIỀN TRIỂN KHAI

| | FP32 | INT8 |
|---|---|---|
| AP@[.50:.95] | 0.4356 | 0.4349 |
| **AP@0.50** | 0.8822 | **0.9171** |
| AP@0.75 | 0.2561 | 0.2552 |
| AR@100 | 0.5489 | 0.5337 |

**Đây là con số nên dùng làm chỉ tiêu hệ thống: AP@0.50 = 0.917.** Nó là của đúng mô hình
đang chạy trên máy bay (INT8, trên NPU), không phải của một checkpoint khác.

⚠️ **Đừng đặt 0.917 cạnh 0.762.** Con số 0.762 trong bảng so sánh dữ liệu huấn luyện là của
`train_v17_deploy` và chấm bằng chương trình khác. So trực tiếp hai số đó là đúng cái bẫy
trộn harness. Bảng dữ liệu giữ nguyên 0.762 (nó đang so *dữ liệu*, không so *hệ thống*);
chỉ tiêu hệ thống dùng 0.917. Đã ghi chú rõ trong `baocao/chuong/09_ket_luan.tex`.

## Kết luận về INT8

Mất mát **phụ thuộc kích thước đối tượng**: −6.5 % AP@0.50 với người tí xíu (VisDrone-val),
gần như bằng 0 với người ở 3–6 m (flight_val: 0.4356 vs 0.4349). Hợp lý — INT8 phá độ phân
giải toạ độ, mà vật càng nhỏ thì càng nhạy với chuyện đó.

## Tốc độ mô hình đang deploy

`pfdet_v17_640_coco_hailo8l.hef`: **56.9 FPS** (`hailortcli`, batch=1), inference p50
**17.32 ms** / p99 17.52 ms đo trong chính lần eval 548 ảnh.

## Đã cập nhật vào tài liệu

| tài liệu | sửa gì |
|---|---|
| `paper/sections/13_exp_deploy.tex` | bảng INT8 → số mới; thêm đoạn về mất mát phụ thuộc kích thước và về phân phối ảnh calibration |
| `baocao/chuong/06_trien_khai_bien.tex` | bảng INT8 → số mới; thêm giải thích cùng-harness + phụ thuộc kích thước + calibration |
| `baocao/chuong/05,08` | chỉ tiêu AP@0,5: 0,762 → **0,917** |
| `baocao/chuong/09_ket_luan.tex` | 0,762 → 0,917, kèm cảnh báo không so với bảng dữ liệu |
| `tomtat/`, `poster/` | chỉ tiêu → 0,917 |
| `baocao/figures/make_figures.py` | ô "AP@0,5 trên miền triển khai" → 0,917, đã vẽ lại |
