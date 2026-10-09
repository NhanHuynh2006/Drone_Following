# HEF trên tập bay THẬT — vì sao chĩa vào người mà không detect (2026-08-27)

Triệu chứng người dùng gặp: **camera chĩa vào người, conf 0.4 không ra gì; hạ xuống 0.2
thì ra nhưng nhiễu xung quanh.**

## Giao thức đo

`data/flight_val` — **92 frame bay thật ~6 m**, mỗi frame 1 người, nhãn gán tay.
Cùng một `decode_scale` + `nms` mà `flight_record.py` / dashboard dùng, chạy 1 lần
inference ở conf sàn 0.02 rồi quét ngưỡng offline. Khớp GT bằng IoU ≥ 0.5.
Script: `flight_sweep.py` (trên Pi), `fp32_sweep.py` (laptop, PyTorch FP32).

## Kết quả

| model | F1 tốt nhất | conf | recall | precision | FP/khung |
|---|---|---|---|---|---|
| `runs/train_v17_deploy/best.pt` FP32 — **CÓ AreaAttention, CÓ COCO** | **0.882** | 0.25 | 0.815 | 0.962 | 0.03 |
| `runs/train_v17_hailo/best.pt` FP32 — bỏ AreaAttention, finetune 20ep | 0.768 | 0.20 | 0.793 | 0.745 | 0.27 |
| `pfdet_v17_640_fair_hailo8l.hef` INT8 | 0.745 | 0.30 | 0.620 | 0.934 | 0.04 |
| `pfdet_v17_640_hailo8l.hef` INT8 ← **đang chạy trên Pi** | 0.697 | 0.30 | 0.576 | 0.883 | 0.08 |

Quét ngưỡng của HEF đang deploy, khớp đúng những gì người dùng thấy:

| conf | recall | precision | FP/khung |
|---|---|---|---|
| 0.20 | 0.750 | 0.365 | **1.30** ← "ra nhưng nhiễu" |
| 0.30 | 0.576 | 0.883 | 0.08 ← **điểm F1 tốt nhất** |
| 0.40 | 0.196 | 0.947 | 0.01 ← "không ra gì" |
| 0.50 | 0.065 | 1.000 | 0.00 |

→ **Ngưỡng deploy nên là 0.30**, không phải 0.4 như mặc định cũ. Đã đổi mặc định
trong `hailo_gcs.py`.

## Nguyên nhân gốc: một lỗi config, KHÔNG phải lỗi INT8

Mất mát chia làm hai bước, và bước lớn hơn **không** nằm ở lượng tử hoá:

```
0.882  train_v17_deploy (FP32)
  │  −0.114   ← finetune Hailo         ★ lớn nhất
0.768  train_v17_hailo  (FP32)
  │  −0.071   ← lượng tử hoá INT8
0.697  HEF đang chạy
```

`configs/train_config_v17_hailo.yaml` để **`extra_enable_epoch: 100`** trong khi
**`epochs: 20`**. Điều kiện bật COCO ở `train_v3.py:819` là

```python
want_coco = extra_enable_epoch <= epoch < extra_disable_epoch   # 100 <= epoch < 360
```

Training chạy epoch 1–20 → **COCO person CHƯA BAO GIỜ được bật.** 20 epoch đó finetune
thuần VisDrone (người tí xíu bay cao), kéo model ngược khỏi domain người-gần mà
`train_v17_deploy` đã học được nhờ COCO.

Log xác nhận: `runs/train_v17_hailo/training_log.csv` cho
`pos_s4=63-65, pos_s8=9, pos_s16=0, pos_s32=0` — **toàn bộ** positive rơi vào P2
(stride 4 = vật thể tí xíu). Không có một mẫu người-gần nào trong suốt quá trình.

Đây cũng là lý do `README_HAILO.md` từng ghi "bỏ attention P5 gần như không mất AP" —
câu đó đúng trên VisDrone-val nhưng **sai trên domain bay thấp**, và thật ra thứ gây mất
không phải attention mà là COCO bị tắt. Đã sửa chú thích trong `README_HAILO.md`.

## Cách sửa (CHƯA CHẠY ĐƯỢC — thiếu data)

1. `configs/train_config_v17_hailo_coco.yaml` — **đã tạo**: `extra_enable_epoch: 0`,
   `extra_disable_epoch: 9999`, `save_dir: runs/train_v17_hailo_coco`.
2. `python train_v3.py --config configs/train_config_v17_hailo_coco.yaml` (20 epoch)
3. `python export_onnx_hailo.py --weights runs/train_v17_hailo_coco/best.pt --img-size 640 ...`
4. `python compile_hef.py --calib-dir <ảnh 3-6m THẬT>` — **không** dùng
   `data/visdrone/train/images`: calib sai domain là một phần lý do INT8 sinh nhiễu.

⛔ **Chặn:** `data/visdrone` và `data/coco_person` **không còn trên đĩa** (chỉ còn
`data/flight_val`). Đã thử chạy, dataset trả 0 ảnh. Phải khôi phục data trước.

## Ghi chú cho việc bay

Với HEF hiện tại ở conf 0.30: bắt được người ở **57.6%** số khung, precision 88.3%.
Ở 30 FPS thì vẫn là ~17 lần phát hiện mỗi giây — đủ để bộ lọc Kalman trong
`follow_drone/` nội suy giữa các lần trượt. Bay thử được, nhưng **nên retrain trước khi
tin vào nó**, vì `train_v17_deploy` cho thấy con số đúng phải là 81.5% recall.
