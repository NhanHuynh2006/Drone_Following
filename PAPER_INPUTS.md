# PAPER_INPUTS.md — Mọi thứ cần để viết báo (index cho phiên Claude viết báo)

> File này gom **kết quả + biểu đồ + thiết kế** để viết bài. Đọc kèm `PROJECT.md` (bản đồ dự án) và
> `V17_DESIGN.md` (thiết kế + SOTA). Ngày chốt snapshot: 2026-07-24.
> ⚠️ **CHƯA HOÀN TẤT:** RT-DETR (cả 2 nhánh) đang/đợi train lại; `BASELINES_TABLE.md` (params/FLOPs/mAP
> đồng nhất) sẽ sinh khi `fix_queue` xong. Xem mục "ĐANG CHỜ" cuối file.

## 1. Đóng góp & câu chuyện
- **Model:** PFDet-Nano **v17** — backbone có **P2 (stride 4)** cho người tí xíu, nhưng **P2 mỏng + LiteHead
  depthwise-sep** → **1.17M params, 4.6 GFLOPs@640** (cắt FLOPs 2.8× so v15 12.9G mà giữ P2). Loss DFL-free
  (STAL+ProgLoss+NWD+CIoU), optimizer MuSGD.
- **Đóng góp chính = Vibration-Consistency:** train-time forward ảnh sạch + ảnh rung, ép nhất quán →
  **0 chi phí inference**. Tăng robustness khi drone rung.
- **NMS-free (Plan B, thành công):** thêm head o2o → bỏ NMS, AP còn cao hơn baseline.
- **Deployment (nhánh riêng, +COCO):** COCO person giúp tầm thấp 3-6m; chạy **Hailo-8L 186 FPS trên Pi5** +
  điều khiển bay **PX4/MAVSDK**. Xem `README_HAILO.md`, `jetson/DEPLOY_GUIDE.md`.

## 2. Kết quả CHÍNH (đọc từ file, đừng chép tay)
### 2a. pfdet v17 — VisDrone-val @640, pycocotools (`runs/paper_queue/SUMMARY.txt`)
| model | AP@.5:.95 | AP50 | AP75 | robust giữ % |
|---|---|---|---|---|
| v17 baseline (light) | 0.1892 | 0.5127 | 0.0811 | 17.1% |
| **v17 +vibration** | 0.1746 | 0.4837 | 0.0728 | **30.1%** |
| **v17 NMS-free** | **0.1954** | **0.5213** | **0.0878** | 24.2% |
| v15 musgd_v3 (mốc so) | 0.2349 | 0.5800 | 0.1305 | 21.0% |

Robustness đường cong (clean→sev1→sev2→sev3): trong SUMMARY.txt, dòng `[robust]`. Vibration ~**1.8× retain**.

### 2b. YOLO baselines — params/GFLOPs/mAP (`runs/paper_queue/BASELINES_TABLE.md`, ĐÃ XONG)
VisDrone-val @640, ultralytics.val. So với **PFDet v17 = 1.17M / 4.6 GFLOPs** (nhỏ nhất, ít FLOPs nhất).

| model | params | GFLOPs | mAP50 P1(no-COCO) | mAP50 P2(+COCO) |
|---|---|---|---|---|
| **PFDet v17** | **1.17M** | **4.6** | 0.513* | 0.502* (deploy) |
| yolo26n | 2.50M | 5.8 | 0.6087 | 0.5979 |
| yolo11n | 2.59M | 6.4 | 0.6212 | 0.6098 |
| yolov10n | 2.71M | 8.4 | 0.6228 | 0.6002 |
| yolov8n | 3.01M | 8.2 | 0.6158 | 0.6114 |
| rtdetr-l | 32.81M | 108.0 | 0.6856 | 0.6930 |

*(P2 +COCO nano thấp hơn P1 — khớp bài học "COCO hại VisDrone-val người tí xíu". RT-DETR thì +COCO nhỉnh hơn.)*

> ⚠️ **CẢNH BÁO ĐO LƯỜNG (CHƯA giải quyết):** `*` = số v17 (0.513/0.502) đo bằng **pycocotools**, còn số YOLO đo
> bằng **ultralytics.val()** — **giao thức KHÁC nhau (conf/max_det/area), KHÔNG so mAP trực tiếp được.**
> So **params/GFLOPs thì OK** (đo cùng cách). Trước khi lên bảng cuối paper: phải eval v17 bằng ultralytics HOẶC
> eval YOLO bằng pycocotools để **1 thước đo duy nhất**. (Efficiency v17 thắng rõ; AP cần đồng nhất mới kết luận.)

### 2c. Deployment — val THẬT 3-6m (`RESULTS_FLIGHT_VAL.md`, 92 frame bay thật)
| model | data | **AP50** | AP@.5:.95 |
|---|---|---|---|
| v17 deploy (+COCO) | VisDrone+COCO | **0.762** | 0.337 |
| v17 light (no-COCO) | VisDrone | 0.143 | 0.055 |

→ **COCO có ích 5.3× cho tầm thấp** — bằng chứng cho nhánh deployment. Demo: `demo_flight/frames/*.jpg`.

## 3. Biểu đồ (đã đẩy lên)
- **pfdet:** `runs/train_v17_*/training_curves.png` (light, vib, nms_free, deploy, vib_deploy) + `runs/train_v15_*`.
- **YOLO:** `runs/yolo/<model>/` và `runs/yolo_deploy/<model>/`: `results.png` (đường cong), `PR_curve.png`,
  `F1_curve.png`, `confusion_matrix.png`, `val_batch*_pred.jpg` (detect định tính).
- **Số thô mỗi epoch:** `results.csv` (YOLO) và `training_log.csv` (pfdet).
- **Demo bay thật:** `demo_flight/frames/demo_*.jpg` (box người, cảnh bay ~6m).

## 4. Thiết kế & tài liệu
- `V17_DESIGN.md` — kiến trúc v17 + nghiên cứu SOTA. `PROJECT.md` — bản đồ toàn dự án + bài học.
- `RESULTS_ROBUSTNESS.md`, `RESULTS_ABLATION.md`, `RESULTS_FLIGHT_VAL.md`.
- Code: `models/pfdet_nano_v17.py`, `vib_consistency.py` (đóng góp), `utils/losses_v15.py`, `utils/losses_v17_e2e.py`.
- Val set thật: `data/flight_val/` (92 ảnh + nhãn + README).

## 5. Venue nhắm
Drones (MDPI, IF 4.8) / IEEE Access / J. Real-Time Image Processing. Hybrid: bảng chính VisDrone-chuẩn +
section "Real-world Deployment" (Hailo/PX4/bay thật).

## 6. ĐANG CHỜ (train sau để tối ưu thời gian — user chốt)
1. **RT-DETR phase-1 + phase-2**: đang/đợi train lại bằng **AdamW lr=1e-4, patience=50** (bản cũ dùng MuSGD →
   sập, số rác). Phase-2 đang chạy; phase-1 do `fix_queue` lo sau `DEPLOY_ALL_DONE`.
2. **`BASELINES_TABLE.md`**: `summarize_baselines.py` tính lại **params/GFLOPs/mAP đồng nhất** cho mọi baseline
   (số cũ ghi 0.00M do bug `model.info`). → đây mới là bảng efficiency dùng cho paper.
3. **flight_val eval** cho vib_deploy + baselines (bổ sung nhánh deployment).
- Marker hoàn tất cuối cùng: `runs/paper_queue/FIX_ALL_DONE`.
