# RESULTS — Val THẬT (chuyến bay 2026-07-21, ~6m)

Val set: `data/flight_val/` — 92 frame đã gán nhãn tay (Roboflow), 1 người/frame, class=person.
Nguồn: `2026-07-21 09-43-12.mkv` (GCS ~6m, POSHOLD). Crop UI (top66/left105), 1fps, 654×1174.
Eval: `run_eval_coco.py`, img-size 640, **device cpu** (không đụng GPU đang train), conf 0.01.

## COCO có ích cho tầm thấp? → CÓ, RẤT RÕ

| Model | Data train | **AP@0.5** | AP@.5:.95 | AP75 | Recall |
|---|---|---|---|---|---|
| `runs/train_v17_deploy/best.pt` | VisDrone **+ COCO** | **0.762** | **0.337** | 0.145 | 1.00 |
| `runs/train_v17_light/best.pt` | VisDrone (no-COCO) | 0.143 | 0.055 | 0.017 | 0.96 |

**deploy > light 5.3× AP@0.5** trên chính domain deployment. light vẫn recall cao (0.96 — thấy người)
nhưng box lỏng/định vị kém (AP75 0.017) vì chỉ học người tí xíu bay cao. → **COCO close-person cần thiết cho 3-6m.**

Đối chiếu: trên VisDrone-val (người tí xíu bay cao) thì NGƯỢC LẠI — deploy 0.5136 < light 0.5245.
→ Khẳng định: **chọn data theo DEPLOYMENT, không theo val set lệch domain** (mục 8 PROJECT.md).

## Hạn chế val set này (nói thẳng cho paper)
- 1 người / 1 cảnh / 1 độ cao (~6m) / 92 frame → **sơ bộ**, chưa đủ mạnh làm benchmark chính.
- Nguồn là quay-lại-màn GCS (có overlay crosshair, chất lượng giảm). Val nghiêm túc cần footage THÔ thẻ SD,
  nhiều người/nhiều cảnh/nhiều độ cao (2-3 buổi bay).

## Hệ quả (chờ no-COCO xong mới làm)
Bằng chứng này MỞ KHOÁ full-pivot (deploy = model chính). Nhưng KHÔNG ngắt paper_queue no-COCO đang chạy.
Sau khi queue no-COCO xong hết (vib→nms_free→eval→6 YOLO baselines) mới thêm: retrain baselines trên
COCO-mix + refinetune vib từ deploy. Xem [[project_paper_hybrid]].
