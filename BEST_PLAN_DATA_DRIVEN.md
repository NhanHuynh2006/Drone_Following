# KẾ HOẠCH CHỐT (theo số liệu thật) — để model thành paper tốt nhất

## 1. Chẩn đoán (từ eval thật, 548 ảnh VisDrone val @640)
| Metric | Giá trị | Ý nghĩa |
|---|---|---|
| AP@0.5 | 0.580 | tìm người tốt |
| **AP@0.5:0.95** | **0.235** | mAP thật — báo cáo con này làm chính |
| **AP@0.75** | **0.130** | ⚠️ định vị LỎNG — nút thắt số 1 |
| AR_small thưa→đông | 0.41 → 0.30 | recall cảnh đông yếu — nút thắt số 2 |

→ Detector **tìm trúng nhưng khoanh không khít**, tệ nhất ở cảnh đông. Mọi thứ dưới đây đánh đúng 2 chỗ này.

## 2. Đóng góp mới = DGS (Density-Guided Supervision): 1 ý tưởng, 2 cơ chế
- **Cơ chế 1 — Density head** (`density_aux.py`): vá recall cảnh đông + regularize. Train-time only.
- **Cơ chế 2 — Crowd-aware localization weighting** (`crowd_loc.py`): tăng box-loss ở vùng đông
  → ép định vị khít hơn đúng chỗ AP75 sụp. Train-time only.
- Cả hai 0 chi phí inference → model deploy Pi/drone không đổi.

Đòn đẩy số (mượn, không phải đóng góp): **Knowledge Distillation** từ teacher balanced 2M
(`distillation.py`) — teacher khoanh khít hơn, truyền xuống student → nâng AP75. Miễn phí lúc inference.

(Tuỳ chọn đẩy số tối đa: **DFL** cho box — thuốc kinh điển cho AP75. Đổi head+loss+decode, nói tôi nếu muốn làm.)

## 3. Vá loss — chèn cơ chế 2 vào `utils/losses_v15.py` (3 dòng, đã test)
**(a)** đầu file, sau dòng `from .box_ops import ...`:
```python
from crowd_loc import crowd_density_weights
```
**(b)** trong `__init__`, sau dòng `self.prog_loss_factor = prog_loss_factor`:
```python
        self.crowd_alpha = 1.0   # DGS cơ chế 2 ; =0 để tắt (cho ablation)
```
**(c)** trong `__call__`, ngay SAU dòng `box_loss_per = box_loss_per * asl_w`:
```python
                    if self.crowd_alpha > 0:
                        cw = crowd_density_weights(gt_t, gi_t, self.img_size, alpha=self.crowd_alpha)
                        box_loss_per = box_loss_per * cw
```
(`gt_t`, `gi_t` đã có sẵn ngay trên đó — không cần thêm gì.)

## 4. Bảng ablation (lõi paper) — mỗi dòng đo bằng eval_coco (AP@.5:.95, AP75, AP_s, AR_s theo mật độ)
| # | density head | crowd-loc (α) | KD | nhắm vào |
|---|---|---|---|---|
| A baseline | ✗ | 0 | ✗ | (mốc 0.235 hiện tại) |
| B | ✓ | 0 | ✗ | recall cảnh đông |
| C (DGS đầy đủ) | ✓ | 1.0 | ✗ | + định vị cảnh đông (AP75) |
| D (final) | ✓ | 1.0 | ✓ | + localization tổng (KD) |
+ baselines: YOLOv8n / 11n / 26n train cùng split (bảng "ta vs giới").

Bật/tắt: A `USE_DENSITY=False, crowd_alpha=0`; B `USE_DENSITY=True, crowd_alpha=0`;
C `USE_DENSITY=True, crowd_alpha=1.0`; D thêm `USE_KD=True`.

## 5. Kỳ vọng (cái gì nên nhúc nhích)
- **C vs B:** AP75 và AP@.5:.95 tăng (nhất là nhóm ≥30 người) — bằng chứng cơ chế 2 chữa localization.
- **B vs A:** AR_small nhóm đông tăng (0.30 → cao hơn).
- **D vs C:** AP@.5:.95 tăng tiếp nhờ KD.
- Báo cáo cả **@320px** (deploy) lẫn @640.

## 6. Thứ tự chạy
1. ✅ A baseline đã có (0.235 / AP75 0.13).
2. Train **C** (density + crowd-loc, α=1.0) — đóng góp đầy đủ, so với A. Đây là run quan trọng nhất.
3. Train **B** (chỉ density, α=0) — để tách đóng góp của cơ chế 2.
4. Train teacher balanced 2M → train **D** (thêm KD).
5. Baselines YOLO + benchmark Pi INT8 + eval 320.
6. Viết: đóng góp = DGS; KD = booster; phân tích size/mật độ = bằng chứng rigorous.

## 7. Tinh chỉnh
- `crowd_alpha`: 1.0 khởi điểm. Nếu box loss áp đảo → giảm 0.5. Nếu cảnh đông vẫn yếu → thử 1.5.
- `sigma_px` trong crowd_density_weights: 64 (≈ bán kính "đông"). Người to/thưa → tăng; tiny dày → giảm 32.
- density head `density_scale_idx`: thử 0 (P2) vì regime của bạn gần như toàn tiny.
- `lambda_density`: 0.5; theo dõi để density loss ~0.1–1× det loss.

## 8. ✅ ĐÃ TRIỂN KHAI (wiring vào pipeline) — sẵn sàng chạy
- `train_v3.py` đọc 2 section config mới: `dgs:` (density head + crowd_alpha) và `kd:` (distillation).
  - `use_density=true` → tự build `PFDetNanoV15Density`, thêm density loss (`lambda_density`).
  - `crowd_alpha` → set vào `criterion.crowd_alpha` (cơ chế 2). =0 để tắt cho ablation.
  - `kd.use_kd=true` → load teacher (freeze), thêm output+feature KD, adapters tự vào optimizer.
  - Cả hai chỉ chạy lúc train; EMA/inference gọi `model(x)` trả về list 4 head như cũ → 0 chi phí FPS.
- Validation: `evaluate(..., coco=...)` — COCO breakdown (AP@.5:.95, AP75, AP_s, AR_s theo mật độ)
  LUÔN chạy ở epoch cuối; bật `eval.coco: true` để chạy mỗi `val_interval`.
- Configs ablation đã tạo (`configs/`):
  | run | file | dgs.use_density | dgs.crowd_alpha | kd.use_kd |
  |---|---|---|---|---|
  | B | `train_config_v15_dgs_B.yaml` | true | 0.0 | false |
  | C (DGS đầy đủ) | `train_config_v15_dgs_C.yaml` | true | 1.0 | false |
  | D (final) | `train_config_v15_dgs_D.yaml` | true | 1.0 | true |
  | teacher 2M | `train_config_v15_balanced_teacher.yaml` | false | 0.0 | false |
  (A baseline = config v15_light gốc, không có section `dgs`/`kd`.)

### Lệnh chạy
```bash
# C — run quan trọng nhất (DGS đầy đủ)
python train_v3.py --config configs/train_config_v15_dgs_C.yaml
# B — tách đóng góp cơ chế 2
python train_v3.py --config configs/train_config_v15_dgs_B.yaml
# teacher cho KD, rồi D (sửa kd.teacher_ckpt trong config D nếu cần)
python train_v3.py --config configs/train_config_v15_balanced_teacher.yaml
python train_v3.py --config configs/train_config_v15_dgs_D.yaml
# eval COCO độc lập 1 checkpoint (AP@.5:.95, AP75, theo mật độ)
python run_eval_coco.py --weights runs/train_v15_dgs_C/best.pt \
    --val-images data/visdrone/val/images --val-labels data/visdrone/val/labels \
    --img-size 640 --profile light --device cuda:0
```
> Cần `pip install pycocotools` cho COCO eval (nếu thiếu, COCO breakdown tự bỏ qua, AP@0.5 vẫn chạy).
