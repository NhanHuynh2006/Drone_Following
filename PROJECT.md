# PROJECT.md — PFDet-Nano (phát hiện người từ drone)

> File này là **bản đồ dự án**: đọc 1 lần hiểu ngay đang làm gì / đã làm gì, khỏi lội hết source.
> **Quy ước: mỗi lần sửa gì đáng kể → cập nhật file này.** (Cập nhật gần nhất: 2026-06-23)

---

## 1. Dự án này là gì?
Phát hiện **người** (1 class) từ **camera drone**.
> ⚠️ **DEPLOYMENT THẬT: bay THẤP 3-5m** → người **to/gần** (KHÔNG phải tí xíu). Khác VisDrone-val (bay cao,
> người tí xíu). Vì vậy: (a) **trộn COCO person** (người gần) là CÓ ÍCH cho use case thật; (b) AP đo trên
> VisDrone-val không phản ánh đúng deployment — lý tưởng cần val set quay ở 3-5m.

Train trên **VisDrone** (data drone có nhãn) + **COCO person** (người gần, hợp tầm thấp).
Hai mục tiêu song song:
1. **Paper hội nghị** — đóng góp chính = **vibration-consistency**. Chỉ viết về MODEL (không có phần deploy).
2. **Deploy realtime** trên **Jetson Nano B01** (GPU Maxwell + **TensorRT FP16**) — sẽ là **DỰ ÁN KHÁC**.

> **2 nhánh model độc lập:**
> | | PAPER | DEPLOY (dự án khác) |
> |---|---|---|
> | model | **v17 NMS-free** (end-to-end, config `train_config_v17_nms_free.yaml`) | v17 thường (`train_config_v17_light.yaml`) |
> | data | **VisDrone CHUẨN, KHÔNG COCO** (so sánh công bằng) | + COCO + footage 3-10m (cảnh thật) |
> | đóng góp | vibration-consistency | — |
> Lưu ý: NMS-free có thể cho AP thấp hơn (head o2o khó train) → phải train thử; nếu tụt thì làm ablation, dùng v17 thường làm headline.

Có sẵn cả hệ thống **bay theo người** (follow drone) — xem `FOLLOW_CONTROL.md`, `follow_drone/`.

---

## 2. Trạng thái hiện tại (TL;DR)
| Hạng mục | Trạng thái |
|---|---|
| Model tốt nhất (clean AP) | **musgd_v3** — `runs/train_v15_light_musgd_v3/best.pt`, AP@0.5 **0.5932** |
| Model đóng góp (vibration) | **E** — `runs/train_v15_vib_E/last.pt`, robustness **retain 34.3%** (baseline 22.4%) |
| Kiến trúc mới (đang train) | **v17** — 1.17M / 4.6 GFLOPs@640 (nhỏ + ít FLOPs hơn YOLO26n, vẫn có P2). `runs/train_v17_light/` |
| Plan B (sẵn, chưa train) | **v17 NMS-free** — flag OFF mặc định, config `train_config_v17_nms_free.yaml` |
| Deploy | ONNX FP32 ở `deploy/` chạy đúng = PyTorch. Demo webcam OK. Jetson TensorRT: scripts ở `jetson/` (chờ benchmark thật) |

---

## 3. Đóng góp khoa học = Vibration-Consistency
- Khi drone bay, camera **rung → ảnh nhoè** lúc phơi sáng → detector thường **sụp** (baseline giữ chỉ 22.4% AP).
- Ý tưởng: lúc train, forward ảnh SẠCH + ảnh RUNG (mô phỏng), **ép dự đoán ảnh rung khớp ảnh sạch**
  (consistency loss). **Train-time only → 0 chi phí inference** (model deploy y hệt baseline).
- Code: `vib_consistency.py` (corrupt_batch + VibConsistencyLoss), benchmark `eval_robustness.py`.
- Kết quả: E giữ **34.3%** (vs 22.4%), riêng rung vừa (sev2) **gấp 2.1×**, chỉ tốn ~6% AP sạch.

---

## 4. Kết quả chính (VisDrone val 548 ảnh, @640, person-only)
**Clean AP & robustness** (chi tiết: `RESULTS_ROBUSTNESS.md`, `RESULTS_ABLATION.md`)
| model | params | GFLOPs@640 | AP@.5:.95 | AP75 | robust retain% |
|---|---|---|---|---|---|
| baseline A (musgd_v3) | 1.0M | 12.9 | 0.235 | 0.131 | 22.4% |
| **E (+vibration)** | 1.0M | 12.9 | 0.221 | 0.120 | **34.3%** |
| **v17 (mới)** | **1.17M** | **4.6** | *(đang train)* | | |
| YOLO26n / YOLO11n (tham chiếu) | 2.4 / 2.6M | 5.4 / 6.5 | — | — | — |

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
jetson/                # build_trt_jetson.py + bench_trt.py (TensorRT) ; DEPLOY_GUIDE.md (PX4 params + bay)
jetson/control/        # follow_px4.cpp — ĐIỀU KHIỂN bay-theo C++ MAVSDK/PX4 (Kalman+feedforward+50Hz, MƯỢT)
jetson/detector_pub.py # detector ONNX -> đẩy target qua UDP cho follow_px4 (kiến trúc companion-computer)
jetson/record_flight.py# CÔNG CỤ RIÊNG (độc lập): thu ảnh+video lúc bay -> làm val set 3-10m thật
jetson/pfdet_v17_*.onnx # model deploy v17 (FP32 ONNX @320/512/640) — TensorRT tự build FP16 trên Jetson
configs/               # train_config_v15_*.yaml, train_config_v17_*.yaml
deploy/                # checkpoint sạch + ONNX để deploy
runs/                  # các lần train (best.pt/last.pt/log/curves)
follow_drone/, FOLLOW_CONTROL.md, drone_follow.py   # hệ thống bay theo người
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
```

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
- **Hạ tầng train (laptop 8GB GPU / 14GB RAM):** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` + tắt `cudnn.benchmark` khi multiscale (đã baked vào train_v3) để khỏi OOM tích lũy; `num_workers ≤ 6` để khỏi cạn RAM giết VS Code; batch nhỏ (v15 batch 6-8 @640, v17 nhẹ hơn batch 12 OK).

---

## 9. Roadmap

### ƯU TIÊN HIỆN TẠI: **DEPLOY TRƯỚC** (paper để sau)

**A. Deploy (đang làm):**
1. ✅ **v17 deploy ĐÃ TRAIN + ĐỔI TÊN** → `runs/train_v17_deploy/best.pt` (epoch 194, **AP@0.5=0.5136**), config `train_config_v17_deploy.yaml` (COCO từ epoch 100, nhắm 3-10m). Dừng ở epoch 230 vì BÃO HOÀ.
2. ✅ **ĐÃ EXPORT ONNX** (`deploy/` + `jetson/pfdet_v17_{320,512,640}.onnx`, FP32; ONNX==PyTorch chênh 2e-5). Test: @512 AP50 0.443 / 51FPS, @320 0.231 / 131FPS (CPU x86; VisDrone-val tí xíu, không phải metric 3-10m). **Sẵn scp lên Jetson** → build FP16 (`build_trt_jetson.py --fp16`).
2. Export ONNX → **TensorRT FP16** + benchmark Jetson (`jetson/`). Cập nhật `build_deploy.py` cho v17 (hiện hardcode v15).
3. Deploy lên drone.

**A2. Điều khiển bay (PX4 + Jetson) — ĐÃ VIẾT XONG (`jetson/control/`, `jetson/DEPLOY_GUIDE.md`):**
- Firmware đổi ArduPilot → **PX4**. Control viết lại bằng **C++ MAVSDK** (nhanh, mượt). Code cũ pymavlink/ArduPilot KHÔNG tái dùng.
- Thuật toán: **Kalman vận-tốc-không-đổi (bù trễ) + PD + velocity feedforward + jerk-limit + 50Hz → PX4 jerk-limited smoothing.** Chống giật (vấn đề velocity-only 10Hz cũ).
- Kiến trúc tách: `detector_pub.py` (ONNX, ~15-30Hz) → UDP → `follow_px4` (C++, 50Hz) → PX4.
- Còn lại: cài MAVSDK trên Jetson, build, **test SITL trước**, calibrate camera (fx/fy), bay test. Xem DEPLOY_GUIDE.md.

**B. Paper (LÀM SAU khi deploy xong) — hướng đã chốt: efficiency + robustness:**
- 3 config v17: `_light` (baseline no-COCO, head thường) / `_nms_free` (no-COCO, NMS-free) / `_deploy` (CÓ COCO, đã train cho deploy).
- v17 baseline **`train_config_v17_light.yaml`** (no-COCO, số chuẩn VisDrone) → AP/FLOPs. (thua v15 ~0.07 AP nhưng FLOPs 2.8× ít hơn → bán efficiency.)
- v17 **NMS-free** (`_nms_free.yaml`, no-COCO) → headline end-to-end nếu AP ~bằng, else ablation.
- **+ Vibration finetune** → robustness = đóng góp chính.
- Baselines: YOLOv8n/11n/26n + v15 → bảng so sánh + đường cong robustness.

---

## 10. Hạ tầng
- **Dev:** laptop RTX 4060 Laptop (8GB, ~7.3GB dùng được) + 14GB RAM, CUDA.
- **Deploy:** Jetson Nano B01 (GPU Maxwell 128-core, ARM A57, TensorRT FP16).
- **Data:** `data/visdrone/{train,val}` (6471 / 548 ảnh, person-only) + `data/coco_person` (2654 ảnh, trộn cho tầm thấp 3-5m).
- **Memory dự án:** `~/.claude/projects/.../memory/` — fact bền vững giữa các phiên Claude.
