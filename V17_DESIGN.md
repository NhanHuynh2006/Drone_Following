# PFDet-Nano v17 — Thiết kế dựa trên SOTA (drone tiny-person, **Jetson Nano B01**)

Mục tiêu: **nhanh nhất + chính xác nhất** cho người-nhìn-từ-drone, deploy **Jetson Nano B01**
(GPU Maxwell 128-core + TensorRT), giữ đóng góp **vibration-consistency**.

> ⚠️ **TARGET = JETSON NANO B01 (GPU), deploy TensorRT FP16** (Maxwell không hỗ trợ INT8 tốt).
>
> 🔄 **CHỈNH HƯỚNG (sau khi soi YOLO26n: 2.4M/5.4G/40.9 mAP):** BỎ ý RepVGG-dense (làm to gấp 3 vô ích).
> v15 ĐÃ bám YOLO26 (STAL/ProgLoss/MuSGD, DFL-free) và còn NHỎ hơn YOLO26n (1.0M) + có P2 (tốt cho người nhỏ
> hơn YOLO26n 3-scale). v17 = **giữ backbone nhỏ v15 + thêm thứ v15 THIẾU = NMS-free end-to-end (YOLO26)**
> + CED nhẹ cho tiny + vibration. Mục tiêu giữ ~1–1.5M params.
> Tham chiếu: **YOLO26** (NMS-free, DFL-free, edge), RemDet (CED tiny-object), YOLOv10 (dual-assign NMS-free).

---

## 1. Chẩn đoán v15 hiện tại (số thật)
| | v15 light (hiện tại) |
|---|---|
| Params / GMACs@640 | 1.0M / 6.5 |
| AP@.5:.95 (person) @640 | 0.235 ; @320 **0.089** (sụp) |
| AP50 @640 | 0.55 |
| Robustness (vibration) | retain 22%→34% (đã có đóng góp) |

**3 nút thắt:**
1. **Tiny-object sụp ở res thấp** (0.235→0.089 khi 640→320). Downsample stride-2 conv phá thông tin người nhỏ.
2. **Định vị lỏng** (AP75 0.12). Box head hồi quy trực tiếp, thiếu phân phối.
3. **Tốc độ Pi** chưa tối ưu: NMS + block phân mảnh + chưa QAT.

---

## 2. SOTA học được (nguồn cuối file)

### A. Thiết kế hiệu quả cho UAV — **RemDet (AAAI 2025)** ~40% mAP VisDrone, 110 FPS
- **CED (Context Enhanced Downsampling):** trộn nhánh ViT + CNN khi hạ mẫu → **giảm mất thông tin vật nhỏ/dày**. (chữa nút thắt #1)
- **GatedFFN:** dùng phép NHÂN (gating) thay FFN → biểu diễn chiều cao **rẻ + độ trễ thấp**.
- **ChannelC2f:** biểu diễn chiều cao chống mất thông tin vật nhỏ.

### B. Backbone cho GPU Jetson (Maxwell + TensorRT)
- **RepVGG (CVPR 2021):** train đa nhánh (3×3 + 1×1 + identity), **deploy fuse về 1 conv 3×3** → GPU chạy Winograg F(2×2,3×3), arithmetic intensity cao. YOLOv6 dùng: RepConv fuse → **+10–15% throughput TensorRT FP16**, RepVGG nhanh hơn ResNet-50 ~83% trên GPU.
- **TRÁNH depthwise/PConv ở đây:** "depthwise conv + channel shuffle TĂNG memory access cost, kém hỗ trợ thiết bị" → CHẬM trên GPU dù ít FLOPs. (PConv chỉ tốt cho CPU Pi — không còn là target.)
- **TensorRT-friendly ops:** conv, BN, SiLU/ReLU, add, concat, resize. Tránh op lạ TRT không fuse được.

### C. Đầu ra hiện đại
- **YOLOv10 NMS-free (dual assignment):** bỏ NMS → **giảm độ trễ deploy + dễ export**. Train 2 nhánh (1-to-many + 1-to-one), deploy chỉ 1-to-one.
- **DFL (Distribution Focal Loss):** hồi quy box theo PHÂN PHỐI → **định vị khít hơn** (chữa nút thắt #2, nâng AP75).
- **Wise-IoU v3:** gradient gain động theo chất lượng anchor — tốt cho vật nhỏ dày.

### D. Nén cho Pi 5 (đã có bằng chứng)
- **INT8 QDQ**: YOLOv11n 195ms→77ms (2.5×) trên Pi 5, **mất accuracy nhỏ NẾU làm đúng** (calibrate kỹ + **QAT**).
- **30% pruning + INT8** là combo chuẩn cho Pi 5 (ARM). PTQ thuần (như tôi thử → hỏng) cần thay bằng **QAT**.
- Runtime: **NCNN** (tối ưu ARM) hoặc ONNX Runtime; cân nhắc **OpenVINO** nếu x86.

---

## 3. Kiến trúc PFDet-Nano v17 (đề xuất)

```
Input (deploy 416–512, train multiscale 384–768)
  │
[Stem] RepConv 3×3 s2  (P1, stride2)            ← giữ res cao sớm
  │
[CED downsample] ──► P2 (stride4)   FasterStage(PConv×N) + GatedFFN
  │                                  ↑ tiny-object info được bảo toàn
[CED downsample] ──► P3 (stride8)   FasterStage(PConv×N) + GatedFFN
  │
[CED downsample] ──► P4 (stride16)  FasterStage + AreaAttention(rẻ)
  │
[CED downsample] ──► P5 (stride32)  FasterStage + AreaAttention
  │
[Neck] BiFPN nhẹ trên P2–P5  (+ GatedFFN trộn kênh rẻ)
  │
[Head] NMS-free decoupled, mỗi scale:
        cls (1 lớp person)  +  box DFL (reg_max=8)
        train: dual (1-to-many + 1-to-one) ; deploy: 1-to-one (bỏ NMS)
  │
Reparameterize toàn bộ về conv đơn khi deploy
```

**Khác biệt cốt lõi vs v15 (tối ưu cho Jetson GPU):**
| thành phần | v15 | **v17 (Jetson)** | lợi ích |
|---|---|---|---|
| backbone block | UIB (depthwise) | **RepVGG dense 3×3 (RepBlock)** | GPU/Winograd nhanh, TRT fuse |
| downsample | strided conv | **CED (ViT+CNN)** | giữ người nhỏ ở res thấp |
| trộn kênh neck | conv thường | **GatedFFN / RepBlock** | rẻ, GPU-friendly |
| box head | hồi quy trực tiếp | **DFL reg_max=8** | AP75 ↑ |
| hậu xử lý | NMS | **NMS-free** (hoặc TRT EfficientNMS) | latency↓ |
| deploy | ONNX | **TensorRT FP16 engine** | Maxwell tối ưu (FP16, không INT8) |
| đóng góp | vibration | **vibration (giữ)** | điểm mới của paper |

---

## 4. Lộ trình triển khai (theo phase, đo sau mỗi phase)
1. **v17-A backbone**: PConv FasterStage + CED downsample (thay UIB). Mục tiêu: GMACs↓, giữ AP. So v15.
2. **v17-B head**: thêm DFL + decouple. Mục tiêu: AP75 ↑.
3. **v17-C NMS-free**: dual-assign train. Mục tiêu: bỏ NMS, latency↓.
4. **v17-D + vibration**: gắn lại vibration-consistency. Mục tiêu: robustness (đóng góp paper).
5. **Deploy Jetson**: reparameterize RepBlocks → ONNX → **TensorRT FP16 engine** (repo có `export.py --format trt`). Benchmark FPS Jetson thật.
6. **So sánh đầy đủ**: vs YOLOv8n/11n/RemDet trên VisDrone-person, + bảng FPS Jetson TensorRT FP16.

## 5. Kỳ vọng (định lượng thận trọng)
- RepVGG-dense + reparam → TensorRT FP16 trên Jetson Nano: tham chiếu YOLOv5n ~27 FPS, YOLOv6n nhanh hơn nhờ RepVGG → mục tiêu **~25–40 FPS @416–512**.
- AP75 +0.02–0.04 (DFL). Tiny-object @ res thấp +0.03–0.05 (CED).
- Robustness giữ retain ~34% (vibration). Clean AP giữ hoặc nhỉnh.
- FP16 gần như **0 mất accuracy** (đã verify FP16 = FP32) — khác hẳn INT8.

## Nguồn
- RemDet (AAAI 2025): https://arxiv.org/abs/2412.10040
- FasterNet/PConv (CVPR 2023) ; RepViT (CVPR 2024)
- "Beyond MACs: Hardware Efficient Architecture Design" (2026): https://arxiv.org/pdf/2603.26551
- YOLOv10 NMS-free (NeurIPS 2024) ; DFL (GFL, NeurIPS 2020)
- Quantized YOLO trên Pi 5 (2025): https://arxiv.org/abs/2506.09300
- Small Object Detection Survey (2025): https://arxiv.org/pdf/2503.20516
- LAF-YOLOv10 (P2 head, Wise-IoU) ; LMW-YOLO (VisDrone 37.2% mAP)
