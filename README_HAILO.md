# SteadyDet-N **v17** → Hailo (Raspberry Pi 5 + AI Kit/HAT)

> Cập nhật cho **v17** (khác v15: đã BỎ LSKBlock, head đổi thành LiteHead, P2 mỏng).
> Đã kiểm ONNX v17 thật: op duy nhất Hailo có thể chê = **AreaAttention ở P5** (xem mục 4). Còn lại sạch.

## 0. Nguyên tắc quan trọng
- Model custom **không cần lib convert riêng**. Hailo Dataflow Compiler (DFC) nhận thẳng ONNX. `hailomz` chỉ cho model Model Zoo.
- Convert chỉ chạy trên **x86_64 Linux** (Ubuntu 20/22), KHÔNG chạy trên Pi. Dùng máy Legion Ubuntu.
- HEF **static**: fix input size lúc export (khuyên **320** cho Pi; hoặc 512 nếu Hailo gánh nổi + cần chính xác hơn).
- **Detection chạy trên Hailo NPU; điều khiển bay (`follow_drone/`, Python pymavlink, ArduPilot GUIDED) vẫn chạy trên CPU Pi 5** — chỉ thay phần inference từ ONNX Runtime sang HailoRT (decode+NMS vẫn CPU numpy như cũ).

## 1. Cài môi trường (trên Legion)
```bash
sudo apt install python3.10-venv python3.10-dev graphviz libgraphviz-dev
python3.10 -m venv ~/hailodfc && source ~/hailodfc/bin/activate
# Tải Hailo Dataflow Compiler .whl từ hailo.ai Developer Zone (đăng ký free), khớp Python version.
pip install hailo_dataflow_compiler-3.*.whl        # Hailo-8/8L -> DFC 3.x (KHÔNG dùng 5.x)
pip install onnx onnxsim onnxruntime pillow
hailo -h
```

## 2. Export ONNX v17 (bước 1)
```bash
python export_onnx_hailo.py \
    --weights runs/train_v17_deploy/best.pt \
    --img-size 320 --output pfdet_v17_hailo.onnx
```
- Script tự nhận **version=v17** (build_model_from_checkpoint), fuse conv+bn, static batch=1, opset 11, ir_version≤8.
- Output: 4 tensor `output_p2/p3/p4/p5`, mỗi cái (1,5,H,W) = cls(1)+box(4). @320 → 80/40/20/10.
- Mở bằng netron.app, ghi lại tên node ở ĐUÔI 4 head (LiteHead kết bằng **Concat** cls+box) — dự phòng cho `--end-nodes`.

## 3. Convert sang HEF (bước 2)
```bash
python compile_hef.py \
    --onnx pfdet_v17_hailo.onnx \
    --hw-arch hailo8l \
    --calib-dir data/visdrone/train/images \
    --img-size 320 --name pfdet_v17
```
- `hailo8l` = AI Kit / AI HAT+ 13 TOPS ; `hailo8` = HAT+ 26 TOPS (`hailortcli scan` nếu không chắc).
- Calibration: ~64 ảnh thật VisDrone (đa dạng). **Lý tưởng: dùng ảnh 3-10m THẬT của bạn** (đúng domain deploy) → quantize sát thực tế hơn.
- ⚠️ **Normalization**: script để `[0,0,0],[255,255,255]` (chip tự chia 255) — ĐÚNG với v17 (RGB /255). Đừng đổi.
- ⚠️ Calib script đang **resize** ảnh; v17 train bằng **letterbox**. Sai lệch nhỏ (calib chỉ cần dải activation), nhưng muốn chuẩn nhất thì letterbox ảnh calib cho khớp.
- Ra `pfdet_v17_hailo8l.hef` → copy qua Pi.

## 4. Nếu PARSE FAIL — v17 CHỈ CÓ 1 thủ phạm: AreaAttention (P5)
v17 đã bỏ LSKBlock, nên **không còn** vấn đề softmax-channel như v15. Op rủi ro giờ **toàn bộ từ `AreaAttentionBlock` ở `bottleneck_p5`**: LayerNorm (ReduceMean/Sub/Pow/Sqrt/Div), MatMul, Softmax, Erf(GELU), Reshape/Transpose 6D — Hailo hay chê mấy cái này.

**3 cách xử lý (theo thứ tự nên thử):**

### ✅ Cách A (đơn giản nhất cho v17): bỏ AreaAttention, finetune ngắn

> ⚠️ **ĐO LẠI 2026-08-27: "gần như không mất AP" là SAI trên domain bay thấp.**
> Trên 92 frame bay thật 3-6m (`data/flight_val`), cùng giao thức, cùng decode:
>
> | checkpoint | F1 tốt nhất | conf | recall | precision |
> |---|---|---|---|---|
> | `train_v17_deploy` (CÓ AreaAttention) | **0.882** | 0.25 | 0.815 | 0.962 |
> | `train_v17_hailo` (bỏ AreaAttention, finetune 20ep) | 0.768 | 0.20 | 0.793 | 0.745 |
> | HEF INT8 từ checkpoint trên | 0.697 | 0.30 | 0.576 | 0.883 |
>
> Mất **0.114 F1 ở bước finetune** — lớn gấp 1.6× mức mất do INT8 (0.071).
>
> **Nguyên nhân KHÔNG phải do bỏ attention, mà do một lỗi config:**
> `configs/train_config_v17_hailo.yaml` để `extra_enable_epoch: 100` trong khi `epochs: 20`.
> `train_v3.py:819` là `want_coco = extra_enable_epoch <= epoch < extra_disable_epoch`
> → **COCO person chưa bao giờ được bật**. 20 epoch đó finetune thuần VisDrone, kéo model
> ngược về domain bay cao và xoá khả năng bắt người gần mà `train_v17_deploy` đã học.
> Log xác nhận: `pos_s4=63, pos_s8=9, pos_s16=0, pos_s32=0` — toàn bộ positive rơi vào P2.
>
> **✅ ĐÃ SỬA VÀ ĐO LẠI (2026-08-27 tối).** Train lại 20 epoch với
> `configs/train_config_v17_hailo_coco.yaml` (`extra_enable_epoch: 0`):
>
> | checkpoint | F1 trên flight_val |
> |---|---|
> | `train_v17_deploy` (giữ attention + COCO) | 0.882 |
> | **`train_v17_hailo_coco`** (bỏ attention, CÓ COCO) | **0.836** |
> | `train_v17_hailo` (bỏ attention, MẤT COCO) | 0.768 |
>
> Bật COCO lấy lại **0.068 trên 0.114** khoảng cách. Vậy con số đúng là: **bỏ AreaAttention
> tốn 0.046 F1** trên domain bay thấp — không phải "gần như không mất" như câu gốc, nhưng
> cũng không phải toàn bộ 0.114 như thoạt nhìn. Phần lớn là do lỗi config, không phải do
> kiến trúc.
>
> Muốn lấy nốt 0.046 đó thì dùng **Cách B** (`ConvAttnBlock`) bên dưới — giữ được ý tưởng
> attention mà Hailo vẫn compile. Chưa làm.
>
> Chi tiết: `runs/retrain_coco_2026-08-27.md`.
>
> Và khi compile lại, dùng `--calib-dir` trỏ ảnh ĐÚNG domain bay (3-6m) thay vì
> `data/visdrone/train/images`, đúng như mục 3 đã khuyên.

P5 (stride 32) đóng góp ÍT cho người tí xíu (3-10m rơi vào P3/P4; tiny ở P2).
```bash
# 1. Tạo config Hailo: copy train_config_v17_deploy.yaml, đổi:
#      use_area_attention: false     + pretrained: "./runs/train_v17_deploy/best.pt"  + epochs: 15  + lr thấp
# 2. Finetune 15-20 epoch  ->  model THUẦN CONV, parse Hailo 100%.
python train_v3.py --config configs/train_config_v17_hailo.yaml
python export_onnx_hailo.py --weights runs/train_v17_hailo/best.pt --img-size 320 --output pfdet_v17_hailo.onnx
```
→ ONNX lúc này **không còn LayerNorm/MatMul/Softmax/Erf** → parse trơn.

### Cách B: thay AreaAttention bằng ConvAttnBlock (SE-style, thuần conv) rồi finetune
Nếu muốn giữ "attention" cho paper: sửa `models/pfdet_nano_v17.py`, thay `bottleneck_p5 = AreaAttentionBlock(...)` bằng:
```python
class ConvAttnBlock(nn.Module):          # Hailo-friendly 100%
    def __init__(self, c):
        super().__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c//4, 1), nn.SiLU(), nn.Conv2d(c//4, c, 1), nn.Sigmoid())
        self.spatial = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, groups=c), nn.Conv2d(c, c, 1))
    def forward(self, x): return x + self.spatial(x * self.se(x))
```
Load weight cũ (strict=False) → finetune 10-20 epoch LR nhỏ → export lại.

### Cách C: parse thử nguyên bản
DFC bản mới đôi khi nuốt được LayerNorm/attention. Cứ chạy `compile_hef.py`, nếu qua thì khỏi sửa.

### Head (LiteHead) — thường KHÔNG cần đụng
Head ra raw: DWConv→PW→ (cls 1×1, box 1×1) → **Concat** 5 kênh. Không decode trong graph (đúng kiểu Hailo thích).
Nếu parser kêu ở đuôi (Concat/Reshape sau head), truyền tên 4 Concat/Conv cuối vào `--end-nodes`.
Decode box + NMS chạy **CPU Pi** (numpy) — tái dùng `decode_predictions_np` + `nms_numpy` như path ONNX.

## 5. Chạy trên Pi 5 (HailoRT)
```python
from hailo_platform import HEF, VDevice, InferVStreams, ConfigureParams, HailoStreamInterface
import numpy as np, cv2

hef = HEF('pfdet_v17_hailo8l.hef')
with VDevice() as dev:
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = dev.configure(hef, cfg)[0]
    with InferVStreams(ng, ng.create_input_vstream_params(), ng.create_output_vstream_params()) as pipe:
        # camera BGR -> letterbox 320 -> RGB -> NHWC uint8 (normalize /255 chip tự lo)
        frame = cv2.cvtColor(letterboxed_320, cv2.COLOR_BGR2RGB)[None]   # (1,320,320,3) uint8
        with ng.activate():
            out = pipe.infer({list(pipe.get_input_vstream_infos())[0].name: frame})
        # out: dict 4 tensor P2..P5 (NHWC) -> transpose về (5,H,W) -> decode_predictions_np + nms_numpy
```
Lưu ý: HailoRT nhận **NHWC uint8, RGB** (khác NCHW float của ONNX). Camera OpenCV là BGR → phải `cvtColor` sang RGB. Output NHWC → nhớ transpose khi đưa vào `decode_predictions_np` (kỳ vọng (5,H,W)).

## 6. ⚠️ Cảnh báo INT8 cho model TÍ HON (v17 chỉ 1.17M)
Detector nano rất nhạy với INT8 (từng thấy PTQ ONNX phá AP xuống ~0). Hailo quantize tốt hơn (per-layer + calib), nhưng vẫn rủi ro:
- Dùng **nhiều ảnh calib đa dạng** (ảnh 3-10m thật càng tốt), **`--opt-level 3` hoặc `4`**.
- **So AP trước/sau HEF**: chạy `test_accuracy_onnx.py` (FP32) rồi so với HEF trên Pi. Rớt >5% AP → tăng opt-level, thêm calib, hoặc cân nhắc **QAT** (Hailo hỗ trợ).
- Nếu INT8 vẫn rớt nặng: đây là giới hạn model tí hon + INT8 — chấp nhận trade-off tốc độ, hoặc dùng model to hơn chút.

## 7. Checklist debug nhanh
| Lỗi | Fix |
|---|---|
| `'tuple' object has no attribute` khi export | (đã sửa) — `build_model_from_checkpoint` trả tuple, script đã unpack |
| `ir_version is higher` | script đã ép ir_version=8 |
| Parser chê LayerNorm/MatMul/Softmax/Erf | = AreaAttention P5 → **Cách A** (bỏ attention + finetune) là nhanh nhất |
| Parser gợi ý end-nodes ở đuôi | copy vào `--end-nodes` (chỉ khi fail nằm CUỐI graph) |
| AP rớt sau quantize | normalization đúng /255 chưa; letterbox calib; opt-level 3/4; thêm calib; QAT |
| HEF chạy output nhảm | NHWC vs NCHW, uint8 vs float, **RGB vs BGR**, thứ tự output vstream |
