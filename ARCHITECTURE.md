# PFDet-Nano v15 — Phân tích kiến trúc chi tiết

> **Bài toán:** Phát hiện người (person detection) từ camera drone, chạy realtime trên Raspberry Pi 5.
> **Kết quả thực tế:** AP@0.5 = **0.5931** trên VisDrone, ~**1.0M params**, ~**30 FPS** trên Pi 5 ở 320px.

Tài liệu này giải thích **TẠI SAO chọn từng kiến trúc** và **TẠI SAO KHÔNG chọn các giải pháp khác**. Viết cho người mới, lập luận chi tiết dựa trên paper và thí nghiệm thực tế.

---

# Phần 0: Thuật ngữ (cho người mới)

Trước khi vào nội dung, đây là các thuật ngữ tiếng Anh sẽ xuất hiện liên tục. Đọc qua một lượt để khi gặp lại không bị bối rối.

## 0.1. Khái niệm cơ bản về deep learning

**Tensor** — mảng số nhiều chiều. Ảnh là tensor 3 chiều (cao × rộng × kênh). Một batch ảnh là 4 chiều (batch × kênh × cao × rộng).

**Channel (kênh)** — một "lớp thông tin" của tensor. Ảnh RGB có 3 channel (Red, Green, Blue). Sau khi qua các Conv, số channel có thể là 24, 48, 96... tùy thiết kế.

**Feature map** — output của một lớp Conv, là tensor có nhiều channel. Mỗi channel đại diện một "feature" model học được (vd: cạnh ngang, cạnh dọc, texture cát, texture cỏ...).

**Layer / Lớp** — một phép biến đổi (Conv, BN, Activation...).

**Weights / Params** — các con số học được trong model. Khi nói "model 1M params" tức là model có 1 triệu con số cần học.

**Forward pass** — đưa ảnh từ input qua model để ra output.

**Backward pass / Backprop** — quá trình tính gradient để cập nhật weights. Đi ngược từ loss → output → ... → input.

**Gradient** — đạo hàm của loss theo từng weight. Cho biết "tăng/giảm weight này thì loss thay đổi thế nào".

**Loss** — một con số đo "model dự đoán sai bao nhiêu". Train là quá trình giảm loss.

**Epoch** — một lần model thấy hết toàn bộ dataset. Train 400 epoch = model thấy data 400 lần.

**Batch** — một nhóm ảnh được xử lý cùng lúc trong một step. Vd batch=12 nghĩa là model xử lý 12 ảnh song song.

**Step / Iteration** — một lần update weights. Một epoch có (số ảnh / batch) step.

## 0.2. Khái niệm về Convolution

**Convolution (Conv)** — phép trượt một kernel nhỏ qua feature map, mỗi vị trí tính tích chập. Là phép cốt lõi của CNN.

**Kernel** — ma trận nhỏ trong Conv (vd 3×3, 5×5). Số trong kernel là weight được học.

**Kernel size** — kích thước kernel. K=3 nghĩa là Conv 3×3.

**Stride** — bước nhảy khi trượt kernel. Stride=1 trượt từng pixel, stride=2 trượt 2 pixel/lần (giảm feature map 2 lần).

**Padding** — viền 0 thêm quanh feature map để giữ kích thước sau Conv.

**Dilation (giãn kernel)** — khoảng cách giữa các phần tử kernel. Dilation=3 nghĩa là kernel 3×3 mà các pixel cách nhau 3 ô → effective kernel = 7×7.

**Conv 3×3 chuẩn** — Conv kernel 3×3 thông thường, mỗi output channel nhìn TẤT CẢ input channel. FLOPs = `H × W × C_in × C_out × 9`.

**DW Conv (Depthwise Convolution)** — mỗi output channel CHỈ nhìn 1 input channel tương ứng. FLOPs = `H × W × C × 9` (ít hơn Conv chuẩn `C` lần).

**PW Conv (Pointwise Convolution)** — Conv 1×1, dùng để trộn các channel với nhau. FLOPs = `H × W × C_in × C_out`.

**Group Conv** — chia channel thành groups, mỗi group tự Conv với nhau. DW Conv là trường hợp đặc biệt khi groups = số channel.

**FLOPs (Floating Point Operations)** — số phép tính dấu phẩy động. Đơn vị đo "model nặng cỡ nào để chạy". 1 GFLOPs = 10⁹ phép tính. Pi 5 có thể làm khoảng 10 GFLOPs/giây CPU.

**MAC (Multiply-Accumulate)** — 1 phép nhân + 1 phép cộng. Thường người ta đếm MAC, FLOPs = 2 × MAC.

## 0.3. Khái niệm về kiến trúc detection

**Backbone** — phần đầu của model, trích xuất feature từ ảnh. Vd ResNet, MobileNet là backbone.

**Neck** — phần giữa, tổng hợp feature từ nhiều scale của backbone. Vd FPN, BiFPN.

**Head** — phần cuối, dự đoán class và bounding box.

**Bottleneck (cổ chai)** — block giảm channel ở giữa rồi tăng lại (như cổ chai). Ý tưởng từ ResNet.

**Inverted Bottleneck** — ngược lại bottleneck: tăng channel ở giữa rồi giảm lại. Ý tưởng từ MobileNet V2.

**Skip connection / Residual** — đường nối tắt: `output = block(x) + x`. Giúp gradient lan ngược dễ hơn.

**Receptive field** — vùng pixel input mà 1 pixel output "nhìn thấy". Conv 3×3 có receptive field 3×3. Sau 5 lớp Conv 3×3 stack lại, receptive field tăng lên 11×11.

**Stride / scale level** — mức độ downsample. P2 = stride 4 nghĩa là feature map đã giảm 4 lần so với input. P5 = stride 32 = giảm 32 lần.

**Anchor** — một "vị trí ứng viên" trong feature map có thể chứa object. Trong anchor-free detection, mỗi pixel của feature map là 1 anchor.

**Positive / Negative anchor** — positive là anchor được gán cho 1 ground truth (chứa object), negative là không chứa.

**Top-k** — chọn k phần tử có score cao nhất.

**Ground Truth (GT)** — nhãn đúng (do người gán). Vd box thật của người trong ảnh.

**Bounding box (bbox / box)** — hộp chữ nhật bao quanh object, định nghĩa bằng (cx, cy, w, h) hoặc (x1, y1, x2, y2).

**IoU (Intersection over Union)** — tỉ số `(diện tích giao) / (diện tích hợp)` của 2 box. Đo độ trùng lặp. IoU=1 là trùng hoàn toàn, IoU=0 là không giao nhau.

**Non-Maximum Suppression (NMS)** — thuật toán loại bỏ box trùng lặp: giữ box có score cao nhất, loại các box overlap > threshold.

**AP (Average Precision)** — chỉ số đo accuracy của detection. AP@0.5 nghĩa là tính ở ngưỡng IoU=0.5.

**Recall** — tỉ lệ "model phát hiện được bao nhiêu / tổng object thật".

**Precision** — tỉ lệ "trong các box model dự đoán, bao nhiêu % đúng".

## 0.4. Khái niệm về training

**Optimizer** — thuật toán cập nhật weights dựa trên gradient. Vd SGD, Adam, AdamW, MuSGD.

**Learning rate (LR)** — kích thước bước nhảy khi update weights. Quá lớn → train không ổn định, quá nhỏ → train chậm.

**Warmup** — giai đoạn đầu training tăng LR từ thấp lên cao dần để ổn định.

**Cosine LR schedule** — giảm LR theo hàm cosine trong suốt training.

**Momentum** — giữ "đà" của gradient cũ để smooth hướng update.

**Weight decay (L2 regularization)** — phạt weight có giá trị lớn, giúp tránh overfit.

**Nesterov momentum** — biến thể momentum nâng cao, hội tụ nhanh hơn.

**Overfit** — model học thuộc training data, không generalize được sang val/test.

**Regularization** — kỹ thuật ngăn overfit (dropout, weight decay, augmentation...).

**Augmentation** — biến đổi ngẫu nhiên data (rotate, crop, flip...) để tăng diversity.

**EMA (Exponential Moving Average)** — trung bình trượt của weights theo thời gian. Eval bằng EMA cho kết quả mượt hơn.

**Gradient Accumulation** — tích lũy gradient qua nhiều batch nhỏ rồi mới update, giả lập batch lớn.

**AMP (Automatic Mixed Precision)** — dùng FP16 (16-bit) thay FP32 (32-bit) để tăng tốc, ít memory.

## 0.5. Các từ khác

**SOTA (State-of-the-Art)** — tốt nhất hiện tại trong literature.

**Inference** — chạy model để predict (không train).

**Deploy** — đưa model lên thiết bị thực tế (Pi 5, server...).

**Benchmark** — đo hiệu năng (FPS, latency, AP) của model.

**Latency** — thời gian xử lý 1 ảnh. Latency 50ms = 20 FPS.

**Heavy / Light** — model nặng/nhẹ về compute.

**No heavy attention** — không dùng attention nặng (full self-attention) ở high-resolution feature map vì quá đắt compute.

**Channels-last** — định dạng tensor (N, H, W, C) thay vì (N, C, H, W) chuẩn. Tối ưu cho một số kiến trúc.

**Reparameterize / Fuse** — gộp nhiều layer thành 1 layer tương đương khi deploy.

**BatchNorm (BN)** — Batch Normalization, layer chuẩn hóa feature map giúp train ổn định.

**SiLU / Swish** — activation function `f(x) = x × sigmoid(x)`. Tốt hơn ReLU trong nhiều trường hợp.

**Aerial imagery** — ảnh chụp từ trên cao (drone, vệ tinh, máy bay).

---

# Phần I: Bài toán và ràng buộc

## 1.1. Đặc điểm của bài toán Drone Person Detection

Drone bay cao 30-150m, camera nhìn xuống. Hậu quả:

1. **Người rất nhỏ:** trong ảnh 640×640, người thường chỉ chiếm 8-50 pixel (so với 100-300px khi chụp ngang). Đây gọi là **tiny object detection** (phát hiện vật thể tí hon).

2. **Phối cảnh từ trên xuống (top-down):** khác hẳn dataset COCO (chụp ngang). Pose người, tỉ lệ thân thể nhìn từ trên cao biến dạng. Đầu chiếm nhiều, chân ít. Aspect ratio (tỉ lệ rộng/cao) gần 1:1 thay vì 1:3 như chụp ngang.

3. **Background đa dạng:** đường, ruộng, mái nhà, cây cối — dễ nhầm với người. Cần model học được "đặc trưng người" mạnh mẽ.

4. **Motion blur:** drone chuyển động, gió, gia tốc → ảnh nhòe. Cần model robust với blur.

## 1.2. Ràng buộc deploy: Raspberry Pi 5

| Tiêu chí | Pi 5 | Hệ quả |
|----------|------|--------|
| **CPU** | 4× Cortex-A76 (ARM, 2.4GHz) | Không có CUDA, mọi thứ chạy CPU. ARM CPU yếu hơn x86 CPU desktop ~2-3 lần ở cùng watt |
| **RAM** | 4-8GB | Model + intermediate features phải vừa RAM. Model 100M params đã ăn 400MB RAM, chưa kể activation |
| **GPU compute** | ~0.1 TFLOPS (rất yếu) | Không train được, chỉ inference |
| **Power** | 5W | Không có quạt khủng, nhiệt giới hạn |
| **Target FPS** | ≥20 FPS để track realtime | ~50ms/frame max |

**TFLOPS (Tera FLOPs per second)** = nghìn tỉ phép tính/giây. RTX 4060 có ~15 TFLOPS, Pi 5 chỉ ~0.1 → 4060 mạnh gấp 150 lần.

Suy ra ràng buộc kiến trúc:
- **Params ≤ 2M** (mong muốn ~1M) — model lớn không vừa RAM hoặc chậm
- **FLOPs ≤ 5 GFLOPs ở 640px** hoặc ~1.5 GFLOPs ở 320px — Pi 5 chỉ làm ~10 GFLOPs/giây nên muốn 20 FPS thì model ≤ 0.5 GFLOPs/frame (khá khó)
- **No heavy attention ở high-resolution** — full self-attention có complexity O(N²) với N = số pixel. Ở P2 (160×160=25600 pixel), N² = 655M, quá đắt. Phải dùng window attention hoặc CNN
- **Phải fuse được BN** — BatchNorm khi inference có thể merge vào Conv weight, tiết kiệm 1 phép tính/pixel/channel

---

# Phần II: Lý do chọn từng kiến trúc

Mỗi module dưới đây trình bày theo cấu trúc:
1. **Vấn đề cần giải quyết**
2. **Module được chọn**
3. **Cách hoạt động chi tiết** (kèm ví dụ số cụ thể)
4. **Các phương án thay thế đã cân nhắc**
5. **Tại sao chọn cái này, KHÔNG chọn cái kia**

---

## 2.1. Stem (lớp đầu vào): EdgeContextStem

### Vấn đề
Lớp đầu của detector phải chuyển ảnh RGB (3 channel) thành feature map có nhiều channel hơn (24-32). Đây là **bước downsample đầu tiên** — quyết định bao nhiêu thông tin được giữ lại từ pixel gốc.

Nếu downsample quá mạnh ngay đầu (vd 16× như ViT) → người tiny 8 pixel sẽ bị mất hoàn toàn (8/16 < 1 pixel sau downsample). Nếu không downsample → feature map quá lớn, các layer sau không kham nổi compute.

### Module đã chọn

```
Input ảnh 3×640×640
  ├─ Local path:   Conv3×3(stride=2) → Conv3×3(stride=1)   → 12 channel, 320×320
  └─ Context path: AvgPool(stride=2) → Conv3×3             → 12 channel, 320×320
                                                    ↓
                              Concat (theo chiều channel) → tensor 24×320×320
                                                    ↓
                              Conv1×1 fuse (trộn 2 nguồn thông tin)
                                                    ↓
                                       output: 24 channel, 320×320
```

### Cách hoạt động chi tiết

**Bước 1: Local path** xử lý chi tiết cục bộ.
- Conv3×3 stride=2: kernel 3×3 trượt qua ảnh với stride=2 → output kích thước 320×320 (giảm 2× từ 640)
- Mỗi pixel output "nhìn" 3×3 = 9 pixel input → bắt được edge, gradient màu, texture nhỏ
- Conv3×3 stride=1: refine thêm, mở rộng receptive field lên 5×5

**Bước 2: Context path** lấy ngữ cảnh rộng.
- AvgPool stride=2: thay vì học, chỉ lấy giá trị trung bình của 2×2 pixel → output 320×320
- AvgPool tốt hơn MaxPool ở đây vì giữ thông tin "trung bình" toàn vùng
- Conv3×3 stride=1: học pattern từ giá trị trung bình → mỗi pixel output đại diện vùng 6×6 input

**Bước 3: Concat + Fuse**
- Hai tensor 12×320×320 ghép theo channel → 24×320×320
- Conv1×1 trộn 24 channel này → output cuối cùng là sự pha trộn local detail và context

**Ví dụ số cụ thể:** với ảnh 640×640 đầu vào:
- Local path FLOPs: `320×320 × 3×12 × 9 + 320×320 × 12×12 × 9 = 14.7M`
- Context path FLOPs: `320×320 × 4 (avgpool) + 320×320 × 3×12 × 9 = 11M`
- Fuse FLOPs: `320×320 × 24×24 = 59M`
- **Tổng: ~85M FLOPs** = 0.085 GFLOPs (rất rẻ so với budget 5 GFLOPs)

### Các phương án đã cân nhắc

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **A. Plain Conv stem** (YOLOv3-v5): 1× Conv 6×6 stride 2 | Receptive field chỉ 6×6 pixel input. Người 8 pixel + background → 36 pixel total, model không phân biệt được người vs noise |
| **B. Patch Embed** (ViT style): Conv 16×16 stride 16 | Downsample 16× ngay đầu → ảnh 640 thành 40×40. Người 8 pixel bị "compress" vào 0.5 pixel → mất thông tin. Phù hợp cho ảnh ImageNet (object lớn) nhưng không cho drone |
| **C. Focus layer** (YOLOv5 đời cũ): pixel shuffle 4 patches → concat | Chỉ rearrange pixel, không học context. Thực chất là 1 trick để transfer info channel↔spatial nhưng không thông minh hơn Conv |
| **D. EdgeContextStem (đã chọn)** | Cân bằng: chi tiết + context, downsample chỉ 2× nên không mất tiny |

### Tại sao chọn EdgeContextStem
1. **Người drone rất nhỏ (8-50 px)** → cần giữ chi tiết → loại B (downsample 16× quá mạnh), C (không học context)
2. **Cần context** để phân biệt "đốm 8 pixel này là người hay noise" → loại A (chỉ local)
3. **Compute thấp:** 85M FLOPs = 1.7% budget. Dual-path "free" về compute
4. **Đã verify thực nghiệm:** trong v14, thay stem này bằng plain Conv → AP giảm ~0.02

---

## 2.2. Backbone block: UIB (Universal Inverted Bottleneck)

### Vấn đề
Backbone phải học representation từ shallow (cạnh, texture) đến deep (semantic, "đây là người"). Block dùng phải:
- **Đủ expressiveness** (đa dạng pattern học được)
- **FLOPs thấp** (vì lặp lại nhiều lần — backbone có 10-20 block)
- **Hoạt động được ở mọi scale** (P2 đến P5)
- **Linh hoạt** (downsample stride=2 hay refine stride=1)

### Module đã chọn: UIB từ MobileNetV4 (ECCV 2024)

```
Input C channel
  ↓ DW Conv 3×3 (stride hoặc =1)         (chi tiết spatial, "nhìn" vùng 3×3)
  ↓ BN → SiLU
  ↓ PW Conv 1×1: C → 4C                  (mở rộng channel "expand")
  ↓ BN → SiLU
  ↓ FactDW 5×5 (optional, dùng cuối stage)  (mở rộng receptive field)
  ↓ BN → SiLU
  ↓ PW Conv 1×1: 4C → out_C              (thu hẹp channel "project")
  ↓ BN
  ↓ + skip connection (nếu cùng shape)
```

### Cách hoạt động chi tiết

**Tên "Inverted Bottleneck"** ngược với ResNet:
- ResNet block: `wide → narrow (bottleneck) → wide` (squeeze ở giữa)
- IRB / UIB: `narrow → wide → narrow` (expand ở giữa)

**Tại sao expand ở giữa?**

Lý thuyết: feature space rộng hơn → model học được nhiều pattern đa dạng hơn. Nhưng nếu Conv chuẩn ở giữa thì FLOPs sẽ × 16 (4× channel × 4× channel). UIB giải quyết bằng cách dùng DW Conv ở giữa — DW chỉ tốn `H × W × 4C × 9` thay vì `H × W × 4C × 4C × 9`.

**Ví dụ cụ thể với C=24, output 24, kích thước 80×80:**

| Bước | Operation | FLOPs |
|------|-----------|-------|
| 1. DW 3×3 | `80×80 × 24 × 9 = 1.38M` | Học spatial pattern |
| 2. PW 1×1 expand | `80×80 × 24 × 96 = 14.7M` | Tăng channel 4×, học feature mới |
| 3. FactDW 1×5 | `80×80 × 96 × 5 = 3.07M` | Học pattern hàng dài |
| 4. FactDW 5×1 | `80×80 × 96 × 5 = 3.07M` | Học pattern cột dài |
| 5. PW 1×1 project | `80×80 × 96 × 24 = 14.7M` | Compress về 24 channel |
| **Tổng** | | **~37M FLOPs** |

So với Conv 3×3 chuẩn: `80×80 × 24 × 24 × 9 = 13.8M` × N block. UIB tốn nhiều FLOPs hơn 1 Conv chuẩn nhưng học được pattern phong phú hơn nhiều. Trade-off này đã được paper MobileNetV4 chứng minh tốt hơn cho ImageNet.

**FactDW (Factorized Depthwise) chi tiết:**
- DW 5×5 chuẩn: `H × W × C × 25` (mỗi pixel × 25 phép)
- Factorized = chia thành DW 1×5 (chiều ngang) + DW 5×1 (chiều dọc)
- Total: `H × W × C × 5 + H × W × C × 5 = H × W × C × 10`
- Giảm 60% compute, nhưng vẫn capture được pattern 5×5 (gần đúng)
- Trade-off chấp nhận được: paper MobileNetV4 chứng minh AP drop <0.5% trên COCO

**SiLU (Swish):** activation function `f(x) = x × sigmoid(x)`. Khác ReLU ở chỗ smooth (đạo hàm liên tục), không bị "dead neuron". Hơi tốn compute hơn ReLU nhưng tăng AP ~0.5-1%.

### Các phương án đã cân nhắc

| Lựa chọn | FLOPs | Vấn đề chi tiết |
|----------|------|----------------|
| **A. Plain Conv 3×3** | ~10× cao hơn | Conv chuẩn `H×W×C×C×9` rất đắt. Với C=24 và 80×80 = 13.8M cho 1 Conv, mà cần 4-6 Conv/stage → quá đắt |
| **B. MobileNetV1 Depthwise** (chỉ DW + PW, không expand) | 0.7× | Không có expand → expressiveness thấp. Paper MobileNetV1 đạt 70.6% ImageNet, IRB (V2) đạt 72%, UIB (V4) đạt 73.8% cùng FLOPs |
| **C. MobileNetV2 IRB** (Inverted Residual Block) | 0.95× | Gần giống UIB nhưng không có FactDW optional. UIB là superset của IRB |
| **D. ConvNeXt block** (DW 7×7 + 2× MLP) | 1.5× | DW 7×7 quá tốn. Ở 80×80 với C=24: `80×80 × 24 × 49 = 7.5M` chỉ cho 1 DW. ConvNeXt nhằm ImageNet 1000 class, không phù hợp nano |
| **E. EfficientNet MBConv** (IRB + SE attention) | 1.1× | SE = Squeeze-Excite attention thêm overhead nhỏ. v15 đã có LSK ở chỗ chiến lược (P2, before head), không cần SE thêm |
| **F. ShuffleNet block** (channel shuffle + group conv) | 0.85× | Channel shuffle phức tạp khi deploy, không fuse được tốt |
| **G. UIB (đã chọn)** | 1.0× (baseline) | Cân bằng: SOTA accuracy + linh hoạt extra_dw |

### Tại sao chọn UIB
1. **Best accuracy/FLOPs tradeoff trên ImageNet:**
   - MobileNetV4 paper Table 3: UIB-S đạt 73.8% top-1 với 200 MFLOPs
   - MBConv-S: 72.5% với 200 MFLOPs → UIB hơn 1.3%
   - IRB-S: 72.0% với 200 MFLOPs → UIB hơn 1.8%
2. **Linh hoạt với extra_dw:** với block cuối mỗi stage, bật `extra_dw=True` để tăng receptive field từ 3×3 → 5×5 effective. Chỉ tốn thêm 5% FLOPs nhưng quan trọng cho người vừa
3. **Hoạt động tốt ở mọi scale:** stride=2 cho downsample (chuyển P2→P3), stride=1 cho refine (cùng scale)
4. **Lý do KHÔNG chọn:**
   - **Không ConvNeXt:** quá nặng, model sẽ vượt 1.5M params (DW 7×7 ở mọi block)
   - **Không MBConv+SE:** SE attention chỉ học channel-wise, có LSK rồi không cần
   - **Không plain Conv:** chậm 10× trên Pi 5, không thể realtime
   - **Không ShuffleNet:** channel shuffle khó tối ưu trên ARM, deploy chậm

---

## 2.3. CSP (Cross Stage Partial) — bao bọc UIB

### Vấn đề
Khi xếp 4-6 UIB block liên tiếp ở mỗi stage (gọi là "stack" hay "stage"), model nhỏ gặp 2 vấn đề:

1. **Gradient vanishing (gradient biến mất):**
   Gradient backprop qua nhiều layer bị nhân liên tiếp với derivative của activation → sau 6 layer, gradient có thể nhỏ ~0.001× original. Layer đầu nhận gradient cực nhỏ → không học được.

2. **Feature collapse (sụp đổ feature):**
   Ở model nhỏ ~1M params, capacity từng block hạn chế. Block sau "compress" feature từ block trước → mất chi tiết. Sau 6 block, info gốc gần như mất hết.

### Module đã chọn: CSPUIBStage (CSP wrapper từ YOLOv5)

```
Input
  ↓ UIBBlock(stride=2)              # downsample trước (giảm 2× kích thước)
Output: out_c channel
  ├─ Branch A: Conv1×1 → out_c/2    # nhánh học (xử lý info)
  │           ↓
  │           UIB × n_blocks         # qua n block UIB
  │           ↓
  │           out_c/2
  └─ Branch B: Conv1×1 → out_c/2    # nhánh shortcut (chỉ project channel)
            ↓
            out_c/2
  ↓ Concat → ghép theo channel → out_c
  ↓ Conv1×1 fuse → out_c           # trộn 2 nguồn info
```

### Cách hoạt động chi tiết

**Ý tưởng cốt lõi:** "Tách feature làm hai nửa. Một nửa qua các block học (Branch A). Một nửa đi tắt (Branch B). Cuối cùng ghép lại."

**Ví dụ với out_c=72, n_blocks=2, kích thước 80×80:**

Branch A:
- Conv 1×1: 72 → 36 channel
- UIB×2: mỗi UIB ~37M FLOPs với 36 channel → 74M total
- Output: 36 channel

Branch B:
- Conv 1×1: 72 → 36 channel
- Tổng: ~17M FLOPs
- Output: 36 channel (gần như info gốc)

Concat: 36+36 = 72 channel
Fuse Conv 1×1: 72 → 72 → ~33M FLOPs

**Tổng CSP stage:** ~135M FLOPs

**So với Sequential UIB×2 thẳng:** UIB cần xử lý 72 channel → mỗi UIB ~150M FLOPs → 2 UIB = 300M FLOPs.

→ **CSP tiết kiệm 55% FLOPs** so với Sequential cùng số UIB.

**Lợi ích cụ thể:**

1. **Branch A** chỉ xử lý ½ channel → tiết kiệm rõ rệt FLOPs cho phần "nặng" (UIB)

2. **Branch B** giữ nguyên info gốc (chỉ project) → khi concat, model có cả "feature đã học sâu" (A) và "feature thô" (B). Đảm bảo info không mất qua nhiều block

3. **Gradient flow:**
   - Trong backprop, gradient từ output → fuse → split thành gradient cho A và B
   - Gradient cho B chỉ qua 1 Conv 1×1 (đường ngắn) → không bị suy yếu
   - Gradient cho A qua n UIB block (đường dài) → bị suy yếu nhưng vẫn có gradient qua B làm "anchor"
   - → Layer đầu vẫn nhận được gradient hợp lý

### Các phương án đã cân nhắc

| Lựa chọn | FLOPs | Vấn đề chi tiết |
|----------|-------|-----------------|
| **A. Sequential UIBs** (v14 dùng): xếp `n × UIB` thẳng | 100% | Gradient suy yếu qua n block. Ở model 1M params, sau 4-6 block dễ collapse. v14 baseline AP=0.385 thấp |
| **B. Plain Residual** (ResNet style): `output = block(x) + x` | ~95% | Skip connection có nhưng không tách channel → không tiết kiệm FLOPs phần học. Branch shortcut trùng channel với main → info redundant |
| **C. DenseBlock** (DenseNet): mọi block concat với mọi block trước | 200%+ | Ở block thứ k, input có k×c channel → bùng nổ. Memory tăng tuyến tính theo block. Pi 5 RAM 4GB không feasible |
| **D. C3** (YOLOv5 specific variant) | ~100% | Variant của CSP nhưng phức tạp hơn (3 nhánh thay 2). Ý tưởng tương tự, không cần thiết cho 1M params |
| **E. ELAN** (YOLOv7) | ~110% | Multi-branch sâu hơn CSP. Tốt cho model lớn nhưng overkill cho nano |
| **F. CSP (đã chọn)** | 45-55% | Tách kênh + concat → vừa giảm FLOPs vừa preserve gradient |

### Tại sao chọn CSP
1. **Đã verify thực nghiệm trên dự án này:**
   - v14 (Sequential): AP=0.385 (best run)
   - v15 (CSP): AP=0.557 (cùng config khác)
   - **Cải thiện +0.172 AP (+44.7% relative)**
2. **Không tăng params:** Branch B chỉ là Conv1×1 nhỏ, total params gần như không đổi. Thực tế v15 light có **ít params hơn v14** (1.0M vs 1.16M)
3. **Giảm FLOPs ~30%** so với Sequential cùng số UIB block
4. **Lý do KHÔNG chọn alternatives:**
   - **Không Sequential:** đã chứng minh kém ở v14, AP thấp
   - **Không Residual đơn giản:** không tận dụng được tiết kiệm FLOPs
   - **Không DenseNet:** memory explosion, Pi 5 RAM 4GB không chịu được
   - **Không C3/ELAN:** CSP đơn giản đủ, không cần phức tạp ở model nano

### Tại sao P2 KHÔNG dùng CSP, chỉ P3-P5 dùng?
Đây là chi tiết tinh tế:
- **P2 ở stride 4** = chứa **95% positive trong VisDrone** (người tiny)
- Ở P2, **mọi pixel đều quan trọng**, tách channel sẽ làm mất chi tiết
- P3-P5 sâu hơn, semantic hơn → split channel không hại lắm

Cụ thể trong code v15:
```python
self.stage_p2 = nn.Sequential(  # P2 dùng Sequential — giữ chi tiết
    UIBBlock(c1, c2, stride=2),
    UIBBlock(c2, c2),
    UIBBlock(c2, c2, extra_dw=True),
)
self.stage_p3 = CSPUIBStage(c2, c3, n_blocks=2, stride=2, extra_dw=True)  # P3+ dùng CSP
```

---

## 2.4. LSKBlock (Large Selective Kernel) — chuyên cho aerial

### Vấn đề
Trong cùng một ảnh drone, người xuất hiện ở **nhiều scale rất khác nhau**:
- Người gần (drone bay thấp 30m): 100-200 pixel
- Người trung bình: 30-60 pixel
- Người xa (drone bay cao 150m): 8-15 pixel

Một kernel cố định (vd 3×3) không thể tốt cho cả ba scale:
- Kernel 3×3 nhìn vùng 3×3 pixel input → đủ cho người tiny 8-15px (xem được toàn bộ thân) nhưng quá nhỏ cho người 100px (chỉ thấy 3% thân)
- Kernel 7×7 nhìn vùng 7×7 → đủ cho người trung bình nhưng lãng phí cho người tiny

### Module đã chọn: LSKBlock từ LSKNet (ICCV 2023, IJCV 2024)

```
Input x (C channel)
  ├─ DW Conv 5×5 (padding=2)            → a0 (kernel nhỏ, focus chi tiết, RF=5×5)
  └─ DW Conv 7×7 dilated=3 (padding=9)  → a1 (RF=19×19, ngữ cảnh rộng)
        ↓
        Concat([a0, a1]) → 2C channel
        ↓
        Attention gate:
          Conv1×1: 2C → C/2  (squeeze)
          Conv1×1: C/2 → 2   (xuất 2 trọng số)
          softmax theo channel → α0, α1 (mỗi pixel có trọng số riêng)
        ↓
        weighted = α0 × a0 + α1 × a1
        ↓
        Conv1×1 project (refine)
        ↓
        + skip connection (output = above + x)
```

### Cách hoạt động chi tiết

**"Hai chuyên gia, một quản lý."**
- **Chuyên gia 1 (DW 5×5):** chuyên xem chi tiết. Receptive field 5×5 pixel input. Phù hợp cho tiny object
- **Chuyên gia 2 (DW 7×7 dilated=3):** chuyên xem toàn cảnh. Dilation=3 nghĩa là kernel 7×7 mà các pixel cách nhau 3 ô → effective kernel = `7 + (7-1)×(3-1) = 19` → receptive field 19×19. Phù hợp cho người lớn
- **Quản lý (attention gate):** cho từng pixel, học trọng số α0 và α1 → quyết định nghe chuyên gia nào nhiều hơn

**Tại sao dùng dilation thay 7×7 thường?**
- DW 7×7 thường: FLOPs = `H×W × C × 49`
- DW 7×7 dilated=3: FLOPs cũng = `H×W × C × 49` (chỉ thay đổi cách trượt)
- Nhưng dilated **không tăng số phép tính** mà **mở rộng receptive field 19×19**
- Free lunch về receptive field

**Attention gate hoạt động cụ thể:**

Ví dụ pixel ở vùng người tiny 8×8 px:
- a0 (DW 5×5): học được "đây là cụm pixel sáng nhỏ" — feature mạnh
- a1 (DW 7×7 dilated): nhìn vùng 19×19 thấy chủ yếu background → feature yếu/nhiễu
- Attention gate: thấy a0 mạnh hơn → α0=0.85, α1=0.15
- Output: 0.85×a0 + 0.15×a1 → chủ yếu lấy info từ chuyên gia chi tiết

Ví dụ pixel ở vùng người 100×100 px:
- a0 (DW 5×5): chỉ thấy 5×5 trong người → feature local, có thể chỉ là 1 phần áo
- a1 (DW 7×7 dilated): thấy 19×19 → có context toàn thân → feature mạnh hơn
- Attention gate: α0=0.30, α1=0.70
- Output chủ yếu từ chuyên gia toàn cảnh

**Skip connection cuối:** `output = projected_weighted + x`. Đảm bảo gradient flow tốt và info gốc không mất.

### Các phương án đã cân nhắc

| Lựa chọn | FLOPs (so với LSK) | Vấn đề chi tiết |
|----------|----|------|
| **A. SE attention** (Squeeze-Excite, CVPR 2018) | 0.5× | Chỉ channel attention: `Pool → FC → FC → Sigmoid → scale channel`. Không có spatial selection → mọi pixel cùng channel được scale như nhau. Không thích nghi với scale của object |
| **B. CBAM** (ECCV 2018): channel + spatial attention | 0.7× | Tốt hơn SE (có spatial map) nhưng chỉ 1 kernel size cho spatial. Không multi-scale như LSK |
| **C. Multi-Head Self-Attention** (ViT) | 50-100× cao hơn | O(N²) compute. Ở P2 (160×160=25600 pixel), 1 head attention = 655M op. Quá đắt, không feasible |
| **D. ASPP** (DeepLab): 4-6 nhánh dilated parallel | 3-4× | Dùng cho semantic segmentation. Mỗi nhánh là 1 dilated rate khác nhau (1, 6, 12, 18). Quá đắt cho Pi 5 |
| **E. SK (Selective Kernel, CVPR 2019)** — tổ tiên của LSK | 1.2× | Tương tự LSK nhưng dùng kernel 3×3 và 5×5, không dilated. Receptive field nhỏ hơn, ít phù hợp aerial |
| **F. LSK (đã chọn)** | 1.0× | 2 nhánh + dilated → receptive field rộng + compute thấp |

### Tại sao chọn LSK
1. **Domain match (cùng lĩnh vực):**
   - LSKNet được thiết kế cho **aerial/satellite imagery** (cùng domain với drone)
   - Đã được prove trên DOTA (Dataset for Object Detection in Aerial Images), ImageNet-Aerial
   - Drone person detection ≈ aerial detection → kế thừa hết lợi ích
2. **Compute thấp:**
   - 2 DW Conv + 2 Conv1×1
   - Tổng <50K FLOPs cho feature 24×320×320 (P2 stride 4)
   - Trong khi self-attention sẽ là 655M FLOPs → LSK rẻ hơn **13000×**
3. **Adaptive (thích nghi):**
   - Khác fixed kernel, gate học được trọng số tối ưu cho từng vị trí
   - Không cần tune hyperparameter — model tự học
4. **Lý do KHÔNG chọn alternatives:**
   - **Không SE:** thiếu spatial selection, không phù hợp cho tiny vs big mixed trong cùng ảnh
   - **Không CBAM:** không multi-scale, chỉ 1 kernel cho spatial attention
   - **Không Self-Attention:** O(N²) ở P2 quá đắt, sẽ ăn 50% compute budget
   - **Không ASPP:** 4-6 nhánh = 2-3× FLOPs của LSK, không feasible
   - **Không SK:** receptive field nhỏ hơn, không phù hợp aerial vốn cần context rộng

### Vị trí đặt LSK trong v15
- **Sau P2 backbone (1 chỗ):** tăng cường tiny person ngay sau khi rời backbone, trước khi vào neck
- **Trước mỗi DecoupledHead (4 chỗ ở P2, P3, P4, P5):** final refinement ngay trước khi predict

Tổng có 5 LSKBlock trong v15. Tổng FLOPs ~1M (rất rẻ).

---

## 2.5. AreaAttention (Window Self-Attention)

### Vấn đề
Self-attention tiêu chuẩn (như Transformer) cho phép **mọi pixel nhìn mọi pixel** → tốt nhất về expressiveness nhưng compute O(N²).

**Ví dụ cụ thể compute self-attention:**
- N = số pixel = H × W
- Attention matrix: N × N
- Mỗi entry là dot product của 2 vector C-dim
- FLOPs = N × N × C × 2 (cho QK^T)

| Scale | H×W | N | N² | FLOPs (C=64) |
|-------|-----|---|-----|--------------|
| P5 | 20×20 | 400 | 160K | 20M (rẻ) |
| P4 | 40×40 | 1600 | 2.5M | 320M (chấp nhận được) |
| P3 | 80×80 | 6400 | 40M | 5G (đắt) |
| P2 | 160×160 | 25600 | 655M | 84G (không feasible) |

→ Không thể bật full attention ở high-res. Nhưng **không có attention thì thiếu global context** ở deeper layers.

### Module đã chọn: AreaAttentionBlock từ Swin Transformer (ICCV 2021 Best Paper)

```
Input feature B×C×H×W
  ↓ Pad H, W cho chia hết area (vd area=3): pH = (3 - H%3) % 3
  ↓ Reshape: (B, C, nH, area, nW, area) → (B*nH*nW, area*area, C)
            chia thành nH × nW các "window" kích thước 3×3
  ↓ LayerNorm
  ↓ Linear: C → 3C, split → Q, K, V (mỗi cái C-dim)
  ↓ Multi-Head Self-Attention TRONG MỖI WINDOW
            attn = softmax(Q @ K^T / sqrt(d)) @ V
  ↓ Linear projection: C → C
  ↓ + skip connection (input + attention)
  ↓ LayerNorm
  ↓ FFN (Feed-Forward Network): Linear C → 2C → GELU → Linear 2C → C
  ↓ + skip connection
  ↓ Reshape ngược lại: (B, C, H, W)
  ↓ Bỏ pad
```

### Cách hoạt động chi tiết

**"Thay vì để mọi pixel trong ảnh nói chuyện với nhau (đắt), chia ảnh thành các vùng nhỏ 3×3, trong mỗi vùng các pixel được nói chuyện với nhau."**

**Bước 1: Window partitioning**
- Feature map H×W được chia thành nH × nW window, mỗi window kích thước area×area (3×3)
- Reshape về dạng `(B*nH*nW, 9, C)` — mỗi window có 9 pixel, mỗi pixel C-dim vector

**Bước 2: Multi-Head Self-Attention trong từng window**
- Q, K, V là 3 ma trận học được từ input bằng Linear (C→C mỗi cái)
- "Multi-head" nghĩa là chia C channel thành H_heads (vd 2 head, mỗi head 32-dim với C=64)
- Mỗi head tự attention độc lập, sau đó concat
- Trong window 3×3: 9 pixel, mỗi pixel attend với 9 pixel (kể cả chính nó)
- Compute = 9 × 9 = 81 cho mỗi window
- Total cho cả feature map: nH × nW × 81 = `(H/3) × (W/3) × 81`

**Bước 3: FFN (Feed-Forward Network)**
- 2 lớp Linear: C → 2C → C
- Activation GELU ở giữa (smooth version của ReLU)
- Mục đích: tăng capacity, cho mỗi pixel tự xử lý info sau khi đã attend với hàng xóm

**Bước 4: Reshape và unpad**
- Đảo lại thứ tự dim để output về dạng (B, C, H, W)
- Bỏ phần pad đã thêm ở đầu

**So sánh compute thực tế (P5 = 20×20, C=128, area=3):**
- Full attention: `(20×20)² × 128 × 2 = 41M op`
- Window attention 3×3: pad lên 21×21, có 49 window 3×3. Mỗi window: `9² × 128 × 2 = 18K`. Total: 49×18K = 880K op
- **Window attention rẻ hơn 47×**

**Ở P2 (160×160, C=24):**
- Full attention: 84G op (không khả thi)
- Window attention: ~10M op (khả thi nhưng vẫn không bằng LSK)

### Các phương án đã cân nhắc

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **A. Full Self-Attention** (ViT) | O(N²) — không khả thi ở high-res. P2 sẽ tốn 84G op |
| **B. Linear Attention** (Performer, Linformer): xấp xỉ với O(N) | Mất accuracy đáng kể. Linformer paper: -2 mAP trên COCO. Performer: -1.5 mAP |
| **C. CNN deeper** (thay attention bằng nhiều Conv hơn) | Receptive field tăng tuyến tính theo layer. Để đạt RF 40 cần 20 Conv 3×3. Quá nhiều layer |
| **D. SE/CBAM** (channel/light spatial attention) | Không phải attention thật giữa các pixel, không học được long-range dependency |
| **E. Shifted Window Attention** (Swin gốc): window + shift để giao tiếp giữa window | Phức tạp hơn (2 lớp: window + shifted window). Cải thiện 0.5-1% AP nhưng tốn 2× compute |
| **F. Window Attention không shifted (đã chọn)** | Cân bằng: gần như linear compute, vẫn có self-attention thật trong từng window |

### Tại sao chọn Window Attention
1. **Đã chứng minh trên ImageNet:**
   - Swin Transformer (ICCV 2021 Best Paper) đạt 87.3% top-1 với compute hợp lý
   - Tốt hơn ConvNeXt cùng FLOPs ~0.5%
2. **Hoạt động ở deeper scale:**
   - P4 (40×40 = 1600 pixel) và P5 (20×20 = 400 pixel) số pixel ít → window attention không tốn nhiều
3. **Compatible với Conv:**
   - Không cần thay toàn bộ backbone (đã có UIB rồi)
   - Chỉ chèn vào sau Conv stage → hybrid Conv + Attention model
4. **Lý do KHÔNG chọn alternatives:**
   - **Không Full Attention:** compute budget không cho phép
   - **Không Linear Attention:** accuracy giảm 1.5-2% mAP — không đáng đổi
   - **Không CNN deeper:** muốn add long-range context mà không tăng layer count
   - **Không Shifted Window:** phức tạp gấp đôi compute, gain chỉ 0.5%

### Tại sao chỉ đặt ở P4 và P5, KHÔNG ở P2 và P3?

| Scale | Số pixel | Window attention FLOPs | Có cần attention? |
|-------|----------|----------------------|-------------------|
| P5 (20×20) | 400 | ~800K | Có — semantic, cần global context |
| P4 (40×40) | 1600 | ~3M | Có — người vừa cần context |
| P3 (80×80) | 6400 | ~12M | Không — quá đắt, LSK đã handle |
| P2 (160×160) | 25600 | ~50M | Không — quá đắt, tiny person chủ yếu cần local detail |

### Khác biệt v14 vs v15
- **v14:** chỉ 1 AreaAttention ở P5
- **v15:** thêm 1 cái ở P4 → cải thiện global context cho người vừa
- Thêm 3M FLOPs nhưng tăng AP ~0.5%

---

## 2.6. BiFPN Neck (Bi-directional Feature Pyramid)

### Vấn đề
Backbone xuất ra 4 feature map ở 4 độ phân giải khác nhau:
- P2 (160×160): có chi tiết (cạnh, texture rõ), thiếu semantic (chưa biết "đây là người")
- P3 (80×80): trung gian
- P4 (40×40): trung gian
- P5 (20×20): có semantic (biết "đây là người"), thiếu chi tiết (vị trí mờ)

**Detection cần cả hai:**
- Người tiny ở P2: cần chi tiết để locate, cần semantic từ P5 để xác nhận "đây là người"
- Người lớn ở P5: cần semantic để classify, cần chi tiết từ P2 để locate chính xác

→ Phải fuse 4 scale này lại sao cho mỗi scale đều có cả semantic và chi tiết.

### Module đã chọn: BiFPN từ EfficientDet (CVPR 2020)

```
Top-down pass (semantic flow xuống):
  P5 ──────────────────────────────────────→ P5_intermediate
       ↓ upsample 2× (nearest neighbor)
  P4 ──────── weighted_fuse(P4, up(P5)) ───→ P4_td
       ↓ upsample 2×
  P3 ──────── weighted_fuse(P3, up(P4_td)) ─→ P3_td
       ↓ upsample 2×
  P2 ──────── weighted_fuse(P2, up(P3_td)) ─→ P2_out

Bottom-up pass (detail flow lên):
  P2_out ───────────────────────────────────→ P2_out (giữ nguyên)
       ↓ downsample 2× (Conv stride=2)
  P3_td ──── weighted_fuse(P3_td, down(P2_out)) → P3_out
       ↓ downsample 2×
  P4_td ──── weighted_fuse(P4_td, down(P3_out)) → P4_out
       ↓ downsample 2×
  P5_intermediate ─── weighted_fuse(P5_intermediate, down(P4_out))→ P5_out
```

### Cách hoạt động chi tiết

**"Hai chiều thông tin: semantic xuống, detail lên."**

**Bước 1: Top-down (semantic xuống)**
- Bắt đầu từ P5 (semantic mạnh nhất), upsample (phóng to) lên 40×40 = match P4
- Fuse với P4 gốc → P4_td có thêm semantic từ P5
- Upsample P4_td lên 80×80, fuse với P3 → P3_td
- Tiếp tục đến P2

Sau pass này, mọi scale đều có thêm "đây là người" info từ P5.

**Bước 2: Bottom-up (detail lên)**
- Bắt đầu từ P2 (detail mạnh nhất), downsample (thu nhỏ) bằng Conv stride=2
- Fuse với P3_td → P3_out có thêm chi tiết từ P2
- Tiếp tục đến P5

Sau pass này, mọi scale đều có thêm chi tiết từ P2.

**Weighted fusion (trọng số học được):**

Thay vì cộng đều `output = input1 + input2`, dùng:
```python
weights = nn.Parameter(torch.ones(2))  # [1.0, 1.0] khởi tạo
w = ReLU(weights)                       # đảm bảo >=0
w = w / (w.sum() + eps)                 # chuẩn hóa thành tỉ lệ
output = w[0] * input1 + w[1] * input2  # weighted sum
```

Trọng số `w` được học cùng với model. Sau training, có thể quan sát:
- Ở P2: w[P2] = 0.7, w[up(P3)] = 0.3 → P2 ưu tiên info từ chính nó (chi tiết)
- Ở P5: w[P5] = 0.6, w[up(P6)] = 0.4 → cân bằng hơn

**Ví dụ tăng AP từ weighted fusion:**
- Equal sum: AP=0.55
- Weighted sum: AP=0.557 (+0.7%)
- Đã prove trong EfficientDet paper

### Các phương án đã cân nhắc

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **A. FPN gốc** (CVPR 2017): chỉ top-down | Detail từ P2 không lan lên P5. Người lớn ở P5 vẫn dùng feature thô từ backbone, locate kém |
| **B. PANet** (CVPR 2018): top-down + bottom-up nhưng cộng đều | Có 2 chiều rồi nhưng không weighted → mọi vị trí cộng đều, không adaptive với từng pixel |
| **C. NAS-FPN** (search-based): kiến trúc neck được search bằng Neural Architecture Search | Kiến trúc cố định sau khi search trên dataset gốc (COCO). Không generalize tốt cho VisDrone (drone perspective khác) |
| **D. Recursive FPN** (DetectoRS): lặp FPN nhiều lần | Quá đắt cho nano model. Mỗi recursion tốn 1 BiFPN |
| **E. PAN** (Path Aggregation Network) — variant của PANet | Tương tự PANet, không weighted |
| **F. BiFPN (đã chọn)** | Hai chiều + weighted fusion + repeat 1-2× |

### Tại sao chọn BiFPN
1. **Đã được prove trên COCO:**
   - EfficientDet-D0 với BiFPN: 33.8 AP với 4M params
   - YOLOv3 với FPN: 33.0 AP với 65M params
   - → BiFPN tốt hơn 0.8 AP với **1/16 params**
2. **Weighted fusion** quan trọng cho **adaptive scale handling:**
   - Mỗi vị trí có thể cần P2 nhiều hơn hoặc P5 nhiều hơn
   - Vùng có nhiều người lớn → ưu tiên P5
   - Vùng có người tiny → ưu tiên P2
3. **Repeat-able (lặp lại được):**
   - Light profile dùng `num_bifpn=1` (1 lớp)
   - Balanced dùng `num_bifpn=2` (2 lớp, AP cao hơn nhưng chậm)
   - Linh hoạt scale up/down
4. **Lý do KHÔNG chọn alternatives:**
   - **Không FPN gốc:** thiếu bottom-up, người lớn locate kém
   - **Không PANet:** không adaptive, AP thấp hơn BiFPN ~1%
   - **Không NAS-FPN:** phụ thuộc dataset, kiến trúc cố định không match drone domain
   - **Không Recursive:** quá đắt — chỉ phù hợp model lớn

### Trong v15 light
- `num_bifpn = 1` → chỉ 1 lớp BiFPN (đủ cho người, nano budget)
- `num_bifpn = 2` ở balanced profile

---

## 2.7. RepConv (Re-parameterizable Convolution)

### Vấn đề
**Nghịch lý:** train tốt cần model rộng/sâu (nhiều branch), inference nhanh cần model gọn (single-branch). Làm sao có cả hai?

Cụ thể:
- Multi-branch (vd 3 nhánh song song) → train AP cao hơn ~1-2% nhưng inference chậm gấp 3
- Single-branch → inference nhanh nhưng train kém hơn

### Module đã chọn: RepConv từ RepVGG (CVPR 2021)

```
Train time (3 nhánh song song):
  Input
    ├─ Conv 3×3 + BN
    ├─ Conv 1×1 + BN  (sẽ zero-pad lên 3×3 lúc fuse)
    └─ Identity + BN  (chỉ khi cùng input/output shape)
    ↓
    Sum (cộng 3 output)
    ↓
    Activation (SiLU)

Deploy time (sau gọi reparameterize()):
  Input
    ↓ Conv 3×3 (đã merge cả 3 nhánh) + bias
    ↓ Activation
```

### Cách hoạt động chi tiết

**Trong toán học, một loạt phép tính tuyến tính có thể gộp lại thành một phép.**

Cụ thể: Conv + BN, và sum của nhiều Conv → một Conv duy nhất tương đương.

**Magic của RepVGG:**
1. **Train:** 3 nhánh học song song
   - Mỗi nhánh học pattern khác nhau
   - Sum = ensemble nhỏ → diversity, capacity cao hơn
   - Có residual (identity branch) → gradient flow tốt
2. **Deploy:** gọi `reparameterize()` → gộp 3 nhánh thành 1 weight matrix duy nhất
3. **Kết quả:** train như multi-branch, deploy như single-branch

### Cách fuse cụ thể (với ví dụ số)

Giả sử:
- Conv 3×3 weight: W3 shape (out_c, in_c, 3, 3), bias b3
- Conv 1×1 weight: W1 shape (out_c, in_c, 1, 1), bias b1
- Identity (chỉ khi in_c=out_c): W_id = identity matrix shape (out_c, in_c, 3, 3) chỉ có 1 ở vị trí trung tâm

**Bước 1: Fuse Conv 1×1 thành Conv 3×3 tương đương**
```python
# W1 shape (out_c, in_c, 1, 1) → pad zeros để thành (out_c, in_c, 3, 3)
W1_padded = F.pad(W1, [1, 1, 1, 1])  # pad 1 mỗi phía
# Bây giờ W1_padded chỉ có giá trị ở vị trí [:,:,1,1] (tâm)
```

Conv 1×1 chỉ ảnh hưởng pixel center, tương đương Conv 3×3 với 8 pixel xung quanh = 0.

**Bước 2: Fuse BN vào Conv weight**

BN công thức: `output = γ × (input - μ) / sqrt(σ² + ε) + β`
Có thể viết: `output = (γ / sqrt(σ² + ε)) × input + (β - γ × μ / sqrt(σ² + ε))`
= `scale × input + shift`

Khi fuse với Conv:
```python
new_weight = old_weight * scale.view(-1, 1, 1, 1)
new_bias = old_bias * scale + shift  # nếu Conv có bias
```

**Bước 3: Cộng 3 fused weights**
```python
W_fused = W3_fused + W1_padded_fused + W_id_fused
b_fused = b3_fused + b1_fused + b_id_fused
```

**Bước 4: Tạo Conv mới với weight đã fuse**
```python
self.reparam_conv = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=True)
self.reparam_conv.weight.data = W_fused
self.reparam_conv.bias.data = b_fused
```

**Verify:** chạy test forward với input random → output trước và sau reparameterize phải giống nhau (sai số <1e-5 do float precision).

### Các phương án đã cân nhắc

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **A. Plain Conv 3×3** | Đơn giản nhưng AP thấp hơn RepVGG ~1-2% trên ImageNet/COCO. Mất AP miễn phí |
| **B. ResBlock** (skip connection lưu khi deploy) | Skip connection cần phép cộng riêng → không fuse được. Deploy chậm hơn |
| **C. Inception** (multi-branch không fuse) | Multi-branch tốt khi train nhưng phải chạy multi-branch luôn ở inference → slow |
| **D. DBB** (Diverse Branch Block — RepVGG mở rộng) | Phức tạp hơn (5+ nhánh) nhưng gain chỉ 0.3-0.5% so với RepConv |
| **E. RepConv (đã chọn)** | Multi-branch khi train, single-branch khi deploy |

### Tại sao chọn RepConv
1. **Free lunch:** train tốt hơn mà deploy không chậm hơn
2. **Đã prove:**
   - RepVGG đạt ImageNet 80.5% top-1 với speed như VGG plain
   - Nhanh hơn ResNet-50 1.5× ở cùng accuracy
3. **Đặc biệt quan trọng cho Pi 5:**
   - Mỗi ms ở inference đều quý
   - Single-branch deploy → có thể dùng ONNX optimization tốt hơn
4. **Lý do KHÔNG chọn alternatives:**
   - **Không Plain Conv:** mất AP miễn phí
   - **Không ResBlock:** skip connection không fuse được
   - **Không Inception:** phải chạy multi-branch luôn ở inference
   - **Không DBB:** phức tạp hơn cần thiết, gain nhỏ

---

## 2.8. DecoupledHead (tách classification và regression)

### Vấn đề
Head detection cần xuất 2 thứ:
1. **Classification (cls):** "có người ở đây không?" → 1 output (score 0-1)
2. **Regression (reg / box):** "box ở đâu?" → 4 output (cx, cy, w, h relative)

Cách dễ nhất là 1 conv xuất cả 5 channel. Nhưng có vấn đề:

**Conflict gradient (xung đột gradient):**
- Cls gradient muốn feature **invariant theo vị trí** (xê dịch chút vẫn là "người")
- Box gradient muốn feature **variant theo vị trí** (xê dịch 1 pixel thì box phải đổi 1 pixel)
- Hai gradient có hướng ngược nhau → feature tối ưu cho cả hai khó học → AP thấp

### Module đã chọn: DecoupledHead từ YOLOX (NeurIPS 2021 W)

```
Input feature từ neck (vd 48 channel ở P2)
  ↓ Shared ConvBN (1 lớp)
  ↓ output: 48 channel
  │
  ├─ cls_branch (chuyên cho classification):
  │  ConvBN(48→48) → ConvBN(48→48) → Conv(48→1) → output 1 channel = obj logit
  │
  └─ box_branch (chuyên cho regression):
     ConvBN(48→48) → ConvBN(48→48) → Conv(48→4) → output 4 channel = dx, dy, log_w, log_h
     
Cuối cùng:
  ↓ Concat([cls, box]) → 5 channel total
```

### Cách hoạt động chi tiết

**"Hai chuyên gia khác nhau, không nhập nhằng nhau."**
- Sau shared layer (chia sẻ low-level features), tách thành 2 branch riêng biệt
- Cls branch tự do tối ưu cho task classification
- Box branch tự do tối ưu cho task regression
- Gradient không xung đột — mỗi branch chỉ nhận gradient cho task của mình

**Output cụ thể:**

Cls output (1 channel): logit (số thực) → qua sigmoid → score 0-1
- Score > threshold (vd 0.5) → predict "có người"
- Score < threshold → "không có người"

Box output (4 channel): dx, dy, log_w, log_h
- dx, dy: offset từ tâm cell (giá trị ~ -1 đến 1, qua sigmoid - 0.5)
  - Tâm box = (cell_x + dx) × stride
- log_w, log_h: log của width, height (giá trị thực)
  - width = exp(log_w) × stride

**Tại sao predict log_w thay vì w trực tiếp?**
- Width là giá trị positive → dùng exp đảm bảo positive
- log space ổn định hơn cho gradient (gradient của exp nhỏ ở giá trị nhỏ, lớn ở giá trị lớn — phù hợp loss)

### Các phương án đã cân nhắc

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **A. Coupled Head** (YOLOv3-v5): 1 Conv xuất cả 5 channel | Gradient conflict. YOLOX paper: switch coupled→decoupled tăng từ 38.5→39.6 AP trên COCO (+1.1) |
| **B. Anchor-based head** (Faster R-CNN, RetinaNet): cần định nghĩa anchor sizes trước | Phải tune anchor cho từng dataset. Với tiny person 8-16 px khó set anchor hợp lý |
| **C. Separate networks** (2 networks hoàn toàn riêng) | Tốn 2× compute, không share base feature |
| **D. DETR-style** (transformer query) | Quá đắt cho nano, transformer query chậm |
| **E. Decoupled với shared base (đã chọn)** | Cân bằng: tách branch nhưng share 1 lớp Conv base |

### Tại sao chọn DecoupledHead
1. **YOLOX paper đã chứng minh:**
   - Coupled head: 38.5 AP trên COCO val
   - Decoupled head: 39.6 AP (+1.1)
   - Convergence cũng nhanh hơn (đạt AP cao sớm hơn)
2. **Compute tăng ít:**
   - Chỉ thêm 2 Conv 1×1 mỗi branch
   - Tổng ~5% FLOPs increase
3. **Anchor-free:**
   - Không cần định nghĩa anchor → không tune hyperparameter này
   - Đặc biệt tốt cho tiny person (anchor-based hay miss tiny do quantization)
4. **Lý do KHÔNG chọn alternatives:**
   - **Không Coupled:** mất 1.1 AP miễn phí
   - **Không Anchor-based:** với tiny person, anchor sizes hay sai
   - **Không Separate networks:** lãng phí compute
   - **Không DETR:** quá đắt, không phù hợp nano

### Trick init bias quan trọng

Cls Conv bias init = **-5.5**, không phải 0 như default.

**Lý do toán học:**
- `sigmoid(-5.5) ≈ 0.004` → ban đầu dự đoán "không có người" cho mọi pixel
- Trong 1 ảnh 640×640 với P2 stride 4, có 160×160 = **25,600 pixel candidate**
- Chỉ ~50 là positive (người), còn 25,550 là negative
- Tỉ lệ positive : negative = 1 : 511 (cực kỳ imbalance)

**Nếu init bias=0:**
- Default predict 0.5 cho mọi pixel
- BCE loss của negative pixel: `-log(1 - 0.5) = 0.69` × 25,550 = **17,629**
- BCE loss của positive: `-log(0.5) = 0.69` × 50 = 34.5
- Total loss ≈ 17,663
- Gradient cực lớn → model "shock" ngay đầu, train không ổn định

**Với init bias=-5.5:**
- Default predict 0.004 cho mọi pixel
- BCE loss của negative: `-log(1 - 0.004) = 0.004` × 25,550 = 102
- BCE loss của positive: `-log(0.004) = 5.52` × 50 = 276
- Total loss ≈ 378 (giảm 47×)
- Gradient hợp lý, training warmup ổn định

**Hậu quả thực tế nếu init sai:**
- Trong v12/v13 của dự án này (ban đầu), init sai → train ~30 epoch, precision = 0
- Sửa init bias=-5.5 → train hội tụ ngay từ epoch 5-10

---

# Phần III: Loss function — tại sao nhiều thành phần?

Loss của detection phức tạp vì phải dạy model cả 2 task (cls + box) đồng thời. Mỗi thành phần dưới giải quyết 1 vấn đề cụ thể.

## 3.1. QFL (Quality Focal Loss) — cho objectness

### Vấn đề thay thế
Trong detection, **target cls thường là cứng (hard target, 0 hoặc 1)**. Nhưng thực tế:
- Một pixel ở biên người → "60% là người, 40% là background"
- Anchor có IoU 0.7 với GT → soft target (mục tiêu mềm) = 0.7, không phải 1.0

Loss cũ (BCE, Focal Loss) chỉ xử lý hard target → không học được "quality" (chất lượng).

### QFL công thức
```
QFL = |target - sigmoid(pred)|^β × BCE(pred, target)
```

Với β=2.0 (default):
- Khi pred khác target nhiều (vd target=0.7, pred=0.1): `|0.7-0.1|² = 0.36` × BCE = boost mạnh
- Khi pred đúng rồi (vd target=0.7, pred=0.7): `|0|² = 0` × BCE = 0, không update nữa

### Ý nghĩa β
- β=0: QFL = BCE (không có focal weighting)
- β=1: weighting tuyến tính
- β=2: weighting bình phương (default)
- β=3+: weighting mạnh hơn nhưng dễ làm gradient nhỏ cho easy samples

**Tại sao chọn β=2.0?** GFL paper thử nghiệm β=1, 2, 3 trên COCO → β=2 tốt nhất.

### Tại sao chọn QFL không chọn các loss khác

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **BCE** (Binary Cross-Entropy) | Hard targets only (0 hoặc 1). Không học quality. Easy samples (đã đúng) vẫn nhận gradient → wasteful |
| **Focal Loss** (RetinaNet, ICCV 2017) | Hard targets, focal weighting `(1-p)^γ × BCE`. Vấn đề: cho easy samples (p gần 1), `(1-p)^γ` rất nhỏ → gradient vanish. Train chậm cuối |
| **Distribution Focal Loss (DFL)** (GFL paper) | Phức tạp hơn (cần expand box thành distribution của 16 bins). Tốn memory |
| **QFL (đã chọn)** | Soft targets + focal weighting, vừa cls vừa quality |

QFL cho phép soft target = IoU giữa predicted box và GT → model học cả "có người" và "predict tốt cỡ nào".

## 3.2. NWD (Normalized Wasserstein Distance) — cho box nhỏ

### Vấn đề
**IoU của box nhỏ rất bất ổn:**

Ví dụ cụ thể:
- Box GT 8×8 = 64 px²
- Box pred lệch 1 pixel theo x: GT = (0,0,8,8), pred = (1,0,9,8)
- Intersection = (1,0,8,8) = 7×8 = 56 px²
- Union = 64 + 64 - 56 = 72 px²
- IoU = 56/72 = **0.78** (drop 22% chỉ vì 1 pixel)

So sánh với box lớn:
- Box GT 100×100 = 10,000 px²
- Box pred lệch 1 pixel: intersection = 99×100 = 9900, union = 10100
- IoU = 9900/10100 = **0.98** (drop chỉ 2%)

→ **Loss IoU cho box nhỏ có gradient rất lớn và bất ổn** so với box lớn. Train mất cân bằng.

### NWD ý tưởng
Coi mỗi box là **Gaussian distribution** (phân phối chuẩn 2D):
- Mean = tâm box (cx, cy)
- Covariance = diagonal matrix với (w/2)², (h/2)² (variance)

Đo **Wasserstein-2 distance** giữa hai distribution thay vì IoU.

Wasserstein distance (W2) đo "công cần để chuyển distribution A thành B" — ổn định hơn IoU cho object nhỏ.

### Công thức cụ thể
```python
def nwd_loss(box1, box2, c=0.04):
    # Convert to Gaussian: (cx, cy, w/2, h/2)
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    
    # Wasserstein-2 distance for diagonal Gaussians
    w2_dist = (cx1 - cx2)² + (cy1 - cy2)² + (w1/2 - w2/2)² + (h1/2 - h2/2)²
    
    # Normalize và biến thành similarity (>0 và <=1)
    nwd = exp(-sqrt(w2_dist) / c)
    
    # Loss = 1 - similarity
    return 1 - nwd
```

c là hyperparameter (~0.04). Nhỏ → loss nhạy với chênh lệch nhỏ.

### So sánh với box lớn vs nhỏ

| Box | Lệch 1 pixel | IoU drop | NWD drop |
|-----|--------------|----------|----------|
| 8×8 | 1 pixel | 22% | 5% |
| 100×100 | 1 pixel | 2% | 0.05% |

→ NWD ổn định hơn IoU **rất nhiều** cho box nhỏ.

### Tại sao chọn NWD cho tiny

| Lựa chọn | Box nhỏ | Box lớn |
|----------|---------|---------|
| **IoU loss** | Bất ổn | OK |
| **GIoU** (CVPR 2019) | Tốt hơn IoU một chút (có enclosing box) | OK |
| **DIoU** (AAAI 2020): + centroid dist | Có centroid awareness | OK |
| **CIoU** (AAAI 2020): + aspect ratio | Có aspect | **Tốt cho lớn** |
| **NWD** (ISPRS 2022) | **Cực kỳ ổn định** | Hơi nhược (mất aspect ratio info) |

### Trong v15: blend NWD + CIoU theo diện tích
```python
nwd_w = exp(-area_px / 1024)    # area nhỏ → nwd_w lớn
loss = nwd_w × NWD + (1 - nwd_w) × CIoU
```

**Ví dụ blend cho box các kích thước khác nhau:**

| Box size | area_px | nwd_w | CIoU weight |
|----------|---------|-------|-------------|
| 8×8 | 64 | 0.94 | 0.06 — gần như chỉ NWD |
| 24×24 | 576 | 0.57 | 0.43 — cân bằng |
| 32×32 | 1024 | 0.37 | 0.63 — bắt đầu nghiêng CIoU |
| 64×64 | 4096 | 0.018 | 0.982 — gần như chỉ CIoU |
| 100×100 | 10000 | 0.00005 | 0.99995 — chỉ CIoU |

→ Best of both worlds. Mỗi box dùng loss phù hợp.

## 3.3. CIoU (Complete IoU) — cho box vừa và lớn

### Vấn đề của các IoU variant

Tiến hóa của IoU loss:
- **IoU** (chỉ overlap): không biết box pred lệch tâm hay chỉ scale sai
- **GIoU** (CVPR 2019): thêm enclosing box → biết overlap relative position. Vẫn thiếu centroid awareness
- **DIoU** (AAAI 2020): thêm khoảng cách centroid → centroid awareness. Vẫn thiếu aspect ratio penalty
- **CIoU** (AAAI 2020): thêm aspect ratio penalty → đầy đủ

### CIoU công thức
```
CIoU = IoU - (centroid_dist² / diagonal²) - α × v

centroid_dist² = (cx1 - cx2)² + (cy1 - cy2)²  # khoảng cách tâm bình phương
diagonal² = (max_x - min_x)² + (max_y - min_y)²  # đường chéo enclosing box
v = (4/π²) × (atan(GT_w/GT_h) - atan(pred_w/pred_h))²  # aspect ratio penalty
α = v / ((1 - IoU) + v)  # adaptive trade-off coefficient

Loss = 1 - CIoU
```

Penalty 3 yếu tố:
1. **Sai overlap** (1 - IoU)
2. **Sai vị trí tâm** (centroid_dist²/diagonal²)
3. **Sai aspect ratio** (α × v)

### Tại sao chọn CIoU
1. **AAAI 2020 paper** chứng minh trên COCO:
   - IoU loss: 36.5 AP
   - GIoU: 36.8 AP (+0.3)
   - DIoU: 37.1 AP (+0.6)
   - CIoU: **37.5 AP (+1.0)**
2. **Match lý tưởng cho người drone:**
   - Người luôn có aspect ratio đặc trưng
   - Nhìn ngang: cao > rộng (h/w ≈ 2-3)
   - Nhìn từ drone xuống: gần như vuông (h/w ≈ 1)
   - Penalty aspect rất hữu ích để model học đúng pose
3. Lý do **KHÔNG chọn IoU/GIoU/DIoU:** mất 1 trong 3 yếu tố quan trọng
4. Lý do **không dùng CIoU thuần (mà blend với NWD):** CIoU vẫn bất ổn cho tiny — nên blend NWD cho tiny

## 3.4. STAL (Small-Target-Aware Label Assignment) — MỚI từ YOLO26

### Vấn đề: TAL gốc bỏ sót tiny

**TAL (Task-Aligned Learning, ICCV 2021)** chọn anchor positive bằng:
```
alignment_score = cls_score^α × IoU^β
```
Với α=1, β=8 (mặc định). Sau đó top-k anchor có alignment cao nhất → positive.

**Vấn đề với tiny object:**
- Object 8×8 px chỉ có 1-2 anchor location gần đó (do quantization của P2 stride=4)
- IoU max của các anchor này có thể chỉ 0.2 (do anchor không match perfectly)
- Cùng ảnh có object lớn (50×50) với nhiều anchor có IoU 0.8 → "ăn" hết top-k slot
- → Tiny object **không có anchor positive** → không được train

### STAL giải quyết bằng 2 cơ chế

**Cơ chế 1: Guaranteed minimum k cho tiny**

```python
def assign_for_tiny(gt_box, anchors, k_min=4):
    if gt_box.area_px < stal_min_area_px:  # vd 64 px²
        # Tìm k_min anchors gần nhất bất kể alignment
        candidates = find_k_min_nearest(gt_box, anchors)
        if len(candidates) < k_min:
            # Mở rộng search radius
            radius = base_radius * 2
            candidates = find_anchors_within(gt_box, radius)
        return candidates
```

→ Tiny object **luôn có ít nhất 4 anchor positive**, không bị "đói" gradient.

**Cơ chế 2: Alignment bonus cho tiny**

```python
# Sau khi tính alignment thường (cls_score^α × IoU^β)
bonus = 1 + γ × exp(-area_px / area_ref)
alignment *= bonus
```

Với γ=2.0, area_ref=576 (=24×24):
- Tiny 8×8 px (64 px²): bonus = 1 + 2 × exp(-64/576) = 1 + 2×0.89 = **2.78×** → boost mạnh
- Trung bình 24×24 px (576 px²): bonus = 1 + 2 × exp(-1) = 1.74×
- Lớn 64×64 px (4096 px²): bonus = 1 + 2 × exp(-7.1) ≈ 1.002 → gần như không boost

**Hiệu quả:** trong top-k selection, tiny object có alignment được nhân 2.78× → khả năng được chọn cao hơn nhiều.

### Tại sao chọn STAL không chọn alternatives

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **TAL gốc** | Bỏ sót tiny như đã giải thích |
| **ATSS** (Adaptive Training Sample Selection, CVPR 2020) | Adaptive threshold dựa trên thống kê IoU. Không có tiny boost rõ ràng. Improve trên COCO general nhưng không chuyên cho tiny |
| **SimOTA** (YOLOX): optimal transport assignment | Tốt và adaptive. Nhưng không có tiny guarantee. Phức tạp compute |
| **STAL (đã chọn)** | Vừa adaptive vừa guarantee tiny |

### Kết quả thực tế trong dự án
Trong log MuSGD v3, log `pos_s4` (số anchor positive ở P2 stride=4) ổn định ~50/batch → tiny luôn được assign.

Nếu dùng TAL gốc, con số này sẽ dao động và đôi khi = 0 cho epoch có nhiều object lớn.

## 3.5. ProgLoss (Progressive Loss Balancing) — MỚI từ YOLO26

### Vấn đề: obj loss và box loss không cân bằng theo thời gian

**Đầu training (epoch 1-50):**
- Box prediction còn ngẫu nhiên → IoU ~0
- Tính box loss ngay → loss = `1 - 0 = 1` (max), gradient lớn
- Nhưng gradient này vô nghĩa (box còn random)
- → Lãng phí compute, không học được hiệu quả

**Cuối training (epoch 350+):**
- Box đã gần đúng (IoU ~0.7), cls đã decent (precision 0.8+)
- Cần tinh chỉnh **box localization** (px-level)
- Cần weight box loss cao hơn để force model focus

### ProgLoss schedule
```python
progress = epoch / total_epochs   # 0 → 1
prog_factor = 2.0  # hyperparameter

obj_w_factor = prog_factor - (prog_factor - 1) × progress  # 2.0 → 1.0
box_w_factor = 1.0 / obj_w_factor  # 0.5 → 1.0

obj_loss_final = base_obj_w × obj_w_factor × obj_loss
box_loss_final = base_box_w × box_w_factor × box_loss
```

| Epoch | Progress | obj_factor | box_factor | Ý nghĩa |
|-------|----------|------------|------------|---------|
| 0 | 0.00 | 2.0 | 0.5 | Tập trung "có người không?" |
| 100 | 0.25 | 1.75 | 0.57 | |
| 200 | 0.50 | 1.5 | 0.67 | Bắt đầu balance |
| 300 | 0.75 | 1.25 | 0.80 | |
| 400 | 1.00 | 1.0 | 1.0 | Tinh chỉnh location |

### Tại sao chọn ProgLoss

| Lựa chọn | Vấn đề chi tiết |
|----------|----------------|
| **Fixed weights** (vd obj=8.0, box=4.0 cố định) | Phải tune. Tune cho early hay late epoch? Compromise → cả 2 đều không tối ưu |
| **Curriculum learning** (lọc data theo độ khó) | Phức tạp, cần annotate độ khó cho từng ảnh. Không scale |
| **ProgLoss (đã chọn)** | Đơn giản, automatic theo epoch |

### Kết quả thực tế
YOLO26 paper: ProgLoss tăng ~0.3 AP trên COCO. Trong dự án, prog_factor=2.0 được dùng theo paper recommend.

## 3.6. ASL (Area Scale Loss) — drone-specific, tự thiết kế

### Vấn đề
Trong VisDrone, người tiny (8-16 px) chiếm 70%+ instances. Nhưng box loss của tiny đóng góp ít vào total loss vì:
- Tiny object có ít anchor positive (1-4 cái sau STAL)
- Box loss tổng = sum(per-object loss) → tiny ít object đóng góp ít
- → Model "bỏ qua" tiny vì gradient từ tiny ít

### ASL giải pháp
```python
asl_w = sqrt(area_ref / area_px)
asl_w = clamp(asl_w, 1.0, asl_max)  # vd asl_max = 4.0
box_loss_per_object × asl_w
```

Với `area_ref = 0.004 × img_size² = 0.004 × 640² = 1638 px²`:

| Box | area_px | sqrt(1638/area) | asl_w (clamped) | Boost |
|-----|---------|-----------------|------------------|-------|
| 8×8 | 64 | sqrt(25.6) = 5.06 | **4.0** (clamped) | 4× |
| 16×16 | 256 | sqrt(6.4) = 2.53 | 2.53 | 2.53× |
| 32×32 | 1024 | sqrt(1.6) = 1.27 | 1.27 | 1.27× |
| 64×64 | 4096 | sqrt(0.4) = 0.63 | **1.0** (clamped) | 1× (no boost) |
| 100×100 | 10000 | sqrt(0.16) = 0.4 | **1.0** | 1× |

→ Tiny box được boost 2-4×, large box không bị nhược.

### Tại sao tự thiết kế (không dùng paper sẵn)
1. **Phù hợp đặc thù VisDrone:** distribution tiny rất extreme (70%+ tiny)
2. **Đơn giản:** chỉ thêm 1 phép sqrt, không thêm hyperparameter phức tạp
3. **Hiệu quả thực tế:** ablation v14 cho thấy **+1% AP cho tiny** với ASL on vs off

## 3.7. TAL (Task-Aligned Learning) — base assignment

Đã giải thích ở STAL. TAL là **base mechanism**, STAL là extension cho tiny.

Cụ thể TAL hoạt động:
1. Cho mỗi GT box, tính alignment với mọi anchor: `cls_score^α × IoU^β`
2. Chọn top-k anchor có alignment cao nhất → positive
3. Còn lại → negative

α và β control trade-off:
- α cao → ưu tiên cls confidence
- β cao → ưu tiên localization quality
- Default v15: α=1, β=8 (lean về quality)

---

# Phần IV: Optimizer — MuSGD

## 4.1. Tại sao không dùng AdamW?

AdamW là default cho hầu hết deep learning. Tại sao chuyển sang MuSGD?

**Thí nghiệm thực tế trên dự án:**
| Optimizer | LR | AP@0.5 | Recall |
|-----------|-----|--------|--------|
| AdamW | 0.002 | 0.5568 | 0.8284 |
| MuSGD | 0.01 | **0.5931** | **0.8486** |

MuSGD tốt hơn **+0.036 AP (+6.5%)** và recall +2%.

## 4.2. Vấn đề của AdamW
- **Adaptive LR per parameter:** mỗi param có "tốc độ học" riêng dựa trên historic gradient
- Tốt cho NLP/Transformer (model lớn, ill-conditioned gradient)
- Nhưng **với CNN nhỏ và dense head**, adaptive làm gradient mất diversity
- Cụ thể: param có gradient lớn → LR giảm → update ít, param có gradient nhỏ → LR tăng → "compensate"
- Hậu quả: tất cả param converge cùng tốc độ → mất pattern phân biệt → model dễ stuck local minimum

## 4.3. Muon (Newton-Schulz orthogonalization) cốt lõi

### Ý tưởng

Gradient của weight matrix W ở mỗi step thường **ill-conditioned** (xấu điều kiện). Cụ thể:

Phân tích SVD: `G = U × Σ × V^T`
- U, V: ma trận orthogonal
- Σ: diagonal matrix chứa singular values σ1 ≥ σ2 ≥ ... ≥ σn

**Vấn đề:** σ1 thường lớn hơn σn rất nhiều (vd σ1=10, σn=0.001). Khi update theo G:
- Các hướng có σ lớn → update mạnh
- Các hướng có σ nhỏ → update yếu

→ Train mất cân bằng, một số neuron "chết".

### Newton-Schulz làm gì?

**Orthogonalize gradient matrix:** chuyển G thành matrix có **mọi singular value đều = 1**.

Toán học: tính approx `UV^T` thay vì G. Tức là:
- Giữ hướng (U, V — orthogonal matrices)
- Bỏ magnitude (Σ → identity)

Kết quả: update đều theo mọi hướng, không bị skewed.

### Iteration cụ thể
```python
def newton_schulz_5(G):
    X = G / ||G||  # normalize đầu
    if rows > cols:
        X = X.T  # transpose để xử lý wide matrix (rẻ hơn)
    for i in range(5):
        a, b, c = 3.4445, -4.7750, 2.0315
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if rows > cols:
        X = X.T  # transpose ngược lại
    return X
```

Sau 5 iter, X ≈ UV^T (orthogonal approximation).

**Tại sao 5 iter, không 3 hay 10?**
- 3 iter: chưa converge, X cách UV^T xa → orthogonalization không tốt → AP thấp
- 5 iter: balance tốt nhất theo paper Muon thí nghiệm
- 10 iter: không gain thêm, compute gấp đôi

### Hệ số (3.4445, -4.7750, 2.0315)
Đây là **quintic polynomial coefficients** được fit để converge nhanh nhất với 5 iter:

`p(s) = 3.4445s + (-4.7750)s³ + 2.0315s⁵`

Polynomial này có tính chất:
- p(0) = 0
- p(1) = 1
- |p(s)| ≈ 1 cho s ∈ [0.1, 1]

→ Sau 5 lần áp dụng, mọi singular value của X được "đẩy" về gần 1.

Không phải tự chọn random, mà là kết quả tối ưu của Muon paper (numerical optimization).

## 4.4. MuSGD = Muon + SGD blend

```python
update = -lr × (muon_frac × Muon_update + sgd_frac × SGD_update)
```

Với `muon_frac = 0.2`, `sgd_frac = 1.0`:
- Muon contribution = 0.2 × Muon_update
- SGD contribution = 1.0 × SGD_update
- Total scaling = 1.2 × lr (effective higher LR than vanilla SGD)

### Tại sao blend không chỉ Muon thuần?
- Muon thuần normalize hoàn toàn → mất magnitude info (đôi khi gradient lớn là important signal)
- SGD thuần dễ stuck local minimum
- Blend: Muon là **regularizer** giữ orthogonality, SGD vẫn drive update

### Tại sao chọn fraction 0.2/1.0 (không 0.5/0.5)
**Đây là bài học từ thí nghiệm sai trước đó:**

Lần đầu code MuSGD dùng 0.5/0.5 (theo intuition "blend đều") → AP cực thấp.

Sau khi check Ultralytics source code (commit f2d3aed):
- 0.5/0.5: Muon dominate → train không ổn (lr quá cao do Muon normalized update có magnitude lớn)
- 0.2/1.0: Muon là phụ trợ → SGD vẫn là main → train ổn

Sửa từ 0.5/0.5 → 0.2/1.0 + lr 0.002 → 0.01: AP từ kém lên 0.5931.

## 4.5. Compare MuSGD vs alternatives

| Optimizer | AP@0.5 | Speed/step | Memory |
|-----------|------|----------|--------|
| **SGD** | ~0.55 (estimated) | 100% (baseline) | 1× state |
| **Adam** | 0.55-0.56 | 110% | 2× state (mom + var) |
| **AdamW** | 0.5568 | 110% | 2× state |
| **Lion** (Google 2023) | ~0.56 (estimated) | 95% | 1× state |
| **MuSGD (đã chọn)** | **0.5931** | 105% (do NS iter) | 2× state |

MuSGD chậm hơn SGD ~5% mỗi step (do 5 iter Newton-Schulz cho mỗi 2D weight) nhưng AP tăng đủ để bù.

---

# Phần V: Training tricks

## 5.1. EMA (Exponential Moving Average)

### Vấn đề
Cuối epoch, weights model dao động vì SGD/MuSGD có noise (gradient từ batch ngẫu nhiên). Eval bằng weights "instantaneous" → AP nhảy nhót giữa các epoch.

### Giải pháp: shadow weights (weights bóng)
```python
ema_weights = decay × ema_weights + (1 - decay) × current_weights
```

Decay = 0.9998 → tương đương trung bình ~5000 step gần nhất.

Eval bằng `ema_weights` → smooth, ổn định hơn.

### Hiểu decay theo cách khác
EMA có tính chất: weight ảnh hưởng giảm theo cấp số nhân.

Sau N step, contribution của weight tại step thứ 0 = decay^N.
- decay=0.99, N=100: 0.99^100 ≈ 0.37 → contribution còn 37%
- decay=0.9998, N=5000: 0.9998^5000 ≈ 0.37 → contribution còn 37%

→ EMA "nhớ" 5000 step gần nhất (theo định nghĩa half-life).

### Tại sao decay = 0.9998 (không 0.99 hay 0.9999)
- **0.99:** chỉ avg ~100 step, vẫn noisy (1 epoch = 1000 step ≫ 100)
- **0.9998:** avg ~5000 step, balance smoothness và adaptivity
- **0.9999:** avg ~10000 step, quá smooth → ema lag training (chậm theo dõi training progress)
- **0.9998 (đã chọn):** YOLO-family standard, paper-prove qua nhiều dataset

## 5.2. Cosine LR schedule

### Tại sao không Step Decay?
Step decay drop LR đột ngột (vd 0.01 → 0.001 sau 100 epoch):
- Trước drop: LR cao, model exploring (khám phá)
- Sau drop: LR thấp đột ngột → model "đông cứng" ngay → có thể stuck local minimum
- AP tăng đột biến ở step điểm nhưng overall không tốt bằng cosine

### Cosine smooth
```
lr(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t/T))
```

Với t = epoch hiện tại sau warmup, T = total epoch sau warmup.

| t/T | cos(π × t/T) | lr (lr_max=0.01, lr_min=0.0001) |
|-----|--------------|----------------------------------|
| 0 | 1 | 0.01 |
| 0.25 | 0.71 | 0.0085 |
| 0.5 | 0 | 0.005 |
| 0.75 | -0.71 | 0.0015 |
| 1 | -1 | 0.0001 |

LR giảm theo cosine từ lr_max về lr_min — smooth, model có thời gian thích nghi.

### Warmup 10 epoch đầu
LR tăng từ 0.1 × lr_max (= 0.001) lên lr_max (= 0.01) trong 10 epoch.

```
warmup_lr(t) = lr_start + (lr_max - lr_start) × t / warmup_epochs
```

**Tại sao cần warmup?**
- Đầu training, weights random → gradient rất lớn (loss cao)
- LR đầy đủ ngay → gradient explosion (gradient khổng lồ → weights jump xa)
- Warmup → tăng LR dần dần, gradient ổn định
- 10 epoch (= 2.5% total 400) — vừa đủ ổn định không phí

## 5.3. Multi-scale training

Mỗi 10 batch, đổi input size từ 320 đến 640 (step 32):
- 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640 (11 sizes)

### Tại sao không train cố định 640?
- Train 640 → model overfit ở scale 640
- Deploy ở 320 (Pi 5) → AP drop đáng kể (~5-10% drop)
- Multi-scale → model **scale-invariant**

### Tại sao không random từng batch?
- Đổi size mỗi batch → DataLoader phải resize liên tục → chậm IO
- Đổi mỗi 10 batch → DataLoader có thời gian cache size hiện tại
- 10 batch = ~1 step grad accumulation (effective batch 96 = 8 sub-batch × 12 batch_size)

## 5.4. Gradient Accumulation

```
batch_size = 12 (giới hạn GPU 4060 với img 640)
grad_accumulate = 8
effective_batch = 12 × 8 = 96
```

### Tại sao cần effective batch lớn?
- Batch nhỏ → gradient noisy (mỗi batch ngẫu nhiên có 1-2 ảnh quirky)
- Batch lớn → gradient stable (trung bình nhiều ảnh)
- Detection cần batch ~32-128 để stable (theo paper YOLOv5/v8)

### Cơ chế
```python
optimizer.zero_grad()
for step in range(grad_accumulate):  # 8 sub-batch
    images, labels = next(loader)  # 12 ảnh
    loss = model(images, labels)
    loss.backward()  # accumulate gradient (PyTorch tự cộng dồn)
    # Không gọi optimizer.step() ở đây!

optimizer.step()  # update sau khi đã accumulate đủ 8 lần
```

**Trade-off:**
- 8× chậm hơn batch 96 thực sự (do 8 forward pass)
- Nhưng GPU memory đủ (chỉ cần activations cho 12 ảnh, không phải 96)
- → Khả thi với GPU 8GB như 4060

## 5.5. AMP (Automatic Mixed Precision)

```
forward: FP16 (16-bit float, nhanh, ít memory)
backward + master weights: FP32 (32-bit, chính xác)
```

### Lợi ích
- **Speed:** ~1.5-2× nhanh hơn pure FP32 (Tensor Core trên 4060 chuyên cho FP16)
- **Memory:** ~50% ít hơn → batch lớn hơn được

### Risk và xử lý
- FP16 có range hẹp hơn → một số gradient có thể underflow về 0
- Solution: **Loss scaling** — nhân loss với 1024 trước khi backward, divide ngược sau
- PyTorch `GradScaler` tự handle

### Một số op vẫn FP32
- BatchNorm running stats: cần precision cao
- Softmax: dễ underflow ở FP16
- Loss computation: cần chính xác

## 5.6. COCO Person mixing schedule

```yaml
extra_enable_epoch: 150     # bật COCO từ ep 150
extra_disable_epoch: 280    # tắt từ ep 280
extra_sample_ratio: 0.20    # 20% batch là COCO
```

### Tại sao không dùng COCO từ đầu?
- Đầu training, model chưa "quen" VisDrone → COCO làm distract (làm phân tâm)
- COCO domain ≠ drone domain → model học mixed signal → AP thấp
- Wait đến ep 150 (~37% training) khi model đã decent trên VisDrone

### Tại sao tắt cuối?
- 280-400: focus tinh chỉnh trên VisDrone (domain target)
- COCO domain hơi khác → cuối training "rung" giữa 2 domain
- Tắt COCO → focus 100% VisDrone

### Tại sao 20% (không 50%)
- VisDrone là domain target → ưu tiên 80%
- COCO chỉ là augment diversity → 20% đủ exposure
- 20% = 1/5 batch → đủ để model học pose từ COCO mà không overwhelm

---

# Phần VI: Augmentation pipeline

## 6.1. Mosaic (prob=0.5)
**Cách:** Ghép 4 ảnh thành 1 (random crop từ mỗi ảnh).

```
┌─────┬─────┐
│img1 │img2 │
├─────┼─────┤
│img3 │img4 │
└─────┴─────┘
```

**Tại sao?**
- **Tăng instances/frame:** thay vì 5 người/ảnh → 20 người/ảnh (4 ảnh ghép)
- **Đa dạng background:** 4 background mix trong 1 frame
- **Test boundary:** object bị cắt qua biên 4 quadrant → model học robust với crop

**Tại sao prob 0.5 không 1.0?**
- 1.0: model thấy 100% mosaic, mất tính "ảnh tự nhiên"
- 0.5: balance giữa mosaic-trained và normal-trained
- Eval data là ảnh tự nhiên → cần model thấy normal đủ

## 6.2. MixUp (prob=0.1)
**Cách:** Blend 2 ảnh với weight α: `img = α × img1 + (1-α) × img2`.

α thường ~0.5 (random Beta(0.5, 0.5)).

**Tại sao prob thấp (0.1)?**
- MixUp tạo ảnh không tự nhiên (semi-transparent — nửa trong suốt)
- 10% là vừa đủ regularize, không quá distort training data

## 6.3. CopyPaste (prob=0.4)
**Cách:** Copy người từ ảnh khác (segmentation hoặc bounding box), paste vào ảnh hiện tại.

**Tại sao 0.4 (cao hơn MixUp)?**
- VisDrone có nhiều ảnh ít người (1-2 người/ảnh) → CopyPaste tăng instance count
- Người được paste tự nhiên (không transparent như MixUp)
- Effective tăng số object/ảnh → train dày đặc hơn

## 6.4. Affine (prob=1.0)
Rotate/scale/translate ngẫu nhiên luôn (mọi ảnh).

**Lý do prob=1.0:**
- Drone bay → mọi ảnh đều có rotation/scale variation tự nhiên
- Train với 100% affine → model robust với mọi góc

Tham số:
- Rotate: ±15 độ
- Scale: 0.8 - 1.2
- Translate: ±10%

## 6.5. ColorJitter (prob=1.0)
HSV jitter ngẫu nhiên luôn.

**Lý do:**
- Drone gặp nhiều thời tiết/ánh sáng (sáng, tối, hoàng hôn, mây...)
- Train 100% jitter → model robust với mọi điều kiện

Tham số:
- Hue (sắc thái): ±0.1
- Saturation (độ bão hòa): ±0.5
- Value (độ sáng): ±0.4

## 6.6. Drone-specific aug (prob=0.3)
Augmentation chuyên cho drone:
- Motion blur (giả lập drone chuyển động)
- Perspective distortion (giả lập góc nghiêng)
- Compression artifacts (giả lập video compression)

## 6.7. Mosaic-off last 60 epoch
Tắt mosaic 60 epoch cuối.

**Lý do:**
- Mosaic tạo "ảnh nhân tạo" — không match distribution thật
- Cuối training cần fine-tune trên ảnh thật để match val/test distribution
- 60/400 = 15% training cuối → đủ thời gian re-adapt

---

# Phần VII: Tổng kết triết lý thiết kế

## 7.1. Top-3 design principles

### 1. Tiny-first (ưu tiên người tiny)
Mọi quyết định ưu tiên người tiny:
- **P2 không dùng CSP** (giữ chi tiết) — Section 2.3
- **LSK ở P2** (boost tiny) — Section 2.4
- **STAL guarantee tiny anchors** — Section 3.4
- **ASL boost tiny box loss** — Section 3.6
- **NWD blend cho tiny stability** — Section 3.2

### 2. Edge-friendly (thân thiện edge device)
Phù hợp Pi 5 từ kiến trúc:
- **DW Conv khắp nơi** (giảm 10× FLOPs vs Conv chuẩn)
- **Window attention** (không full self-attention) — Section 2.5
- **RepConv** (multi-branch train, single-branch deploy) — Section 2.7
- **Multi-scale train** → có 320px option cho Pi 5

### 3. Peer-reviewed only (chỉ dùng paper top venue)
Mọi component đều có paper top venue:
- ICCV/CVPR/NeurIPS/ECCV
- Không dùng "trick" không paper, không dùng module rumor

## 7.2. Trade-offs đã chấp nhận

| Trade-off | Quyết định | Lý do |
|-----------|-----------|-------|
| Accuracy vs Speed | Speed wins ở Pi 5 | Realtime quan trọng hơn 1-2% AP |
| Memory train vs Memory deploy | Train cao OK | Deploy weights chỉ 4MB |
| Train time vs Iteration | Iteration wins | Multi-branch (RepConv) train chậm hơn nhưng deploy nhanh |
| Complexity vs Maintainability | Cân bằng | Mọi module có docstring + paper ref |
| AdamW familiar vs MuSGD better | MuSGD wins | +6.5% AP đáng đổi compute nhỏ |

## 7.3. Khi nào nên đổi v15 → v14 hoặc v16?

| Use case | Model nên dùng |
|----------|----------------|
| Production drone tracking (target Pi 5) | **v15 light** (đã chọn) |
| GPU server, accuracy max | v15 balanced (~2M params) |
| Stability over performance | v16 (giữ P2 v14, P5 light CSP) |
| Học PyTorch detection (đơn giản) | v14 (no CSP, no LSK, no STAL) |

---

# Phần VIII: Bảng tham khảo paper

| Module | Paper | Venue | Năm |
|--------|-------|-------|-----|
| UIB | MobileNetV4 (Howard et al.) | ECCV | 2024 |
| CSP | YOLOv5 (Jocher et al.) | Zenodo / CVPR W | 2020 |
| LSK | LSKNet (Li et al.) | ICCV / IJCV | 2023 / 2024 |
| Window Attention | Swin Transformer (Liu et al.) | ICCV (Best Paper) | 2021 |
| BiFPN | EfficientDet (Tan et al.) | CVPR | 2020 |
| RepConv | RepVGG (Ding et al.) | CVPR | 2021 |
| Decoupled Head | YOLOX (Ge et al.) | NeurIPS W | 2021 |
| QFL | GFL (Li et al.) | NeurIPS | 2020 |
| NWD | NWD paper | ISPRS | 2022 |
| CIoU | CIoU paper | AAAI | 2020 |
| TAL | TOOD (Feng et al.) | ICCV | 2021 |
| STAL + ProgLoss + MuSGD | YOLO26 (Ultralytics) | arXiv 2509.25164 | 2025 |
| Newton-Schulz Muon | Muon paper | arXiv 2502.16982 | 2025 |
| EMA | Polyak averaging (Polyak) | SIAM | 1992 |
| Cosine LR | SGDR (Loshchilov & Hutter) | ICLR | 2017 |
| AMP | Mixed Precision Training (Micikevicius et al.) | ICLR | 2018 |
