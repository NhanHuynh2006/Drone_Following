# Số đo THẬT trên Pi 5 + Hailo-8L — 2026-08-24

Đo trên chính con Pi deploy (`raspi@192.168.196.114`, hostname `pi5`), không phải mô phỏng.

| hạng mục | giá trị |
|---|---|
| Board | Raspberry Pi 5, Debian 12 bookworm, kernel 6.12.93, aarch64 |
| NPU | HAILO8L, HailoRT / firmware **4.20.0** |
| PCIe | **5 GT/s (Gen2) x1 — downgraded** từ cap 8 GT/s x4. `config.txt` chưa có dòng `pciex1_gen=3` |
| Python | 3.11.2 · numpy 1.26.4 · cv2 5.0.0 · hailo_platform 4.20.0 |

## Inference thuần (NPU)

Hai cách đo độc lập, khớp nhau:

| model | `hailortcli benchmark` (hw_only) | gọi đồng bộ `InferVStreams` (p50) |
|---|---|---|
| `pfdet_v17_320_hailo8l.hef` | **112.55 FPS** (HW latency 8.40 ms) | 8.88 ms → 112.7 FPS |
| `pfdet_v17_640_hailo8l.hef` | **55.78 FPS** | 18.00 ms → 55.6 FPS |

## End-to-end (capture → preprocess → inference → decode+NMS), @640

Đo bằng `flight_record.py --source synth` (nguồn ảnh giả, 0 người → decode/NMS ở mức sàn):

| chặng | p50 |
|---|---|
| capture | ~0 ms (nguồn synth; camera thật sẽ khác) |
| preprocess (letterbox + BGR→RGB) | 1.2 ms |
| inference | 17.8 ms |
| decode + NMS | 0.65 ms |
| **tổng** | **19.96 ms → 50 FPS** |

p95 = 20.2 ms, p99 = 20.34 ms. Nhiệt 41–45 °C, `get_throttled = 0x0` suốt bài đo.

> Chi phí ghi video + ghi jpg **KHÔNG** nằm trong con số trên (log riêng ở
> `ms.record_overhead`). Lúc bay thật, deploy không ghi đĩa nên không được cộng vào.

## ⚠️ Lệch với paper: 186 FPS @320

`PROJECT.md:60,270` và `paper/sections/13_exp_deploy.tex:9` ghi **186 FPS @320**.
Đo lại trên đúng phần cứng này ra **112.55 FPS** — lệch **1.65×**.

Con số 186 theo `PAPER_INPUTS.md:179` là "inference-only, EMA" đo bằng EMA trong
`hailo_demo_cam.py`. EMA trên vòng lặp đồng bộ dễ đếm thiếu (ví dụ EMA khởi động
trong lúc warm-up, hoặc trung bình trượt không tính frame chậm). Bản đo hôm nay dùng
**chính `hailortcli benchmark` của Hailo** làm trọng tài nên khó cãi hơn.

Chưa loại trừ: HEF khác, thiết bị khác (Hailo-8 26 TOPS thay vì 8L 13 TOPS), hoặc
PCIe Gen3. Nhưng file HEF đo hôm nay **chính là** `pfdet_v17_hailo8l.hef` mà
`PROJECT.md:270` trỏ tới.

**Việc cần làm trước khi nộp:** hoặc sửa 186 → 112.55 (kèm nói rõ đo bằng
`hailortcli benchmark`), hoặc tìm lại được điều kiện cho ra 186 và ghi rõ điều kiện đó.
Để nguyên là rủi ro reviewer chạy lại không ra.

## Chưa test được

- **Camera**: Pi CHƯA cắm camera nào (không có V4L2 device trên bus USB, `rpicam-hello`
  không thấy CSI). 17 node `/dev/video19-35` là ISP/codec nội bộ của Pi 5, không phải camera.
- Vì vậy `capture` và AP thực tế chưa đo được. Cắm camera rồi chạy `./run.sh check`.

---

# Độ chính xác model 640 đã train (thêm 2026-08-24)

**Checkpoint gốc của `pfdet_v17_640_hailo8l.hef`**: `runs/train_v17_hailo/best.pt`
(v17 / profile light / `use_area_attention=False` — nhánh "Cách A" thuần conv của
`README_HAILO.md`, epoch 14). Xác nhận bằng cách đọc op trong `pfdet_v17_hailo_640.onnx`:
không còn MatMul/Softmax/Erf/ReduceMean.

Giao thức: **548 ảnh VisDrone val, @640, conf 0.01, person-only**, và **cùng một**
`decode_predictions_np` + `nms_numpy` (DIoU) mà `train_v3.py:evaluate` dùng.

| chỉ số | FP32 (RTX 4060) | INT8 (HEF trên Hailo-8L) | rớt tương đối |
|---|---|---|---|
| AP@[.50:.95] | 0.1734 | **0.1516** | **−12.6 %** |
| AP@0.50 | 0.4812 | **0.4585** | **−4.7 %** |
| AP@0.75 | 0.0703 | 0.0493 | −29.9 % |
| AP small | 0.1710 | 0.1493 | −12.7 % |
| AR@100 | 0.3014 | 0.2852 | −5.4 % |

Latency inference đo trong chính lần eval này: p50 **17.64 ms**, p99 17.85 ms → 57 FPS.

**Đọc kết quả:** model vẫn detect được sau khi lượng tử hoá — AP@0.5 chỉ mất 4.7 %.
Nhưng **AP@0.75 mất 29.9 %**, tức INT8 làm **box kém khít** chứ không làm mất object.
Hợp lý cho detector 1.17M: điểm số (obj) chịu quantise tốt, còn hồi quy toạ độ mất
độ phân giải. `README_HAILO.md` mục 6 đặt ngưỡng "rớt >5 % AP → tăng opt-level":
AP tổng rớt 12.6 % nên **đáng thử `--opt-level 4` + thêm ảnh calib** nếu cần khít box.
Với use case bay bám người ở 3–5 m thì AP@0.5 mới là thứ quan trọng, và nó ổn.

## 3 lỗi trong đường đo HEF đã sửa trong lần này

Cả ba đều làm số HEF **sai lệch**, không phải lỗi hiệu năng:

1. **`hailo_eval_val.py` decode sai encoding.** Dùng `tanh(dx) + 0.5` trong khi model
   train theo `sigmoid(dx) − 0.5` (`utils/box_ops.py:136`, tài liệu hoá ở
   `models/pfdet_nano_v14.py:552`) → tâm box lệch nửa ô lưới.
2. **NMS khác đường chuẩn.** Bản cũ tự viết IoU-NMS, không có `pre_nms_topk=500`, trong
   khi paper dùng **DIoU**-`nms_numpy` có top-k. Vừa sai so sánh vừa chậm ~40×
   (548 ảnh ước tính >3 giờ, sau khi sửa còn ~7 phút). Nay import thẳng qua
   `box_ops_np.py` — bản chép numpy-only của `utils/box_ops.py`.
3. **`score()` không đưa GT qua letterbox.** Nhãn `.txt` chuẩn hoá theo ảnh gốc
   1920×1080, còn dự đoán chuẩn hoá theo canvas 640×640. Ảnh không vuông nên lệch thật:
   GT `cy=0.25` phải thành `0.3594`, `h=0.10` phải thành `0.0563`. Không sửa thì AP của
   HEF bị dìm oan.

⚠️ Dòng "So với FP32 trong UNIFIED.json" mà script cũ in ra là **so nhầm model**
(`v17-baseline` / `v17-deploy` là checkpoint khác, không phải cái dựng ra HEF này).
Nay đổi thành cờ `--fp32-ap / --fp32-ap50` để phải truyền số của đúng checkpoint.

---

# ⭐ GIẢI ĐƯỢC chỗ lệch 186 FPS (bổ sung 2026-08-24, sau khi test batch size)

Con số 186 **KHÔNG sai**. Nó là **throughput theo lô (batch)**, không phải tốc độ
xử lý từng khung hình. Đo bằng `hailortcli benchmark --batch-size N` trên đúng HEF 320:

| batch | FPS (hw_only) | Latency (hw) |
|---|---|---|
| **1** | **112.55** | **8.40 ms** |
| **4** | **188.78** | 17.05 ms |
| 8 | 214.63 | 28.35 ms |
| 16 | 231.82 | 50.48 ms |

`batch=4 → 188.78 FPS` ≈ con số 186 trong paper.

**Vì sao deploy thật vẫn phải dùng batch=1:** batch=4 nghĩa là phải gom đủ 4 khung hình
mới chạy được một lượt. Ở camera 30 fps, riêng việc chờ đủ lô đã tốn **133 ms**, cộng
17 ms xử lý. Vòng điều khiển bay bám người không chịu được độ trễ đó. Throughput cao
mà latency cao là đánh đổi sai cho use case này.

## Cách phát biểu đúng trong paper

Bản hiện tại (`paper/sections/13_exp_deploy.tex:9`) ghi trần "sustains 186 FPS at
320×320" — không nói batch. Reviewer chạy `hailortcli benchmark` mặc định (batch=1)
sẽ ra 112.55 và nghi ngờ toàn bộ mục.

Nên sửa thành đại ý:

> 112.6 FPS (8.40 ms/frame) at 320×320 in single-frame mode, 55.8 FPS at 640×640;
> peak throughput reaches 188.8 FPS at batch 4, but single-frame operation is used
> for deployment because batching adds 133 ms of frame-accumulation delay at 30 fps.
> End-to-end (capture → preprocess → inference → decode/NMS) is 19.96 ms p50,
> 20.34 ms p99 at 640×640 — 3.7× the camera rate, so the control loop is
> latency-bound, not detection-bound.

Phát biểu này mạnh hơn bản cũ: có latency, có p99, có lý do vì sao đủ nhanh — thay vì
một con số trần mà người khác chạy lại không ra.

## Bối cảnh: 112 FPS có thấp không? Không.

| | tốc độ | so với camera 30 fps |
|---|---|---|
| @320 inference | 112.55 FPS | dư 3.7× |
| @640 inference | 55.78 FPS | dư 1.9× |
| @640 **end-to-end** | 50 FPS (19.96 ms) | dư 1.7× |

Model không phải nút cổ chai. Với bài toán bay bám, thứ quyết định là **latency và p99**
(khung chậm nhất làm drone bám hụt), và p99 đo được chỉ **17.85 ms** — rất ổn định,
không throttle (`0x0`) suốt bài đo.

---

# Head-to-head với YOLO26n trên CÙNG phần cứng (2026-08-24)

**Lý do đo:** nghi ngờ YOLO26n chạy 80 FPS @640 trong khi SteadyDet chỉ 55.78 — nếu đúng thì
luận điểm "4.6 GFLOPs nên hiệu quả" của bài sụp, vì trên phần cứng đích lại chậm hơn.

**Con số 80 FPS đó đo trên một con Pi KHÁC**, không phải con này. Nhiều khả năng con đó
gắn **Hailo-8 (26 TOPS)** thay vì **Hailo-8L (13 TOPS)** — gấp đôi sức tính.

## Cách đo (để tái lập được)

Cả hai HEF compile bằng **đúng một lệnh, đúng một bộ tham số**:

```bash
python compile_hef.py --onnx <model>.onnx --hw-arch hailo8l \
    --calib-dir data/visdrone/val/images --img-size 640 --name <ten>
```
- YOLO26n: `runs/yolo/yolo26n/weights/best.pt` (đúng baseline trong bảng paper),
  export ONNX @640 opset 11 `nms=False`, cắt ở 6 conv head
  (`one2one_cv2.{0,1,2}` box 4ch + `one2one_cv3.{0,1,2}` cls 1ch) để Hailo chỉ nhận
  phần conv — **giống SteadyDet, decode + NMS đều chạy CPU**. Không cắt thì đồ thị còn
  TopK/Tile/Mod của NMS-free, Hailo không parse được.
- Đo bằng `hailortcli benchmark --batch-size 1` trên cùng con Pi, cùng lúc.

## Kết quả

| model | tham số (M) | GFLOPs | **contexts** | **FPS @640** | latency (hw) |
|---|---|---|---|---|---|
| **SteadyDet-N** | **1.17** | **4.6** | **4** | **63.66** | **15.26 ms** |
| YOLO26n | 2.38 | 5.2 | 6 | 37.46 | 24.79 ms |

**SteadyDet-N nhanh hơn 1.70×**, với 49% tham số và 88% compute. Kết luận ngược hẳn
với lo ngại ban đầu.

**Cơ chế: `contexts`, không phải GFLOPs.** Hailo-8L có bộ nhớ trên chip hữu hạn; model
không vừa thì compiler chia thành nhiều *context*, mỗi lần chuyển context phải nạp lại
trọng số. SteadyDet vừa **4** context, YOLO26n cần **6**. Đây mới là biến quyết định thông
lượng trên NPU — nên **GFLOPs thấp hơn không tự động nhanh hơn, và ngược lại**.

(Nhánh P2 stride-4 của SteadyDet ngốn 30.7% activation nhưng vẫn vừa 4 context, nên nó
không phải là thứ giết throughput như tôi phỏng đoán lúc đầu.)

## Cập nhật số deploy: HEF cũ bị compile kém

`pfdet_v17_640_hailo8l.hef` (dựng 15/7) chỉ đạt **55.78 FPS**. Recompile bằng pipeline
hiện tại ra `pfdet_v17_640_fair_hailo8l.hef` = **63.66 FPS** (+14%), cùng 4 context.
→ **Số nên dùng cho paper là 63.66 FPS / 15.26 ms @640**, và nên build lại HEF 320 tương tự.

## ⚠️ Hai HEF này CHỈ dùng để so TỐC ĐỘ, không được lấy AP

Cả hai calibrate bằng `data/visdrone/val/images` — **chính tập dùng để chấm AP**. Với
benchmark tốc độ thì vô hại (ảnh calib không ảnh hưởng thời gian chạy), nhưng lấy AP từ
chúng là **rò rỉ dữ liệu**. Muốn có AP thì phải compile lại với `--calib-dir` trỏ vào
**tập train**. Số AP INT8 ở mục trên (0.1516 / 0.4585) lấy từ HEF cũ, không dính vấn đề này.
