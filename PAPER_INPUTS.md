# PAPER_INPUTS.md — Bản đồ "mỗi bảng trong bài báo ← file kết quả nào"

> Paper: `paper/` (SOICT 2026, Springer CCIS/LNCS). Đọc kèm `PROJECT.md` và `V17_DESIGN.md`.
> Snapshot: 2026-07-25. **Quy tắc: mọi số trong paper phải truy được về 1 file dưới đây.**

## 0. Venue (SOICT 2026 — soict.org, HUST Hà Nội)
| | |
|---|---|
| Deadline | abstract **09/09/2026**, full paper **16/09/2026**, notification 12/10/2026, camera-ready 23/10/2026 |
| Page limit | **12 trang KHÔNG tính references** (submission). Camera-ready full 12-15 trang TÍNH cả refs |
| Template | Springer **CCIS/LNCS** (`llncs.cls`), PDF **không số trang** |
| Review | **SINGLE-BLIND** → GIỮ tên tác giả + affiliation |
| Nộp | EasyChair. Proceedings: Springer CCIS (Scopus/DBLP/EI) |

## 1. Câu chuyện & đóng góp  (ĐỔI KHUNG 2026-07-27b — MODEL đứng tên chính)

**Tựa:** *SteadyDet-N: A 4.6-GFLOP Person Detector That Leads Every Compact Baseline Under
UAV Vibration*. Đóng góp **(i) = model + chế độ train tính chung**, rồi mới tới protocol,
kỹ thuật, 2 kết quả âm.

⚠️ Đã cân nhắc và BÁC phương án "bỏ rung, làm bài kiến trúc thuần": clean AP của SteadyDet-N
thua YOLO26n 29% mà chỉ rẻ hơn 21% FLOPs, và bài không có module mới → chắc chắn reject.
Rung là trục duy nhất model thắng.

- **Claim chính:** detector nhỏ trên drone **sụp dưới rung** (thứ benchmark che);
  **vibration-aware training** (rung làm augmentation, train-time only, 0 chi phí inference)
  vá được; và hệ quả là **thứ hạng đảo chiều trên trục triển khai**.
- **5 đóng góp:**
  1. giao thức VisDrone-Shake **có seed + 3 seed + error bar**
  2. **vibration-aware training**: retain 16.8 → 36.2%, + đối chứng 120 epoch không rung (16.9%)
  3. **kết quả ÂM**: consistency (30.0%) THUA augmentation (36.2%) mà tốn 1.8× thời gian train
  4. **ranh giới áp dụng**: lặp lại trên SteadyDet-W (22.2 → 39.4%), **không có tác dụng gì**
     trên YOLO11n; đã bác 3 giả thuyết (COCO-pretrain / object scale / headroom)
  5. **KIẾN TRÚC GIỜ LÀ ĐÓNG GÓP** — nhưng phát biểu trên trục mCP, không phải clean AP

- **Vì sao (5) hợp lệ** (câu hỏi của tác giả 2026-07-27 "vậy đóng góp chính là gì, tôi viết model mà"):
  trên clean AP thì 4 con YOLO nano đều thắng SteadyDet-N (0.2768 vs 0.1892) → claim kiến trúc
  ở đó là tự sát. Nhưng trên **mCP** (AP trung bình khi ĐANG RUNG = điều kiện bay thật):

  | model | M | GFLOPs | clean AP | sev2 | **mCP** |
  |---|---|---|---|---|---|
  | SteadyDet-W + vib-aware | 1.00 | 12.9 | 0.2219 | 0.0839 | **0.0874** ← nhất bảng |
  | RT-DETR-l | 32.81 | 108.0 | 0.2502 | 0.0389 | 0.0641 |
  | **SteadyDet-N + vib-aware** | **1.17** | **4.6** | 0.1708 | **0.0518** | **0.0617** |
  | YOLO11-n | 2.59 | 6.4 | **0.2768** | 0.0210 | 0.0557 |
  | YOLOv10-n / v8-n / 26-n | 2.5–3.0 | 5.8–8.4 | ~0.27 | ~0.021 | 0.0507–0.0538 |
  | SteadyDet-N baseline | 1.17 | 4.6 | 0.1892 | 0.0123 | 0.0318 ← bét bảng |

  **Kết luận đo được:** SteadyDet-N+vib thắng CẢ 4 YOLO nano trên mCP, và thắng RT-DETR-l
  (28× tham số) **ở sev2** (5.4σ). ⚠️ **KHÔNG** thắng RT-DETR trên mCP trung bình (0.0617 <
  0.0641) và ở **sev3 chỉ 0.64σ = trong nhiễu, KHÔNG được claim.** Bài đã viết đúng mức này.

- **Vì sao so sánh này công bằng:** kết quả âm (4) chính là thứ hợp thức hoá nó. Vì kỹ thuật
  **không** nâng được YOLO11n (mCP 0.0557 → 0.0542), nên "của ta có xử lý" vs "YOLO11n có xử lý"
  vẫn là ta thắng. Nếu kỹ thuật nâng mọi model thì so sánh đó ăn gian. Chỉ mới thử trên YOLO11n
  → với 3 baseline còn lại thì phản biện này CHƯA có câu trả lời, và bài nói thẳng như vậy.

## 2. Nguồn số — MỘT thước đo duy nhất
### 2a. `runs/paper_queue/UNIFIED.json`  ← Bảng 2, 3, 4 (bảng chính)
Sinh bởi **`paper_eval_unified.py`**. Đây là file DUY NHẤT dùng cho mọi bảng accuracy trên VisDrone-val.
Nó vá 2 lỗ hổng của các file cũ:
1. **1 giao thức cho MỌI model** (pfdet *và* ultralytics): cùng loader letterbox 640, cùng
   `conf=0.01 / nms_iou=0.5 / max_det=300`, cùng `eval_coco.coco_eval` (pycocotools).
   *(File cũ: pfdet đo pycocotools, YOLO đo `ultralytics.val()` → KHÔNG so được.)*
2. **Seed cố định + 3 seed → mean ± std.** `corrupt_batch()` random hướng blur/nhiễu/sáng nên
   `eval_robustness.py` không seed cho ra retain khác nhau mỗi lần chạy (đo được 17.0 / 17.1 / 17.6
   cho cùng 1 checkpoint) → số 1 lần đo là 1 lần rút thăm.

Chạy lại: `python paper_eval_unified.py --seeds 0 1 2 --out runs/paper_queue/UNIFIED.json`
(resume được: model đã có trong JSON thì bỏ qua). Log: `runs/paper_queue/unified_eval.log`.

### 2a-bis. Bảng số ĐÃ ĐO (VisDrone-val 548 @640, 1 giao thức, 3 seed)
| model | params(M) | GFLOPs | clean AP | AP50 | AP_s | retain% |
|---|---|---|---|---|---|---|
| RT-DETR-l | 32.81 | 108.0 | 0.2502† | 0.5909 | 0.2447 | 25.6 ± 1.1 |
| YOLOv10-n | 2.71 | 8.4 | 0.2730 | 0.6043 | 0.2700 | 19.7 ± 1.4 |
| YOLO11-n | 2.59 | 6.4 | **0.2768** | **0.6119** | 0.2741 | 20.1 ± 1.2 |
| YOLOv8-n | 3.01 | 8.2 | 0.2684 | 0.6033 | 0.2648 | 19.5 ± 0.8 |
| YOLO26-n | 2.50 | 5.8 | 0.2675 | 0.5898 | 0.2646 | 19.0 ± 1.0 |
| SteadyDet-W | 1.00 | 12.9 | 0.2349 | 0.5800 | 0.2324 | 22.2 ± 0.9 |
| SteadyDet-W +vib (best) | 1.00 | 12.9 | 0.2207 | 0.5561 | 0.2188 | **30.5 ± 0.6** |
| SteadyDet-W +vib (last) | 1.00 | 12.9 | 0.2210 | 0.5505 | 0.2189 | 33.4 ± 0.3 ⚠️ khác quy tắc ckpt |
| SteadyDet-N | 1.17 | 4.6 | 0.1892 | 0.5127 | 0.1872 | 16.8 ± 0.7 |
| SteadyDet-N NMS-free | 1.17 | 4.6 | 0.1954 | 0.5213 | 0.1939 | 23.4 ± 0.8 |
| SteadyDet-N +vib | 1.17 | 4.6 | 0.1746 | 0.4837 | 0.1728 | **30.0 ± 0.6** |
| SteadyDet-N deploy(+COCO) | 1.17 | 4.6 | 0.1781 | 0.5016 | — | 16.6 ± 0.4 |
| SteadyDet-N deploy +vib | 1.17 | 4.6 | 0.1774 | 0.4946 | — | 30.2 ± 0.2 |

† RT-DETR bị thiệt vì predictor gốc dùng `LetterBox(scale_fill=True)` = **kéo giãn**, còn loader repo
letterbox đệm xám → nhận hình học chưa calibrate. Native harness cho 0.3340. **retain% KHÔNG bị ảnh
hưởng** (là tỉ số 2 pass cùng pipeline). Paper ghi rõ bằng footnote — không được im lặng.

**Điểm mấu chốt cho paper:** MỌI detector đều sụp (mất 74-83% AP). YOLO nano 19-20% bất kể thế hệ.
RT-DETR 25.6% (28× params, 23× FLOPs) vẫn **thua** v17+vib 30.0% (1.17M). → retain% KHÔNG phải hàm
đơn điệu của clean AP → phải đo và tối ưu riêng. Đây là lập luận "đóng góp trực giao".

### 2b. `runs/paper_queue/BASELINES_TABLE.md` ← cột params / GFLOPs
Sinh bởi `summarize_baselines.py` (tính trực tiếp, không dùng `model.info` vì bug trả `None`).
SteadyDet-N = **1.171M / 4.6 GFLOPs**. Chỉ dùng file này cho **params/GFLOPs**; mAP trong đó đo bằng
`ultralytics.val()` → **KHÔNG dùng cho bảng paper**, dùng UNIFIED.json.

### 2c. `runs/paper_queue/SUMMARY_DEPLOY.txt` + `RESULTS_FLIGHT_VAL.md` ← Bảng 5 (bay thật)
92 frame gán nhãn tay, `data/flight_val/`, ~6m, từ `2026-07-21 09-43-12.mkv`. Cùng `coco_eval`.
| training data | vib-cons | AP50 | AP | AP75 | AR100 |
|---|---|---|---|---|---|
| VisDrone+COCO | – | 0.7613 | 0.3369 | 0.1448 | 0.5076 |
| VisDrone+COCO | ✓ | **0.8798** | **0.3885** | **0.2620** | 0.5087 |
| VisDrone only | – | 0.1435 | 0.0547 | 0.0166 | 0.4315 |
| VisDrone only | ✓ | 0.0909 | 0.0283 | 0.0093 | 0.3772 |

→ 2 hiệu ứng ĐỘC LẬP: (a) **domain data**: +COCO ×5.3 AP50 (và benchmark xếp hạng ĐẢO CHIỀU);
(b) **vibration-consistency trên rung THẬT**: AP75 0.1448→0.2620 (**+81%**) mà recall gần như không đổi
→ không phải "bắt được nhiều người hơn" mà là "box chặt hơn" = đúng dấu hiệu nhìn xuyên nhoè.
→ Nhánh no-COCO vib TỆ HƠN (0.0909 < 0.1435): **báo thẳng** — cả 2 đều thất bại ở domain này,
so sánh giữa 2 cái thất bại thì vô nghĩa; vib là robustness treatment, KHÔNG phải domain adaptation.

### 2d. Biểu đồ
- pfdet: `runs/train_v17_*/training_curves.png`, `runs/train_v15_*`.
- YOLO: `runs/yolo/<model>/` + `runs/yolo_deploy/<model>/`: `results.png`, `PR_curve.png`, `F1_curve.png`.
- Hình paper: `paper/figures/fig1..4*.pdf` + script `.py` sinh ra chúng (chạy trên máy, không chạy trên Overleaf).
- Demo bay: `demo_flight/frames/demo_*.jpg`.

## 3. TRẠNG THÁI 2026-07-27 — thí nghiệm đã XONG HẾT

Không còn job nào đang train. Mọi cron `@reboot` đã tắt (`#DISABLED-2026-07-26`,
backup ở `runs/crontab_backup_2026-07-26.txt`), nên bật lại máy KHÔNG tự chạy lại gì.

### Đã đo xong, đã vào paper
| thí nghiệm | kết quả | vào mục nào |
|---|---|---|
| v17 + rung-augmentation | retain **36.2 ± 0.7%** (baseline 16.8) | Bảng 1, 4.2 |
| v17 finetune 120ep KHÔNG rung (đối chứng) | **16.9 ± 0.7%** → train thêm vô nghĩa | Bảng 1, 4.2 |
| v15 (SteadyDet-W) + augmentation | **39.4 ± 0.3%** (baseline 22.2) | Bảng 1, generality |
| YOLO11n + augmentation (pretrained) | 20.1 → **20.1%** — không nhúc nhích | Bảng 1+2, 4.2 |
| YOLO11n + consistency | 20.1 → **19.2%** | 4.2 |
| **YOLO11n from scratch** (bác giả thuyết COCO) | 21.9 → **21.5%**, đối chứng 21.4 | 4.2 |
| bảng baselines + cột mCP | xem mục 1 | Bảng 2, 4.4 |
| bay thật 6 dòng, 1 code path | `runs/paper_queue/FLIGHT.json` | **ĐÃ CẮT khỏi bài** |

### Mục 4.5 (bay thật) — ĐÃ CẮT khỏi bản nộp
Quyết định của tác giả 2026-07-27 để vừa 12 trang. Lý do: 92 frame / 1 người / 1 cảnh /
quay màn hình GCS = bằng chứng yếu nhất bài. File `paper/sections/12_exp_flight.tex`
**vẫn còn và đã cập nhật đủ 6 dòng số thật**; `\input` đã comment trong `main.tex`.
Sau chuyến bay đầu tháng 8 (theo `FLIGHT_DATA_SPEC.md`) thì đưa lại — lúc đó sẽ mạnh hơn nhiều.

Số bay thật đã đo (1 code path, `flight_eval_unified.py`):

| train data | xử lý | AP50 | AP | AP75 | AR100 |
|---|---|---|---|---|---|
| VisDrone+COCO | không | 0.7613 | 0.3369 | 0.1448 | 0.5076 |
| VisDrone+COCO | vib-aware | 0.7747 | 0.3792 | 0.2583 | 0.5315 |
| VisDrone+COCO | consistency | 0.8798 | 0.3885 | 0.2620 | 0.5087 |
| VisDrone only | không | 0.1435 | 0.0547 | 0.0166 | 0.4315 |
| VisDrone only | vib-aware | 0.0579 | 0.0207 | 0.0076 | 0.4076 |
| VisDrone only | consistency | 0.0909 | 0.0283 | 0.0093 | 0.3772 |

**Bootstrap 1000 lượt trên frame (ghép cặp), khoảng tin cậy 95%** — có trong `FLIGHT.json`:

| so sánh | AP50 | AP | AP75 |
|---|---|---|---|
| vib-aware − baseline | +0.013 [−0.032, +0.056] | **+0.042 [+0.011, +0.074]** | **+0.108 [+0.010, +0.211]** |
| consistency − baseline | **+0.119 [+0.071, +0.173]** | **+0.051 [+0.012, +0.087]** | **+0.110 [+0.012, +0.214]** |
| consistency − vib-aware | **+0.106 [+0.058, +0.161]** | +0.009 [−0.023, +0.044] | +0.002 [−0.074, +0.073] |

⚠️ Cả hai xử lý đều thắng baseline THẬT (khoảng không chứa 0 ở AP và AP75). Hai xử lý chỉ
tách nhau ở **AP50** — và khoảng đó KHÔNG chứa 0, nên trong 92 frame này là thật. Nhưng thứ
tự đó **ngược** với VisDrone-Shake (augmentation thắng 6.2 điểm). 1 người / 1 cảnh / 1 độ cao
không quyết được cái nào tổng quát hơn → cần tập bay lớn hơn, đừng cãi thêm từ 92 frame này.

### Còn phải làm (người)
- [ ] Compile Overleaf → **đếm số trang thật**. Ước lượng ~12.25, giới hạn 12.
      Nếu dư, cắt theo thứ tự trong `paper/README_OVERLEAF.txt`.
- [ ] Điền `\TODO`: tên+email GVHD, ORCID (`01_titlepage.tex`), Acknowledgements (`main.tex`).
- [ ] Bay đầu tháng 8 theo `FLIGHT_DATA_SPEC.md` → đưa lại mục 4.5.
- [x] **Hailo: ĐÃ ĐO XONG (2026-08-24 + 2026-08-27) → mục 4.6 đã viết lại, bỏ TODO.**

## 6. Cần thu thập gì trên Hailo/Pi 5 (nếu muốn giữ mục 4.6)

> ✅ **XONG 2026-08-27.** `paper/sections/13_exp_deploy.tex` đã viết lại bằng số đo thật,
> không còn `\TODO`. Nguồn số: `runs/pi_hailo_measured_2026-08-24.md` (AP INT8 + FPS NPU)
> và `runs/pi_camera_measured_2026-08-27.md` (e2e với camera thật). Tóm tắt cái đã thay:
>
> | claim cũ | claim mới |
> |---|---|
> | "sustains 186 FPS @320" (không nói batch) | 112.55 FPS @320 / 63.66 @640, **batch=1**, đo bằng `hailortcli benchmark`; nói rõ 186 là batch=4 và latency tăng 8.40→17.05 ms |
> | không có e2e | **30.0 FPS e2e, p50 32.04 ms, p99 36.15 ms** với camera thật, 2 phút, 48.5°C, không throttle |
> | không có AP INT8 | AP@[.50:.95] 0.1734→0.1516 (−12.6%), AP@0.50 0.4812→0.4585 (−4.7%), AP@0.75 −29.9% |
> | không nói nút cổ chai | tính toán 19.55 ms / ngân sách 33.3 ms = duty 59% → **camera-bound, không phải model-bound** |

*(Ghi chú lịch sử — vì sao con số 186 từng nguy hiểm)*
Con số cũ trong paper là **186 FPS @320, inference-only, EMA** — đo bởi
`hailo_demo_cam.py`. Reviewer của một bài deployment sẽ hỏi 4 thứ mà con số đó KHÔNG
trả lời. Đã viết sẵn 2 script:

### 6a. AP của model INT8 trên Hailo — `hailo_eval_val.py` 🔴 quan trọng nhất
Không có số này thì con số FPS vô nghĩa: repo này từng thấy INT8 PTQ làm AP sập
**0.18 → 0.009**. Hai bước để giữ **cùng một giao thức đo** với cả bài báo:
```bash
# TRÊN PI (chỉ chạy model, dump dự đoán)
python hailo_eval_val.py --hef pfdet_v17_640_hailo8l.hef --img-size 640 \
    --val-images data/visdrone/val/images --out hef_preds_640.json
# COPY về laptop rồi chấm bằng CHÍNH eval_coco.coco_eval của paper
python hailo_eval_val.py --score hef_preds_640.json \
    --val-images data/visdrone/val/images --val-labels data/visdrone/val/labels
```
Nó tự in luôn phần trăm AP mất so với FP32 trong `UNIFIED.json`.

### 6b. Latency end-to-end + p99 + nhiệt — `hailo_bench_latency.py`
```bash
python hailo_bench_latency.py --hef pfdet_v17_640_hailo8l.hef --img-size 640 \
    --minutes 10 --source cam        # --source synth nếu chưa cắm camera
```
In bảng từng chặng (capture / preprocess / inference / decode+NMS / total) với
**p50 / p95 / p99**, FPS theo từng phút, nhiệt SoC và `vcgencmd get_throttled`.
Ba lý do phải có p99 chứ không phải mean: bộ điều khiển chết vì frame **chậm nhất**;
decode+NMS phụ thuộc **số box** nên cảnh đông người là lúc latency phình; Pi 5 tản
nhiệt bị động nên FPS 10 giây đầu **không phải** FPS lúc bay.

### 6c. AP của model no-attention (0.875M) vs model đầy đủ (1.17M)
Compile Hailo phải **bỏ windowed attention** (dataflow compiler không hỗ trợ reshape của
nó) → model deploy KHÁC model trong bảng chính. Phải đo `runs/train_v17_hailo/best.pt`
bằng `paper_eval_unified.py` để biết mất bao nhiêu AP.

### 6d. Xác nhận claim "0 chi phí inference"
Paper viết *"the exported graph is structurally identical to the baseline"*. Kiểm bằng số:
export ONNX cả 2 (baseline và +vib) rồi so số node/param — phải trùng khít. Đây là claim
trung tâm, nên có 1 dòng bằng chứng thì tốt.

~~**Nếu không kịp đo 6a–6b thì CẮT HẲN mục 4.6** và bỏ con số 186 FPS.~~ → **Không cần
cắt nữa: 6a và 6b đều đã đo xong**, mục 4.6 giờ mang số thật kèm p99, nhiệt độ và
điều kiện đo.

## 4. Đã vá trong phiên 2026-07-25
- ✅ `paper_eval_unified.py` — 1 giao thức + 3 seed cho mọi model (xem 2a).
- ✅ `train_v3.py`: thêm `vib.mode: aug` + `vib.aug_prob` → chạy được ablation augment-only.
- ✅ `configs/train_config_v17_vib_augonly.yaml` — đối chứng, cùng schedule với `train_config_v17_vib.yaml`.
- ✅ Checkpoint rule đồng nhất: đo CẢ `best.pt` và `last.pt` cho v15_vib_E, paper dùng cùng quy tắc
  với baseline (best theo clean val AP) — không được lấy last cho model mình, best cho baseline.
- ✅ `train_yolo_vibcons.py` — áp vib-consistency lên detector KHÔNG phải của mình (ghi đè
  `model.loss()` của ultralytics, không sửa thư viện). Box consistency dùng **KL trên bin DFL** vì
  YOLO biểu diễn box bằng phân phối. Đã smoke-test đủ 3 epoch + val + save.
- ✅ `ablation_queue.sh` + cron @reboot — chạy tuần tự 3 thí nghiệm rồi tự đo lại + sinh bảng.
- ✅ `configs/train_config_v17_ft_control.yaml` — đối chứng finetune không rung.
- ✅ `paper/PENDING_ABLATION.md` — văn sẵn cho từng kết quả có thể xảy ra.
- ✅ `paper/` — bản viết lại: kiến trúc rút từ 3 trang → 1 trang, thêm error bar, thêm bảng 1 giao
  thức, thêm kết quả bay thật của vibration (trước đây BỎ SÓT), Limitations nói thẳng mục 3.1.

## 5. Code liên quan
`models/pfdet_nano_v17.py`, `vib_consistency.py` (đóng góp), `utils/losses_v15.py`,
`utils/losses_v17_e2e.py`, `eval_coco.py`, `eval_robustness.py`, `paper_eval_unified.py`,
`summarize_baselines.py`, `train_yolo_baselines*.py`.
