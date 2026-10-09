# PENDING_ABLATION.md — sửa paper thế nào khi 3 thí nghiệm về đích

> Hàng chờ: `ablation_queue.sh` (đã chạy, có cron @reboot). Theo dõi:
> `tail -f runs/paper_queue/ablation_queue.log`
> Xong hết khi có `runs/paper_queue/ABLATION_ALL_DONE`.
> Số cuối: `sed -n '/^====/,$p' runs/paper_queue/unified_eval.log | tail -25`
> Dòng LaTeX: `runs/paper_queue/TABLES.tex` (sinh bởi `paper/make_tables.py`).

Mốc so sánh đã có (VisDrone-val, 1 giao thức, 3 seed):

| model | clean AP | retain% |
|---|---|---|
| v17 baseline | 0.1892 | 16.8 ± 0.7 |
| **v17 + vib-consistency** | 0.1746 | **30.0 ± 0.6** |
| yolo11n baseline | 0.2768 | 20.1 ± 1.2 |

---

## 1. `v17-vib-augonly` — ablation QUAN TRỌNG NHẤT

Tách "model thấy ảnh rung" khỏi "consistency term". Đọc `retain%` của nó rồi chọn 1 trong 3 nhánh.

### Nhánh A — augonly rõ ràng THẤP HƠN vibcons (ví dụ ~20-25% vs 30.0%)
**Đây là kết quả tốt.** Consistency term có giá trị riêng, không chỉ là augmentation.

- `sections/09_exp_main.tex`: thêm 2 dòng vào bảng chính (dán từ `TABLES.tex`), và thêm đoạn:
  > \subsubsection{The consistency term, not the corrupted inputs, produces the gain.}
  > Training on the identical corruption as plain augmentation, under an identical schedule,
  > learning rate and augmentation pipeline, reaches only $X\%$ retention against $30.0 \pm 0.6\%$
  > for vibration-consistency. Exposure to corrupted inputs therefore accounts for roughly
  > $Y$ of the $13.2$-point improvement and the consistency term for the remaining $Z$. This is the
  > decomposition the method claims: augmentation supplies hard inputs, while the consistency term
  > supplies the statement that the two views depict the same scene.
- `sections/14_limitations.tex`: **XOÁ HẲN** mục `\subsubsection{The augmentation control is missing.}`
  (giữ lại 1 câu về đối chứng finetune nếu `ft_control` chưa xong).
- `sections/07_vibcon.tex`: trong mục "Why consistency rather than augmentation", đổi câu
  *"This is an argument, not yet a measurement..."* thành
  *"Section~\ref{sec:mainresults} measures this decomposition directly."*
- `sections/02_abstract.tex`: thêm 1 câu sau câu kết quả:
  > An augmentation-only control under an identical schedule reaches only $X\%$, isolating the
  > consistency term as the source of most of the gain.
- `sections/03_introduction.tex`: đóng góp (ii) thêm *", of which an augmentation-only control
  attributes the majority to the consistency term itself"*.

### Nhánh B — augonly XẤP XỈ vibcons (chênh trong khoảng ±2×std, tức < ~1.5 điểm)
**Phải đổi khung bài, đây là tình huống trung thực khó.** KHÔNG được im lặng.

- Đóng góp chính chuyển thành: **giao thức VisDrone-Shake + phát hiện cả lớp detector đều sụp +
  cách vá rẻ**; consistency term trở thành *một cách hiện thực hoá*, không phải cơ chế then chốt.
- Đổi title bỏ chữ "Consistency": ví dụ
  *"Vibration-Aware Training: Zero-Cost Robustness for Tiny-Person Detection On-Board Vibrating UAVs"*.
- `09_exp_main.tex` thêm đoạn:
  > \subsubsection{The gain comes from the corruption, not from the consistency term.}
  > An augmentation-only control under an identical schedule reaches $X\%$ retention, statistically
  > indistinguishable from the $30.0 \pm 0.6\%$ of vibration-consistency. We report this because it
  > changes what the paper claims: the source of the robustness is exposure to the corruption, and
  > the consistency formulation is one way of obtaining it rather than the mechanism responsible.
  > The practical conclusion --- that vibration robustness is available at zero inference cost by a
  > training-time change --- is unaffected; the mechanistic explanation is.
- `14_limitations.tex`: đổi mục 1 thành *"the consistency term is not identified as the mechanism"*.
- Bỏ chữ "self-distillation" ra khỏi abstract (không còn là điểm bán).
- **Giữ nguyên** tất cả kết quả khác: bay thật, generality, bảng baseline. Bài vẫn nộp được.

### Nhánh C — augonly CAO HƠN vibcons
Báo thẳng: augmentation đơn thuần tốt hơn. Đổi headline model thành augonly, giữ vibcons làm ablation
âm. Điều này vẫn là paper hợp lệ (kết quả âm có kiểm soát), và trung thực hơn nhiều so với giấu đi.

---

## 2. `v17-ft-control` — đối chứng "train thêm 120 epoch, KHÔNG rung"

- **Nếu retain% ≈ baseline (~16.8%)**: tốt. Thêm 1 câu vào `09_exp_main.tex`:
  > Continuing to finetune the baseline for the same $120$ epochs without vibration leaves retention
  > at $X\%$, so the improvement is not an artefact of the additional training.
  Và xoá câu *"Relatedly, the regulariser is applied as a finetuning stage..."* trong `14_limitations.tex`.
- **Nếu retain% tăng đáng kể**: trừ phần đó ra khỏi gain đã báo, và nói rõ trong `09_exp_main.tex`
  rằng $X$ trong số $13.2$ điểm là do train thêm.

---

## 3. `yolo11n-vibcons` — generality NGOÀI họ model của mình

⚠️ Kiểm tra trước: phải có `runs/paper_queue/A3_yolo_vibcons_done`.
Nếu là `A3_yolo_vibcons_SUSPECT` thì nhánh rung **không thực sự chạy** → số vô nghĩa, đừng dùng.

- **Nếu retain% > 20.1% (baseline yolo11n)**: đây là nâng cấp lớn nhất của bài.
  - `sections/10_exp_generality.tex`: thêm 1 bảng/2 dòng và viết:
    > Applied without modification to YOLO11-n --- a detector we did not design, with a different
    > backbone, a distributional box representation and a different loss and assignment scheme ---
    > the same regulariser raises retention from $20.1 \pm 1.2\%$ to $X\%$. The box consistency term
    > operates on the DFL bin distribution rather than on direct coordinates, which is the only change
    > the transfer required. The claim is therefore not about our architecture.
  - **XOÁ** mục `\subsubsection{Scope of this evidence.}` (không còn đúng).
  - `14_limitations.tex`: xoá câu *"The generality evidence covers two architectures from one family"*.
  - `02_abstract.tex` + `03_introduction.tex`: đóng góp (iii) đổi thành generality trên **3 detector,
    2 họ**, trong đó có 1 detector không phải của tác giả. Đây là câu đáng giá nhất thêm được.
  - `sections/11_exp_baselines.tex`: đổi câu *"Establishing that directly is the experiment we most
    want and have not run"* thành *"Section~\ref{sec:generality} establishes this directly on
    YOLO11-n."*
- **Nếu KHÔNG cải thiện**: báo thẳng trong `10_exp_generality.tex`, và thu hẹp claim về đúng họ
  model của mình. Nói rõ khác biệt (DFL vs regression trực tiếp) như giả thuyết giải thích.

### Vì sao YOLO11-n mà không phải YOLO26-n (câu này PHẢI có trong bài)
Reviewer sẽ hỏi "26 mới hơn, sao không dùng?" nếu bài không nói lý do. Ghép câu sau vào
đoạn generality (đã viết sẵn, chỉ cần dán):

> We apply it to YOLO11-n rather than to the newest generation for two reasons. First, it
> is the strongest of the nano baselines on this task, on both clean accuracy
> ($0.2768$ AP$_{50:95}$) and retention ($20.1 \pm 1.2\%$), so the transfer is tested
> against the hardest of them rather than the weakest. Second, and more importantly, its
> box head is \emph{distributional}: it predicts a $16$-bin distribution per box side,
> where our detector regresses coordinates directly. The box-consistency term therefore
> had to be reformulated as a Kullback--Leibler divergence between bin distributions,
> which makes the transfer a test of the idea rather than a re-use of our own
> parameterisation. YOLO26-n and YOLOv10-n regress directly, as we do, and would have
> made the transfer easier but less informative; they are also end-to-end detectors whose
> training forward returns separate one-to-many and one-to-one branches, so applying the
> regulariser to them requires deciding which branch to constrain, a design choice we
> leave to future work.

Số liệu chống lưng cho câu trên (đo thật, `paper_eval_unified.py` + ultralytics):

| model | reg_max | box | end2end | clean AP | retain% |
|---|---|---|---|---|---|
| yolo11n | 16 | **phân phối DFL** | không | **0.2768** | **20.1 ± 1.2** |
| yolov8n | 16 | phân phối DFL | không | 0.2684 | 19.5 ± 0.8 |
| yolo26n | 1 | regression trực tiếp (như SteadyDet) | **có** | 0.2675 | 19.0 ± 1.0 |
| yolov10n | 16 | phân phối DFL | **có** | 0.2730 | 19.7 ± 1.4 |

Lưu ý: yolo26n/yolov10n forward lúc train trả `{one2many, one2one}` chứ không phải
`{boxes, scores, feats}` → `train_yolo_vibcons.py` sẽ raise `NotImplementedError` với
thông báo rõ ràng. Muốn chạy chúng phải mở rộng `vib_consistency_dict()`.

**Nếu còn thời gian sau khi yolo11n xong:** chạy thêm yolov8n (cùng cấu trúc DFL, không
end2end → script chạy được ngay, chỉ đổi `--weights`). Hai detector ngoài họ thì claim
generality thành "2 họ, 4 detector" — mạnh nhất có thể, và rẻ vì không phải sửa code.

---

## 3-bis. QUYẾT ĐỊNH: có bỏ SteadyDet-W khỏi bài không?

Chỉ trả lời được SAU khi có kết quả `yolo11n-vibcons`. Ba vai trò đang phân bố như sau:

| model | có cặp baseline/+vib? | gánh claim gì |
|---|---|---|
| SteadyDet-N | ✅ 16.8 → 30.0% | kết quả chính |
| SteadyDet-W | ✅ 22.2 → 30.5% | **generality (đóng góp iii)** |
| Nano NMS-free | ❌ chỉ 23.4%, KHÔNG có bản +vib | không gánh claim nào → **đã cắt khỏi bài** |
| yolo11n | ✅ 20.1 → ? | generality NGOÀI họ model của mình |

- **Nếu `yolo11n-vibcons` CẢI THIỆN** (> 20.1%): generality do YOLO11n gánh, mạnh hơn Wide
  nhiều vì đó là detector của người khác. Lúc này **bỏ SteadyDet-W được**, tiết kiệm ~0.3 trang:
  xoá 2 dòng khối "SteadyDet-W" trong bảng ở `09_exp_main.tex`, xoá mục
  "The same regulariser on a different architecture" + "A note on checkpoint selection",
  xoá cặp `("v15-baseline","v15-vibcons",...)` trong `paper/figures/make_fig4_tradeoff.py`,
  và sửa đóng góp (iii) trong abstract + intro thành YOLO11n.
  ⚠️ Nếu bỏ Wide thì **mất luôn** mục "A note on checkpoint selection" — đó là điểm cộng
  trung thực với reviewer. Cân nhắc chuyển 1 câu của nó sang Limitations.
- **Nếu KHÔNG cải thiện**: Wide là bằng chứng generality DUY NHẤT → **tuyệt đối giữ**, và
  phải nói thẳng trong `10`/`09` rằng kỹ thuật chưa transfer sang họ model khác, kèm giả
  thuyết (DFL vs regression trực tiếp).

**KHÔNG BAO GIỜ** thay Wide bằng NMS-free: NMS-free không có bản +vib nên là một điểm đơn,
không phải một cặp, nên không chứng minh được gì về regulariser.

---

## 4. Sau khi sửa xong — bắt buộc

1. `python paper/make_tables.py` → dán lại MỌI dòng bảng (đừng chép tay).
2. `cd paper/figures && python make_fig1_teaser.py` (hình đọc số từ JSON, tự cập nhật).
3. Đối chiếu số: script ở cuối phiên trước — mọi `0.XXXX` trong `sections/` phải truy được về
   `UNIFIED.json` hoặc flight_val.
4. Đếm lại số trang trên Overleaf. Thêm 3 kết quả này sẽ **tăng ~0.5-0.8 trang** → cắt theo
   thứ tự trong `README_OVERLEAF.txt`. Lưu ý: nhánh A cho phép xoá mục Limitations #1 (tiết kiệm
   ~0.25 trang) và nhánh 3-tốt cho phép xoá "Scope of this evidence" (~0.15 trang).
5. Cập nhật `PAPER_INPUTS.md` mục 3 (bỏ dòng đã xong).
