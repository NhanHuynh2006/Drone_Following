# Nội dung chi tiết từng slide — bảo vệ nghiệm thu SV2026-474

File này giải thích **từng chữ trên mỗi slide nghĩa là gì, vì sao có số đó, và nói
ra sao cho đúng** — để bạn hiểu bản chất chứ không học thuộc lòng. Dùng cùng với:

- `SteadyDet-N_nghiem_thu.pptx` — bản trình chiếu chính (25 slide, có video).
- `kich_ban_15phut.md` — bảng bấm giờ tổng và thứ tự cắt nếu thiếu giờ.
- Ghi chú trong mỗi slide (Notes/Presenter View) — lời dẫn ngắn gọn hơn bản này,
  dùng để đọc lướt lúc đang nói; bản này dùng để **học hiểu trước khi lên**.

Quy ước đọc: mỗi mục có **Trên slide** (nguyên văn chữ/bảng/số hiện ra),
**Nghĩa là gì** (giải thích để hiểu, không phải học vẹt), và **Nói sao cho đúng**
(câu nói gợi ý, có thể diễn đạt lại bằng lời của bạn miễn giữ đúng số liệu).

---

## Slide 1 — Trang bìa

**Trên slide:** Logo trường HCMUTE, logo ISLab, tên khoa, dòng "BÁO CÁO NGHIỆM
THU ĐỀ TÀI NCKH SINH VIÊN — MÃ SỐ SV2026-474", tên đề tài, chủ nhiệm/thành
viên/GVHD, địa điểm và tháng.

**Nói:** Chào Hội đồng, giới thiệu tên, mã số đề tài, tên đề tài, người hướng
dẫn. Khoảng 20 giây, không cần đọc hết chữ trên bìa vì Hội đồng tự đọc được.

---

## Slide 2 — Vấn đề và khoảng trống nghiên cứu

**Trên slide:**
- 4 gạch đầu dòng: UAV dùng nhiều trong giám sát/cứu hộ/quay phim nhưng người
  lái vừa lái vừa phải giữ người trong khung hình → cánh quạt làm camera rung
  liên tục (ảnh nhoè, nhiễu, dao động sáng) → đo 5 bộ phát hiện nhỏ gọn thì tất
  cả **mất 74–83% độ chính xác khi rung** → các bộ đánh giá chuẩn (benchmark)
  chỉ đo trên ảnh sạch nên không ai thấy lỗ hổng này.
- Ảnh minh hoạ: ảnh sạch cạnh ảnh rung ở 3 mức.
- Khối kết luận: "Cần một bộ phát hiện chịu được rung, đủ nhẹ để chạy ngay
  trên máy bay."

**Nghĩa là gì:** Đây là *lỗ hổng nghiên cứu* mà đề tài lấp vào. Các mô hình
phát hiện người phổ biến (YOLO, RT-DETR...) được đo trên ảnh tĩnh, sạch. Khi
gắn lên UAV thực tế, cánh quạt rung làm ảnh mờ nhoè — đây là loại suy giảm
ảnh **khác** với làm mờ ngẫu nhiên, nó có hướng (theo trục rung) và đi kèm
nhiễu + dao động sáng. Số 74–83% là do chính nhóm đo bằng thực nghiệm (không
phải trích dẫn), chứng minh vấn đề có thật chứ không suy diễn.

**Nói sao cho đúng:** "Khoảng trống không nằm ở việc chưa có mô hình phát hiện
người tốt — mà ở chỗ *chưa ai đo độ chính xác của các mô hình đó khi camera
rung*, trong khi đó là điều kiện thật khi gắn lên UAV."

**Nếu bị hỏi:** *"Sao biết là 74–83%, đo thế nào?"* → Trả lời: dùng giao thức
VisDrone-Shake tự dựng (giải thích ở slide 5), đo trên 5 mô hình YOLO/RT-DETR
công khai, lặp 3 lần với seed cố định.

---

## Slide 3 — Mục tiêu và tổng quan hệ thống

**Trên slide:**
- 3 ô mục tiêu: **≥ 70%** (chỉ tiêu AP@0,5 nhận dạng người, thời gian thực) ·
  **Bám tự động** (giữ khoảng cách và hướng bay phù hợp) · **Trên máy bay**
  (camera – xử lý ảnh – điều khiển, không cần Internet).
- Sơ đồ khối toàn hệ thống: Camera → NPU Hailo-8L chạy SteadyDet-N INT8 →
  OC-SORT (theo vết) → Chọn & khoá mục tiêu → Ước lượng khoảng cách (pinhole +
  Kalman) → 3 bộ PID → Lớp an toàn → MAVLink 30 Hz → Pixhawk 6C (ArduPilot).
  Có nhánh "Tay điều khiển RC (người lái)" nối ngược vào, ghi "ưu tiên cao
  nhất".
- Chú thích dưới: "Raspberry Pi 5 + NPU Hailo-8L + camera USB, nối Pixhawk 6C
  (ArduPilot) qua MAVLink 30 Hz".

**Nghĩa là gì:** Đây là 3 mục tiêu **trích đúng từ thuyết minh đã duyệt**
(không phải nhóm tự đặt ra), nên khi đối chiếu ở slide 24 nó phải khớp với 3
dòng này. Sơ đồ là toàn bộ đường đi của một khung hình: từ lúc camera chụp
tới lúc máy bay nhận lệnh.

**Nói sao cho đúng:** Đọc theo mũi tên từ trái qua phải, nhấn mạnh **tay điều
khiển RC luôn có quyền cao nhất** — đây là nguyên tắc an toàn xuyên suốt toàn
bộ đề tài, sẽ nhắc lại ở slide 19–20.

**Nếu bị hỏi:** *"Vì sao chọn Hailo-8L chứ không phải Jetson?"* → Rẻ hơn, tiêu
thụ điện ít hơn, gắn thẳng lên Raspberry Pi 5 qua khe M.2 mà không cần máy
tính rời; đo thật được 63,66 FPS ở ảnh 640 (slide 16).

---

## Slide 4 — Dữ liệu và giao thức đánh giá

**Trên slide:**
- 4 gạch đầu dòng: VisDrone-person (huấn luyện + so sánh chuẩn, người rất
  nhỏ) · COCO-person (bổ sung người cự ly gần, đúng miền triển khai) · 92
  khung bay thật gán nhãn tay (chỉ dùng để đánh giá) · VisDrone-Shake (3 mức
  rung mô phỏng; nhoè không làm dịch vật nên nhãn giữ nguyên).
- Ảnh minh hoạ 2 miền dữ liệu (VisDrone bay cao / dữ liệu bay thật ~6 m).
- Khối "Kỷ luật đo": cố định seed, lặp 3 lần, báo cáo độ lệch chuẩn, dùng
  chung một chương trình đo. Ví dụ: không cố định seed → cùng một mô hình ra
  17,0 / 17,1 / 17,6%.

**Nghĩa là gì:** Đây là nền tảng phương pháp luận cho **mọi** con số sau này
trong bài. Ba nguồn dữ liệu phục vụ ba việc khác nhau:
1. VisDrone-person: để **so sánh công bằng** với 5 mô hình đối chứng (ai cũng
   train/test trên cùng bộ này).
2. COCO-person: bù lại vì VisDrone chụp từ rất cao, người cực nhỏ — không
   giống cảnh UAV bám người ở 3–6 m. Thêm ảnh người cỡ lớn này giúp mô hình
   học đúng kích thước mục tiêu thật khi bay thấp.
3. 92 khung bay thật: đây là tập **duy nhất đến từ chính chuyến bay thật**,
   dùng làm "trọng tài cuối cùng" — không dùng để train, chỉ để kiểm tra.

Con số "17,0/17,1/17,6%" rất quan trọng khi trình bày: nó chứng minh **nếu
không cẩn thận về phương pháp đo, kết luận khoa học có thể sai** — chỉ vì
chọn ngẫu nhiên khác nhau mà cùng một mô hình cho kết quả lệch nhau gần 1
điểm phần trăm, đủ để đảo ngược xếp hạng giữa các mô hình gần nhau.

**Nói sao cho đúng:** "Ba việc tách bạch: VisDrone để so sánh công bằng, COCO
để đúng miền triển khai, 92 khung bay thật để kiểm chứng cuối cùng — và tất
cả đều đo bằng kỷ luật cố định seed, lặp 3 lần."

---

## Slide 5 — Giao thức đánh giá độ bền với rung

**Trên slide:**
- Bảng 3 mức rung (đây là thước đo cốt lõi của cả đề tài):

| Mức | Nhoè chuyển động | Nhiễu σ | Dao động sáng | Tình huống bay |
|---|---|---|---|---|
| 1 | 5 px | 0,02 | ±10% | bay treo, gió nhẹ |
| 2 | 9 px | 0,04 | ±20% | **bay tiến thông thường — lúc bám người** |
| 3 | 15 px | 0,06 | ±30% | gió mạnh, cơ động gấp |

- 3 gạch đầu dòng định nghĩa: Nhãn giữ nguyên (nhoè không làm dịch vật) · retain
  = AP khi rung / AP ảnh sạch · mCP = AP trung bình trên cả ba mức rung.
- Khối "Kỷ luật đo áp cho MỌI mô hình".

**Nghĩa là gì — đây là slide quan trọng nhất về phương pháp, phải hiểu kỹ:**

- **Vì sao có 3 mức, không phải 1?** Vì rung không phải một mức cố định —
  tuỳ tình huống bay (treo tại chỗ, bay tiến, gió mạnh) mà mức rung khác
  nhau. Mức 2 được đánh dấu quan trọng nhất vì **đó chính là lúc UAV đang bám
  người** — tình huống vận hành thật của đề tài.
- **Ba tham số của mỗi mức nghĩa là gì?**
  - *Nhoè chuyển động (motion blur length, đơn vị pixel)*: độ dài vệt mờ do
    camera dịch chuyển trong lúc cảm biến phơi sáng. Mức 1 chỉ 5 px (mờ nhẹ),
    mức 3 tới 15 px (mờ nặng, vật thể kéo vệt dài).
  - *Nhiễu cảm biến σ*: độ lệch chuẩn nhiễu Gauss cộng vào ảnh, mô phỏng hạt
    nhiễu khi ánh sáng yếu + rung làm camera phải tăng ISO.
  - *Dao động sáng*: độ sáng tổng thể ảnh dao động ±10% đến ±30%, mô phỏng
    hiện tượng độ sáng chớp nhẹ do rung ảnh hưởng tới bộ đo sáng tự động.
  - Ba hiệu ứng này được tổng hợp từ **mô hình vật lý thật của rung động
    UAV** (không phải tự đặt số cho đẹp) — xem Chương 1 báo cáo, mục suy
    giảm ảnh do rung động.
- **"Nhãn giữ nguyên" nghĩa là gì và vì sao quan trọng?** Khi làm nhoè ảnh
  (blur), vị trí hộp bao quanh người **không đổi** — người vẫn đứng y nguyên
  chỗ cũ trong ảnh, chỉ là ảnh bị mờ. Vì vậy không cần người ngồi gán nhãn
  lại, tự động áp phép biến đổi lên ảnh gốc đã có nhãn là xong. Đây là lý do
  giao thức này **dựng được nhanh và đo được hàng chục mô hình**.
- **retain và mCP khác nhau thế nào?**
  - *mCP* (mean Corrupted Precision) = độ chính xác trung bình **khi đang
    rung** — trả lời câu "bay thật thì mô hình chính xác bao nhiêu?"
  - *retain* = tỷ lệ phần trăm **giữ lại được** so với lúc đứng yên (ảnh
    sạch) — trả lời câu "mô hình mất bao nhiêu phần trăm khi rung?" Một mô
    hình có thể AP sạch rất cao nhưng retain rất thấp (tức là "giòn", dễ vỡ
    khi gặp rung) — đây chính là hiện tượng nhóm phát hiện ở slide 2.
- **Kỷ luật đo (cố định seed, lặp 3 lần):** là nguyên tắc thống kê bắt buộc để
  kết quả đáng tin. Ví dụ số 17,0/17,1/17,6% minh hoạ: nếu chỉ chạy 1 lần với
  seed ngẫu nhiên, con số đo được có thể lệch tới 0,6 điểm phần trăm — đủ để
  làm sai thứ hạng giữa hai mô hình gần nhau. Nhóm xử lý bằng cách luôn chạy 3
  lần với 3 seed cố định khác nhau, báo cáo **trung bình ± độ lệch chuẩn**.

**Nói sao cho đúng:** "Trước khi đo độ bền, phải có thước đo chuẩn. Nhóm xây
giao thức VisDrone-Shake ba mức, dựa trên mô hình vật lý thật của rung động,
không đổi nhãn. Mức 2 là mức quan trọng nhất vì tương ứng lúc máy bay đang
bám người. Hai chỉ số retain và mCP dùng xuyên suốt toàn bài."

---

## Slide 6 — Mô hình đề xuất: SteadyDet-N

**Trên slide:**
- 4 ô số liệu: **1,17 M** tham số (nhỏ nhất nhóm so sánh) · **4,6 GFLOPs**
  (rẻ hơn SteadyDet-W 2,8 lần) · **Nhánh P2** (giữ cho người nhỏ nhưng làm
  mỏng) · **LiteHead** (đầu dự đoán tách chiều sâu, dùng chung).
- Sơ đồ kiến trúc: Backbone → FPN neck (từ trên xuống) → Head, với nhánh P2
  vẽ nét đứt để nhấn mạnh nó mỏng hơn các nhánh khác.
- Chú thích: "Chi phí phân bổ không đối xứng: nhánh P2 (nét đứt) mỏng, tính
  toán dồn về các mức sâu".

**Nghĩa là gì — đây là đóng góp kiến trúc chính của đề tài:**

- **Vì sao cần nhánh P2?** P2, P3, P4, P5 là các mức độ phân giải khác nhau
  của bản đồ đặc trưng (feature map) trong mạng — P2 có độ phân giải cao
  nhất (bước nhảy/stride = 4), nhìn thấy chi tiết nhỏ nhất. Người chụp từ UAV
  ở xa có thể chỉ cao 8–16 pixel; nếu mô hình chỉ bắt đầu dự đoán từ P3 trở đi
  (như phần lớn YOLO bản nano) thì người nhỏ bị **mất thông tin** sau các
  lớp giảm mẫu, dễ bỏ sót.
- **Vì sao phải "làm mỏng" nhánh P2?** Vì chi phí tính toán của một lớp tỷ lệ
  thuận với *diện tích* bản đồ đặc trưng. P2 có độ phân giải gấp đôi P3 ở mỗi
  chiều → diện tích gấp 4 lần → cùng một lớp đặt ở P2 tốn **gấp 16 lần** so
  với đặt ở P3, và gấp 64 lần so với P4. Đây là lý do các mô hình khác né
  tránh P2 hoàn toàn. SteadyDet-N không né tránh mà **vẫn giữ P2 nhưng thiết
  kế nó mỏng đi** (ít kênh hơn, ít lớp tính toán hơn) để chi phí vẫn chấp
  nhận được, rồi dồn phần "dày" (nhiều tham số, nhiều phép tính) về các mức
  sâu hơn (P4, P5) — nơi chi phí mỗi lớp rẻ hơn nhiều.
- **LiteHead là gì?** Đầu dự đoán (phần cuối mạng, ra toạ độ hộp + điểm tin
  cậy) dùng tích chập tách theo chiều sâu (depthwise separable) và **dùng
  chung trọng số giữa các mức** thay vì mỗi mức có một đầu riêng như kiến
  trúc cũ — giảm thêm tham số mà không giảm độ chính xác đáng kể.
- **So với SteadyDet-W:** SteadyDet-W là phiên bản "rộng" (Wide) ra đời
  trước, chính xác hơn một chút nhưng tốn 12,9 GFLOPs. SteadyDet-N (Narrow/
  Nano) là bản tối ưu cho thiết bị biên, rẻ hơn 2,8 lần. Cả hai đều xuất
  hiện trong bảng đối chứng ở slide 9 và 11 để chứng minh phương pháp huấn
  luyện nhận biết rung hoạt động trên **cả hai** kiến trúc, không phải chỉ
  một trường hợp may mắn.

**Nói sao cho đúng:** "Điểm thiết kế cốt lõi: chi phí một lớp tỷ lệ với diện
tích bản đồ đặc trưng, nên thay vì né tránh độ phân giải cao như các mô hình
khác, SteadyDet-N vẫn giữ nhánh P2 cho người nhỏ nhưng làm mỏng nó, dồn tính
toán về các mức sâu."

### Lý do quan trọng nhất — một nguyên lý duy nhất giải thích mọi khối trong sơ đồ

Trước khi có SteadyDet-N, nhóm đã có bản **SteadyDet-W** — ít tham số hơn
(1,00 triệu, ít hơn cả SteadyDet-N) nhưng **tốn 12,9 GFLOPs**. Mổ xẻ chi phí
từng lớp phát hiện ra: **riêng head đặt ở P2 một mình đã ngốn 28% tổng chi
phí**, chỉ vì dùng 5 lớp tích chập đầy đủ trên đúng bản đồ lớn nhất (160×160).
→ Bằng chứng sống: *số tham số ít không có nghĩa là chạy nhanh* — vấn đề nằm
ở **chỗ đặt** phép tính, không phải **số lượng** phép tính.

Từ đó ra một nguyên tắc duy nhất, áp dụng nhất quán cho **mọi** khối trong sơ
đồ: **ở tầng bản đồ to (tốn) thì làm mỏng hết mức; ở tầng bản đồ nhỏ (rẻ) thì
mới "chơi sang".** Đây là lý do nên nói đầu tiên nếu Hội đồng hỏi "vì sao
thiết kế như vậy" — không liệt kê đặc điểm, mà chỉ ra một phép tính chi phí
duy nhất chi phối toàn bộ quyết định.

### Vì sao UIB (Universal Inverted Bottleneck) chứ không phải tích chập thường

Mỗi khối UIB bên trong là: tích chập **depthwise** 3×3 (xử lý riêng từng
kênh, rẻ) → mở rộng kênh bằng tích chập **1×1** → thu hẹp lại bằng 1×1. Lý do
chọn kiểu tách này thay vì một lớp tích chập thường, bằng đúng công thức chi
phí ở Chương 1: tích chập thường 3×3 từ $C_{in}$ sang $C_{out}$ kênh tốn
$3^2 \times C_{in} \times C_{out}$ phép tính mỗi điểm ảnh; tách thành
depthwise + 1×1 chỉ còn $3^2 \times C_{in} + C_{in}\times C_{out}$ — **rẻ hơn
gần 9 lần** khi $C_{out}$ lớn. UIB dùng ở **mọi** khối trong toàn mạng vì đó
là cách rẻ nhất để vừa nhìn được không gian xung quanh (depthwise) vừa trộn
được thông tin giữa các kênh (1×1).

### Vì sao CSP-UIB ở P3–P5 — và nhánh "đi tắt" là gì

Ở ba tầng P3, P4, P5 (mỗi tầng có 2 khối UIB), bản đồ được **chẻ đôi theo
kênh** thành hai nhánh trước khi vào 2 khối UIB đó:
- **Nhánh xử lý:** một nửa kênh đi qua lần lượt 2 khối UIB — nhánh "làm việc
  nặng".
- **Nhánh đi tắt (skip):** nửa kênh còn lại **không đụng tới gì cả**, đi
  thẳng một mạch, bỏ qua hoàn toàn 2 khối UIB.

Sau đó hai nhánh được ghép lại (concat) và trộn bằng một tích chập 1×1. Lợi
ích kép: (1) chỉ một nửa kênh phải chạy qua phần tốn kém → giảm gần một nửa
chi phí của cả cụm; (2) nhánh đi tắt giữ nguyên thông tin gốc, cho tín hiệu
sửa lỗi (gradient) một đường ngắn, sạch khi huấn luyện — giúp mạng học ổn
định hơn khi có nhiều khối xếp chồng. **Vì sao không bọc CSP ở P2?** Vì P2
chỉ có đúng 1 khối (đã làm mỏng tối đa) — chẻ đôi/ghép lại chỉ có lợi khi có
từ 2 khối trở lên để "rút gọn"; với 1 khối, thêm bước chẻ/ghép chỉ tốn thêm
mà không tiết kiệm được gì.

### Vì sao Conv 1×1 — và vì sao neck chỗ dùng Conv 1×1, chỗ dùng UIB

Trong cột FPN neck, hộp `n5` ghi "Conv 1×1" còn `n4`, `n3`, `n2` ghi "UIB" —
khác nhau vì:
- `n5` chỉ nhận **một nguồn duy nhất** từ Backbone, chỉ cần "đổi số kênh cho
  khớp" (192→48 kênh), không có gì để trộn — Conv 1×1 là cách rẻ nhất làm
  đúng việc đó (tích chập 1×1 áp dụng tại mỗi điểm ảnh chính là phép tuyến
  tính trộn kênh tại điểm đó, không có phép nào rẻ hơn cho đúng việc "chỉ đổi
  kênh, giữ nguyên không gian"; dùng tích chập 3×3 cho việc này tốn gấp 9 lần
  mà không được lợi gì thêm).
- `n4`, `n3`, `n2` đều có **hai mũi tên chụm vào** trước khi tới hộp: một từ
  mức trên phóng to lên (mang ngữ cảnh tổng quát), một từ Backbone chiếu
  ngang qua (mang chi tiết thô, đúng vị trí). Hai luồng này được **cộng**
  lại — nhưng cộng thô hai nguồn khác nhau chưa chắc đã hợp lý ngay, nên cần
  thêm một khối UIB **sau phép cộng** để mạng tự học cách hoà trộn hai nguồn
  thành một đặc trưng thống nhất. Ở `n2`, trước khi cộng còn có thêm một Conv
  1×1 phụ để ép 48 kênh của `n3` xuống 24 kênh cho khớp với nhánh P2 mỏng.

### Vì sao không dùng khối phổ biến khác (ResNet, self-attention khắp nơi...)

- **Khối ResNet** (2 lớp tích chập 3×3 đầy đủ): dùng tích chập thường chứ
  không depthwise — tốn hơn UIB nhiều lần cho cùng kích thước đầu ra, không
  hợp ngân sách 33,3 ms trên máy nhúng không quạt.
- **Self-attention ở khắp mọi tầng** (không chỉ riêng P5): chi phí
  self-attention tăng theo **bình phương** số ô trong bản đồ. P5 chỉ có
  20×20=400 ô (rẻ); P2 có 160×160=25.600 ô — gấp 64 lần số ô, mà chi phí
  tăng theo bình phương nên sẽ đắt hơn khoảng 4.000 lần. Đặt self-attention
  ở P2 sẽ phá nát ngân sách tính toán — đây là hệ quả trực tiếp của đúng
  nguyên lý gốc (chi phí tỷ lệ diện tích bản đồ), áp dụng luôn cho cả phép
  attention chứ không chỉ tích chập.
- **Tích chập lớn (5×5, 7×7) thay depthwise 3×3:** tốn hơn theo đúng công
  thức $k^2$ mà không cần thiết, vì receptive field rộng hơn đã được bù bằng
  cách xếp nhiều lớp 3×3 liên tiếp qua các tầng.

---

## Slide 7 — Quá trình huấn luyện

**Trên slide:**
- 4 gạch đầu dòng: Huấn luyện từ đầu, không dùng trọng số có sẵn (khác YOLO
  khởi tạo từ COCO) · 300 epoch trên VisDrone-person + COCO-person · 120
  epoch tinh chỉnh có rung (phần tăng độ bền) · AP sạch chỉ giảm nhẹ.
- Đồ thị 2 phần: trái là hàm mất mát (loss, thang log) theo epoch; phải là
  AP@0,5 kiểm định trên ảnh sạch theo epoch. Có điểm gãy rõ ở epoch 300 (ranh
  giới giữa train gốc và tinh chỉnh rung).

**Nghĩa là gì:**

- **"Huấn luyện từ đầu" là điểm khác biệt quan trọng cần nói rõ khi so sánh
  với YOLO** (sẽ gặp lại ở slide 9, 12): các mô hình YOLO đối chứng dùng
  trọng số đã huấn luyện sẵn trên bộ COCO đầy đủ (hàng triệu ảnh, nhiều lớp
  vật thể) rồi mới tinh chỉnh cho bài toán người. SteadyDet-N khởi tạo ngẫu
  nhiên, chỉ học từ đúng dữ liệu của đề tài. Đây là lý do chính khiến
  SteadyDet-N thua về AP trên ảnh sạch — **không phải vì kiến trúc kém**.
- **Hai giai đoạn huấn luyện:** 300 epoch đầu là huấn luyện "gốc" bình
  thường trên ảnh sạch (VisDrone + COCO), cho ra mô hình baseline. Sau đó
  300 epoch này được **đóng băng làm điểm xuất phát**, rồi chạy tiếp 120
  epoch tinh chỉnh (fine-tune) có đưa rung vào — đây chính là bước áp dụng
  phương pháp đề xuất ở slide 8.
- **Đọc đồ thị:** Ở epoch 300, hàm mất mát nhảy vọt lên rồi giảm trở lại —
  điều này *dự kiến*, vì đột ngột mô hình phải học thêm trên ảnh khó hơn
  (ảnh rung). AP trên ảnh sạch chỉ giảm nhẹ trong 120 epoch sau, chứng tỏ mô
  hình không "quên" khả năng cũ, chỉ học thêm kỹ năng chịu rung.

**Nói sao cho đúng:** "Khác với các mô hình YOLO khởi tạo từ COCO, SteadyDet-N
huấn luyện từ đầu hoàn toàn. 300 epoch gốc rồi 120 epoch tinh chỉnh có rung —
đúng giai đoạn áp dụng phương pháp đề xuất, và chi phí đổi lấy là AP sạch chỉ
giảm nhẹ."

---

## Slide 8 — Phương pháp: huấn luyện nhận biết rung

**Trên slide:**
- Sơ đồ quy trình: Lô ảnh x + nhãn y → (xác suất p=0,5) → nếu có: qua Toán tử
  rung 𝒯 (nhoè L,θ + nhiễu σ + sáng b) → Mạng f_θ; nếu không: giữ nguyên x̂=x
  → tính hàm mất mát phát hiện L_det(f_θ(x̂), y), **nhãn giữ nguyên**.
- 4 gạch đầu dòng: Thay 50% mỗi lô bằng ảnh rung mô phỏng · Nhãn giữ nguyên ·
  Hàm mất mát không đổi, kiến trúc không đổi · **Không tốn chi phí khi suy
  luận** — chỉ thay dữ liệu huấn luyện · Mô hình và hàm mất mát giữ nguyên,
  chỉ dữ liệu đưa vào là khác.

> **Đã bỏ khỏi slide:** biến thể đối chứng "ràng buộc nhất quán" (huấn luyện
> cả ảnh sạch lẫn ảnh rung qua cùng mạng, ép hai dự đoán giống nhau, dùng
> stop-gradient) — không đủ thời gian giải thích trong 15 phút nên đã cắt
> khỏi bullet, ghi chú và bảng kết quả. Vẫn còn đầy đủ trong báo cáo (Chương
> 3, phương trình `eq:vibcons`) và trong `kich_ban_thuyet_trinh.md` nếu Hội
> đồng hỏi sâu — xem phần giải thích cơ chế ở cuối mục này.

### Giải thích từng ký hiệu trong sơ đồ (đọc từ trái sang phải)

| Ký hiệu | Đọc là | Nghĩa là gì | Tại sao có mặt ở đó |
|---|---|---|---|
| $x$ | x | Một lô ảnh **sạch** (chưa xử lý gì) lấy từ tập huấn luyện | Điểm bắt đầu của mọi lượt huấn luyện |
| $y$ | y | Nhãn thật đi kèm — vị trí hộp bao quanh từng người trong ảnh | Dùng để tính xem mô hình đoán đúng/sai bao nhiêu |
| $p=0{,}5$ | xác suất 0,5 | Tung đồng xu: 50% ảnh đi vào nhánh rung, 50% giữ nguyên | Đây là bước "gieo xúc xắc" quyết định ảnh này có bị làm rung hay không |
| $\mathcal{T}$ | toán tử rung T (chữ T viết hoa kiểu toán) | Một **hàm số**, nhận ảnh sạch vào, trả ra ảnh đã rung | Chính là công thức rung ở slide 5 ($I_{rung} = \gamma(I_{sạch}*k_{L,\theta})+n$), áp trực tiếp lên ảnh |
| $L$ | L | Độ dài vệt nhoè (đơn vị: pixel) | Một tham số của $\mathcal{T}$ — quyết định nhoè nhẹ hay nặng |
| $\theta$ (bên trong $\mathcal{T}$) | theta — **góc** nhoè | Hướng của vệt mờ (vài độ, vài chục độ...) | Tham số thứ hai của $\mathcal{T}$ |
| $\sigma$ | sigma | Độ lệch chuẩn nhiễu cộng vào ảnh (mô phỏng hạt nhiễu cảm biến) | Tham số thứ ba của $\mathcal{T}$ |
| $b$ | b | Mức dao động độ sáng (± bao nhiêu %) | Tham số thứ tư của $\mathcal{T}$ |
| $f_\theta$ | f theta | Chính là mạng SteadyDet-N, với $\theta$ ở đây là **trọng số mạng** (hàng triệu con số mạng đang học) | Đây là mạng duy nhất, không có mạng phụ nào khác — ảnh dù rung hay sạch đều đi qua đúng một mạng này |
| $\hat{x}$ | x mũ (x-hat) | Ảnh **thực sự** được đưa vào mạng ở lượt đó — có thể là bản đã rung (nếu rơi vào nhánh "có"), hoặc chính là $x$ không đổi (nếu rơi vào nhánh "không": $\hat{x}=x$) | Ký hiệu chung cho "ảnh đầu vào thật sự", vì nó có thể là một trong hai trường hợp tuỳ xác suất |
| $\mathcal{L}_{\det}$ | L-det | Hàm mất mát phát hiện — công thức chấm điểm "mô hình đoán sai bao nhiêu" | Đúng hàm mất mát bình thường, không thêm thành phần nào mới |

**Lưu ý quan trọng — một chữ $\theta$ dùng cho HAI nghĩa khác nhau, dễ gây
nhầm:** $\theta$ ở bên trong $\mathcal{T}$ (góc nhoè) và $\theta$ ở $f_\theta$
(trọng số mạng) là **hai thứ hoàn toàn khác nhau**, chỉ tình cờ dùng chung
một chữ cái Hy Lạp theo quy ước toán học phổ biến (mỗi công thước tự quy định
ký hiệu riêng trong phạm vi của nó). Nếu Hội đồng hỏi, cứ nói thẳng: *"Đây là
cách dùng lại ký hiệu phổ biến trong tài liệu khoa học, hai chữ θ này độc lập
với nhau, không liên quan gì nhau."*

**Vì sao mũi tên "không: $\hat x = x$" lại vòng xuống dưới rồi mới vào $f$?**
Đây là nhánh khi *không* rơi vào 50% bị rung — ảnh giữ nguyên, bỏ qua hoàn
toàn khối $\mathcal{T}$, đi thẳng một đường vòng xuống dưới hộp $\mathcal{T}$
rồi nhập lại vào cùng mạng $f_\theta$. Vẽ vòng xuống chỉ để không phải vẽ đè
lên khối $\mathcal{T}$, không có ý nghĩa gì khác ngoài chuyện trình bày hình.

**Nghĩa là gì — đây là đóng góp phương pháp chính, cần hiểu thật kỹ vì Hội
đồng sẽ hỏi nhiều:**

- **Cơ chế hoạt động:** Với xác suất 50%, mỗi ảnh trong lô huấn luyện được đi
  qua toán tử rung 𝒯 (chính là công thức rung ở slide 5) trước khi đưa vào
  mạng. Nhãn (vị trí hộp bao) **không đổi** vì rung không làm vật dịch chỗ.
  Hàm mất mát tính như bình thường — không có gì đặc biệt, không thêm nhánh
  mạng, không thêm thành phần loss nào.
- **Vì sao hiệu quả mà "không tốn chi phí khi suy luận"?** Vì toán tử rung
  𝒯 chỉ chạy **lúc huấn luyện**, trên dữ liệu — không phải một phần của
  kiến trúc mạng. Khi mô hình đã huấn luyện xong và đem đi suy luận thực tế
  (trên máy bay), nó chạy y hệt một mô hình bình thường, không có bước xử lý
  thêm nào, không có phép tính thêm nào. Đây là lý do mục "0 ms" ở bảng kết
  quả slide 9 — khác hẳn hướng tiếp cận "khử nhoè bằng một mạng riêng" (vốn
  thêm một mạng nữa vào đường xử lý, làm tăng độ trễ thật).
- **Vì sao gọi là "huấn luyện nhận biết rung" chứ không phải chỉ là data
  augmentation thông thường?** Vì phép tăng cường này **mô phỏng đúng vật lý
  của rung UAV** (không phải random crop/flip chung chung), và được áp dụng
  có chủ đích để giải quyết đúng lỗ hổng đã đo ở slide 2.
- **Biến thể đối chứng "ràng buộc nhất quán" là gì và vì sao kém hơn?** Đây
  là một ý tưởng phức tạp hơn mà nhóm *đã thử trước khi chọn phương án đơn
  giản*: đưa cả ảnh sạch (x) và ảnh rung (x̃) qua **cùng một mạng trong cùng
  một lần forward**, tính loss phát hiện trên nhánh sạch, rồi thêm một thành
  phần loss ép buộc dự đoán trên ảnh rung phải "giống" dự đoán trên ảnh sạch
  (có dừng gradient ở nhánh sạch để tránh mạng tự phá chính nó). Ý tưởng nghe
  hợp lý (khai thác thêm tín hiệu giám sát) nhưng **tốn gấp đôi lượt truyền
  xuôi** (forward pass) nên tốn 1,8× thời gian huấn luyện, và quan trọng hơn
  — **kết quả thực nghiệm lại kém hơn** phương án đơn giản (xem slide 9,
  dòng "ràng buộc nhất quán" chỉ đạt retain 30,0% so với 36,2% của phương án
  chính). Đưa biến thể này vào báo cáo thể hiện nhóm **đã thử nghiệm có hệ
  thống**, không chỉ chọn bừa phương án đầu tiên nghĩ ra.

**Nói sao cho đúng:** "Cơ chế rất đơn giản có chủ đích: thay một nửa ảnh mỗi
lô bằng bản rung, nhãn giữ nguyên, không đổi gì về kiến trúc hay hàm mất mát.
Vì chỉ tác động lên dữ liệu huấn luyện, khi suy luận thực tế mô hình chạy y
hệt bình thường — không tốn thêm chi phí tính toán nào."

**Nếu bị hỏi:** *"Tại sao không dùng một mạng khử nhoè riêng?"* → Vì thêm một
mạng là thêm một lượt suy luận nữa vào đường xử lý thời gian thực, trong khi
tài nguyên trên Raspberry Pi 5 + Hailo-8L có hạn; hướng "làm bộ phát hiện tự
bền hơn qua dữ liệu" không tốn gì thêm lúc chạy.

---

## Slide 9 — Kết quả 1: Hiệu quả của huấn luyện nhận biết rung ⭐

**Đây là bảng kết quả quan trọng nhất của toàn bài — đọc kỹ từng dòng.**

**Trên slide (bảng đầy đủ — đã bỏ dòng "ràng buộc nhất quán" khỏi bản trình
chiếu để đủ giờ, xem ghi chú cuối mục này):**

| SteadyDet-N | AP ảnh sạch | AP mức rung 2 | retain (%) | Kết luận |
|---|---|---|---|---|
| Gốc | 0,1892 | 0,0123 | 16,8 ± 0,7 | gần như không phát hiện được |
| + 120 epoch ảnh sạch | 0,1823 | 0,0125 | 16,9 ± 0,7 | huấn luyện thêm không cải thiện |
| **+ huấn luyện nhận biết rung** | 0,1708 | **0,0518** | **36,2 ± 0,7** | **cao nhất** |
| SteadyDet-W: gốc → rung | 0,2349→0,2219 | 0,0258→0,0839 | 22,2→39,4 | tái lập được |
| YOLO11-n: gốc → rung | 0,2588→0,2579 | 0,0248→0,0227 | 21,9→21,5 | không cải thiện |

4 ô số liệu lớn: **16,8% → 36,2%** (tỷ lệ giữ độ chính xác khi rung) · **×4,2**
(AP ở mức rung của bay tiến) · **−9,7%** (mức suy giảm AP trên ảnh sạch) ·
**0 ms** (chi phí suy luận tăng thêm).

**Nghĩa là gì — đọc từng dòng theo đúng logic suy luận khoa học (rất quan
trọng khi trình bày, đây là cách "chứng minh bằng loại trừ"):**

1. **Dòng "Gốc" (16,8%):** Mô hình baseline, không xử lý gì — gần như mù khi
   gặp rung mức 2 (chỉ giữ được 16,8% độ chính xác so với lúc đứng yên).
2. **Dòng "+ 120 epoch ảnh sạch" (16,9%):** Đây là **đối chứng**, trả
   lời câu hỏi "có phải chỉ cần huấn luyện lâu hơn là tự nhiên bền hơn?" —
   Câu trả lời là **không**: huấn luyện thêm 120 epoch nữa nhưng vẫn trên ảnh
   sạch thì retain gần như không đổi (16,8% → 16,9%, trong sai số thống kê).
   Điều này loại trừ giả thuyết "cứ train lâu là tốt".
3. **Dòng "+ huấn luyện nhận biết rung" (36,2%) — kết quả chính:** Cao nhất
   trong tất cả, với chi phí huấn luyện rẻ nhất trong các phương án xử lý rung,
   và **0 ms chi phí thêm lúc suy luận**.
4. **Hai dòng cuối (SteadyDet-W, YOLO11-n) — đây là bước kiểm chứng tính tổng
   quát, cực kỳ quan trọng để trả lời "có phải chỉ ăn may trên một kiến trúc?"**
   - Trên **SteadyDet-W** (kiến trúc khác, nhưng cùng họ do nhóm thiết kế):
     retain tăng từ 22,2% lên 39,4% — **phương pháp tái lập được** trên kiến
     trúc thứ hai.
   - Trên **YOLO11-n** (kiến trúc hoàn toàn khác họ, của Ultralytics): retain
     gần như không đổi (21,9% → 21,5%) — phương pháp **không có tác dụng**.
     → Kết luận trung thực: phương pháp hiệu quả và tái lập được trên họ kiến
     trúc SteadyDet, nhưng **không khẳng định** nó tổng quát cho mọi kiến trúc
     khác. Đây là điểm nhóm *chủ động nói rõ giới hạn* — một dấu hiệu tốt về
     tính khoa học, không nên né tránh khi Hội đồng hỏi.

**4 ô số liệu lớn tóm tắt:**
- 16,8% → 36,2%: con số headline của cả đề tài.
- ×4,2: AP tuyệt đối ở mức rung 2 tăng từ 0,0123 lên 0,0518 — gấp 4,2 lần.
- −9,7%: cái giá phải trả — AP trên ảnh sạch giảm từ 0,1892 xuống 0,1708,
  tức giảm 9,7% *tương đối* (không phải giảm 9,7 điểm phần trăm tuyệt đối).
- 0 ms: không có phép tính nào thêm lúc suy luận thật (đã giải thích ở slide 8).

**Nói sao cho đúng:** "Xin đọc theo đúng thứ tự suy luận: dòng 1 là baseline
gần như mù khi rung. Dòng 2 loại trừ giả thuyết train lâu hơn là đủ. Dòng 3 là
đối chứng phức tạp hơn nhưng kém hơn. Dòng 4 là phương pháp đề xuất, cao nhất
và rẻ nhất. Hai dòng cuối kiểm chứng: tái lập được trên kiến trúc thứ hai của
nhóm, nhưng không có tác dụng trên YOLO11-n — nhóm nói rõ giới hạn này."

**Câu hỏi khó có thể gặp:**
- *"Vì sao AP ảnh sạch giảm dần qua các dòng (0,1892 → 0,1708)?"* → Vì càng
  thêm ràng buộc/rung vào huấn luyện, mô hình càng phải "chia sẻ" khả năng
  giữa ảnh sạch và ảnh rung, nên AP sạch giảm nhẹ — đây chính là sự đánh đổi
  (trade-off) đã nêu, và mức đánh đổi 9,7% được xem là chấp nhận được so với
  mức tăng 4,2 lần ở phía rung.
- *"Vì sao không có tác dụng trên YOLO11-n?"* → Giả thuyết nêu trong báo cáo:
  khác biệt nằm ở kiến trúc/hàm mất mát (SteadyDet dùng NWD+CIoU+QFL và nhánh
  P2 bước 4, còn YOLO11-n không có các yếu tố này) — nhưng với một mô hình
  không phản hồi, chưa đủ bằng chứng để khẳng định chắc chắn nguyên nhân, nên
  nhóm chỉ phát biểu đúng phạm vi: *tái lập được trên họ kiến trúc của đề tài,
  không tổng quát ngoài phạm vi đã thử*.

---

## Slide 10 — Minh hoạ trực quan: phát hiện ở bốn mức rung

**Trên slide:**
- Ảnh ghép 2 hàng × 4 cột: cùng MỘT ảnh VisDrone có 55 người, cùng ngưỡng tin
  cậy 0,30. Hàng trên = mô hình gốc, hàng dưới = mô hình sau huấn luyện nhận
  biết rung. 4 cột = Ảnh sạch / Rung mức 1 / Rung mức 2 / Rung mức 3. Mỗi ảnh
  có nhãn "phát hiện đúng N/55 người" ở góc dưới.
- 4 ô số liệu: **40 → 40** (Ảnh sạch) · **34 → 34** (Rung mức 1) · **20 → 30**
  (Rung mức 2) · **0 → 7** (Rung mức 3).

**Nghĩa là gì:** Đây là minh hoạ **bằng mắt** cho đúng con số retain 16,8% →
36,2% ở slide 9 — để Hội đồng *thấy* chứ không chỉ *nghe số*. Quan trọng: hai
hàng **giống hệt nhau** ở ảnh sạch và mức 1 (40→40, 34→34) — chứng tỏ huấn
luyện nhận biết rung **không làm hỏng khả năng ở điều kiện nhẹ**, chỉ khác
biệt rõ từ mức 2 trở đi (20 so với 30 người bắt được) và mức 3 (0 so với 7).
Đây đúng là hình ảnh của đường cong AP theo mức rung ở slide 12 — hai đường
chập vào nhau lúc đầu, tách ra dần về sau.

**Lưu ý quan trọng khi nói:** Ảnh này là ảnh VisDrone-val (đông người, không
phải cảnh bám 1 người của đề tài) — dùng để minh hoạ hiện tượng rung ảnh
hưởng độ chính xác, không phải cảnh vận hành thật của UAV bám người.

**Nói sao cho đúng:** "Đây là minh hoạ trực quan cho đúng bảng vừa rồi. Cùng
một ảnh, cùng ngưỡng. Hai mô hình giống hệt nhau ở ảnh sạch và mức 1, chỉ
khác biệt từ mức 2 — đúng là mức bay tiến khi bám người."

---

## Slide 11 — Xếp hạng độ bền khi rung của 11 cấu hình

**Trên slide:** Biểu đồ cột ngang, 11 thanh xếp theo retain (%) giảm dần, có
thanh sai số (error bar ± độ lệch chuẩn):
1. SteadyDet-W + huấn luyện rung — 39,4%
2. SteadyDet-N + huấn luyện rung — 36,2%
3. SteadyDet-N + ràng buộc nhất quán — 30,0%
4. RT-DETR-l — 25,6%
5. SteadyDet-W (gốc) — 22,2%
6. YOLO11-n — 20,1%
7. YOLOv10-n — 19,7%
8. YOLOv8-n — 19,5%
9. YOLO26-n — 19,0%
10. SteadyDet-N + 120 epoch không rung — 16,9%
11. SteadyDet-N (gốc) — 16,8%

**Nghĩa là gì:** Đây là "bức tranh toàn cảnh" gộp tất cả các cấu hình đã nhắc
ở slide 9 và các mô hình đối chứng ở slide 12 vào **một hình duy nhất**, xếp
theo độ bền. Hai quan sát quan trọng:
- **Hai thanh dẫn đầu là hai mô hình của đề tài** sau khi áp dụng phương pháp
  — cao hơn hẳn mọi mô hình công khai khác, kể cả RT-DETR-l (mô hình lớn
  nhất, 108 GFLOPs).
- **Năm mô hình đối chứng công khai (RT-DETR-l, 4×YOLO) nằm sát nhau ở
  19–26%** dù khác nhau rất xa về kích thước và năm ra đời — tức là **chỉ
  tăng kích thước/độ phức tạp mô hình không tự nhiên làm nó bền với rung
  hơn**. Đây là luận điểm khoa học cốt lõi: độ bền trước rung là một trục
  tối ưu **riêng biệt**, không tự động đi kèm với độ chính xác hay độ lớn mô
  hình.
- Hai thanh cuối (16,8–16,9%) là mô hình gốc và đối chứng "train thêm" —
  thấp nhất, củng cố lại kết luận ở slide 9.

**Nói sao cho đúng:** "Nhìn toàn cảnh: năm mô hình công khai trải dài nhiều
năm phát triển nhưng retain gần như ngang nhau quanh 19–26%, chứng tỏ độ bền
trước rung không tự nhiên cải thiện theo kiến trúc mới hay mô hình lớn hơn —
phải chủ động xử lý bằng phương pháp, như hai thanh dẫn đầu của đề tài."

### Vì sao so sánh bằng retain chứ không so AP (mCP) tuyệt đối — câu hỏi rất
### hay, nhiều người trực giác nghĩ AP tuyệt đối mới "chuẩn" hơn

**Ví dụ đời thường trước:** hai vận động viên marathon. A chạy 100km đường
bằng hết 10 giờ (nhanh); B chạy hết 12 giờ (chậm hơn A). Cho cả hai chạy
tiếp trên địa hình gồ ghề: A hết 40 giờ (chậm đi **4 lần**), B hết 15 giờ
(chậm đi chỉ **1,25 lần**). Nếu so **thời gian tuyệt đối** trên địa hình khó:
A 40h, B 15h → tưởng B thắng — nhưng B chỉ "thắng" vì A sụp đổ hoàn toàn, chứ
không phải B bền hơn. Nếu so **tỷ lệ giữ được tốc độ** (retain): A giữ 10/40
= 25%, B giữ 12/15 = 80% → B mới thật sự bền hơn.

**Áp đúng số liệu thật — chứng minh so AP tuyệt đối sẽ ra kết luận sai:**

| Mô hình | AP sạch | mCP (AP tuyệt đối khi rung) | retain |
|---|---|---|---|
| RT-DETR-l | 0,2502 | **0,0641** | 25,6% |
| SteadyDet-N + rung | 0,1708 | 0,0617 | **36,2%** |

Nếu chỉ nhìn cột mCP: RT-DETR-l (0,0641) **cao hơn** SteadyDet-N (0,0617) —
tưởng RT-DETR-l chịu rung tốt hơn. **Sai**, vì RT-DETR-l chỉ cao hơn tuyệt
đối nhờ nó xuất phát từ vạch cao hơn hẳn (AP sạch 0,2502) và to gấp 28 lần
(nhiều "sức chịu đựng" hơn) — giống vận động viên A chạy nhanh trên đường
bằng nhưng sụp đổ trên địa hình khó. Nhìn retain thì ngược lại: RT-DETR-l chỉ
giữ được 25,6% độ chính xác **của chính nó**, còn SteadyDet-N giữ được
36,2% — tỷ lệ "sụp đổ" của RT-DETR-l khi rung nặng hơn hẳn.

**Bản chất của retain, nói thật gọn:** retain không đo "mô hình chính xác bao
nhiêu" (đó là việc của AP/mCP) — retain đo **"mô hình trung thành với chính
nó tới mức nào khi điều kiện xấu đi"**. Hai việc này độc lập: một mô hình có
thể AP sạch rất cao nhưng retain rất thấp (giòn, dễ vỡ khi rung — đúng là
trường hợp cả 5 mô hình đối chứng ở bảng trên, retain chỉ quanh 19–26% dù AP
sạch của chúng cao hơn SteadyDet-N).

**Vì sao slide này buộc phải dùng retain?** Vì mục đích của slide là xếp
hạng **độ bền** — nếu dùng AP tuyệt đối thì mô hình nào vốn mạnh sẵn (nhiều
tham số, nhiều "vốn" để mất) sẽ luôn trông có vẻ "bền" hơn chỉ vì nó có nhiều
vốn hơn, chứ không phải vì đặc trưng nó học được thực sự ít nhạy cảm với
rung hơn. retain chia AP khi rung cho AP sạch của **chính mô hình đó**, tách
riêng được yếu tố "bền" ra khỏi yếu tố "vốn mạnh sẵn" — đây là lý do bắt
buộc phải dùng retain để so sánh độ bền một cách công bằng.

---

## Slide 12 — Kết quả 2: Thứ hạng đảo ngược khi có rung

**Trên slide:**
- Bảng so 6 mô hình:

| Mô hình | GFLOPs | AP sạch | mCP |
|---|---|---|---|
| RT-DETR-l | 108,0 | 0,2502 | 0,0641 |
| YOLO11-n | 6,4 | 0,2768 | 0,0557 |
| YOLOv10-n | 8,4 | 0,2730 | 0,0538 |
| YOLOv8-n | 8,2 | 0,2684 | 0,0522 |
| YOLO26-n | 5,8 | 0,2675 | 0,0507 |
| **SteadyDet-N + rung** | **4,6** | 0,1708 | **0,0617** |

- Đồ thị đường: AP@[.50:.95] theo 4 mức (sạch/mức1/mức2/mức3) cho 4 đường:
  SteadyDet-N có rung, SteadyDet-N gốc, YOLO11-n, RT-DETR-l. Hai đường
  SteadyDet-N có rung và gốc chập nhau lúc đầu rồi tách; SteadyDet-N có rung
  **vượt lên trên** YOLO11-n và RT-DETR-l từ mức 2.
- 2 gạch đầu dòng: ảnh sạch SteadyDet-N thấp hơn cả năm mô hình đối chứng;
  khi có rung mCP cao hơn cả 4 YOLO nano với chi phí thấp nhất, từ mức 2 vượt
  cả RT-DETR-l (28 lần tham số).

**Nghĩa là gì — đây là "cú đảo ngược thứ hạng" đặt tên cho slide:**

- **Nhìn cột AP sạch:** SteadyDet-N (0,1708) thấp nhất trong 6 mô hình — nếu
  chỉ nhìn cột này, SteadyDet-N có vẻ "thua" rõ ràng.
- **Nhìn cột mCP (độ chính xác khi đang rung — tức khi UAV đang bay thật):**
  SteadyDet-N (0,0617) lại **cao nhất** trong toàn bảng, kể cả hơn RT-DETR-l
  (0,0641... khoan, thực ra RT-DETR-l 0,0641 > 0,0617 — cần xem lại: theo
  bảng RT-DETR-l có mCP 0,0641, cao hơn SteadyDet-N 0,0617 về con số tuyệt
  đối. Điểm mấu chốt không phải "cao nhất tuyệt đối" mà là theo **đồ thị
  theo từng mức rung**: SteadyDet-N vượt lên trên RT-DETR-l **từ mức 2 trở
  đi** dù RT-DETR-l có mCP trung bình nhỉnh hơn một chút do tính trung bình
  cả 3 mức. Khi nói, nên bám đúng câu đã viết trong ghi chú: *"mCP cao hơn cả
  bốn mô hình YOLO bản nano"* (đúng, 0,0617 > 0,0557/0,0538/0,0522/0,0507) và
  *"từ mức 2 trở đi vượt RT-DETR-l"* (đúng theo đồ thị đường, dù mCP trung
  bình 3 mức của RT-DETR-l nhỉnh hơn do mức 1 nó mạnh hơn) — **không nói
  "mCP cao nhất tuyệt đối trong bảng"** để tránh tự mâu thuẫn với chính số
  trong bảng.
- **Vì sao đáng chú ý:** SteadyDet-N chỉ 4,6 GFLOPs — **rẻ nhất** trong toàn
  bảng (so với RT-DETR-l 108,0 GFLOPs tức gấp 23 lần, và các YOLO-n 5,8–8,4
  GFLOPs) — nhưng lại bền với rung nhất trong nhóm mô hình nhẹ (4 YOLO nano),
  và cạnh tranh được với mô hình lớn hơn nhiều lần ở các mức rung cao.

**Nói sao cho đúng:** "Nếu chỉ nhìn ảnh sạch, SteadyDet-N thua cả năm mô hình
— nhóm không né tránh điều này. Nhưng khi có rung, đúng điều kiện vận hành
thật, SteadyDet-N vượt cả bốn mô hình YOLO bản nano với chi phí tính toán
thấp nhất, và từ mức rung 2 trở đi còn vượt cả RT-DETR-l — mô hình có số
tham số gấp 28 lần."

**Nếu bị hỏi — rất quan trọng, cần nhớ kỹ:**
- *"So sánh YOLO không qua xử lý rung với SteadyDet-N có xử lý rung, vậy có
  công bằng không?"* → Đã kiểm chứng: áp đúng phương pháp huấn luyện nhận
  biết rung lên YOLO11-n thì retain **không tăng** (20,1% → 20,1%, xem slide
  9/11), tức là đưa YOLO11-n lên "trạng thái tốt nhất của nó" cũng không đổi
  thứ hạng — nên so sánh ở đây vẫn là so sánh đúng, mỗi mô hình ở trạng thái
  tốt nhất của nó.
- *"Vì sao AP sạch SteadyDet-N thấp hơn?"* → Trả lời bằng đúng lý do ở slide
  7: YOLO khởi tạo từ trọng số COCO có sẵn, SteadyDet-N huấn luyện từ đầu.

---

## Slide 13 — Phát hiện người trong cảnh đông

**Trên slide:**
- 3 ảnh VisDrone-val có hộp phát hiện màu xanh: ngã tư giờ cao điểm (119
  người), quảng trường (68 người), sân thể thao trường học (55 người).
- 2 gạch đầu dòng bên trái: 3 ảnh có 119/68/55 người, phần lớn chỉ cao vài
  chục pixel; hộp xanh là kết quả ở ngưỡng 0,30, **bằng chính mô hình đang
  chạy trên máy bay**.
- 2 gạch đầu dòng bên phải: đo trên cả 548 ảnh VisDrone-val → AP@0,5 = 0,514;
  phạm vi đề tài là bám **một người** — cảnh đông thì bộ chọn mục tiêu lấy
  người gần nhất rồi khoá theo mã định danh.

**Nghĩa là gì:** Slide này trả lời trực tiếp mối lo "cảnh bay thử trong video
chỉ có 1 người, vậy mô hình có phát hiện được nhiều người cùng lúc không?" —
Câu trả lời: **có**, đây là bằng chứng. Mô hình phát hiện (bước nhận diện)
hoạt động tốt trên cảnh đông tới hơn trăm người; con số AP@0,5 = 0,514 đo
trên toàn bộ 548 ảnh VisDrone-val (không chỉ 3 ảnh chọn lọc) khẳng định điều
này ở quy mô lớn.

**Điểm cần nói rõ để không bị hiểu lầm (rất quan trọng):** Đây là khả năng
*phát hiện* (detection), khác với khả năng *bám* (tracking + follow) của toàn
hệ thống. **Phạm vi đề tài theo thuyết minh chỉ là bám MỘT người** — nên dù
mô hình nhìn thấy cả trăm người trong ảnh, bộ chọn mục tiêu (target selector)
vẫn chỉ chọn và khoá đúng một người (gần nhất khi bắt đầu), những người khác
chỉ hiện hộp xám (đã phát hiện, chưa khoá) chứ máy bay không bám theo họ.

**Nói sao cho đúng:** "Ngoài cảnh bay thử một người, mô hình còn được đo trên
548 ảnh đô thị đông người. Ba ảnh minh hoạ có 119, 68, 55 người — mô hình bắt
được phần lớn. Nhưng cần nói rõ: đây là năng lực phát hiện của mô hình, còn
phạm vi bám của đề tài vẫn là một người theo đúng thuyết minh."

**Nếu bị hỏi:** *"Vậy sao không mở rộng bám nhiều người luôn?"* → Đây chính
là một hướng phát triển đã nêu ở slide 24 ("bám nhiều mục tiêu"), nằm ngoài
phạm vi đăng ký ban đầu của đề tài.

---

## Slide 14 — Kết quả 3: Kiểm chứng trên dữ liệu bay thật

**Trên slide:**
- 2 biểu đồ cột: (1) so sánh AP các mức trên tập "Huấn luyện VisDrone +
  COCO-person" vs "Huấn luyện chỉ VisDrone", mỗi tập có 3 cột (không xử lý /
  huấn luyện rung / ràng buộc nhất quán). (2) tương tự.
- 3 gạch đầu dòng: thêm COCO-person → AP@0,5 gấp 5,3 lần so với chỉ VisDrone
  → chọn dữ liệu theo miền triển khai, không theo bộ benchmark · huấn luyện
  nhận biết rung → AP@0,75 tăng +78%/+81%, khoảng tin cậy bootstrap 95% không
  chứa 0 · hạn chế: 1 người, 1 cảnh, 1 độ cao.

**Nghĩa là gì:**

- **Phát hiện quan trọng thứ nhất — chọn dữ liệu theo miền triển khai:** Khi
  chỉ huấn luyện trên VisDrone (người chụp từ rất cao, rất nhỏ), mô hình học
  sai "kích thước kỳ vọng" của người khi áp dụng vào cảnh bay thấp 3–6 m (nơi
  người chiếm diện tích lớn hơn nhiều). Thêm COCO-person (ảnh người cỡ bình
  thường, gần camera) vào tập huấn luyện giúp mô hình học đúng miền triển
  khai thật, kết quả AP@0,5 trên 92 khung bay thật **tăng gấp 5,3 lần**. Đây
  là bài học phương pháp luận: *chọn dữ liệu huấn luyện theo nơi mô hình sẽ
  được dùng, không theo bộ benchmark phổ biến nhất*.
- **Phát hiện quan trọng thứ hai — huấn luyện nhận biết rung cải thiện độ
  khít hộp trên dữ liệu bay thật:** AP@0,75 (ngưỡng IoU cao, đo độ khít hộp
  chứ không chỉ "có thấy người hay không") tăng 78–81% tuỳ nhóm dữ liệu. Vì
  tập chỉ có 92 khung (nhỏ), nhóm dùng **khoảng tin cậy bootstrap 95%** để
  kiểm tra mức tăng này có phải do may rủi ngẫu nhiên không — kết quả khoảng
  tin cậy của *hiệu số* hai mô hình **không chứa 0**, nghĩa là về mặt thống
  kê, khác biệt này **có ý nghĩa**, không phải nhiễu.
- **Hạn chế được nêu rõ chủ động:** Tập 92 khung bay thật chỉ có 1 người, 1
  cảnh, 1 độ cao (~6 m) — đây là tập **sơ bộ**, không đủ lớn để làm benchmark
  chính thức. Đây là lý do nhóm đề xuất "mở rộng tập bay thật" làm hướng phát
  triển ở slide 24.

**Nói sao cho đúng:** "Hai bài học từ dữ liệu bay thật: một, chọn dữ liệu
theo miền triển khai làm AP tăng 5,3 lần, quan trọng hơn cả việc đổi mô hình;
hai, huấn luyện nhận biết rung làm hộp khít hơn rõ rệt, có ý nghĩa thống kê.
Nhóm cũng nêu rõ hạn chế: tập này mới 1 người, 1 cảnh, 1 độ cao."

---

## Slide 15 — Đưa mô hình lên NPU Hailo-8L

**Trên slide:**
- Phụ đề quy trình: PyTorch → ONNX → trình biên dịch luồng dữ liệu → lượng tử
  hoá INT8 → tệp HEF chạy trên máy bay.
- 4 bước đánh số: (1) Xuất ONNX chỉ dùng phép tính Hailo hỗ trợ, NMS để ngoài
  đồ thị · (2) Trình biên dịch luồng dữ liệu phân tích và chia context · (3)
  Lượng tử hoá INT8 bằng 64 ảnh hiệu chuẩn lấy từ miền triển khai · (4) Sinh
  tệp HEF rồi chép sang Raspberry Pi 5.
- 1 gạch đầu dòng: 4 context so với 6 của YOLO26-n → 63,66 / 37,46 FPS.
- 2 đồ thị: trái so FPS @640 batch=1 giữa SteadyDet-N (63,66 FPS, 4 context)
  và YOLO26-n (37,46 FPS, 6 context); phải so độ trễ phần cứng theo kích
  thước lô (1/4/16).

**Nghĩa là gì — khái niệm "context" là điểm kỹ thuật quan trọng nhất ở đây:**

- **Vì sao không chạy thẳng mô hình PyTorch trên máy bay?** Vì NPU Hailo-8L
  là phần cứng chuyên dụng, chỉ chạy được định dạng nhị phân riêng (HEF), và
  chỉ hỗ trợ một tập con các phép toán nhất định (không có NMS phần cứng, ví
  dụ) — nên phải qua pipeline chuyển đổi 4 bước.
- **"Context" là gì và vì sao quyết định tốc độ?** NPU Hailo có bộ nhớ trên
  chip (on-chip SRAM) giới hạn. Nếu toàn bộ mô hình (trọng số + bản đồ đặc
  trưng trung gian) không vừa bộ nhớ này, trình biên dịch phải **chia mô hình
  thành nhiều phần (context)**, và mỗi lần chuyển từ context này sang context
  khác, **trọng số phải nạp lại qua đường truyền PCIe** — đây là thao tác
  chậm. SteadyDet-N thiết kế gọn (ít tham số, kiến trúc "mỏng" như slide 6)
  nên chỉ cần **4 context**, trong khi YOLO26-n cần **6 context** dù số phép
  tính (GFLOPs) của hai mô hình không chênh lệch nhiều như vậy. Đây là lý do
  chênh lệch tốc độ (63,66 so với 37,46 FPS, tức 1,70 lần) **lớn hơn nhiều**
  so với chênh lệch về GFLOPs thuần tuý — một phát hiện thực nghiệm quan
  trọng về đặc thù phần cứng NPU, không chỉ là "mô hình nhỏ thì nhanh".
- **64 ảnh hiệu chuẩn lấy từ miền triển khai:** Lượng tử hoá INT8 cần một tập
  ảnh mẫu nhỏ để ước lượng khoảng giá trị (dải động) của từng lớp, từ đó tính
  hệ số tỷ lệ lượng tử hoá. Nhóm chủ động lấy 64 ảnh **từ đúng miền triển
  khai** (cảnh bay thấp, không phải ảnh VisDrone ngẫu nhiên) để lượng tử hoá
  chính xác nhất cho tình huống sử dụng thật.

**Nói sao cho đúng:** "Mô hình huấn luyện xong chưa chạy được ngay trên NPU,
phải qua 4 bước. Điểm đáng chú ý nhất: tốc độ trên Hailo phụ thuộc số
context — số lần phải nạp lại trọng số qua PCIe — chứ không chỉ phụ thuộc số
phép tính. SteadyDet-N chỉ cần 4 context so với 6 của YOLO26-n nên nhanh hơn
1,70 lần dù GFLOPs không chênh nhiều như vậy."

---

## Slide 16 — Kết quả 4: Mô hình chạy thật trên máy bay (INT8) ⭐

**Đây là slide trả lời trực tiếp chỉ tiêu của thuyết minh — slide then chốt
thứ hai sau slide 9.**

**Trên slide:**
- 4 ô số liệu: **0,917** (AP@0,5 trên bay thật INT8 — chỉ tiêu ≥ 0,70) ·
  **0,880** (F1 ở ngưỡng vận hành 0,25) · **63,66 FPS** (suy luận NPU @640,
  112,55 @320) · **×1,70** (nhanh hơn YOLO26-n trên cùng NPU).
- 2 biểu đồ cột FP32 (GPU) vs INT8 (Hailo-8L) trên hai tập: VisDrone-val (548
  ảnh, người rất nhỏ) và Bay thật ~6m (92 khung, người lớn) — mỗi tập 3 nhóm
  cột AP@[.5:.95]/AP@0,5/AP@0,75.
- 4 gạch đầu dòng: vượt chỉ tiêu 70% · tổn thất INT8 phụ thuộc kích thước vật
  (−6,5% người rất nhỏ, ≈0 ở cự ly 3–6m) · recall 0,837/precision 0,928 ·
  nhanh hơn nhờ kiến trúc hợp NPU.

**Nghĩa là gì:**

- **Đây là con số "chốt" trả lời chỉ tiêu ≥ 70% của thuyết minh:** 0,917 đo
  trên **đúng mô hình INT8 đang chạy trên máy bay thật** (không phải mô
  phỏng, không phải FP32 trên máy tính), trên 92 khung bay thật gán nhãn tay.
  Vượt chỉ tiêu 21,7 điểm phần trăm.
- **F1 = 0,880 ở ngưỡng 0,25:** Đây là điểm cân bằng giữa recall (0,837 — bắt
  được 83,7% số người thật có trong ảnh) và precision (0,928 — trong số báo
  "có người", 92,8% là đúng). Recall cao hơn hay thấp hơn precision tuỳ ứng
  dụng; ở đây ưu tiên không bỏ sót (recall) nhỉnh hơn một chút so với không
  báo nhầm.
- **Hai biểu đồ FP32 vs INT8 — đây là bằng chứng lượng tử hoá không "ăn" mất
  độ chính xác đáng kể, NHƯNG có khác biệt theo kích thước vật:**
  - Trên VisDrone-val (người rất nhỏ, chụp từ cao): INT8 mất khoảng 6,5% so
    với FP32 — lượng tử hoá ảnh hưởng rõ hơn khi vật quá nhỏ (vài pixel),
    vì sai số làm tròn toạ độ chiếm tỷ lệ lớn so với kích thước hộp.
  - Trên Bay thật ~6m (người lớn, đúng cự ly vận hành của UAV): tổn thất gần
    **bằng 0** — đây là *tin tốt quan trọng nhất*: ở đúng điều kiện sử dụng
    thật, lượng tử hoá INT8 gần như không mất gì.
- **×1,70 nhanh hơn YOLO26-n:** Nhắc lại kết quả slide 15 (do ít context
  hơn), đặt cạnh độ chính xác để khẳng định: không phải đánh đổi tốc độ lấy
  độ chính xác, mà đạt cả hai.

**Nói sao cho đúng:** "Đây là con số trả lời trực tiếp thuyết minh: 0,917 đo
trên đúng mô hình đang bay, vượt xa chỉ tiêu 70%. Lượng tử hoá INT8 gần như
không mất gì ở cự ly 3–6 mét — đúng cự ly bám thực tế — chỉ mất ở người rất
nhỏ ngoài phạm vi vận hành chính."

**Nếu bị hỏi:** *"Chỉ tiêu 70% đo như thế nào, có đáng tin không?"* → AP@0,5
theo đúng chuẩn đánh giá COCO, trên 92 khung bay thật gán nhãn tay (không
phải tự động), đo bằng chính file HEF INT8 chạy trên Hailo-8L thật — không
phải con số mô phỏng hay suy luận lý thuyết.

---

## Slide 17 — Chọn ngưỡng tin cậy để vận hành

**Trên slide:**
- Đồ thị đường: F1 theo ngưỡng tin cậy (0,20 / 0,25 / 0,30 / 0,40) trên 92
  khung bay thật — đỉnh ở ngưỡng 0,25 (F1=0,880), giảm dần sau đó, tụt mạnh ở
  0,40 (F1=0,551).
- 4 gạch đầu dòng: ngưỡng 0,25 → F1 0,880 · ngưỡng 0,30 → 0,823, ngưỡng 0,40
  → 0,551 (bỏ sót nhiều) · cấu hình bay chốt 0,25, ghi thẳng trong
  config.yaml · tham số vận hành chọn theo số đo trên miền triển khai, không
  theo mặc định.

**Nghĩa là gì:** Ngưỡng tin cậy (confidence threshold) là tham số quyết định
mô hình phải "chắc chắn" tới mức nào mới báo "có người" — ngưỡng thấp thì bắt
được nhiều hơn nhưng dễ báo nhầm, ngưỡng cao thì chắc chắn hơn nhưng dễ bỏ
sót. Đây là một ví dụ cụ thể, nhỏ nhưng minh hoạ **nguyên tắc phương pháp
luận xuyên suốt đề tài**: mọi tham số vận hành đều chọn bằng cách quét và đo
trên đúng dữ liệu thật (92 khung bay thật), chứ không dùng giá trị mặc định
của framework hay copy từ bài báo khác.

**Nói sao cho đúng:** "Một ví dụ nhỏ cho nguyên tắc chọn tham số bằng số đo:
quét bốn ngưỡng trên dữ liệu bay thật, 0,25 cho F1 cao nhất, nên chốt làm cấu
hình vận hành."

---

## Slide 18 — Tốc độ đầu-cuối: hệ thống bị giới hạn bởi camera

**Trên slide:**
- Biểu đồ thanh ngang phân rã độ trễ: Chờ camera (12,47 ms) + Suy luận NPU
  (17,72 ms) + phần nhỏ giải mã/NMS, tổng gần tới ngân sách 33,3 ms (ứng với
  30 Hz).
- 4 gạch đầu dòng: 30,0 FPS đầu-cuối · độ trễ trung vị 32 ms, p99 36 ms ·
  tính toán chỉ chiếm 59% ngân sách → giới hạn nằm ở camera · sửa 2 lỗi cấu
  hình camera: 11,1 → 30,0 FPS.

**Nghĩa là gì — đây là một phát hiện kỹ thuật thú vị, không phải chỉ là số
liệu suông:** Trực giác thông thường sẽ nghĩ "mô hình AI là phần chậm nhất".
Thực tế đo được: suy luận NPU chỉ chiếm 17,72/33,3 ≈ 59% ngân sách thời gian;
phần còn lại là **chờ camera trả khung hình**. Ban đầu hệ thống chỉ chạy được
11,1 FPS — nhóm đã **tìm ra và sửa 2 lỗi cấu hình camera cụ thể** (không nói
chi tiết ở slide này, nhưng có trong báo cáo: lỗi buffer size và lỗi tự phơi
sáng tự hạ tốc độ khung) để đưa tốc độ lên đúng 30,0 FPS — bằng đúng tốc độ
camera danh định.

**Nói sao cho đúng:** "Chạy đủ 30 FPS trên máy bay, nhưng thú vị là nút cổ
chai không phải mô hình — suy luận chỉ chiếm 59% ngân sách — mà là cấu hình
camera. Sau khi tìm và sửa hai lỗi, tốc độ tăng từ 11,1 lên 30,0 FPS."

---

## Slide 19 — Vòng điều khiển bám người và lớp an toàn

**Trên slide:**
- Ảnh minh hoạ 3 khung hình từ video bay thử, có vẽ hộp bám + vệt quỹ đạo.
- 5 gạch đầu dòng: theo vết OC-SORT → khoá mục tiêu → tìm lại người cũ sau
  khi mất · khoảng cách camera lỗ kim + Kalman, giới hạn 3–15 m · 3 bộ PID →
  lệnh vận tốc MAVLink 30 Hz, khớp nhịp detector · thang an toàn FOLLOW →
  COAST → HOVER, chỉ gửi lệnh khi GUIDED · giữ mục tiêu 83,5% thời gian, sai
  số hướng trung vị 5,5°.

**Nghĩa là gì — tóm tắt toàn bộ chuỗi xử lý điều khiển, mỗi cụm từ cần hiểu:**

- **OC-SORT:** thuật toán theo vết đa đối tượng (đã giải thích lý thuyết ở
  Chương 1) — nối các hộp phát hiện rời rạc giữa các khung thành "quỹ đạo"
  của từng người, gán mỗi người một mã định danh (ID) ổn định.
- **Khoá mục tiêu + tìm lại người cũ:** hệ thống chỉ bám đúng một ID đã chọn;
  nếu người đó biến mất tạm thời (ra khỏi khung, bị che khuất), hệ thống có
  cơ chế **chờ và tìm lại đúng người đó** (dựa vào vị trí gần đúng) thay vì tự
  động chuyển sang bám người khác.
- **Camera lỗ kim + Kalman:** mô hình hình học ước lượng khoảng cách từ chiều
  cao hộp bao (đã có công thức ở Chương 1), rồi làm mượt kết quả đo (vốn có
  nhiễu) bằng bộ lọc Kalman — giới hạn hoạt động 3–15 m vì ngoài khoảng này
  ước lượng không còn tin cậy.
- **3 bộ PID:** ba kênh điều khiển độc lập — tiến/lùi (theo khoảng cách),
  quay hướng (theo sai số ngang trong ảnh), độ cao (giữ cố định) — mỗi kênh
  một bộ PID riêng, tổng hợp thành lệnh vận tốc gửi xuống Pixhawk.
- **Thang an toàn FOLLOW → COAST → HOVER:** ba trạng thái theo mức độ "tự
  tin" về vị trí người: đang thấy rõ (FOLLOW) → vừa mất dấu, bám theo dự đoán
  (COAST) → mất quá lâu, đứng yên chờ (HOVER). Sẽ giải thích chi tiết hơn ở
  slide 20.
- **83,5% và 5,5°:** đo trên **toàn bộ clip 4 phút 43** của video bay thử
  (không phải chỉ đoạn 18 giây chiếu ở slide 21–22) — tỷ lệ thời gian giữ
  được khoá mục tiêu, và sai số hướng trung vị khi đang bám.

**Nói sao cho đúng:** "Phần điều khiển tóm tắt ngắn: phát hiện nối thành quỹ
đạo bằng OC-SORT, khoá một người và tìm lại được khi mất dấu ngắn, ước lượng
khoảng cách bằng camera lỗ kim và Kalman, ba bộ PID sinh lệnh gửi 30 Hz, có
thang an toàn ba mức và chỉ gửi lệnh khi máy bay ở GUIDED."

---

## Slide 20 — Nhịp lệnh 30 Hz và lớp an toàn

**Trên slide:**
- Biểu đồ: độ trễ từ ánh sáng vào cảm biến tới lúc máy bay nhận lệnh, theo 4
  nhịp gửi (10/20/30/50 Hz) — 30 Hz là điểm nhóm chọn.
- Sơ đồ thang trạng thái: FOLLOW ⇄ COAST (mất mục tiêu, < 3s) ⇄ HOVER (mất >
  3s); có mũi tên "người lái lấy lại quyền" dẫn ra ngoài sơ đồ.
- 5 gạch đầu dòng: 10 Hz cộng thêm 50ms chờ vô ích · đường truyền đo thật 0%
  mất gói tới 50 Hz, khứ hồi p99 7,73ms · chọn 30 Hz vì khớp nhịp detector ·
  mất người <3s bám theo dự đoán nửa vận tốc, sau đó HOVER, không tự bay về ·
  lệnh chỉ gửi khi GUIDED.

**Nghĩa là gì — giải thích kỹ phần "vì sao 30 Hz" vì đây là quyết định kỹ
thuật có cơ sở đo đạc, không phải chọn tuỳ ý:**

- **Vì sao không phải 10 Hz (phổ biến trong nhiều hệ thống khác)?** Ở 10 Hz,
  mỗi kết quả phát hiện mới phải **chờ trung bình 50 ms** mới tới lượt được
  gửi đi (vì hệ thống gửi theo chu kỳ cố định), cộng thẳng vào độ trễ phản
  ứng mà không đổi lấy lợi ích gì.
- **Vì sao không phải nhanh hơn 30 Hz (ví dụ 50 Hz)?** Vì bộ phát hiện
  (detector) chỉ **sinh ra kết quả mới ở đúng 30 Hz** (khớp FPS camera/NPU đã
  đo ở slide 16/18) — gửi lệnh nhanh hơn 30 Hz chỉ lặp lại cùng một kết quả
  phát hiện cũ nhiều lần, không mang thêm thông tin mới, trong khi đường
  truyền vẫn phải xử lý thêm gói tin không cần thiết.
- **→ 30 Hz là điểm "khớp một-một" tối ưu:** mỗi lệnh gửi đi mang đúng một kết
  quả phát hiện mới, không lãng phí cũng không bỏ lỡ.
- **Đường truyền chịu được:** đo thật cho thấy ở tần suất 50 Hz vẫn **0% mất
  gói**, khứ hồi (round-trip) p99 chỉ 7,73 ms — tức băng thông/độ trễ truyền
  không phải giới hạn, giới hạn thật là tốc độ sinh dữ liệu của detector.
- **Thang an toàn, giải thích chi tiết hơn slide 19:**
  - FOLLOW: đang thấy người, bám bình thường.
  - COAST (< 3 giây mất dấu): không thấy người nhưng chưa lâu — máy bay vẫn
    di chuyển theo **vị trí dự đoán** (ngoại suy từ vận tốc cũ) nhưng **giảm
    một nửa tốc độ** để thận trọng hơn.
  - HOVER (> 3 giây mất dấu): mất quá lâu, không còn tin vào dự đoán nữa —
    máy bay **đứng yên tại chỗ**, không tự ý bay về điểm cất cánh (RTL) hay
    làm hành động nào khác ngoài dự kiến của người lái.
  - **Người lái luôn có thể lấy lại quyền** bất cứ lúc nào chỉ bằng cách gạt
    công tắc chế độ bay sang chế độ khác (rời GUIDED) — phần mềm lập tức im
    lặng, không gửi lệnh nữa.

**Nói sao cho đúng:** "Nhiều hệ thống dùng 10 Hz theo thói quen, nhưng đo thật
cho thấy điều đó cộng thêm 50 ms vô ích. Nhóm chọn 30 Hz vì khớp đúng nhịp
sinh dữ liệu của detector. Về an toàn, mất người dưới 3 giây thì giảm nửa tốc
độ bám theo dự đoán, lâu hơn thì đứng yên chờ — và người lái luôn lấy lại
được quyền ngay lập tức bằng công tắc chế độ bay."

---

## Slide 21 — Video 1: Phát hiện người trên video bay thử

**Trên slide:** Khung video (bấm để phát), chú thích "Hộp xanh = người được
phát hiện · dấu thập = tâm khung hình", ghi rõ bản đầy đủ 4 phút 43 nộp kèm
hồ sơ tại `video/video_bay_thu.mp4`.

**Nội dung video:** Đoạn 18 giây cắt liên tục (không ghép, không tua) từ
chuyến bay thật ngày 21/07/2026, chạy qua mô hình SteadyDet-N — chỉ hiển thị
kết quả **phát hiện thô**, chưa qua bước theo vết/bám.

**Nói sao cho đúng:** "Đây là mô hình chạy trên chính video bay thử, mỗi hộp
xanh là một người được phát hiện. Đoạn này 18 giây liên tục, không ghép nối."
→ Phát video khoảng 15–20 giây rồi chuyển sang slide 22.

---

## Slide 22 — Video 2: Vòng bám người trên chính đoạn bay đó

**Trên slide:** Khung video (bấm để phát), chú thích màu: xanh lá = đang khoá
· xám = đã phát hiện, chưa khoá · vàng = quỹ đạo · không hộp = khung đó không
thấy ai. Có dòng chữ: "Chuyến bay 21/07 ở chế độ giữ vị trí; vòng bám chạy
trên video của chuyến bay đó, lệnh quay hướng được tính ở từng khung. Giữ mục
tiêu 83,5% là số đo trên toàn clip 4 phút 43."

**Nghĩa là gì — ĐÂY LÀ ĐIỂM QUAN TRỌNG NHẤT CẦN NÓI ĐÚNG, không được để Hội
đồng hiểu nhầm:**

Cùng đoạn 18 giây vừa xem ở slide 21, nhưng giờ chạy qua **toàn bộ chương
trình điều khiển** (OC-SORT → chọn/khoá mục tiêu → ước lượng khoảng cách →
PID → thang an toàn) để tính ra lệnh quay hướng mà máy bay *sẽ* gửi.

**Sự thật cần nói rõ, chủ động, không chờ bị hỏi:** Chuyến bay 21/07/2026 này
được thực hiện ở **chế độ giữ vị trí (position hold)**, tức **người lái điều
khiển bằng tay**, máy bay không tự bay theo người. Đoạn video phần 2 là kết
quả chạy **lại (replay) ngoại tuyến** đoạn video đã quay, qua đúng các mô-đun
phần mềm của `follow_drone/`, để tính ra lệnh quay hướng **sẽ** được gửi —
nhưng trong chuyến bay thật đó, lệnh **không** được gửi xuống máy bay. Đây là
cách kiểm chứng toàn bộ chuỗi xử lý phần mềm hoạt động đúng trên dữ liệu bay
thật, **không phải bằng chứng máy bay đã tự động bay bám người**.

Lý do: phần cứng nối với Pixhawk 6C mới kiểm tra xong liên lạc ngày 27/08 (sau
chuyến bay 21/07); bay tự động hoàn toàn (chế độ GUIDED nhận lệnh) là **bước
tiếp theo**, chưa thực hiện trong phạm vi thời gian của đề tài.

**Nói sao cho đúng (nói chủ động câu này trước khi Hội đồng kịp hỏi):**
"Xin lưu ý rõ: chuyến bay này bay ở chế độ giữ vị trí bằng tay. Vòng bám được
chạy lại trên video của chính chuyến bay đó để kiểm chứng toàn bộ chuỗi xử lý
— từ phát hiện, theo vết, tới lệnh quay hướng — hoạt động đúng trên dữ liệu
thật. Lệnh được tính ra nhưng chưa gửi xuống máy bay trong chuyến bay này.
Phần cứng nối Pixhawk đã kiểm tra xong, bay tự động hoàn toàn là bước kế
tiếp."

**Nếu bị hỏi (gần như chắc chắn sẽ bị hỏi):**
- *"Vậy máy bay đã tự động bám người thật chưa?"* → Chưa. Đây là kiểm chứng
  phần mềm trên dữ liệu thật ngoại tuyến, không phải chuyến bay tự động.
- *"Vì sao không bay tự động luôn cho chắc?"* → Vì an toàn bay cần quy trình
  nghiêm ngặt (xem slide 23, 7 tầng kiểm thử) và cần thời gian chuẩn bị thêm
  sau khi phần cứng mới kiểm tra xong liên lạc (27/08), nằm trong kế hoạch kế
  tiếp chứ không kịp trong phạm vi báo cáo này.

---

## Slide 23 — Bảy tầng kiểm thử trước khi bay

**Trên slide:** Bảng 7 tầng, đi từ không cần phần cứng đến bay thử ngoài bãi:

| Tầng | Nội dung | Cách kiểm thử | Kết quả |
|---|---|---|---|
| 1 | Logic chọn mục tiêu + an toàn | Kiểm thử tự động, không cần phần cứng | 25/25 đạt |
| 2 | Mô hình trên NPU | 548 ảnh VisDrone-val + 92 khung bay thật, trên Pi | AP 0,917 · F1 0,880 |
| 3 | Camera + suy luận | Đo liên tục 2 phút, camera thật | 30,0 FPS · p99 36ms · 48,5°C |
| 4 | Kiểm tra môi trường | HailoRT, thư viện, HEF, camera, nhiệt, đĩa, NTP | 10 OK / 0 lỗi |
| 5 | Liên lạc với máy bay | Đọc trạng thái, khứ hồi lệnh tới 50 Hz | 94–100 gói/s · 0% mất gói |
| 6 | Toàn bộ vòng bám, không cất cánh | Chạy đủ camera/NPU/theo vết/PID/giám sát — chặn mọi lệnh | 29,9 Hz · 19,93 ms/khung |
| 7 | Bay thử ngoài bãi | Bay thử, chạy vòng bám trên video | Giữ mục tiêu 83,5% |

**Nghĩa là gì:** Đây là nguyên tắc kỹ thuật phần mềm chuẩn — **kiểm thử phân
tầng** (layered testing): mỗi tầng chỉ được kiểm tra khi tầng dưới nó đã đạt,
sao cho khi có lỗi, biết ngay lỗi nằm ở tầng nào thay vì phải dò toàn hệ
thống phức tạp cùng lúc. Chú ý tầng 6 đặc biệt quan trọng về an toàn: **chạy
đủ mọi thành phần phần mềm (camera, NPU, theo vết, PID, bảng giám sát) nhưng
cố tình chặn không cho gửi bất kỳ lệnh điều khiển nào xuống máy bay** — đây
là bước kiểm tra "mọi thứ chạy đúng tốc độ, đúng logic" mà **không có rủi ro
thật** (máy bay không nhận lệnh sai nếu phần mềm có lỗi) — mới tới tầng 7 là
bay thử thật ngoài bãi.

**Nói sao cho đúng:** "Hệ thống không được mang ra bãi ngay. Bảy tầng, tầng
dưới đạt mới lên tầng trên — từ kiểm thử tự động không cần phần cứng, tới
chạy đủ mọi thành phần nhưng chặn lệnh (tầng 6), rồi mới bay thử ngoài bãi."

---

## Slide 24 — Mức độ đạt mục tiêu và kết luận

**Trên slide:**
- Bảng đối chiếu 3 mục tiêu (đúng 3 mục tiêu đã nêu ở slide 3) với kết quả
  đạt được:

| Mục tiêu (thuyết minh) | Chỉ tiêu | Kết quả đạt được | Đánh giá |
|---|---|---|---|
| Mô hình nhận dạng người thời gian thực | AP ≥ 70% | AP@0,5 = 0,917 (INT8, bay thật) | **Vượt chỉ tiêu** |
| Thuật toán bám giữ khoảng cách, hướng bay | Hoạt động ổn định | Giữ mục tiêu 83,5% · sai số hướng 5,5° | Đạt |
| Tích hợp camera–xử lý ảnh–điều khiển bay | Chạy trên máy bay | 30,0 FPS đầu-cuối, đã bay thử thực tế | Đạt |

- 4 kết luận đánh số: (1) rung là lỗ hổng thật và đo được, mọi mô hình nhỏ
  gọn mất 74–83% AP · (2) huấn luyện nhận biết rung: 16,8%→36,2%, không tốn
  chi phí khi chạy · (3) chạy thật trên Pi5+Hailo-8L: 0,917 AP và 30 FPS ·
  (4) toàn bộ quy trình tái lập được: cố định seed, một chương trình đo.
- Hộp "Hướng phát triển": mở rộng tập bay thật (nhiều độ cao/người/cảnh) ·
  hiệu chuẩn mức rung bằng dữ liệu IMU thật · bám nhiều mục tiêu và tránh vật
  cản.

**Nghĩa là gì:** Đây là slide "chốt sổ" — đối chiếu lại đúng 3 mục tiêu đăng
ký trong thuyết minh (slide 3) để chứng minh đề tài hoàn thành đúng cam kết,
rồi tóm tắt 4 đóng góp khoa học chính theo thứ tự quan trọng giảm dần. Ba
hướng phát triển không phải nói cho có — mỗi hướng gắn trực tiếp với một hạn
chế đã nêu trong bài: mở rộng tập bay thật (gắn với hạn chế ở slide 14), hiệu
chuẩn IMU thật (gắn với việc rung mới mô phỏng chứ chưa đo bằng cảm biến quán
tính thật), bám nhiều mục tiêu (gắn với giới hạn "chỉ một người" ở slide 13).

**Nói sao cho đúng:** "Đối chiếu thuyết minh: mục tiêu độ chính xác 70% đạt
91,7%, vượt chỉ tiêu. Hai mục tiêu còn lại đều đạt. Bốn kết luận chính..." →
đọc theo đúng 4 gạch đầu dòng, đây là phần tóm tắt nên không cần thêm diễn
giải ngoài những gì đã nói ở các slide trước.

---

## Slide 25 — Cảm ơn Hội đồng

**Trên slide:** "CẢM ƠN HỘI ĐỒNG ĐÃ LẮNG NGHE", dòng phụ "Em xin sẵn sàng trả
lời câu hỏi của Hội đồng", thông tin đề tài/người trình bày/GVHD, logo ISLab.

**Nói:** Câu cảm ơn ngắn, mời Hội đồng đặt câu hỏi. Đây là lúc dùng tới phần
câu hỏi thường gặp bên dưới.

---

## Tổng hợp câu hỏi Hội đồng hay hỏi — và câu trả lời đã chuẩn bị sẵn

1. **"Vì sao AP trên ảnh sạch của SteadyDet-N thấp hơn YOLO?"**
   → Các mô hình YOLO khởi tạo từ trọng số huấn luyện sẵn trên COCO (hàng
   triệu ảnh), còn SteadyDet-N huấn luyện từ đầu, chỉ trên dữ liệu của đề
   tài. Trục đánh giá chính của đề tài là **độ chính xác khi rung (mCP)**,
   nơi SteadyDet-N dẫn đầu nhóm mô hình nhẹ (slide 12).

2. **"So sánh với YOLO không qua xử lý rung có công bằng không?"**
   → Đã kiểm chứng bằng cách áp đúng phương pháp huấn luyện nhận biết rung
   lên YOLO11-n: retain không tăng (20,1% → 20,1%, xem slide 9/11). Vậy mỗi
   mô hình đã được so sánh ở đúng trạng thái tốt nhất có thể đạt được của nó.

3. **"Rung mô phỏng (VisDrone-Shake) có giống rung thật trên máy bay không?"**
   → Tham số mô phỏng dựa trên mô hình vật lý của rung động nhưng **chưa
   hiệu chuẩn bằng dữ liệu IMU thật** (cảm biến quán tính trên Pixhawk). Tuy
   nhiên đã kiểm chứng gián tiếp: trên 92 khung bay thật, xử lý rung cho
   cùng chiều hướng cải thiện (hộp khít hơn ~80% ở AP@0,75, slide 14). Hiệu
   chuẩn bằng IMU thật là hướng phát triển đã nêu ở slide 24.

4. **"Chỉ tiêu 70% được đo chính xác như thế nào?"**
   → AP@0,5 theo đúng chuẩn đánh giá COCO, đo bằng **chính tệp HEF INT8**
   đang chạy trên máy bay, trên 92 khung bay thật gán nhãn tay: kết quả 0,917
   (slide 16).

5. **"Nếu máy bay mất dấu người thì xử lý thế nào, có nguy hiểm không?"**
   → Dưới 3 giây: bám theo vị trí dự đoán với nửa vận tốc (COAST). Trên 3
   giây: đứng yên tại chỗ chờ (HOVER), **không** tự động bay về điểm cất
   cánh. Người lái có thể lấy lại toàn quyền bất cứ lúc nào bằng cách gạt
   công tắc chế độ bay (slide 20).

6. **"Vì sao chọn NPU Hailo-8L thay vì Jetson hay bo mạch khác?"**
   → Hailo-8L rẻ hơn, tiêu thụ điện ít hơn, gắn thẳng vào khe M.2 của
   Raspberry Pi 5 mà không cần máy tính rời. Đo thật đạt 63,66 FPS ở ảnh
   640×640 (slide 15–16), đủ dư cho yêu cầu 30 Hz của vòng điều khiển.

7. **"Máy bay đã thực sự tự động bay bám người chưa, hay chỉ mô phỏng?"**
   → **Chưa bay tự động.** Chuyến bay 21/07/2026 ở chế độ giữ vị trí, người
   lái điều khiển bằng tay. Vòng bám trong video (slide 22) là chạy lại
   ngoại tuyến trên video đã quay để kiểm chứng phần mềm — lệnh được tính ra
   nhưng chưa gửi xuống máy bay trong chuyến bay đó. Phần cứng nối Pixhawk đã
   kiểm tra xong liên lạc (27/08), bay tự động hoàn toàn là bước kế tiếp sau
   đề tài. **Đây là câu trả lời quan trọng nhất phải nói đúng và chủ động,
   không chờ bị hỏi** — đã nói trước ở slide 22.

8. **"Tại sao không bám được nhiều người cùng lúc dù mô hình phát hiện được
   cả trăm người?"**
   → Phạm vi đề tài theo thuyết minh là bám **một người**. Mô hình phát hiện
   (slide 13) và khả năng bám (toàn hệ thống điều khiển) là hai năng lực
   khác nhau — bộ chọn mục tiêu chủ động chỉ khoá một người. Bám nhiều mục
   tiêu là hướng phát triển đã nêu (slide 24).

9. **"Giao thức đo lường có đảm bảo khách quan, không tự chọn số đẹp không?"**
   → Mọi mô hình (kể cả của đề tài) đều đo bằng **đúng một chương trình đo
   duy nhất**, cố định seed, lặp 3 lần, báo cáo trung bình ± độ lệch chuẩn
   (slide 4, 5, 9, 11). Các khác biệt có ý nghĩa còn được kiểm bằng khoảng
   tin cậy bootstrap (slide 14).

---

*File này dựng tự động từ nội dung thật trong* `SteadyDet-N_nghiem_thu.pptx`
*(bảng, chữ, ghi chú) kèm giải thích thêm dựa trên báo cáo tổng kết. Nếu sau
này slide có sửa nội dung, chạy lại việc trích xuất để đối chiếu, tránh file
này lệch với bản trình chiếu thật.*
