# Kịch bản thuyết trình ~15 phút — bảo vệ nghiệm thu SV2026-474

Dùng cho bộ slide **`SteadyDet-N_nghiem_thu.pptx`** (24 slide, PowerPoint 16:9).
Mỗi slide đã có sẵn lời dẫn trong ô **ghi chú của người trình bày** (Notes) —
bật Presenter View là vừa đọc được vừa thấy đồng hồ.

Tổng 14:30, còn dư cho phần hỏi đáp. Trọng tâm vẫn là kết quả mô hình và kết
quả nhận diện: slide 9–16 chiếm gần 6 phút.

## Bảng bấm giờ

| # | Slide | Thời lượng | Mốc đồng hồ | Trọng tâm |
|---|---|---|---|---|
| 1 | Trang bìa | 0:20 | 0:20 | Chào Hội đồng, tên đề tài, mã số, thành viên |
| 2 | Vấn đề và khoảng trống | 0:45 | 1:05 | Rung làm bộ phát hiện mất **74–83% AP**; benchmark chỉ đo ảnh sạch |
| 3 | Mục tiêu và tổng quan hệ thống | 0:40 | 1:45 | Ba mục tiêu, chỉ tiêu ≥ 70%; đi nhanh theo mũi tên sơ đồ |
| 4 | Dữ liệu: hai miền | 0:35 | 2:20 | VisDrone bay cao + COCO cự ly gần; 92 khung bay thật chỉ để đánh giá |
| 5 | Giao thức đánh giá độ bền với rung | 0:45 | 3:05 | Ba mức rung; nhãn giữ nguyên; cố định seed, lặp 3 lần |
| 6 | Mô hình SteadyDet-N | 0:40 | 3:45 | 1,17 M tham số, 4,6 GFLOPs, giữ P2 nhưng làm mỏng |
| 7 | Quá trình huấn luyện | 0:35 | 4:20 | Train từ đầu 300 epoch + 120 epoch tinh chỉnh rung |
| 8 | Phương pháp huấn luyện nhận biết rung | 0:35 | 4:55 | Thay 50% mỗi lô, nhãn không đổi, **0 ms chi phí khi chạy** |
| 9 | **Kết quả 1 — hiệu quả phương pháp** | 1:05 | 6:00 | Đọc bảng từ trên xuống; **16,8% → 36,2%**, AP mức 2 ×4,2; hai đối chứng |
| 10 | **Kết quả — xếp hạng retain 11 cấu hình** | 0:40 | 6:40 | Hai thanh dài nhất là mô hình của nhóm; đối chứng chỉ 19–22% |
| 11 | **Kết quả 2 — so với 5 mô hình nhẹ** | 1:00 | 7:40 | Ảnh sạch thua (nói thẳng); khi rung vượt 4 YOLO nano, từ mức 2 vượt RT-DETR-l |
| 12 | **Phát hiện trong cảnh đông người** | 0:35 | 8:15 | Ba ảnh VisDrone 119 / 68 / 55 người; AP@0,5 = 0,514 trên 548 ảnh |
| 13 | **Kết quả 3 — bay thật 92 khung** | 0:45 | 9:00 | COCO-person ×5,3 lần; AP@0,75 +78–81%; nêu rõ hạn chế |
| 14 | Đưa mô hình lên NPU Hailo-8L | 0:40 | 9:40 | 4 bước ONNX → HEF; 4 context so với 6 → nhanh 1,70× |
| 15 | **Kết quả 4 — INT8 trên máy bay** | 0:50 | 10:30 | **AP@0,5 = 0,917** vượt chỉ tiêu; F1 0,880 |
| 16 | Chọn ngưỡng tin cậy vận hành | 0:30 | 11:00 | 0,25 cho F1 0,880; 0,40 tụt còn 0,551 |
| 17 | Tốc độ đầu-cuối | 0:30 | 11:30 | 30 FPS, p99 36 ms, nút cổ chai là camera |
| 18 | Vòng điều khiển và lớp an toàn | 0:35 | 12:05 | OC-SORT → PID → MAVLink; thang FOLLOW/COAST/HOVER |
| 19 | Nhịp lệnh 30 Hz và lớp an toàn | 0:45 | 12:50 | 10 Hz cộng thêm 50 ms vô ích; 0% mất gói tới 50 Hz; GUIDED mới gửi lệnh |
| 20 | Video 1 — phát hiện người | 0:25 | 13:15 | Đoạn 18 giây, phát trọn rồi chuyển |
| 21 | Video 2 — vòng bám người | 0:25 | 13:40 | Cùng đoạn bay đó. Nói rõ: bay giữ vị trí, vòng bám chạy lại trên video |
| 22 | Bảy tầng kiểm thử | 0:35 | 14:15 | Tầng dưới đạt mới lên tầng trên; 25/25 phép kiểm tra tự động |
| 23 | Mức độ đạt mục tiêu và kết luận | 0:40 | 14:55 | Đối chiếu 3 mục tiêu; 4 kết luận; hướng phát triển |
| 24 | Cảm ơn | 0:10 | 15:05 | Mời Hội đồng đặt câu hỏi |

## Nếu chỉ còn 10 phút

Bỏ theo thứ tự này, bỏ tới đâu đủ giờ thì dừng — đúng 8 slide được thêm vào so
với bản 10 phút:

1. **Slide 16** (ngưỡng tin cậy) — gộp một câu vào slide 15.
2. **Slide 7** (quá trình huấn luyện) — gộp một câu vào slide 8.
3. **Slide 22** (bảy tầng kiểm thử) — nói miệng một câu ở slide 23.
4. **Slide 14** (đưa lên NPU Hailo) — gộp vào slide 15.
5. **Slide 10** (xếp hạng retain) — slide 11 đã có bảng so sánh.
6. **Slide 19** (nhịp 30 Hz) — gộp một câu vào slide 18.
7. **Slide 5** (giao thức đo rung) — gộp một câu vào slide 4.
8. **Slide 20** (video phát hiện) — sang thẳng slide 21.

**Không bỏ slide 9, 11, 12, 13, 15 và 21.** Đó là phần Hội đồng chấm.

## Chuẩn bị trước khi lên

- Mở **Presenter View** để thấy ghi chú và đồng hồ.
- Thử phát video ở slide 20 và 21 **trên đúng máy chiếu** trước khi bắt đầu.
  Hai đoạn phim nằm **bên trong** `SteadyDet-N_nghiem_thu.pptx` — mở đúng file
  này, không phải bản `_nhe` hay bản PDF. Cách phát: bấm **F5** vào chế độ trình
  chiếu rồi bấm vào khung video. Ở chế độ soạn thảo video trông y như một tấm
  ảnh — đó là bình thường.
- Dự phòng: `SteadyDet-N_nghiem_thu.pdf` mở được mà không cần PowerPoint
  (video thành ảnh tĩnh — khi đó dùng slide 18 để nói thay).

## Hai đoạn phim trên slide

Cắt từ **cùng một khoảng** của chuyến bay (phút 1:38–1:56), nên Hội đồng thấy
đúng cảnh đó hai lần: slide 20 là phát hiện thô, slide 21 là vòng bám chạy trên
chính cảnh ấy. Đoạn được chọn là 18 giây liên tục mà vòng bám giữ khoá tốt nhất
— không lần nào rơi về HOVER, FOLLOW 79,4% số khung — **cắt liền một mạch,
không ghép, không tua**.

**Nói trước khi chiếu slide 21:** "Chuyến bay này bay ở chế độ giữ vị trí; vòng
bám được chạy trên video của chính chuyến bay đó, lệnh quay hướng tính ở từng
khung." Đúng như mục 7.3 báo cáo — Hội đồng cầm báo cáo trên tay nên nói trước
vẫn hơn để bị hỏi.

Con số **83,5% giữ mục tiêu** và **5,5° sai số hướng** là đo trên **toàn bộ clip
4 phút 43**, bản đầy đủ nộp kèm hồ sơ tại `video/video_bay_thu.mp4`.

## Ba bản slide

| Tệp | Dung lượng | Khi nào dùng |
|---|---|---|
| `SteadyDet-N_nghiem_thu.pptx` | ~14 MB | **Bản trình bày chính.** Hai đoạn phim 18 giây nhúng sẵn, copy một file duy nhất qua USB là chạy được. |
| `SteadyDet-N_nghiem_thu_nhe.pptx` | ~3 MB | Gửi mail cho Hội đồng/GVHD. Slide 20 và 21 là ảnh có liên kết tới `../video/phan1_phat_hien.mp4` và `../video/phan2_vong_bam.mp4` nên chỉ bấm được khi giữ nguyên cây thư mục. |
| `SteadyDet-N_nghiem_thu.pdf` | ~1 MB | Bản dự phòng khi máy chiếu không có PowerPoint. |

Dựng lại cả ba:

```bash
cd nckh_SV2026-474/slides
PYTHONPATH=<thư mục có python-pptx> ../../hailo_env/bin/python make_pptx_nghiemthu.py
PYTHONPATH=<thư mục có python-pptx> ../../hailo_env/bin/python make_pptx_nghiemthu.py --nhe
libreoffice --headless --convert-to pdf SteadyDet-N_nghiem_thu_nhe.pptx
# hai đoạn phim: cd ../video && python3 make_video.py
```

## Câu hỏi Hội đồng hay hỏi

Đã ghi sẵn trong ô ghi chú của slide cuối. Bản đầy đủ 7 câu nằm ở
[`kich_ban_thuyet_trinh.md`](kich_ban_thuyet_trinh.md) (kịch bản của bộ slide
LaTeX `slides.pdf`, vẫn giữ để tham khảo).
