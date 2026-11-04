# Kịch bản thuyết trình 10 phút — bảo vệ nghiệm thu SV2026-474

Dùng cho bộ slide **`SteadyDet-N_nghiem_thu.pptx`** (15 slide, PowerPoint 16:9).
Mỗi slide đã có sẵn lời dẫn trong ô **ghi chú của người trình bày** (Notes) —
bật chế độ Presenter View là đọc được trong lúc nói.

Trọng tâm theo yêu cầu: **kết quả mô hình và kết quả nhận diện** (slide 7–10,
tổng 3 phút 40 trên 10 phút). Các phần còn lại nói lướt.

## Bảng bấm giờ

| # | Slide | Thời lượng | Mốc đồng hồ | Trọng tâm |
|---|---|---|---|---|
| 1 | Trang bìa | 0:20 | 0:20 | Chào Hội đồng, tên đề tài, mã số, thành viên |
| 2 | Vấn đề và khoảng trống | 0:45 | 1:05 | Rung làm bộ phát hiện mất **74–83% AP**; benchmark chỉ đo ảnh sạch |
| 3 | Mục tiêu và tổng quan hệ thống | 0:40 | 1:45 | Ba mục tiêu, chỉ tiêu ≥ 70%; đi nhanh theo mũi tên sơ đồ |
| 4 | Dữ liệu và giao thức đánh giá | 0:35 | 2:20 | Hai miền dữ liệu; cố định seed, lặp 3 lần, một chương trình đo |
| 5 | Mô hình SteadyDet-N | 0:40 | 3:00 | 1,17 M tham số, 4,6 GFLOPs, giữ P2 nhưng làm mỏng |
| 6 | Huấn luyện nhận biết rung | 0:35 | 3:35 | Thay 50% mỗi lô, nhãn không đổi, **0 ms chi phí khi chạy** |
| **7** | **Kết quả 1 — hiệu quả phương pháp** | **1:05** | **4:40** | Đọc bảng từ trên xuống; **16,8% → 36,2%**, AP mức 2 ×4,2; hai đối chứng |
| **8** | **Kết quả 2 — so với 5 mô hình nhẹ** | **1:00** | **5:40** | Ảnh sạch thua (nói thẳng); khi rung vượt 4 YOLO nano, từ mức 2 vượt RT-DETR-l |
| **9** | **Kết quả 3 — bay thật 92 khung** | **0:45** | **6:25** | COCO-person ×5,3 lần; AP@0,75 +78–81%; nêu rõ hạn chế |
| **10** | **Kết quả 4 — INT8 trên máy bay** | **0:50** | **7:15** | **AP@0,5 = 0,917** vượt chỉ tiêu; F1 0,880; nhanh hơn YOLO26-n 1,70× |
| 11 | Tốc độ đầu-cuối | 0:30 | 7:45 | 30 FPS, p99 36 ms, nút cổ chai là camera |
| 12 | Vòng điều khiển và an toàn | 0:35 | 8:20 | Nói lướt: OC-SORT → PID → MAVLink 30 Hz; thang FOLLOW/COAST/HOVER |
| 13 | Video bay thử | 0:50 | 9:10 | Phát **45–60 giây** phần vòng bám rồi dừng |
| 14 | Mức độ đạt mục tiêu và kết luận | 0:40 | 9:50 | Đối chiếu 3 mục tiêu; 4 kết luận; hướng phát triển |
| 15 | Cảm ơn | 0:10 | 10:00 | Mời Hội đồng đặt câu hỏi |

## Nếu bị trễ giờ

Cắt theo thứ tự này, cắt tới đâu đủ giờ thì dừng:

1. **Slide 11** (tốc độ) — bỏ hẳn, gộp một câu vào slide 10: *"chạy đầy đủ trên máy bay được 30 FPS"*.
2. **Slide 13** (video) — rút còn 25 giây, chỉ phát đoạn khoá mục tiêu.
3. **Slide 4** (dữ liệu) — chỉ nói một câu: *"hai miền dữ liệu, 92 khung bay thật chỉ dùng để đánh giá"*.

**Không cắt slide 7–10.** Đó là phần Hội đồng chấm.

## Chuẩn bị trước khi lên

- Mở **Presenter View** để thấy ghi chú và đồng hồ.
- Thử phát video ở slide 13 **trên đúng máy chiếu** trước khi bắt đầu.
  Video nằm **bên trong** `SteadyDet-N_nghiem_thu.pptx` (64 MB) — mở đúng file này,
  không phải bản `_nhe` hay bản PDF. Cách phát: bấm **F5** vào chế độ trình chiếu rồi
  bấm vào khung video, hoặc ở chế độ soạn thảo bấm vào video rồi nhấn nút Play.
  Ở chế độ soạn thảo video trông y như một tấm ảnh — đó là bình thường.
- Dự phòng: `SteadyDet-N_nghiem_thu.pdf` mở được mà không cần PowerPoint
  (video thành ảnh tĩnh — khi đó dùng slide 12 để nói thay).

## Hai bản slide

| Tệp | Dung lượng | Khi nào dùng |
|---|---|---|
| `SteadyDet-N_nghiem_thu.pptx` | ~64 MB | **Bản trình bày chính.** Video nhúng sẵn, copy một file duy nhất qua USB là chạy được. |
| `SteadyDet-N_nghiem_thu_nhe.pptx` | ~2 MB | Gửi mail cho Hội đồng/GVHD. Slide 13 là ảnh có liên kết tới `../video/video_bay_thu.mp4` nên chỉ bấm được khi giữ nguyên cây thư mục. |
| `SteadyDet-N_nghiem_thu.pdf` | ~1 MB | Bản dự phòng khi máy chiếu không có PowerPoint. |

Dựng lại cả ba:

```bash
cd nckh_SV2026-474/slides
PYTHONPATH=<thư mục có python-pptx> ../../hailo_env/bin/python make_pptx_nghiemthu.py
PYTHONPATH=<thư mục có python-pptx> ../../hailo_env/bin/python make_pptx_nghiemthu.py --nhe
libreoffice --headless --convert-to pdf SteadyDet-N_nghiem_thu_nhe.pptx
```

## Câu hỏi Hội đồng hay hỏi

Đã ghi sẵn trong ô ghi chú của slide 15. Bản đầy đủ 7 câu nằm ở
[`kich_ban_thuyet_trinh.md`](kich_ban_thuyet_trinh.md) (kịch bản 12–15 phút của
bộ slide LaTeX `slides.pdf`, vẫn giữ để tham khảo).
