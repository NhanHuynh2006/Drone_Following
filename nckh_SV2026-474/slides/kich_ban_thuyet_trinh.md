# Kịch bản thuyết trình — đề tài SV2026-474 (khoảng 12–15 phút)

Mỗi slide ghi thời lượng gợi ý và ý chính cần nói. Số liệu đều khớp với báo cáo tổng kết.

| Slide | Thời lượng | Nói gì |
|---|---|---|
| 1. Trang bìa | 0:20 | Chào Hội đồng, giới thiệu tên đề tài, mã số SV2026-474, thành viên và giảng viên hướng dẫn. |
| 2. Nội dung | 0:15 | Trình bày theo 8 phần. |
| 3. Đặt vấn đề | 0:45 | UAV dùng nhiều trong giám sát, cứu hộ; người lái vừa lái vừa giữ khung hình rất khó. Cần UAV tự nhận dạng và bám người, xử lý ngay trên máy bay. |
| 4. Khoảng trống | 0:45 | Cánh quạt làm camera rung → nhoè, nhiễu, dao động sáng. Bộ chuẩn chỉ đo ảnh sạch. Đo của nhóm: mọi bộ phát hiện nhỏ gọn mất 74–83% độ chính xác khi rung. |
| 5. Mục tiêu | 0:40 | Ba mục tiêu theo thuyết minh, chỉ tiêu ≥ 70%. Phương pháp: đối chứng, một chương trình đo, đo trên phần cứng thật. |
| 6. Tổng quan hệ thống | 0:40 | Đi theo mũi tên: camera → phát hiện trên NPU → OC-SORT → khoá mục tiêu → khoảng cách → PID → an toàn → MAVLink → Pixhawk. Tay điều khiển luôn ưu tiên. |
| 7. Dữ liệu | 0:30 | Hai miền: VisDrone bay cao, người rất nhỏ; bay thật 6 m, người lớn gấp ~64 lần diện tích. 92 khung bay thật chỉ dùng để đánh giá. |
| 8. VisDrone-Shake | 0:40 | Ba mức rung; nhoè không làm dịch vật nên nhãn giữ nguyên. Nhấn mạnh: phải cố định seed (không cố định seed thì 17,0/17,1/17,6%) và dùng một chương trình đo. |
| 9. SteadyDet-N | 0:50 | 1,17 M tham số, 4,6 GFLOPs. Giữ P2 cho người nhỏ nhưng làm mỏng vì chi phí tỷ lệ diện tích bản đồ; rẻ hơn bản trước 2,8 lần. |
| 10. Huấn luyện nhận biết rung | 0:50 | Thay 50% lô bằng ảnh rung, tính loss bình thường, không đổi mô hình khi chạy. Biến thể nhất quán phức tạp hơn nhưng kém hơn. |
| 11. Kết quả chính | 1:00 | Đọc bảng từ trên xuống: gốc 16,8% → thêm 120 epoch không rung vẫn 16,9% (không phải do train thêm) → nhất quán 30,0% → đề xuất 36,2%. AP ở mức 2 tăng 4,2 lần. Tái lập trên SteadyDet-W, không tác dụng trên YOLO11-n. |
| 12. So sánh 5 mô hình | 0:50 | Trên ảnh sạch nhóm thua, nói thẳng. Khi rung, mô hình nhỏ nhất vượt cả 4 YOLO nano; ở mức 2 vượt RT-DETR-l lớn gấp 28 lần. |
| 13. Bay thật | 0:40 | Dữ liệu cự ly gần tăng AP@0,5 gấp 5,3 lần → chọn dữ liệu theo miền triển khai. Xử lý rung làm hộp khít hơn ~80%, khoảng tin cậy không chứa 0. |
| 14. Triển khai NPU | 0:45 | INT8 trên máy bay đạt AP@0,5 = 0,917, vượt chỉ tiêu. Tổn thất INT8 phụ thuộc kích thước vật. Nhanh hơn YOLO26-n 1,70 lần nhờ ít context. |
| 15. Độ trễ | 0:35 | 30 FPS đầu-cuối, p99 36 ms; tính toán chỉ 59% ngân sách → giới hạn là camera. Sửa 2 lỗi camera: 11,1 → 30,0 FPS. |
| 16. Điều khiển | 0:45 | Vòng lặp 10 bước; OC-SORT, khoá và tìm lại người cũ, pinhole + Kalman, 3 PID (đọc bảng hệ số). |
| 17. 30 Hz và an toàn | 0:40 | 30 Hz khớp nhịp detector, 10 Hz thêm 50 ms trễ vô ích. Thang an toàn FOLLOW → COAST → HOVER; chỉ gửi lệnh khi GUIDED. |
| 18. Vòng bám người | 0:45 | Chương trình bám người xử lý 3.961 khung của chuyến bay: khoá người đi bộ, giữ mục tiêu 83,5% thời gian, sai số hướng trung vị 5,5°; mất phát hiện ngắn thì COAST theo dự đoán rồi bám lại. |
| 19. Video bay thử | 1:00–4:43 | Bấm nút phát video. Phần 1: phát hiện người trên video bay thử. Phần 2: vòng bám người (hộp xanh lá = mục tiêu khoá, vệt vàng = quỹ đạo, bảng góc trái = trạng thái và lệnh). Nếu thời gian ngắn, tua tới phần 2. |
| 20. Đánh giá mục tiêu | 0:40 | Đọc cột kết quả: vượt chỉ tiêu độ chính xác, đạt các mục tiêu còn lại. |
| 21. Kết luận | 0:40 | Bốn kết luận chính và hướng phát triển. |
| 22. Cảm ơn | 0:10 | Cảm ơn Hội đồng, sẵn sàng trả lời câu hỏi. |

## Câu hỏi Hội đồng có thể hỏi và gợi ý trả lời

1. **Vì sao AP sạch của SteadyDet-N thấp hơn YOLO?** — Đúng, trên ảnh sạch mô hình thua (YOLO khởi tạo từ COCO, mô hình nhóm huấn luyện từ đầu). Đề tài không cạnh tranh ở trục đó; trục triển khai là độ chính xác khi rung (mCP), ở đó mô hình dẫn đầu nhóm nano với chi phí thấp nhất.
2. **So sánh mô hình có xử lý với YOLO không xử lý có công bằng không?** — Nhóm đã áp dụng đúng phương pháp lên YOLO11-n và nó không tăng (20,1% → 20,1%), nên đây là so sánh mỗi mô hình ở trạng thái tốt nhất của nó.
3. **Rung mô phỏng có giống rung thật không?** — Tham số được chọn để bao quát điều kiện bay, chưa hiệu chuẩn bằng IMU. Kiểm chứng trên 92 khung bay thật cho cùng chiều hướng (hộp khít hơn ~80%). Nhật ký bay đã ghi rung thật cùng dòng với chất lượng phát hiện để hiệu chuẩn tiếp.
4. **Chỉ tiêu 70% đo thế nào?** — AP@0,5 theo chuẩn COCO của đúng mô hình INT8 đang chạy trên máy bay, trên 92 khung bay thật gán nhãn tay: 0,917. F1 ở ngưỡng vận hành 0,25 là 0,880.
5. **Khoảng cách ước lượng sai số bao nhiêu?** — Mô hình pinhole giả định chiều cao người 1,7 m; sai số tăng theo bình phương khoảng cách nên có bộ lọc Kalman với phương sai đo thay đổi theo khoảng cách và giới hạn 3–15 m.
6. **Nếu mất người thì máy bay làm gì?** — Dưới 3 s bám theo dự đoán với nửa vận tốc, sau đó đứng yên (HOVER) chờ; không tự RTL; người lái lấy lại quyền bằng công tắc bất cứ lúc nào.
7. **Vì sao dùng Hailo-8L thay vì Jetson?** — Giá thấp, công suất nhỏ, lắp trực tiếp lên Raspberry Pi 5; đo thật cho 63,66 FPS @640, dư ngân sách 30 Hz.
