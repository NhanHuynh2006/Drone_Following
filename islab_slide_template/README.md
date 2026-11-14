# Khung slide ISLab

Khung trình chiếu dùng chung cho mọi báo cáo của ISLab (Intelligent Systems Laboratory),
có hai bản cùng bộ màu, cùng bố cục, cùng hoạ tiết:

| Bản | Mở file nào | Xem trước |
|---|---|---|
| PowerPoint | `pptx/ISLab_mau.pptx` | `pptx/ISLab_mau.pdf` |
| LaTeX (beamer) | `latex/mau.tex` + `latex/beamerthemeISLab.sty` | `latex/mau.pdf` |

Chép cả thư mục này sang dự án mới là dùng được, không phụ thuộc gì bên ngoài.

## Nhận diện

- **Màu** lấy thẳng từ logo: xanh "IS" `#30649E`, xanh "Lab" `#6B95C5`, bản đậm `#244E7C`
  cho thanh tiêu đề, cam `#C86A2C` làm màu nhấn (số liệu, kết luận), nền khối `#EFF3F9`.
- **Hoạ tiết** ở trang bìa, trang chuyển mục và trang kết: bản đồ lưới, quỹ đạo nét đứt có
  điểm mốc, một quadrotor đang quét cảm biến và khung bám mục tiêu màu cam. Gói gọn việc
  phòng thí nghiệm làm: dẫn đường, lập bản đồ, phát hiện và theo dõi đối tượng.
- **Trang nội dung**: thanh tiêu đề xanh đậm, logo trên nền trắng ở góc phải, dải hai sắc
  của logo ngay dưới thanh, chân trang ghi tên phòng thí nghiệm, tên ngắn của báo cáo và
  số trang.
- Phông Arial (PowerPoint) và Helvetica (LaTeX), khổ 16:9.

## Bộ mẫu có gì (10 trang, hai bản giống nhau)

1. Bìa
2. Nội dung trình bày
3. Chuyển mục 01
4. **Hướng nghiên cứu của ISLab** — dùng lại được cho mọi báo cáo
5. **Hệ thống phần cứng của ISLab** — dùng lại được cho mọi báo cáo
6. Chuyển mục 02
7. Trang chữ và hình
8. Trang số liệu và bảng
9. Trang khối nhấn mạnh
10. Kết thúc

## Dùng bản PowerPoint

Khung nằm trong Slide Master nên không phải vẽ lại trên từng trang:

1. Chép `ISLab_mau.pptx`, đổi tên, xoá các trang không dùng.
2. **Home → New Slide** rồi chọn một trong bảy layout: *ISLab - Bìa*, *Nội dung*,
   *Chuyển mục*, *Hai cột*, *Kết thúc*, *Chỉ tiêu đề*, *Trống*.
3. Đổi tên ngắn ở chân trang: **View → Slide Master**, sửa dòng chữ dưới cùng của trang
   master trên cùng.
4. Màu của phòng thí nghiệm đã nằm trong bảng màu theme, hình và biểu đồ vẽ mới tự theo.

Muốn dựng slide bằng mã (số liệu, bảng, hình sinh tự động) thì dùng `pptx/islab_slides.py`;
`pptx/dung_mau.py` là ví dụ đầy đủ:

```python
from islab_slides import *
prs = tao_bai(ten_ngan="Báo cáo tiến độ tháng 11")
trang_bia(prs, "Tên báo cáo", "Loại báo cáo", "Người trình bày",
          "ISLab — Intelligent Systems Laboratory", "Tháng 11 năm 2026")
s = trang(prs, "Kết quả", phu="Bộ dữ liệu X", kieu="noi_dung")
gach(s, ["Ý thứ nhất", "Ý có **chỗ nhấn**", (1, "Ý cấp hai")])
prs.save("bao_cao.pptx")
```

Các hàm có sẵn: `trang_bia`, `trang_muc`, `trang`, `trang_ket`, `gach`, `the` (thẻ có số),
`dai_so_lieu` (ô số liệu), `khoi` (khối có thanh tên), `bang`, `anh`, `chu_thich`, `nhan`,
`chu`. Cần `python-pptx` từ 1.0 và `Pillow`.

## Dùng bản LaTeX

```latex
\documentclass[aspectratio=169,10pt]{beamer}
\usepackage[vietnamese]{babel}
\usetheme{ISLab}
\title{...} \subtitle{...} \author{...} \institute{...} \date{...}
\ISLabTenNgan{Tên ngắn in ở chân trang}
\begin{document}
\ISLabTrangBia
\ISLabTrangMuc{01}{Tên phần}{Mô tả ngắn}
\begin{frame}{Tiêu đề}\framesubtitle{Dòng phụ} ... \end{frame}
\ISLabTrangKet{Lời cảm ơn}{Hỏi và đáp}{Thông tin liên hệ}
\end{document}
```

Biên dịch **hai lần** bằng `pdflatex`. Khối dùng lại: `\ISLabThe[cao]{01}{Tiêu đề}{Nội dung}`,
`\ISLabSo{36,2\%}{nhãn}`, `\ISLabNhan{chữ}`, cùng `block`, `alertblock`, `exampleblock`
và `\alert{}` của beamer đã theo màu ISLab.

Nếu file `.tex` không nằm cạnh thư mục `assets`, khai báo trước `\usetheme`:
`\newcommand{\ISLabAssets}{duong/dan/toi/assets}`.

## Sửa khung

| Muốn đổi | Sửa ở đâu |
|---|---|
| Màu | Đầu file `beamerthemeISLab.sty` và từ điển `_MAU` trong `islab_slides.py` |
| Hoạ tiết bìa, chuyển mục | `\ISLabNenBia`, `\ISLabNenMuc` trong `.sty`, rồi chạy `pptx/xuat_nen.sh` để bản PowerPoint nhận ảnh nền mới |
| Logo | Thay `assets/logo_islab.png` (giữ tỷ lệ 805 × 351) |
| Bộ mẫu PowerPoint | `python pptx/dung_mau.py` |

Bản PowerPoint đã được kiểm bằng LibreOffice (xuất ra `ISLab_mau.pdf`); chưa mở thử trên
Microsoft PowerPoint.
