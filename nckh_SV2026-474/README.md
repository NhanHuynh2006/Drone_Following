# Hồ sơ nghiệm thu đề tài NCKH sinh viên — mã số SV2026-474

**Máy bay không người lái tự động theo dõi con người bằng trí tuệ nhân tạo**

| Sản phẩm | Tệp | Theo mẫu |
|---|---|---|
| Báo cáo tổng kết | `baocao/main.pdf` | Phụ lục HD viết báo cáo tổng kết (PL01, PL02, PL03) |
| Bài tóm tắt 2 trang, 2 cột | `tomtat/tomtat.docx` (dựng từ file mẫu Word), `tomtat/tomtat.pdf` | Mẫu + Hướng dẫn định dạng tóm tắt |
| Poster triển lãm | `poster_a3/poster_a3.pdf` (A3 ngang, đóng ở Phụ lục E) và `poster_a3/poster_a1.pdf` (A1 để báo cáo) | Mẫu BM07A |
| Standee | `standee/standee.pdf` (0,6 × 1,6 m) | Mẫu BM07B |
| Slide báo cáo hội đồng | `slides/slides.pdf` + `slides/kich_ban_thuyet_trinh.md` | — |
| Video bay thử | `video/video_bay_thu.mp4` (4 phút 43 giây): phần 1 phát hiện người, phần 2 vòng bám người | dựng từ chuyến bay 21/07/2026 |

## Biên dịch lại

```bash
cd baocao && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ../tomtat && pdflatex tomtat && <venv có python-docx>/bin/python build_docx.py
cd ../poster_a3 && pdflatex poster_a3 && gs -o poster_a1.pdf -sDEVICE=pdfwrite -dDEVICEWIDTHPOINTS=2384 -dDEVICEHEIGHTPOINTS=1684 -dFIXEDMEDIA -dPDFFitPage poster_a3.pdf
cd ../standee && pdflatex standee
cd ../slides && pdflatex slides && pdflatex slides
cd ../hinh && /usr/bin/python3 make_figures.py
cd ../video && GRACE=1.5 REACQ=1.0 CONF=0.30 IMG=512 /usr/bin/python3 replay_follow.py && /usr/bin/python3 make_video.py
```

Thông tin chung (tên, MSSV, GVHD...) đặt một chỗ ở `thongtin.tex`.

## Thông tin cá nhân đã điền (16/09/2026)

Trong `thongtin.tex`: Nam, dân tộc Kinh, ngày sinh 14/08/2006, lớp 24119FIE4, điện thoại 0937395656.
Ảnh 3×4 đã cắt về tỷ lệ 3:4 tại `hinh/anh_3x4.jpg` và chèn vào cả `tomtat/tomtat.docx` lẫn `tomtat/tomtat.pdf`.
Logo trường `hinh/logo_truong.png`, logo Khoa `hinh/logo_khoa.jpg`.

Lớp của cả hai thành viên: 24119FIE4.

## Còn lại

1. Chữ ký của sinh viên và nhận xét + chữ ký của giảng viên hướng dẫn ở trang PL03 (ghi tay trên bản in).
2. Hợp đồng đã ký: scan thành `tailieu_goc/hop_dong.pdf` rồi biên dịch lại báo cáo, hợp đồng sẽ tự động đóng vào Phụ lục F ngay sau thuyết minh. Nếu không scan thì kẹp bản photo vào cuối cuốn khi đóng quyển.

## Tệp dùng để IN (28/09/2026)

Bản gốc `poster_a3.pdf` / `standee.pdf` dùng font Type 1 nhúng; một số máy in và máy RIP
ở tiệm in xử lý sai loại font này nên chữ tiếng Việt bị vỡ dấu. Vì vậy có thêm bản
**đã chuyển chữ thành nét vẽ (outline)** — nội dung y hệt, không phụ thuộc font, in ở đâu cũng đúng:

| Nhu cầu | Tệp nên đưa tiệm in | Khổ |
|---|---|---|
| Poster đóng vào cuốn báo cáo | `poster_a3/poster_a3_in.pdf` | A3 ngang, 420 × 297 mm |
| Poster triển lãm | `poster_a3/poster_a1_in.pdf` | A1 ngang, 841 × 594 mm |
| Standee | `standee/standee_in.pdf` | 600 × 1600 mm |
| Dự phòng nếu tiệm in vẫn lỗi | `poster_a3/poster_a3_300dpi.pdf` (đã raster 300 dpi) | A3 ngang |

Khi in nhớ chọn **"Actual size" / "Kích thước thực" / tỷ lệ 100%**, KHÔNG chọn "Fit to page"
hay "Shrink oversized pages"; chọn đúng khổ giấy A3 và hướng giấy ngang.
Gửi thẳng tệp PDF cho tiệm in (không chụp màn hình, không gửi qua ứng dụng nén ảnh).

## In lại các trang đã đổi thuật ngữ

Sau khi đổi thuật ngữ chuyên ngành sang tiếng Anh (backbone, neck, head, seed, benchmark…),
53 trong 111 trang của báo cáo có thay đổi:

- `baocao/trang_in_lai.pdf` — chỉ gồm đúng 53 trang đó, xếp theo thứ tự, khổ A4.
- `baocao/DANH_SACH_TRANG_IN_LAI.txt` — bảng đối chiếu *trang PDF* ↔ *số trang in ở chân trang*.

Cuốn báo cáo in một mặt nên chỉ cần rút tờ cũ ra và thay tờ mới vào đúng vị trí.
