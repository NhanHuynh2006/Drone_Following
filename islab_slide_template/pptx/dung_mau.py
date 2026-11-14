#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dung_mau.py — dựng ISLab_mau.pptx: bộ slide mẫu, cùng nội dung với ../latex/mau.tex.

Đây cũng là ví dụ cách viết một bài báo cáo bằng islab_slides.py: chép file này,
đổi nội dung, chạy lại.

    python dung_mau.py
"""
import os

from pptx.enum.text import PP_ALIGN
from pptx.util import Inches

from islab_slides import *

OUT = os.path.join(HERE, "ISLab_mau.pptx")

prs = tao_bai(ten_ngan="Tên ngắn của báo cáo")

# ------------------------------------------------------------------ bìa -----
trang_bia(prs, "Tên đề tài hoặc tên báo cáo\nđặt ở đây, tối đa ba dòng",
          "Loại báo cáo — ví dụ: báo cáo tiến độ, seminar, nghiệm thu",
          "Họ và tên người trình bày", "ISLab — Intelligent Systems Laboratory",
          "Tháng 11 năm 2026",
          ghi_chu="Ghi chú người nói đặt ở đây: lời chào, tên người trình bày, tên báo cáo.")

# ------------------------------------------------------------- mục lục ------
s = trang(prs, "Nội dung trình bày")
muc = [("01", "Giới thiệu", "Bối cảnh, bài toán và mục tiêu"),
       ("02", "Phương pháp", "Hệ thống, mô hình, thuật toán"),
       ("03", "Thực nghiệm", "Dữ liệu, cách đo, cấu hình"),
       ("04", "Kết quả", "Số liệu chính và so sánh"),
       ("05", "Thảo luận", "Hạn chế và điều còn bỏ ngỏ"),
       ("06", "Kết luận", "Đóng góp và hướng tiếp theo")]
khe, cao = Inches(0.4), Inches(1.12)
cot = (RONG - khe) // 2
for i, (so, ten, mo_ta) in enumerate(muc):
    the(s, LE + (i // 3) * (cot + khe), TRAN + Inches(0.75) + (i % 3) * (cao + Inches(0.22)),
        cot, cao, so, ten, mo_ta, co=15)

# ---------------------------------------------- giới thiệu phòng thí nghiệm --
trang_muc(prs, "01", "Giới thiệu phòng thí nghiệm",
          "Hai trang dùng lại được cho mọi báo cáo: hướng nghiên cứu và hệ thống phần cứng của ISLab.")

s = trang(prs, "Hướng nghiên cứu của ISLab",
          phu="Hệ thống tự hành và robot thông minh, từ trên không, mặt đất đến dưới nước")
huong = [
    ("01", "Autonomous UAVs",
     "Điều khiển, dẫn đường, nhận thức môi trường, SLAM, phát hiện và theo dõi đối tượng cho drone tự hành."),
    ("02", "Autonomous Vehicles",
     "Sensor fusion, computer vision, localization, mapping và environment perception cho xe tự hành."),
    ("03", "Mobile Robotics & Lifelong AI",
     "SLAM, định vị, xây dựng bản đồ, tự hành và tránh vật cản trên Clearpath Husky A200 và Unitree Go2 EDU."),
    ("04", "Underwater Robotics – ROV/AUV",
     "Thị giác dưới nước, điều khiển và dẫn đường ROV, kiểm tra đường ống, docking tự động, định vị và SLAM dưới nước."),
    ("05", "Intelligent & Assistive Robotics",
     "Human–robot interaction, computer vision, NLP và robot hỗ trợ con người."),
]
cao, khe = Inches(0.94), Inches(0.14)
for i, (so, ten, mo_ta) in enumerate(huong):
    the(s, LE, TRAN + Inches(0.05) + i * (cao + khe), RONG, cao, so, ten, mo_ta, co=15.5)

s = trang(prs, "Hệ thống phần cứng của ISLab", phu="Thiết bị thành viên trực tiếp làm việc")
khe = Inches(0.3)
cot = (RONG - 2 * khe) // 3
y, cao = TRAN + Inches(0.95), Inches(2.75)
khoi(s, LE, y, cot, cao, "Tính toán biên", ["NVIDIA Jetson AGX Orin 64GB"], co=17)
khoi(s, LE + cot + khe, y, cot, cao, "Cảm biến",
     ["Ouster OS1-32 3D LiDAR", "Intel RealSense D435i", "Xsens/Movella MTi-630R IMU",
      "Duro high-precision GNSS"], co=17)
khoi(s, LE + 2 * (cot + khe), y, cot, cao, "Nền tảng robot",
     ["Clearpath Husky A200", "Unitree Go2 EDU", "Các nền tảng robot nghiên cứu khác"], co=17)

# ------------------------------------------------- các kiểu trang nội dung ---
trang_muc(prs, "02", "Các kiểu trang nội dung", "Chép kiểu trang cần dùng, thay chữ và hình.")

s = trang(prs, "Trang chữ và hình", phu="Dòng phụ: phạm vi, bộ dữ liệu hoặc điều kiện đo",
          kieu="hai_cot")
gach(s, ["Mỗi trang chỉ nói một ý, viết thành câu ngắn.",
         "Dùng **màu nhấn** cho con số hoặc kết luận cần nhớ, mỗi trang không quá hai chỗ.",
         "Gạch đầu dòng cấp hai dành cho điều kiện hoặc ví dụ:",
         (1, "cỡ chữ nhỏ hơn một bậc;"),
         (1, "không lồng quá hai cấp.")], idx=1)
p2 = [p for p in s.placeholders if p.placeholder_format.idx == 2][0]
x2 = p2.left
p2._element.getparent().remove(p2._element)        # cột phải đặt hình thay cho khung chữ
khung = hop(s, x2, TRAN + Inches(0.15), LE + RONG - x2, Inches(4.2), fill=NHAT, line=VIEN)
chu(s, khung.left, khung.top + Inches(1.9), khung.width, Inches(0.4),
    "anh(slide, \"hinh.png\", x, y, w, h)", co=14, mau=MO, canh=PP_ALIGN.CENTER)
chu_thich(s, khung, "Chú thích hình đặt ngay dưới hình")

s = trang(prs, "Trang số liệu và bảng", phu="Bốn ô số liệu phía trên, bảng chi tiết phía dưới")
dai_so_lieu(s, TRAN + Inches(0.1), [("36,2%", "chỉ số chính"), ("× 4,2", "mức cải thiện"),
                                    ("63,7 FPS", "tốc độ trên thiết bị biên"),
                                    ("0 ms", "chi phí tăng thêm")])
bang(s, LE, TRAN + Inches(1.5), RONG, Inches(2.5), [
    ["Cấu hình", "Chỉ số A", "Chỉ số B", "Ghi chú"],
    ["Đối chứng", "0,189", "16,8", "mốc so sánh"],
    ["Phương án 1", "0,182", "16,9", "không cải thiện"],
    ["@Phương án đề xuất", "@0,171", "@**36,2**", "@**tốt nhất**"],
], rong_cot=[3, 1.6, 1.6, 2.4], co=15)

s = trang(prs, "Trang khối nhấn mạnh")
khe = Inches(0.4)
cot = (RONG - khe) // 2
khoi(s, LE, TRAN + Inches(0.2), cot, Inches(1.5), "Khối thường",
     "Dùng cho định nghĩa, giả thiết, cấu hình thí nghiệm.", co=15)
khoi(s, LE, TRAN + Inches(2.0), cot, Inches(1.5), "Khối ví dụ",
     "Dùng cho ví dụ minh hoạ hoặc ghi chú triển khai.", mau=IS, co=15)
khoi(s, LE + cot + khe, TRAN + Inches(0.2), cot, Inches(1.5), "Khối cần chú ý",
     "Dùng cho hạn chế, điều kiện áp dụng, điều chưa kiểm chứng.", mau=CAM, co=15)
tf = chu(s, LE + cot + khe, TRAN + Inches(2.0), cot, Inches(1.5),
         "Công thức: chèn bằng Insert → Equation của PowerPoint, hoặc xuất ảnh từ LaTeX "
         "rồi đặt bằng anh().", co=15, gian=1.1)
nhan(s, LE + cot + khe, TRAN + Inches(3.05), "Nhãn nhấn", w=Inches(1.4))

# ---------------------------------------------------------------- kết -------
trang_ket(prs, lien_he="Họ và tên người trình bày\nISLab — Intelligent Systems Laboratory\n"
                       "email@truong.edu.vn")

prs.save(OUT)
print("Đã lưu:", OUT, "—", len(prs.slides), "slide")
