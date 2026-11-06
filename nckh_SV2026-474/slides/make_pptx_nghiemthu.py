#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pptx_nghiemthu.py — dựng bộ slide PowerPoint bảo vệ nghiệm thu (10 phút).

Vì sao có file này: Hội đồng nghiệm thu dùng PowerPoint, còn slides.tex chỉ ra PDF.
Bộ này rút từ 22 slide (12–15 phút) xuống 15 slide vừa đúng 10 phút, dồn thời
lượng vào phần kết quả mô hình và kết quả nhận diện (slide 7–10, ~3 phút 35).

Chạy:
    PYTHONPATH=<thư mục có python-pptx> hailo_env/bin/python make_pptx_nghiemthu.py

Ảnh: các hình .pdf trong ../hinh và sơ đồ TikZ sodo_*.tex được render sang PNG
trước vào thư mục tạm (xem hàm chuan_bi_anh), nên không cần sửa gì trong ../hinh.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
GOC = os.path.dirname(HERE)                     # nckh_SV2026-474/
HINH = os.path.join(GOC, "hinh")
VIDEO = os.path.join(GOC, "video")
OUT = os.path.join(HERE, "SteadyDet-N_nghiem_thu.pptx")
OUT_NHE = os.path.join(HERE, "SteadyDet-N_nghiem_thu_nhe.pptx")
# --nhe: không nhúng video (file ~3 MB, tiện gửi mail) — slide 13 thành ảnh có
# liên kết tới ../video/video_bay_thu.mp4, phải giữ nguyên cây thư mục mới bấm được.
NHE = "--nhe" in sys.argv

# ---- bảng màu, lấy đúng từ slides.tex để đồng bộ với báo cáo ----------------
DAM = RGBColor(0x0B, 0x3C, 0x78)      # xanh navy — thanh tiêu đề
XANH = RGBColor(0x2A, 0x78, 0xD6)     # xanh nhấn
CAM = RGBColor(0xEB, 0x68, 0x34)      # cam — số liệu quan trọng
NHAT = RGBColor(0xEE, 0xF4, 0xFC)     # nền khối nhạt
CHU = RGBColor(0x1A, 0x1A, 0x1A)
MO = RGBColor(0x5A, 0x64, 0x72)
TRANG = RGBColor(0xFF, 0xFF, 0xFF)
VIEN = RGBColor(0xC8, 0xD4, 0xE4)

FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)
LE = Inches(0.55)                      # lề trái/phải
TRAN = Inches(1.30)                    # mép trên vùng nội dung
DAY = Inches(6.95)                     # mép dưới vùng nội dung


# ============================================================ tiện ích vẽ ===
def txt(slide, x, y, w, h, size=16, bold=False, color=CHU, align=PP_ALIGN.LEFT,
        anchor=MSO_ANCHOR.TOP, space=1.0):
    """Hộp chữ trống, trả về text_frame để đổ nội dung."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    p = tf.paragraphs[0]
    p.alignment = align
    p.line_spacing = space
    r = p.add_run()
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), bold, color, FONT
    return tf


def dong(tf, text, size=16, bold=False, color=CHU, bullet=None, space=1.0,
        truoc=6, align=PP_ALIGN.LEFT):
    """Thêm một dòng. `bullet` là ký tự đầu dòng (None = không có)."""
    p = tf.paragraphs[0] if (len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs) \
        else tf.add_paragraph()
    p.alignment = align
    p.line_spacing = space
    p.space_before = Pt(truoc)
    if bullet:
        r = p.add_run()
        r.text = bullet + "  "
        r.font.size, r.font.bold, r.font.name = Pt(size), True, FONT
        r.font.color.rgb = XANH
    # **đậm** và **|cam|** cho phép nhấn giữa dòng
    for phan in re.split(r"(\*\*.+?\*\*)", text):
        if not phan:
            continue
        nhan = phan.startswith("**")
        noi_dung = phan.strip("*") if nhan else phan
        mau = CAM if nhan else color
        r = p.add_run()
        r.text = noi_dung
        r.font.size = Pt(size)
        r.font.bold = bold or nhan
        r.font.color.rgb = mau
        r.font.name = FONT
    return p


def hop(slide, x, y, w, h, fill=NHAT, line=None, radius=True):
    sp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(0.75)
    sp.shadow.inherit = False
    if radius:
        sp.adjustments[0] = 0.06
    sp.text_frame.word_wrap = True
    return sp


def anh(slide, path, x, y, w, h, giua=True):
    """Chèn ảnh vừa khít khung (x,y,w,h), giữ nguyên tỷ lệ, canh giữa khung."""
    from PIL import Image
    iw, ih = Image.open(path).size
    ti_le = min(w / iw, h / ih)
    nw, nh = int(iw * ti_le), int(ih * ti_le)
    px = x + (w - nw) // 2 if giua else x
    py = y + (h - nh) // 2 if giua else y
    return slide.shapes.add_picture(path, px, py, nw, nh)


def chu_thich(slide, x, y, w, text, size=11):
    tf = txt(slide, x, y, w, Inches(0.3), size=size, color=MO, align=PP_ALIGN.CENTER)
    tf.paragraphs[0].runs[0].text = text


def chu_thich_duoi(slide, pic, text, size=11, cach=Inches(0.06)):
    """Chú thích đặt ngay dưới mép dưới của ảnh, không đè vào chữ trong ảnh."""
    y = pic.top + pic.height + cach
    if y + Inches(0.3) > DAY + Inches(0.4):
        y = DAY + Inches(0.1)
    return chu_thich(slide, pic.left - Inches(0.4), y, pic.width + Inches(0.8), text, size)


def trang_moi(prs, tieu_de, phu=None, so=None, ghi_chu=None):
    """Slide trắng + thanh tiêu đề navy + số trang."""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    thanh = hop(s, 0, 0, W, Inches(1.02), fill=DAM, radius=False)
    tf = thanh.text_frame
    tf.margin_left, tf.margin_top = Inches(0.55), Inches(0.14)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = tieu_de
    r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(27), True, TRANG, FONT
    if phu:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = phu
        r2.font.size, r2.font.color.rgb, r2.font.name = Pt(13), RGBColor(0xBF, 0xD6, 0xF2), FONT
    if so is not None:
        t = txt(s, W - Inches(1.1), H - Inches(0.42), Inches(0.6), Inches(0.3),
                size=11, color=MO, align=PP_ALIGN.RIGHT)
        t.paragraphs[0].runs[0].text = str(so)
    if ghi_chu:
        s.notes_slide.notes_text_frame.text = ghi_chu
    return s


def dai_so_lieu(slide, y, muc, cao=Inches(1.02)):
    """Dải KPI ngang: [(số, nhãn), ...] — dùng cho các slide kết quả."""
    n = len(muc)
    khe = Inches(0.16)
    rong = (W - 2 * LE - khe * (n - 1)) // n
    for i, (so, nhan) in enumerate(muc):
        x = LE + i * (rong + khe)
        hop(slide, x, y, rong, cao, fill=NHAT, line=VIEN)
        tf = txt(slide, x, y + Inches(0.12), rong, Inches(0.45), size=25, bold=True,
                 color=CAM, align=PP_ALIGN.CENTER)
        tf.paragraphs[0].runs[0].text = so
        t2 = txt(slide, x + Inches(0.08), y + Inches(0.60), rong - Inches(0.16),
                 Inches(0.36), size=11.5, color=MO, align=PP_ALIGN.CENTER, space=0.95)
        t2.paragraphs[0].runs[0].text = nhan


def bang(slide, x, y, w, h, du_lieu, rong_cot=None, co_chu=12.5, cao_dau=Inches(0.42)):
    """Bảng gọn: hàng đầu là tiêu đề navy, các hàng sau kẻ nhạt."""
    hang, cot = len(du_lieu), len(du_lieu[0])
    g = slide.shapes.add_table(hang, cot, x, y, w, h).table
    if rong_cot:
        tong = sum(rong_cot)
        for i, phan in enumerate(rong_cot):
            g.columns[i].width = Emu(int(w * phan / tong))
    g.rows[0].height = cao_dau
    for i in range(1, hang):
        g.rows[i].height = Emu(int((h - cao_dau) / (hang - 1)))
    for i, hg in enumerate(du_lieu):
        for j, o in enumerate(hg):
            cell = g.cell(i, j)
            cell.text = ""
            cell.margin_left = cell.margin_right = Inches(0.08)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = DAM if i == 0 else (
                NHAT if o.startswith("@") else TRANG)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            for phan in re.split(r"(\*\*.+?\*\*)", o.lstrip("@")):
                if not phan:
                    continue
                nhan = phan.startswith("**")
                r = p.add_run()
                r.text = phan.strip("*") if nhan else phan
                r.font.size = Pt(co_chu)
                r.font.name = FONT
                r.font.bold = (i == 0) or nhan
                r.font.color.rgb = TRANG if i == 0 else (CAM if nhan else CHU)
    return g


# ===================================================== chuẩn bị ảnh =========
def chuan_bi_anh(tmp):
    """Render hình .pdf và sơ đồ TikZ .tex sang .png trong thư mục tạm."""
    for ten, clip, giay in (("poster1", "phan1_phat_hien.mp4", 3.0),
                            ("poster2", "phan2_vong_bam.mp4", 1.5)):
        duong = os.path.join(VIDEO, clip)
        if os.path.exists(duong):
            subprocess.run(["ffmpeg", "-y", "-ss", str(giay), "-i", duong, "-frames:v", "1",
                            os.path.join(tmp, ten + ".png")],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for ten in ("duong_cong_rung", "bay_that", "int8", "do_tre", "retain",
                "hailo", "nguong_conf", "nhip_lenh", "huan_luyen", "camera_fps"):
        subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile",
                        os.path.join(HINH, ten + ".pdf"), os.path.join(tmp, ten)],
                       check=True)
    bia = (r"\documentclass[border=4pt]{standalone}"
           r"\usepackage[T5]{fontenc}\usepackage[utf8]{inputenc}"
           r"\usepackage[vietnamese]{babel}"
           r"\usepackage{mathptmx}\usepackage{helvet}"
           r"\renewcommand{\familydefault}{\sfdefault}"
           r"\usepackage{amsmath,graphicx}\usepackage{tikz}"
           r"\usetikzlibrary{arrows.meta,positioning,shapes.geometric,fit,calc,backgrounds}"
           r"\definecolor{dam}{HTML}{0B3C78}\definecolor{xanh}{HTML}{2A78D6}"
           r"\definecolor{cam}{HTML}{EB6834}\definecolor{nhat}{HTML}{EEF4FC}"
           r"\begin{document}\input{%s}\end{document}")
    for ten in ("sodo_hethong", "sodo_kientruc", "sodo_huanluyen_rung",
                "sodo_vongdieukhien", "sodo_antoan"):
        nguon = os.path.join(HINH, ten + ".tex")
        if ten == "sodo_huanluyen_rung":
            # Chỉ giữ phần (a) — phương pháp đề xuất. Phần (b) là đối chứng, trên
            # slide bị thu nhỏ thì mũi tên "dừng gradient" cắt chéo qua chữ; nội
            # dung phần (b) đã nói bằng một gạch đầu dòng bên phải.
            goc = open(nguon, encoding="utf-8").read()
            dau = goc.index(r"\begin{scope}[yshift=-3.75cm]")
            cuoi = goc.index(r"\end{scope}") + len(r"\end{scope}")
            goc = goc[:dau] + goc[cuoi:]
            goc = goc.replace("(a) Huấn luyện nhận biết rung (phương pháp đề xuất)",
                              "Huấn luyện nhận biết rung — phương pháp đề xuất")
            nguon = os.path.join(tmp, ten + "_a.tex")
            with open(nguon, "w", encoding="utf-8") as f:
                f.write(goc)
        tex = os.path.join(tmp, "w_" + ten + ".tex")
        with open(tex, "w", encoding="utf-8") as f:
            f.write(bia % nguon)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", tex],
                       cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["pdftoppm", "-r", "220", "-png", "-singlefile",
                        os.path.join(tmp, "w_" + ten + ".pdf"), os.path.join(tmp, ten)],
                       check=True)
    return lambda ten: os.path.join(tmp, ten + ".png")


def doc_thongtin():
    s = open(os.path.join(GOC, "thongtin.tex"), encoding="utf-8").read()
    def lay(k):
        m = re.search(r"\\newcommand\{\\" + k + r"\}\{(.*?)\}\s*$", s, re.M)
        return m.group(1) if m else ""
    return lay


# ========================================================== các slide =======
def slide_video(prs, tieu_de, phu, so, ghi_chu, ten_phim, poster, chu_duoi, lien_ket):
    """Một slide = một đoạn video. Bản chính nhúng thẳng phim, bản --nhe dùng ảnh có liên kết."""
    s = trang_moi(prs, tieu_de, phu=phu, so=so, ghi_chu=ghi_chu)
    kx, ky, kw, kh = Inches(2.45), TRAN + Inches(0.15), Inches(8.45), Inches(4.3)
    phim = os.path.join(VIDEO, ten_phim)
    if os.path.exists(phim) and not NHE:
        s.shapes.add_movie(phim, kx, ky, kw, kh, poster_frame_image=poster,
                           mime_type="video/mp4")
        # Trong PowerPoint phải ở chế độ trình chiếu (F5) hoặc bấm nút Play thì mới chạy;
        # ở chế độ soạn thảo video trông y như một tấm ảnh.
        nhan = hop(s, kx + kw - Inches(2.35), ky + Inches(0.12), Inches(2.2), Inches(0.42),
                   fill=CAM, line=None)
        tfn = nhan.text_frame
        tfn.vertical_anchor = MSO_ANCHOR.MIDDLE
        pn = tfn.paragraphs[0]
        pn.alignment = PP_ALIGN.CENTER
        rn = pn.add_run()
        rn.text = "▶  Bấm để phát"
        rn.font.size, rn.font.bold, rn.font.color.rgb, rn.font.name = Pt(13), True, TRANG, FONT
    else:
        pic = anh(s, poster, kx, ky, kw, kh)
        pic.click_action.hyperlink.address = lien_ket
    tf = txt(s, LE, DAY - Inches(0.95), W - 2 * LE, Inches(1.0), size=15,
             align=PP_ALIGN.CENTER, space=1.2)
    for i, (text, co, mau) in enumerate(chu_duoi):
        dong(tf, text, size=co, color=mau, truoc=0 if i == 0 else 6, align=PP_ALIGN.CENTER)
    return s


def dung_slide(prs, P, tt):
    # ---------------------------------------------------------------- 1. bìa
    s = prs.slides.add_slide(prs.slide_layouts[6])
    hop(s, 0, 0, W, Inches(2.30), fill=DAM, radius=False)
    logo = s.shapes.add_picture(os.path.join(HINH, "logo_hcmute.png"),
                                LE, Inches(0.40), height=Inches(0.66))
    t = txt(s, logo.left + logo.width + Inches(0.45), Inches(0.46),
            W - logo.width - Inches(1.5), Inches(0.6), size=13.5,
            color=RGBColor(0xBF, 0xD6, 0xF2), anchor=MSO_ANCHOR.MIDDLE)
    t.paragraphs[0].runs[0].text = "KHOA " + tt("khoa").upper() + "  •  BỘ MÔN " + tt("bomon").upper()
    t2 = txt(s, LE, Inches(1.35), W - 2 * LE, Inches(0.6), size=15, color=TRANG)
    t2.paragraphs[0].runs[0].text = "BÁO CÁO NGHIỆM THU ĐỀ TÀI NCKH SINH VIÊN  —  MÃ SỐ " + tt("madetai")

    t3 = txt(s, LE, Inches(2.75), W - 2 * LE, Inches(1.5), size=34, bold=True,
             color=DAM, align=PP_ALIGN.CENTER, space=1.12)
    t3.paragraphs[0].runs[0].text = tt("TENDETAI")
    hop(s, Inches(4.6), Inches(4.35), Inches(4.1), Inches(0.06), fill=CAM, radius=False)

    t4 = txt(s, Inches(3.0), Inches(4.75), Inches(7.3), Inches(1.6), size=16,
             align=PP_ALIGN.CENTER, space=1.35)
    dong(t4, "Chủ nhiệm đề tài:  " + tt("chunhiem"), size=16, bold=True, truoc=0,
         align=PP_ALIGN.CENTER)
    dong(t4, "Thành viên:  " + tt("thanhvien"), size=15, color=MO, align=PP_ALIGN.CENTER)
    dong(t4, "Giảng viên hướng dẫn:  " + tt("gvhd"), size=15, color=MO,
         align=PP_ALIGN.CENTER)
    t5 = txt(s, LE, H - Inches(0.75), W - 2 * LE, Inches(0.35), size=13, color=MO,
             align=PP_ALIGN.CENTER)
    t5.paragraphs[0].runs[0].text = "TP. Hồ Chí Minh, " + tt("thangnam")
    s.notes_slide.notes_text_frame.text = (
        "⏱ 0:20 — Kính chào Hội đồng. Em là Huỳnh Lê Thành Nhân, chủ nhiệm đề tài "
        "mã số SV2026-474. Hôm nay em xin báo cáo kết quả đề tài UAV tự động bám người "
        "bằng trí tuệ nhân tạo, hướng dẫn bởi PGS.TS. Lê Mỹ Hà.")

    # ------------------------------------------------- 2. vấn đề & khoảng trống
    s = trang_moi(prs, "Vấn đề và khoảng trống nghiên cứu", so=2, ghi_chu=(
        "⏱ 0:45 — UAV dùng nhiều trong giám sát, cứu hộ, quay phim; người lái vừa phải "
        "lái vừa phải giữ người trong khung hình. Vấn đề cốt lõi: cánh quạt làm camera "
        "rung liên tục, ảnh bị nhoè và nhiễu. Nhóm đo thử 5 bộ phát hiện nhỏ gọn phổ biến: "
        "tất cả đều mất 74 đến 83 phần trăm độ chính xác khi rung — trong khi các bộ đánh "
        "giá chuẩn chỉ đo trên ảnh sạch nên không ai thấy điều này."))
    pic = anh(s, os.path.join(HINH, "rung_minh_hoa.png"), LE, TRAN + Inches(0.35),
              Inches(6.3), Inches(4.1))
    chu_thich_duoi(s, pic, "Ảnh sạch (trái) và ảnh khi khung bay rung ở ba mức (phải)")
    tf = txt(s, Inches(7.1), TRAN + Inches(0.25), Inches(5.7), Inches(4.4), size=16.5, space=1.15)
    dong(tf, "UAV ngày càng dùng nhiều trong giám sát, cứu hộ, quay phim — "
             "nhưng người lái vừa lái vừa phải giữ đối tượng trong khung hình.",
         size=16.5, bullet="▸", truoc=0)
    dong(tf, "Cánh quạt làm camera **rung liên tục**: ảnh nhoè, nhiễu, dao động sáng.",
         size=16.5, bullet="▸", truoc=14)
    dong(tf, "Nhóm đo 5 bộ phát hiện nhỏ gọn: tất cả mất **74–83% độ chính xác** khi rung.",
         size=16.5, bullet="▸", truoc=14)
    dong(tf, "Các bộ đánh giá chuẩn **chỉ đo trên ảnh sạch** → không ai thấy lỗ hổng này.",
         size=16.5, bullet="▸", truoc=14)
    hop(s, Inches(7.1), TRAN + Inches(3.85), Inches(5.7), Inches(1.15), fill=NHAT, line=VIEN)
    t = txt(s, Inches(7.35), TRAN + Inches(4.05), Inches(5.2), Inches(0.9), size=15.5,
            bold=True, color=DAM, space=1.1)
    t.paragraphs[0].runs[0].text = "→ Cần một bộ phát hiện chịu được rung, đủ nhẹ để chạy ngay trên máy bay."

    # --------------------------------------------- 3. mục tiêu & hệ thống
    s = trang_moi(prs, "Mục tiêu và tổng quan hệ thống", so=3, ghi_chu=(
        "⏱ 0:40 — Ba mục tiêu theo thuyết minh, chỉ tiêu độ chính xác từ 70% trở lên. "
        "Sơ đồ bên dưới đi theo mũi tên: camera, phát hiện người trên NPU, theo vết OC-SORT, "
        "khoá mục tiêu, ước lượng khoảng cách, ba bộ PID, lớp an toàn, rồi gửi lệnh MAVLink "
        "xuống Pixhawk. Tay điều khiển của người lái luôn được ưu tiên."))
    dai_so_lieu(s, TRAN - Inches(0.08), [
        ("≥ 70%", "Chỉ tiêu AP@0,5 nhận dạng người, thời gian thực"),
        ("Bám tự động", "Giữ khoảng cách và hướng bay phù hợp"),
        ("Trên máy bay", "Camera – xử lý ảnh – điều khiển, không cần Internet"),
    ])
    pic = anh(s, P("sodo_hethong"), LE, TRAN + Inches(1.10), W - 2 * LE, Inches(4.75))
    chu_thich_duoi(s, pic,
                   "Raspberry Pi 5 + NPU Hailo-8L + camera USB, nối Pixhawk 6C (ArduPilot) qua MAVLink 30 Hz")

    # ------------------------------------------------ 4. dữ liệu & giao thức
    s = trang_moi(prs, "Dữ liệu và giao thức đánh giá", so=4, ghi_chu=(
        "⏱ 0:35 — Hai miền dữ liệu: VisDrone bay cao người rất nhỏ, và dữ liệu bay thật ở "
        "6 mét người lớn hơn khoảng 64 lần diện tích. 92 khung bay thật chỉ dùng để đánh giá, "
        "không dùng huấn luyện. Nhóm dựng bộ VisDrone-Shake ba mức rung; nhoè không làm dịch "
        "vật nên nhãn giữ nguyên. Điểm quan trọng về phương pháp: cố định seed, lặp ba lần và "
        "dùng đúng một chương trình đo — không cố định seed thì kết quả nhảy 17,0 / 17,1 / 17,6%."))
    pic = anh(s, os.path.join(HINH, "mau_du_lieu.png"), LE, TRAN + Inches(0.25),
              Inches(6.6), Inches(4.0))
    chu_thich_duoi(s, pic, "Hai miền dữ liệu: VisDrone (bay cao) và dữ liệu bay thật ở ~6 m")
    tf = txt(s, Inches(7.35), TRAN + Inches(0.2), Inches(5.45), Inches(4.5), size=16, space=1.15)
    dong(tf, "**VisDrone-person** — huấn luyện và so sánh chuẩn; người rất nhỏ.",
         size=15.5, bullet="▸", truoc=0)
    dong(tf, "**COCO-person** — bổ sung người cự ly gần, đúng miền triển khai.",
         size=15.5, bullet="▸", truoc=12)
    dong(tf, "**92 khung bay thật** gán nhãn tay — chỉ dùng để đánh giá.",
         size=15.5, bullet="▸", truoc=12)
    dong(tf, "**VisDrone-Shake**: 3 mức rung mô phỏng; nhoè không làm dịch vật nên "
             "nhãn giữ nguyên.", size=15.5, bullet="▸", truoc=12)
    hop(s, Inches(7.35), TRAN + Inches(3.60), Inches(5.45), Inches(1.80), fill=NHAT, line=VIEN)
    t = txt(s, Inches(7.62), TRAN + Inches(3.80), Inches(4.9), Inches(1.45), size=14,
            color=DAM, space=1.2)
    dong(t, "Kỷ luật đo: cố định seed, lặp 3 lần, báo cáo độ lệch chuẩn, dùng chung "
            "một chương trình đo.", size=14, bold=True, truoc=0)
    dong(t, "Không cố định seed → cùng một mô hình ra 17,0 / 17,1 / 17,6%.",
         size=13, color=MO, truoc=6)

    # ------------------------------ 5. giao thức đo độ bền với rung (MỚI)
    s = trang_moi(prs, "Giao thức đánh giá độ bền với rung",
                  phu="VisDrone-Shake — ba mức rung mô phỏng theo mô hình vật lý của khung bay nhiều cánh quạt",
                  so=5, ghi_chu=(
        "⏱ 0:45 — Để đo được độ bền thì trước hết phải có thước đo. Nhóm dựng bộ VisDrone-Shake "
        "gồm ba mức rung: mức 1 ứng với bay treo gió nhẹ, mức 2 là bay tiến thông thường — mức "
        "quan trọng nhất vì đó là lúc máy bay bám người, mức 3 là gió mạnh hoặc cơ động. Nhoè "
        "chuyển động không làm vật dịch chỗ nên nhãn giữ nguyên, không phải gán lại. Hai chỉ số "
        "dùng xuyên suốt: retain là tỷ lệ giữ được độ chính xác khi rung, mCP là AP trung bình "
        "trên ba mức. Và kỷ luật đo: cố định seed, lặp ba lần, báo cáo độ lệch chuẩn."))
    bang(s, LE, TRAN + Inches(0.05), W - 2 * LE, Inches(2.35), [
        ["Mức rung", "Nhoè chuyển động", "Nhiễu cảm biến σ", "Dao động sáng", "Tương ứng tình huống bay"],
        ["Mức 1", "5 px", "0,02", "± 10%", "bay treo, gió nhẹ"],
        ["@Mức 2", "@9 px", "@0,04", "@± 20%", "@**bay tiến thông thường — lúc bám người**"],
        ["Mức 3", "15 px", "0,06", "± 30%", "gió mạnh, cơ động gấp"],
    ], rong_cot=[1.5, 1.9, 1.9, 1.6, 4.6], co_chu=13.5)
    tf = txt(s, LE, TRAN + Inches(2.70), Inches(6.0), Inches(2.3), size=16, space=1.2)
    dong(tf, "**Nhãn giữ nguyên**: nhoè không làm vật dịch chỗ nên không phải gán lại.",
         size=16, bullet="▸", truoc=0)
    dong(tf, "**retain** = AP khi rung / AP ảnh sạch — tỷ lệ giữ được độ chính xác.",
         size=16, bullet="▸", truoc=14)
    dong(tf, "**mCP** = AP trung bình trên cả ba mức rung.", size=16, bullet="▸", truoc=14)
    hop(s, Inches(7.0), TRAN + Inches(2.70), Inches(5.8), Inches(2.2), fill=NHAT, line=VIEN)
    t = txt(s, Inches(7.28), TRAN + Inches(2.92), Inches(5.3), Inches(1.9), size=14.5,
            color=DAM, space=1.2)
    dong(t, "Kỷ luật đo áp cho MỌI mô hình", size=15.5, bold=True, truoc=0)
    dong(t, "Cố định seed · lặp 3 lần · báo cáo ± độ lệch chuẩn · dùng chung một chương trình đo.",
         size=14.5, truoc=8)
    dong(t, "Không cố định seed thì cùng một mô hình ra 17,0 / 17,1 / 17,6% — chênh lệch đủ lớn "
            "để kết luận sai.", size=13.5, color=MO, truoc=8)

    # ------------------------------------------------------- 5. SteadyDet-N
    s = trang_moi(prs, "Mô hình đề xuất: SteadyDet-N", so=6, ghi_chu=(
        "⏱ 0:40 — SteadyDet-N chỉ 1,17 triệu tham số và 4,6 GFLOPs. Điểm thiết kế chính: "
        "giữ lại nhánh độ phân giải cao P2 cho người nhỏ, nhưng làm nhánh này mỏng đi vì chi "
        "phí tính toán tỷ lệ với diện tích bản đồ đặc trưng, rồi dồn tính toán về các mức sâu. "
        "Nhờ vậy rẻ hơn bản SteadyDet-W trước đó 2,8 lần mà vẫn giữ khả năng thấy người nhỏ."))
    dai_so_lieu(s, TRAN - Inches(0.08), [
        ("1,17 M", "tham số — nhỏ nhất trong nhóm so sánh"),
        ("4,6 GFLOPs", "rẻ hơn SteadyDet-W 2,8 lần"),
        ("Nhánh P2", "giữ cho người nhỏ, nhưng làm mỏng"),
        ("LiteHead", "đầu dự đoán tách chiều sâu, dùng chung"),
    ])
    pic = anh(s, P("sodo_kientruc"), Inches(3.3), TRAN + Inches(1.05), Inches(6.7), Inches(4.55))
    chu_thich_duoi(s, pic,
                   "Chi phí phân bổ không đối xứng: nhánh P2 (nét đứt) mỏng, tính toán dồn về các mức sâu")

    # ---------------------------------------- 7. quá trình huấn luyện (MỚI)
    s = trang_moi(prs, "Quá trình huấn luyện", so=7, ghi_chu=(
        "⏱ 0:35 — Mô hình huấn luyện từ đầu, không dùng trọng số có sẵn: 300 epoch trên "
        "VisDrone-person cộng COCO-person, rồi 120 epoch tinh chỉnh có đưa rung vào. Đồ thị bên "
        "trái là hàm mất mát, bên phải là AP kiểm định trên ảnh sạch. Điểm cần chú ý: khi bắt "
        "đầu tinh chỉnh rung, mất mát nhảy lên rồi hạ xuống, còn AP sạch chỉ giảm nhẹ — tức là "
        "mô hình học thêm được độ bền mà gần như không đánh đổi độ chính xác."))
    pic = anh(s, P("huan_luyen"), LE, TRAN + Inches(0.3), Inches(7.6), Inches(4.3))
    chu_thich_duoi(s, pic, "Trái: hàm mất mát · Phải: AP@0,5 kiểm định trên ảnh sạch")
    tf = txt(s, Inches(8.45), TRAN + Inches(0.3), Inches(4.35), Inches(4.6), size=16,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "**Huấn luyện từ đầu**, không dùng trọng số có sẵn — khác với các mô hình YOLO "
             "khởi tạo từ COCO.", size=16, bullet="▸", truoc=0)
    dong(tf, "**300 epoch** trên VisDrone-person + COCO-person.", size=16, bullet="▸", truoc=16)
    dong(tf, "**120 epoch** tinh chỉnh có rung — phần tăng độ bền.", size=16, bullet="▸", truoc=16)
    dong(tf, "AP sạch chỉ giảm nhẹ, tức **đổi rất ít để lấy độ bền**.", size=16, bullet="▸", truoc=16)

    # ------------------------------------------- 6. huấn luyện nhận biết rung
    s = trang_moi(prs, "Phương pháp: huấn luyện nhận biết rung", so=8, ghi_chu=(
        "⏱ 0:35 — Ý tưởng rất đơn giản: trong mỗi lô huấn luyện, thay 50% ảnh bằng ảnh đã "
        "áp mô hình vật lý của rung, nhãn giữ nguyên, hàm mất mát tính bình thường. Mô hình "
        "khi chạy không thay đổi gì nên không tốn thêm một phép tính nào lúc suy luận. "
        "Nhóm cũng thử biến thể ràng buộc nhất quán, phức tạp hơn và tốn 1,8 lần thời gian "
        "huấn luyện nhưng kết quả kém hơn."))
    pic = anh(s, P("sodo_huanluyen_rung"), LE, TRAN + Inches(0.25), W - 2 * LE, Inches(2.5))
    chu_thich_duoi(s, pic, "Một nửa số ảnh mỗi lô đi qua toán tử rung; phần còn lại giữ nguyên")
    tf = txt(s, LE, TRAN + Inches(3.35), Inches(6.0), Inches(2.4), size=16.5, space=1.2)
    dong(tf, "Thay **50% mỗi lô** bằng ảnh rung mô phỏng.", size=16.5, bullet="▸", truoc=0)
    dong(tf, "**Nhãn giữ nguyên** — nhoè không làm dịch vật.", size=16.5, bullet="▸", truoc=16)
    dong(tf, "Hàm mất mát **không đổi**, kiến trúc **không đổi**.", size=16.5, bullet="▸", truoc=16)
    tf2 = txt(s, Inches(7.0), TRAN + Inches(3.35), Inches(5.8), Inches(2.4), size=16.5, space=1.2)
    dong(tf2, "**Không tốn chi phí khi suy luận** — chỉ thay dữ liệu huấn luyện.",
         size=16.5, bullet="▸", truoc=0)
    dong(tf2, "Biến thể ràng buộc nhất quán (đối chứng): tốn **1,8×** thời gian huấn "
              "luyện, kết quả **kém hơn**.", size=16.5, bullet="▸", truoc=16)

    # ======================================================= PHẦN KẾT QUẢ ===
    # ------------------------------- 7. kết quả chính trên VisDrone-person
    s = trang_moi(prs, "Kết quả 1 — Hiệu quả của huấn luyện nhận biết rung",
                  phu="VisDrone-person · retain = tỷ lệ giữ được độ chính xác khi rung · 3 seed, báo cáo ± độ lệch chuẩn",
                  so=9, ghi_chu=(
        "⏱ 1:05 — Đây là bảng kết quả quan trọng nhất, xin Hội đồng đọc từ trên xuống. "
        "Mô hình gốc chỉ giữ được 16,8% độ chính xác khi rung — gần như mù khi bay tiến. "
        "Đối chứng 1: huấn luyện thêm 120 epoch trên ảnh sạch vẫn 16,9%, chứng tỏ cái lợi "
        "không đến từ việc train lâu hơn. Đối chứng 2: ràng buộc nhất quán được 30,0%, tốn "
        "hơn mà kém hơn. Phương pháp đề xuất đạt 36,2%, tức AP ở mức rung bay tiến tăng 4,2 lần, "
        "đổi lại AP trên ảnh sạch chỉ giảm 9,7%. Hai dòng cuối là kiểm chứng: tái lập được "
        "trên SteadyDet-W, còn trên YOLO11-n thì không có tác dụng."))
    bang(s, LE, TRAN - Inches(0.05), W - 2 * LE, Inches(3.5), [
        ["SteadyDet-N", "AP ảnh sạch", "AP mức rung 2", "retain (%)", "Kết luận"],
        ["Gốc", "0,1892", "0,0123", "16,8 ± 0,7", "“mù” khi bay tiến"],
        ["+ 120 epoch ảnh sạch", "0,1823", "0,0125", "16,9 ± 0,7", "huấn luyện thêm vô ích"],
        ["+ ràng buộc nhất quán", "0,1746", "0,0392", "30,0 ± 0,6", "kém hơn, tốn 1,8×"],
        ["@+ huấn luyện nhận biết rung", "@0,1708", "@**0,0518**", "@**36,2 ± 0,7**", "@**tốt nhất**"],
        ["SteadyDet-W: gốc → rung", "0,2349 → 0,2219", "0,0258 → 0,0839", "22,2 → 39,4", "tái lập được"],
        ["YOLO11-n: gốc → rung", "0,2588 → 0,2579", "0,0248 → 0,0227", "21,9 → 21,5", "không tác dụng"],
    ], rong_cot=[3.0, 1.7, 1.7, 1.6, 2.4], co_chu=12.5)
    dai_so_lieu(s, TRAN + Inches(3.75), [
        ("16,8% → 36,2%", "tỷ lệ giữ độ chính xác khi rung"),
        ("× 4,2", "AP ở mức rung của bay tiến"),
        ("−9,7%", "AP trên ảnh sạch — cái giá phải trả"),
        ("0 ms", "chi phí thêm khi chạy trên máy bay"),
    ], cao=Inches(1.12))

    # ------------------------- 10. xếp hạng độ bền của 11 cấu hình (MỚI)
    s = trang_moi(prs, "Xếp hạng độ bền khi rung của 11 cấu hình",
                  phu="retain = tỷ lệ giữ được độ chính xác khi rung · trung bình ± độ lệch chuẩn trên 3 seed",
                  so=10, ghi_chu=(
        "⏱ 0:40 — Toàn cảnh một hình: mười một cấu hình được đo bằng đúng một chương trình đo. "
        "Hai thanh dài nhất là hai mô hình của nhóm sau khi huấn luyện nhận biết rung. Nhóm "
        "giữa là năm mô hình đối chứng, retain chỉ quanh 19 đến 22 phần trăm. Hai thanh ngắn "
        "nhất ở dưới là mô hình gốc của nhóm và bản huấn luyện thêm 120 epoch không rung — "
        "chứng tỏ cái lợi đến từ phương pháp chứ không từ việc huấn luyện lâu hơn."))
    pic = anh(s, P("retain"), LE, TRAN + Inches(0.2), W - 2 * LE, Inches(4.5))
    chu_thich_duoi(s, pic, "Hai thanh đầu là SteadyDet-W và SteadyDet-N sau huấn luyện nhận biết rung")

    # --------------------------------------- 8. so sánh với 5 mô hình nhẹ
    s = trang_moi(prs, "Kết quả 2 — Thứ hạng đảo ngược khi có rung",
                  phu="So sánh với 5 mô hình nhẹ phổ biến, cùng một chương trình đo, cùng dữ liệu",
                  so=11, ghi_chu=(
        "⏱ 1:00 — Trên ảnh sạch, nhóm thua và em xin nói thẳng điều đó: YOLO11-n đạt 0,2768 "
        "còn mô hình của nhóm 0,1708, vì các mô hình YOLO khởi tạo từ COCO còn mô hình nhóm "
        "huấn luyện từ đầu. Nhưng trục đánh giá đúng với bài toán bay là độ chính xác khi rung, "
        "cột mCP. Ở đó SteadyDet-N vượt cả bốn mô hình YOLO bản nano trong khi chi phí tính toán "
        "thấp nhất — 4,6 GFLOPs. Nhìn đồ thị bên trái, từ mức rung 2 trở đi mô hình của nhóm "
        "vượt cả RT-DETR-l, là mô hình lớn gấp 28 lần."))
    pic = anh(s, P("duong_cong_rung"), LE, TRAN + Inches(0.15), Inches(6.6), Inches(4.4))
    chu_thich_duoi(s, pic,
                   "AP theo mức rung — SteadyDet-N vượt lên từ mức 2 (mức rung của bay tiến)")
    bang(s, Inches(7.25), TRAN + Inches(0.2), Inches(5.55), Inches(3.3), [
        ["Mô hình", "GFLOPs", "AP sạch", "mCP"],
        ["RT-DETR-l", "108,0", "0,2502", "0,0641"],
        ["YOLO11-n", "6,4", "0,2768", "0,0557"],
        ["YOLOv10-n", "8,4", "0,2730", "0,0538"],
        ["YOLOv8-n", "8,2", "0,2684", "0,0522"],
        ["YOLO26-n", "5,8", "0,2675", "0,0507"],
        ["@SteadyDet-N + rung", "@**4,6**", "@0,1708", "@**0,0617**"],
    ], rong_cot=[2.6, 1.1, 1.2, 1.2], co_chu=12.5)
    tf = txt(s, Inches(7.25), TRAN + Inches(3.7), Inches(5.55), Inches(1.5), size=15, space=1.15)
    dong(tf, "Ảnh sạch: mô hình của nhóm **thua** — nhóm không né tránh.",
         size=15, bullet="▸", truoc=0)
    dong(tf, "Khi rung: **vượt cả 4 YOLO nano** với chi phí thấp nhất; từ mức 2 vượt "
             "RT-DETR-l (lớn gấp **28×**).", size=15, bullet="▸", truoc=10)

    # ------------------- phát hiện trong cảnh đông người (MỚI)
    s = trang_moi(prs, "Phát hiện người trong cảnh đông",
                  phu="Ảnh VisDrone-val chụp từ UAV · SteadyDet-N nhánh triển khai, ảnh vào 640, ngưỡng tin cậy 0,30",
                  so=12, ghi_chu=(
        "⏱ 0:35 — Ngoài cảnh bay thử chỉ có một người, mô hình còn được đo trên 548 ảnh "
        "VisDrone-val là cảnh đô thị đông người chụp từ UAV. Ba ảnh này có 119, 68 và 55 người, "
        "phần lớn chỉ cao vài chục điểm ảnh, và mô hình bắt được phần lớn trong số đó. Xin lưu ý "
        "đây là ảnh tĩnh của bộ dữ liệu công khai, dùng để so sánh với các mô hình khác; còn bài "
        "toán bám người của đề tài chỉ khoá một người, trong cảnh đông thì bộ chọn mục tiêu lấy "
        "người gần nhất rồi khoá theo mã định danh."))
    anh(s, os.path.join(HINH, "dong_nguoi.png"), LE, TRAN + Inches(0.3),
        W - 2 * LE, Inches(2.95))
    tf = txt(s, LE, TRAN + Inches(3.45), Inches(6.1), Inches(2.1), size=16, space=1.2)
    dong(tf, "Ba ảnh có **119, 68 và 55 người**, phần lớn chỉ cao vài chục điểm ảnh.",
         size=16, bullet="▸", truoc=0)
    dong(tf, "Hộp xanh là phát hiện ở ngưỡng 0,30 — **cùng mô hình đang chạy trên máy bay**.",
         size=16, bullet="▸", truoc=14)
    tf2 = txt(s, Inches(7.0), TRAN + Inches(3.45), Inches(5.8), Inches(2.1), size=16, space=1.2)
    dong(tf2, "Đo trên cả **548 ảnh VisDrone-val**: AP@0,5 = **0,514**.",
         size=16, bullet="▸", truoc=0)
    dong(tf2, "Đề tài chỉ bám **một người**: cảnh đông thì bộ chọn mục tiêu lấy người gần "
              "nhất rồi khoá theo mã định danh.", size=16, bullet="▸", truoc=14)

    # --------------------------------------- 9. kiểm chứng trên bay thật
    s = trang_moi(prs, "Kết quả 3 — Kiểm chứng trên dữ liệu bay thật",
                  phu="92 khung ảnh bay thật ở độ cao ~6 m, gán nhãn tay, chỉ dùng để đánh giá",
                  so=13, ghi_chu=(
        "⏱ 0:45 — Hai kết luận từ dữ liệu bay thật. Thứ nhất, thêm COCO-person làm AP@0,5 "
        "tăng 5,3 lần so với chỉ dùng VisDrone — bài học là chọn dữ liệu theo miền triển khai "
        "chứ không theo bộ benchmark. Thứ hai, huấn luyện nhận biết rung làm hộp bao khít hơn "
        "rõ rệt, AP@0,75 tăng 78 đến 81%, và khoảng tin cậy bootstrap 95% không chứa 0 nên đây "
        "không phải ngẫu nhiên. Em cũng xin nêu hạn chế: tập này mới có một người, một cảnh, một độ cao."))
    anh(s, P("bay_that"), LE, TRAN + Inches(0.15), Inches(7.3), Inches(4.3))
    tf = txt(s, Inches(8.15), TRAN + Inches(0.35), Inches(4.65), Inches(4.3), size=16, space=1.2)
    dong(tf, "Thêm **COCO-person**: AP@0,5 **gấp 5,3 lần** so với chỉ VisDrone.",
         size=16, bullet="▸", truoc=0)
    dong(tf, "→ Chọn dữ liệu theo **miền triển khai**, không theo bộ benchmark.",
         size=16, truoc=4, color=MO)
    dong(tf, "Huấn luyện nhận biết rung: AP@0,75 **+78% / +81%** — hộp bao khít hơn hẳn.",
         size=16, bullet="▸", truoc=14)
    dong(tf, "Khoảng tin cậy bootstrap 95% **không chứa 0**.", size=16, truoc=4, color=MO)
    dong(tf, "Hạn chế đã ghi rõ: 1 người, 1 cảnh, 1 độ cao.", size=16, bullet="▸", truoc=14)

    # ------------------------------ 13. đưa mô hình lên NPU Hailo-8L (MỚI)
    s = trang_moi(prs, "Đưa mô hình lên NPU Hailo-8L",
                  phu="PyTorch → ONNX → trình biên dịch luồng dữ liệu → lượng tử hoá INT8 → tệp HEF chạy trên máy bay",
                  so=14, ghi_chu=(
        "⏱ 0:40 — Mô hình huấn luyện xong chưa chạy được trên NPU. Quy trình bốn bước: xuất "
        "ONNX với các phép tính mà Hailo hỗ trợ, đưa qua trình biên dịch, lượng tử hoá về số "
        "nguyên 8 bit bằng 64 ảnh hiệu chuẩn lấy từ chính miền triển khai, rồi sinh tệp HEF. "
        "Điểm đáng nói nằm ở đồ thị bên trái: tốc độ trên NPU phụ thuộc số lần mô hình phải "
        "nạp lại lên chip, gọi là context. Mô hình của nhóm chỉ cần 4 context trong khi "
        "YOLO26-n cần 6, nên nhanh hơn 1,70 lần dù cùng chạy trên một con chip."))
    pic = anh(s, P("hailo"), LE, TRAN + Inches(0.25), Inches(7.3), Inches(4.3))
    chu_thich_duoi(s, pic, "Trái: ít context hơn nên nhanh hơn · Phải: tăng kích thước lô để đo giới hạn phần cứng")
    tf = txt(s, Inches(8.15), TRAN + Inches(0.3), Inches(4.65), Inches(4.5), size=15.5,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "Xuất **ONNX** chỉ dùng phép tính Hailo hỗ trợ; NMS để ngoài đồ thị.",
         size=15.5, bullet="1.", truoc=0)
    dong(tf, "Trình biên dịch luồng dữ liệu phân tích và chia **context**.",
         size=15.5, bullet="2.", truoc=14)
    dong(tf, "Lượng tử hoá **INT8** bằng **64 ảnh hiệu chuẩn** lấy từ miền triển khai.",
         size=15.5, bullet="3.", truoc=14)
    dong(tf, "Sinh tệp **HEF** rồi chép sang Raspberry Pi 5 trên máy bay.",
         size=15.5, bullet="4.", truoc=14)
    dong(tf, "**4 context** so với 6 của YOLO26-n → **63,66 / 37,46 FPS**.",
         size=15.5, bullet="▸", truoc=16)

    # ------------------------------------ 10. INT8 trên NPU + nhận diện
    s = trang_moi(prs, "Kết quả 4 — Mô hình chạy thật trên máy bay (INT8)",
                  phu="Raspberry Pi 5 + NPU Hailo-8L · đo trên chính phần cứng gắn trên UAV",
                  so=15, ghi_chu=(
        "⏱ 0:50 — Đây là con số trả lời trực tiếp chỉ tiêu của thuyết minh. Mô hình INT8 "
        "đang chạy trên máy bay đạt AP@0,5 bằng 0,917 trên 92 khung bay thật, vượt xa chỉ tiêu "
        "70%. F1 ở ngưỡng vận hành 0,25 là 0,880 với recall 0,837 và precision 0,928. "
        "Lượng tử hoá INT8 gần như không mất gì ở cự ly 3 đến 6 mét là cự ly bám thực tế; "
        "chỉ mất 6,5% với người rất nhỏ. Về tốc độ, NPU chạy 112,55 FPS ở 320 và 63,66 FPS ở 640, "
        "nhanh hơn YOLO26-n 1,70 lần nhờ chỉ cần 4 context thay vì 6."))
    dai_so_lieu(s, TRAN - Inches(0.08), [
        ("0,917", "AP@0,5 trên bay thật (INT8) — chỉ tiêu ≥ 0,70"),
        ("0,880", "F1 ở ngưỡng vận hành 0,25"),
        ("63,66 FPS", "suy luận NPU @640 (112,55 @320)"),
        ("× 1,70", "nhanh hơn YOLO26-n trên cùng NPU"),
    ])
    anh(s, P("int8"), LE, TRAN + Inches(1.15), Inches(7.3), Inches(3.8))
    tf = txt(s, Inches(8.15), TRAN + Inches(1.35), Inches(4.65), Inches(3.6), size=15.5, space=1.2)
    dong(tf, "**Vượt chỉ tiêu 70%** của thuyết minh, đo trên đúng mô hình đang bay.",
         size=15.5, bullet="▸", truoc=0)
    dong(tf, "Tổn thất INT8 phụ thuộc kích thước vật: **−6,5%** với người rất nhỏ, "
             "**≈ 0** ở cự ly 3–6 m.", size=15.5, bullet="▸", truoc=12)
    dong(tf, "recall 0,837 · precision 0,928 — bỏ sót ít, báo nhầm càng ít.",
         size=15.5, bullet="▸", truoc=12)
    dong(tf, "Nhanh hơn nhờ kiến trúc hợp NPU: **4 context** thay vì 6.",
         size=15.5, bullet="▸", truoc=12)

    # ------------------------------- 15. chọn ngưỡng tin cậy vận hành (MỚI)
    s = trang_moi(prs, "Chọn ngưỡng tin cậy để vận hành",
                  phu="Đo trên 92 khung bay thật, bằng chính mô hình INT8 đang chạy trên máy bay",
                  so=16, ghi_chu=(
        "⏱ 0:30 — Ngưỡng tin cậy quyết định hệ thống nhạy hay chắc. Nhóm quét bốn ngưỡng trên "
        "dữ liệu bay thật: 0,25 cho F1 cao nhất là 0,880; lên 0,30 còn 0,823; lên 0,40 tụt "
        "xuống 0,551 vì bỏ sót quá nhiều. Vì vậy cấu hình bay chốt ở 0,25. Đây là ví dụ cho "
        "thấy tham số vận hành được chọn bằng số đo trên đúng miền triển khai, không chọn theo "
        "giá trị mặc định."))
    pic = anh(s, P("nguong_conf"), LE, TRAN + Inches(0.25), Inches(7.0), Inches(4.3))
    tf = txt(s, Inches(7.9), TRAN + Inches(0.3), Inches(4.9), Inches(4.5), size=16,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "Ngưỡng **0,25** → F1 **0,880** (recall 0,837 · precision 0,928).",
         size=16, bullet="▸", truoc=0)
    dong(tf, "Ngưỡng 0,30 → 0,823; ngưỡng 0,40 → **0,551** vì bỏ sót nhiều.",
         size=16, bullet="▸", truoc=16)
    dong(tf, "Cấu hình bay chốt **0,25**, ghi thẳng trong config.yaml.",
         size=16, bullet="▸", truoc=16)
    dong(tf, "Tham số vận hành chọn theo **số đo trên miền triển khai**, không theo mặc định.",
         size=16, bullet="▸", truoc=16)

    # --------------------------------------------- 11. tốc độ & độ trễ
    s = trang_moi(prs, "Tốc độ đầu-cuối: hệ thống bị giới hạn bởi camera", so=17,
                  ghi_chu=(
        "⏱ 0:30 — Chạy đầy đủ trên máy bay được 30 FPS, độ trễ trung vị 32 ms và p99 36 ms. "
        "Điều đáng nói là phần tính toán chỉ chiếm 59% ngân sách thời gian, nút cổ chai là camera "
        "chứ không phải mô hình. Nhóm đã tìm và sửa hai lỗi cấu hình camera, tốc độ tăng từ "
        "11,1 lên 30,0 FPS."))
    pic = anh(s, P("do_tre"), LE, TRAN + Inches(0.2), Inches(8.0), Inches(4.9))
    chu_thich_duoi(s, pic, "Phân rã độ trễ p50 trên máy bay (@640, camera USB thật, 3.594 khung)")
    tf = txt(s, Inches(8.85), TRAN + Inches(0.3), Inches(3.95), Inches(4.6), size=16.5,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "**30,0 FPS** đầu-cuối trên máy bay.", size=16.5, bullet="▸", truoc=0)
    dong(tf, "Độ trễ trung vị **32 ms**, p99 **36 ms**.", size=16.5, bullet="▸", truoc=18)
    dong(tf, "Tính toán chỉ chiếm **59% ngân sách** → giới hạn nằm ở camera.",
         size=16.5, bullet="▸", truoc=18)
    dong(tf, "Sửa 2 lỗi cấu hình camera: **11,1 → 30,0 FPS**.", size=16.5, bullet="▸", truoc=18)

    # ------------------------------------------- 12. điều khiển & an toàn
    s = trang_moi(prs, "Vòng điều khiển bám người và lớp an toàn", so=18, ghi_chu=(
        "⏱ 0:35 — Phần điều khiển em xin nói ngắn. Kết quả phát hiện ở từng khung được nối "
        "thành quỹ đạo bằng OC-SORT, khoá đúng một người và tìm lại được người cũ sau khi mất. "
        "Khoảng cách ước lượng bằng mô hình camera lỗ kim kết hợp bộ lọc Kalman, rồi ba bộ PID "
        "sinh lệnh vận tốc gửi xuống Pixhawk ở 30 Hz. Lớp an toàn có thang FOLLOW – COAST – HOVER, "
        "và chỉ gửi lệnh khi máy bay ở chế độ GUIDED, người lái lấy lại quyền bất cứ lúc nào."))
    pic = anh(s, os.path.join(HINH, "vongbam_khung.png"), LE, TRAN + Inches(0.2),
              Inches(8.0), Inches(4.8))
    chu_thich_duoi(s, pic, "Vòng bám chạy trên 3.961 khung của chuyến bay thử")
    tf = txt(s, Inches(8.85), TRAN + Inches(0.2), Inches(3.95), Inches(4.9), size=15.5,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "Theo vết **OC-SORT** → khoá mục tiêu → tìm lại người cũ sau khi mất.",
         size=15.5, bullet="▸", truoc=0)
    dong(tf, "Khoảng cách: **camera lỗ kim + Kalman**, giới hạn 3–15 m.",
         size=15.5, bullet="▸", truoc=16)
    dong(tf, "**3 bộ PID** → lệnh vận tốc **MAVLink 30 Hz**, khớp nhịp detector.",
         size=15.5, bullet="▸", truoc=16)
    dong(tf, "Thang an toàn **FOLLOW → COAST → HOVER**; chỉ gửi lệnh khi ở GUIDED.",
         size=15.5, bullet="▸", truoc=16)
    dong(tf, "Giữ mục tiêu **83,5%** thời gian, sai số hướng trung vị **5,5°**.",
         size=15.5, bullet="▸", truoc=16)

    # --------------------------- 18. nhịp lệnh 30 Hz và lớp an toàn (MỚI)
    s = trang_moi(prs, "Nhịp lệnh 30 Hz và lớp an toàn",
                  phu="Chọn nhịp bằng số đo, không theo thói quen · người lái luôn giữ quyền chế độ bay",
                  so=19, ghi_chu=(
        "⏱ 0:45 — Nhiều hệ thống gửi lệnh ở 10 Hz theo thói quen. Nhóm đo thật: ở 10 Hz, riêng "
        "việc chờ tới lượt gửi đã cộng trung bình 50 ms vào độ trễ, trong khi đường truyền chịu "
        "được 50 Hz với 0% mất gói. Nhóm chọn 30 Hz vì khớp đúng nhịp của detector — nhanh hơn "
        "nữa thì detector cũng không sinh thêm thông tin. Về an toàn: thang ba mức FOLLOW, "
        "COAST, HOVER; mất người dưới 3 giây thì bám theo dự đoán với nửa vận tốc, sau đó đứng "
        "yên chờ; và lệnh chỉ được gửi khi máy bay đang ở chế độ GUIDED, người lái gạt công tắc "
        "là lấy lại quyền ngay lập tức."))
    pic = anh(s, P("nhip_lenh"), LE, TRAN + Inches(0.2), Inches(6.4), Inches(3.3))
    chu_thich_duoi(s, pic, "Độ trễ từ ánh sáng vào cảm biến tới lúc máy bay nhận lệnh")
    pic2 = anh(s, P("sodo_antoan"), LE, TRAN + Inches(3.95), Inches(6.4), Inches(1.5))
    tf = txt(s, Inches(7.3), TRAN + Inches(0.25), Inches(5.5), Inches(4.9), size=15.5,
             space=1.25, anchor=MSO_ANCHOR.MIDDLE)
    dong(tf, "10 Hz cộng thêm **50 ms** chờ tới lượt gửi mà không đổi lại gì.",
         size=15.5, bullet="▸", truoc=0)
    dong(tf, "Đường truyền đo thật: **0% mất gói tới 50 Hz**, khứ hồi p99 7,73 ms.",
         size=15.5, bullet="▸", truoc=14)
    dong(tf, "Chọn **30 Hz** vì khớp nhịp detector — nhanh hơn cũng không có thêm thông tin.",
         size=15.5, bullet="▸", truoc=14)
    dong(tf, "Mất người **< 3 s**: bám theo dự đoán với nửa vận tốc; sau đó **HOVER** chờ, "
             "không tự bay về.", size=15.5, bullet="▸", truoc=14)
    dong(tf, "Lệnh chỉ gửi khi máy bay ở **GUIDED** — gạt công tắc là người lái lấy lại quyền.",
         size=15.5, bullet="▸", truoc=14)

    # ------------------------------------------- 13. video phần 1: phát hiện
    slide_video(
        prs, "Video 1 — Phát hiện người trên video bay thử",
        "Trích 18 giây liên tục của chuyến bay 21/07/2026, độ cao ≈ 6 m · SteadyDet-N chạy trên từng khung",
        20,
        "⏱ 0:25 — Đây là mô hình chạy trên chính video bay thử: mỗi hộp xanh là một người "
        "được phát hiện, dấu thập là tâm khung hình. Đoạn này cắt liên tục 18 giây, không "
        "ghép nối, không tua.",
        "phan1_phat_hien.mp4", P("poster1"),
        [("Hộp xanh = người được phát hiện  ·  dấu thập = tâm khung hình", 15, CHU),
         ("Bản đầy đủ 4 phút 43 nộp kèm hồ sơ: video/video_bay_thu.mp4", 13, MO)],
        "../video/phan1_phat_hien.mp4")

    # --------------------------------------- 14. video phần 2: vòng bám người
    slide_video(
        prs, "Video 2 — Vòng bám người trên chính đoạn bay đó",
        "Phát hiện → OC-SORT → khoá mục tiêu → lệnh quay hướng · trạng thái FOLLOW / COAST / HOVER",
        21,
        "⏱ 0:25 — Vẫn đúng đoạn bay vừa xem, nhưng chạy qua chương trình bám người. Hộp xanh "
        "lá là mục tiêu đang khoá, hộp xám là người đã phát hiện nhưng chưa khoá, hộp cam là "
        "vị trí dự đoán khi mất phát hiện, vệt vàng là quỹ đạo; bảng góc trái hiển thị trạng "
        "thái và lệnh quay hướng tính được. Đây là đoạn 18 giây mà vòng bám giữ khoá liên tục "
        "nhất, không lần nào rơi về HOVER. NẾU HỘI ĐỒNG HỎI: chuyến bay này bay ở chế độ giữ vị "
        "trí, vòng bám chạy lại trên video của chuyến bay đó nên lệnh được TÍNH chứ chưa gửi "
        "xuống máy bay — đúng như mục 7.3 của báo cáo. Phần cứng nối Pixhawk đã kiểm tra xong "
        "ngày 27/08, bay tự động là bước kế tiếp.",
        "phan2_vong_bam.mp4", P("poster2"),
        [("Xanh lá = đang khoá  ·  xám = đã phát hiện, chưa khoá  ·  vàng = quỹ đạo  ·  không hộp = khung đó không thấy ai", 14, CHU),
         ("Chuyến bay 21/07 ở chế độ giữ vị trí; vòng bám chạy trên video của chuyến bay đó, "
          "lệnh quay hướng được tính ở từng khung. Giữ mục tiêu 83,5% là số đo trên toàn clip 4 phút 43.", 12.5, MO)],
        "../video/phan2_vong_bam.mp4")

    # ----------------------------------- 21. bảy tầng kiểm thử (MỚI)
    s = trang_moi(prs, "Bảy tầng kiểm thử trước khi bay",
                  phu="Mỗi tầng chỉ kiểm thử khi tầng dưới đã đạt, để có lỗi là biết ngay lỗi thuộc tầng nào",
                  so=22, ghi_chu=(
        "⏱ 0:35 — Hệ thống không được mang ra bãi bay ngay. Bảy tầng, tầng dưới đạt mới lên "
        "tầng trên: logic chọn mục tiêu và lớp an toàn kiểm thử tự động 25 trên 25 phép; mô "
        "hình trên NPU; camera cộng suy luận chạy liên tục hai phút; kiểm tra môi trường trước "
        "mỗi lần bay; liên lạc với mạch bay; toàn bộ vòng bám chạy nhưng chặn mọi lệnh; cuối "
        "cùng mới bay thử ngoài bãi. Cách này giúp khi có lỗi thì biết ngay lỗi nằm ở tầng nào."))
    bang(s, LE, TRAN + Inches(0.05), W - 2 * LE, Inches(5.0), [
        ["Tầng", "Nội dung", "Cách kiểm thử", "Kết quả"],
        ["1", "Logic chọn mục tiêu và lớp an toàn", "Kiểm thử tự động, không cần phần cứng", "**25/25 đạt**"],
        ["@2", "@Mô hình trên NPU", "@548 ảnh VisDrone-val và 92 khung bay thật, chạy ngay trên Pi", "@**AP 0,917 · F1 0,880**"],
        ["3", "Camera + suy luận", "Đo liên tục 2 phút với camera thật", "30,0 FPS · p99 36 ms · 48,5 °C"],
        ["4", "Kiểm tra môi trường", "HailoRT, thư viện, HEF, camera, nhiệt, đĩa, đồng hồ NTP", "10 OK / 0 lỗi"],
        ["5", "Liên lạc với máy bay", "Đọc trạng thái, đo khứ hồi lệnh tới 50 Hz", "94–100 gói/s · 0% mất gói"],
        ["@6", "@Toàn bộ vòng bám, không cất cánh", "@Chạy đủ camera, NPU, theo vết, PID, bảng giám sát — chặn mọi lệnh", "@**29,9 Hz · 19,93 ms/khung**"],
        ["7", "Bay thử ngoài bãi", "Bay thử, thu video; chạy vòng bám trên video bay thử", "Giữ mục tiêu 83,5%"],
    ], rong_cot=[0.9, 3.4, 5.4, 2.8], co_chu=12)

    # ------------------------------------------ 14. đạt mục tiêu & kết luận
    s = trang_moi(prs, "Mức độ đạt mục tiêu và kết luận", so=23, ghi_chu=(
        "⏱ 0:40 — Đối chiếu với thuyết minh: mục tiêu độ chính xác đặt 70% thì đạt 91,7%, "
        "vượt chỉ tiêu. Mục tiêu thuật toán bám và mục tiêu tích hợp đều đạt, đã bay thử thực tế. "
        "Bốn kết luận: rung là lỗ hổng thật và đo được; phương pháp đề xuất không tốn chi phí "
        "khi chạy mà nâng tỷ lệ giữ độ chính xác lên gấp đôi; mô hình chạy thật trên NPU vượt "
        "chỉ tiêu; và toàn bộ quy trình tái lập được. Hướng phát triển: mở rộng tập bay thật "
        "nhiều độ cao, nhiều người, và hiệu chuẩn mức rung bằng dữ liệu IMU thật."))
    bang(s, LE, TRAN - Inches(0.05), W - 2 * LE, Inches(2.5), [
        ["Mục tiêu theo thuyết minh", "Chỉ tiêu", "Kết quả đạt được", "Đánh giá"],
        ["Mô hình nhận dạng người thời gian thực", "AP ≥ 70%", "**AP@0,5 = 0,917** (INT8, bay thật)", "**Vượt chỉ tiêu**"],
        ["Thuật toán bám giữ khoảng cách, hướng bay", "Hoạt động ổn định",
         "Giữ mục tiêu 83,5% · sai số hướng 5,5°", "Đạt"],
        ["Tích hợp camera – xử lý ảnh – điều khiển bay", "Chạy trên máy bay",
         "30,0 FPS đầu-cuối, đã bay thử thực tế", "Đạt"],
    ], rong_cot=[3.6, 1.7, 3.3, 1.6], co_chu=13)
    tf = txt(s, LE, TRAN + Inches(2.75), Inches(7.4), Inches(2.6), size=15.5, space=1.2)
    dong(tf, "Rung là lỗ hổng **thật và đo được**: mọi bộ phát hiện nhỏ gọn mất 74–83% AP.",
         size=15.5, bullet="1.", truoc=0)
    dong(tf, "Huấn luyện nhận biết rung: **16,8% → 36,2%**, không tốn chi phí khi chạy.",
         size=15.5, bullet="2.", truoc=10)
    dong(tf, "Chạy thật trên Pi 5 + Hailo-8L: **0,917 AP** và **30 FPS**.",
         size=15.5, bullet="3.", truoc=10)
    dong(tf, "Toàn bộ quy trình **tái lập được**: cố định seed, một chương trình đo.",
         size=15.5, bullet="4.", truoc=10)
    hop(s, Inches(8.3), TRAN + Inches(2.70), Inches(4.5), Inches(2.55), fill=NHAT, line=VIEN)
    tf = txt(s, Inches(8.55), TRAN + Inches(2.88), Inches(4.0), Inches(2.2), size=14.5, space=1.2)
    dong(tf, "Hướng phát triển", size=16, bold=True, color=DAM, truoc=0)
    dong(tf, "Mở rộng tập bay thật: nhiều độ cao, nhiều người, nhiều cảnh.",
         size=14, bullet="▸", truoc=8)
    dong(tf, "Hiệu chuẩn mức rung bằng dữ liệu IMU thật từ nhật ký bay.",
         size=14, bullet="▸", truoc=8)
    dong(tf, "Bám nhiều mục tiêu và tránh vật cản.", size=14, bullet="▸", truoc=8)

    # ------------------------------------------------------- 15. cảm ơn
    s = prs.slides.add_slide(prs.slide_layouts[6])
    hop(s, 0, 0, W, H, fill=DAM, radius=False)
    t = txt(s, LE, Inches(2.55), W - 2 * LE, Inches(1.0), size=40, bold=True,
            color=TRANG, align=PP_ALIGN.CENTER)
    t.paragraphs[0].runs[0].text = "CẢM ƠN HỘI ĐỒNG ĐÃ LẮNG NGHE"
    hop(s, Inches(5.6), Inches(3.75), Inches(2.1), Inches(0.06), fill=CAM, radius=False)
    t2 = txt(s, LE, Inches(4.1), W - 2 * LE, Inches(1.2), size=18,
             color=RGBColor(0xBF, 0xD6, 0xF2), align=PP_ALIGN.CENTER, space=1.4)
    dong(t2, "Em xin sẵn sàng trả lời câu hỏi của Hội đồng.", size=18,
         color=RGBColor(0xBF, 0xD6, 0xF2), truoc=0, align=PP_ALIGN.CENTER)
    dong(t2, tt("madetai") + "  ·  " + tt("chunhiem") + "  ·  GVHD: " + tt("gvhd"),
         size=15, color=RGBColor(0x9A, 0xBB, 0xE0), truoc=10, align=PP_ALIGN.CENTER)
    s.notes_slide.notes_text_frame.text = (
        "⏱ 0:10 — Em xin hết phần trình bày, cảm ơn Hội đồng.\n\n"
        "CÂU HỎI THƯỜNG GẶP (xem thêm kich_ban_thuyet_trinh.md):\n"
        "• Vì sao AP sạch thấp hơn YOLO? → YOLO khởi tạo từ COCO, mô hình nhóm train từ đầu; "
        "trục cạnh tranh của đề tài là độ chính xác khi rung (mCP), ở đó nhóm dẫn đầu nhóm nano.\n"
        "• So sánh có công bằng không? → Đã áp đúng phương pháp lên YOLO11-n, nó không tăng "
        "(20,1% → 20,1%), nên mỗi mô hình được so ở trạng thái tốt nhất của nó.\n"
        "• Rung mô phỏng có giống rung thật? → Chưa hiệu chuẩn bằng IMU, nhưng kiểm chứng trên "
        "92 khung bay thật cho cùng chiều hướng (hộp khít hơn ~80%).\n"
        "• Chỉ tiêu 70% đo thế nào? → AP@0,5 chuẩn COCO của đúng mô hình INT8 đang bay, "
        "trên 92 khung gán nhãn tay: 0,917.\n"
        "• Mất người thì sao? → Dưới 3 s bám theo dự đoán với nửa vận tốc, sau đó HOVER chờ; "
        "không tự RTL; người lái lấy lại quyền bằng công tắc.\n"
        "• Vì sao Hailo-8L thay vì Jetson? → Rẻ, ít điện, gắn thẳng lên Pi 5; đo thật 63,66 FPS @640.")


def main():
    tmp = tempfile.mkdtemp(prefix="slide_nghiemthu_")
    try:
        P = chuan_bi_anh(tmp)
        prs = Presentation()
        prs.slide_width, prs.slide_height = W, H
        dung_slide(prs, P, doc_thongtin())
        duong = OUT_NHE if NHE else OUT
        prs.save(duong)
        print("Đã lưu:", duong)
        print("Số slide:", len(prs.slides.__iter__.__self__._sldIdLst))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
