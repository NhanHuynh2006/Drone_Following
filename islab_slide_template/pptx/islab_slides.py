#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
islab_slides.py — khung slide PowerPoint chung của ISLab (Intelligent Systems Laboratory).

Cùng bộ màu, cùng bố cục với bản LaTeX (../latex/beamerthemeISLab.sty). Hai ảnh nền
trang bìa và trang chuyển mục được vẽ từ chính mã TikZ của bản LaTeX (xuat_nen.sh)
nên hai bản trùng hoạ tiết.

Khung được đặt vào SLIDE MASTER và các LAYOUT, không phải vẽ lại trên từng slide:
mở file bằng PowerPoint, bấm "New Slide" là có sẵn thanh tiêu đề, logo, chân trang.

Dùng trong mã:

    from islab_slides import *
    prs = tao_bai(ten_ngan="Báo cáo tiến độ tháng 11")
    trang_bia(prs, "Tên báo cáo", "Loại báo cáo", "Người trình bày",
              "ISLab — Intelligent Systems Laboratory", "Tháng 11 năm 2026")
    s = trang(prs, "Tiêu đề trang", phu="Dòng phụ", kieu="noi_dung")
    gach(s, ["Ý thứ nhất", "Ý có **chỗ nhấn**", (1, "Ý cấp hai")])
    prs.save("bao_cao.pptx")

Cần: python-pptx >= 1.0, Pillow.
"""
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml import parse_xml
from pptx.oxml.ns import nsdecls, qn
from pptx.shapes.shapetree import SlideShapes
from pptx.util import Emu, Inches, Pt

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(os.path.dirname(HERE), "assets")
LOGO = os.path.join(ASSETS, "logo_islab.png")

# ---- bảng màu: lấy thẳng từ logo (trùng với beamerthemeISLab.sty) -----------
_MAU = dict(DAM="244E7C", IS="30649E", LAB="6B95C5", CAM="C86A2C", NHAT="EFF3F9",
            VIEN="C6D6E9", MO="5A6472", PHU="BFD6F2", CHU="1A1A1A", TRANG="FFFFFF")
DAM, IS, LAB, CAM, NHAT, VIEN, MO, PHU, CHU, TRANG = (
    RGBColor.from_string(_MAU[k]) for k in
    ("DAM", "IS", "LAB", "CAM", "NHAT", "VIEN", "MO", "PHU", "CHU", "TRANG"))

FONT = "Arial"
W, H = Inches(13.333), Inches(7.5)
LE = Inches(0.55)            # lề trái/phải
THANH = Inches(1.02)         # chiều cao thanh tiêu đề
TRAN = Inches(1.35)          # mép trên vùng nội dung
DAY = Inches(6.85)           # mép dưới vùng nội dung
RONG = W - 2 * LE            # bề rộng vùng nội dung

# Layout theo thứ tự trong file
L_BIA, L_NOI_DUNG, L_MUC, L_HAI_COT, L_KET, L_TIEU_DE, L_TRONG = range(7)


def cm(x):
    """Đổi cm trên trang LaTeX 16 x 9 cm sang đơn vị của trang PowerPoint 13,333 x 7,5 inch."""
    return Emu(int(x * 13.333 / 16 * 914400))


# ================================================================ dựng khung ==
def _kieu_ph(ph, co, dam=False, mau="1A1A1A", canh="l", neo="t", goi_y=None):
    """Đặt kiểu chữ cho một placeholder của layout (không đầu dòng, không lề trong)."""
    tx = ph._element.txBody
    bp = tx.find(qn("a:bodyPr"))
    for c in list(bp):
        bp.remove(c)
    for k in ("lIns", "tIns", "rIns", "bIns"):
        bp.set(k, "0")
    bp.set("anchor", neo)
    bp.set("wrap", "square")
    bp.append(parse_xml("<a:normAutofit %s/>" % nsdecls("a")))
    ls = tx.find(qn("a:lstStyle"))
    if ls is None:
        ls = parse_xml("<a:lstStyle %s/>" % nsdecls("a"))
        bp.addnext(ls)
    for c in list(ls):
        ls.remove(c)
    ls.append(parse_xml(
        '<a:lvl1pPr %s marL="0" indent="0" algn="%s"><a:lnSpc><a:spcPct val="100000"/></a:lnSpc>'
        '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:buNone/>'
        '<a:defRPr sz="%d" b="%d"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
        '<a:latin typeface="%s"/></a:defRPr></a:lvl1pPr>'
        % (nsdecls("a"), canh, int(co * 100), int(dam), mau, FONT)))
    if goi_y is not None:
        ph.text_frame.text = goi_y


def _dat(ph, x, y, w, h):
    ph.left, ph.top, ph.width, ph.height = int(x), int(y), int(w), int(h)


def _them_ph(layout, idx, ten, x, y, w, h):
    """Thêm một placeholder chữ mới vào layout."""
    tree = layout._element.cSld.spTree
    sid = max(int(e.get("id")) for e in tree.iter(qn("p:cNvPr"))) + 1
    sp = parse_xml(
        '<p:sp %s><p:nvSpPr><p:cNvPr id="%d" name="%s"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
        '<p:nvPr><p:ph type="body" sz="quarter" idx="%d"/></p:nvPr></p:nvSpPr>'
        '<p:spPr><a:xfrm><a:off x="%d" y="%d"/><a:ext cx="%d" cy="%d"/></a:xfrm></p:spPr>'
        '<p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:endParaRPr lang="vi-VN"/></a:p></p:txBody></p:sp>'
        % (nsdecls("p", "a"), sid, ten, idx, int(x), int(y), int(w), int(h)))
    tree.append(sp)
    return [p for p in layout.placeholders if p.placeholder_format.idx == idx][0]


def _ph(doi_tuong, idx):
    for p in doi_tuong.placeholders:
        if p.placeholder_format.idx == idx:
            return p
    return None


def _xoa_ph(doi_tuong, idxs):
    for p in list(doi_tuong.placeholders):
        if p.placeholder_format.idx in idxs:
            p._element.getparent().remove(p._element)


def _nen_anh(layout, duong):
    """Ảnh nền tràn trang cho một layout, ẩn phần khung của master."""
    layout._element.set("showMasterSp", "0")
    tree = layout._element.cSld.spTree
    pic = SlideShapes(tree, layout).add_picture(duong, 0, 0, W, H)
    pic.name = "ISLab nền"
    tree.remove(pic._element)
    tree.insert(2, pic._element)          # ngay sau nvGrpSpPr, grpSpPr: nằm dưới mọi thứ


def _khong_bong(sp):
    """Tắt bóng đổ ở cả hai chỗ: effectLst rỗng và effectRef của style (LibreOffice chỉ
    nghe chỗ thứ hai)."""
    sp.shadow.inherit = False
    st = sp._element.find(qn("p:style"))
    if st is not None:
        st.find(qn("a:effectRef")).set("idx", "0")


def _hinh(shapes, kieu, x, y, w, h, mau, ten):
    sp = shapes.add_shape(kieu, int(x), int(y), int(w), int(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = mau
    sp.line.fill.background()
    _khong_bong(sp)
    sp.name = ten
    return sp


def _chu_tinh(shapes, x, y, w, h, co, mau, canh, ten):
    tb = shapes.add_textbox(int(x), int(y), int(w), int(h))
    tb.name = ten
    tf = tb.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = canh
    return tb, p


def _theme(master):
    """Đưa màu và phông của phòng thí nghiệm vào theme: bảng chọn màu của PowerPoint
    và mọi hình/biểu đồ vẽ mới sẽ mặc định theo màu ISLab."""
    part = master.part.part_related_by(RT.THEME)
    s = part.blob.decode("utf-8")
    bang = dict(dk2=_MAU["DAM"], lt2=_MAU["NHAT"], accent1=_MAU["IS"], accent2=_MAU["CAM"],
                accent3=_MAU["LAB"], accent4=_MAU["DAM"], accent5=_MAU["MO"],
                accent6=_MAU["VIEN"], hlink=_MAU["IS"], folHlink=_MAU["LAB"])
    for ten, ma in bang.items():
        s = re.sub(r"(<a:%s>\s*<a:srgbClr val=\")[0-9A-Fa-f]{6}" % ten, r"\g<1>" + ma, s)
    s = re.sub(r'(<a:(?:majorFont|minorFont)>\s*<a:latin typeface=")[^"]*', r"\g<1>" + FONT, s)
    s = re.sub(r'(<a:clrScheme name=")[^"]*', r"\g<1>ISLab", s)
    s = re.sub(r'(<a:fontScheme name=")[^"]*', r"\g<1>ISLab", s)
    part._blob = s.encode("utf-8")


def _master(prs):
    m = prs.slide_master
    _theme(m)
    for p in list(m.placeholders):                       # bỏ ngày, chân trang, số trang mặc định
        if p._element.find(".//" + qn("p:ph")).get("type") in ("dt", "ftr", "sldNum"):
            p._element.getparent().remove(p._element)

    # ---- kiểu chữ mặc định của tiêu đề và thân bài ---------------------------
    tx = m._element.find(qn("p:txStyles"))
    for c in list(tx):
        tx.remove(c)
    tx.append(parse_xml(
        '<p:titleStyle %s><a:lvl1pPr algn="l"><a:lnSpc><a:spcPct val="95000"/></a:lnSpc>'
        '<a:spcBef><a:spcPts val="0"/></a:spcBef><a:buNone/>'
        '<a:defRPr sz="2700" b="1" kern="1200"><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
        '<a:latin typeface="%s"/></a:defRPr></a:lvl1pPr></p:titleStyle>' % (nsdecls("p", "a"), FONT)))
    cap = []
    for i, (le, co, truoc) in enumerate(
            [(285750, 2000, 1000), (628650, 1700, 500), (914400, 1500, 400),
             (1200150, 1400, 300), (1485900, 1400, 300)], start=1):
        cap.append(
            '<a:lvl%dpPr marL="%d" indent="-%d" algn="l"><a:lnSpc><a:spcPct val="105000"/></a:lnSpc>'
            '<a:spcBef><a:spcPts val="%d"/></a:spcBef><a:buClr><a:srgbClr val="%s"/></a:buClr>'
            '<a:buSzPct val="80000"/><a:buFont typeface="%s"/><a:buChar char="&#9642;"/>'
            '<a:defRPr sz="%d" kern="1200"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
            '<a:latin typeface="%s"/></a:defRPr></a:lvl%dpPr>'
            % (i, le, 285750 if i == 1 else 228600, truoc, _MAU["LAB"], FONT, co, _MAU["CHU"], FONT, i))
    tx.append(parse_xml("<p:bodyStyle %s>%s</p:bodyStyle>" % (nsdecls("p", "a"), "".join(cap))))
    tx.append(parse_xml(
        '<p:otherStyle %s><a:lvl1pPr><a:defRPr sz="1600"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
        '<a:latin typeface="%s"/></a:defRPr></a:lvl1pPr></p:otherStyle>'
        % (nsdecls("p", "a"), _MAU["CHU"], FONT)))

    # ---- hình tĩnh: thanh tiêu đề, dải màu, logo, chân trang -----------------
    tree = m._element.cSld.spTree
    truoc = set(tree)
    sh = SlideShapes(tree, m)
    _hinh(sh, MSO_SHAPE.RECTANGLE, 0, 0, W, THANH, DAM, "ISLab thanh tiêu đề")
    _hinh(sh, MSO_SHAPE.RECTANGLE, 0, THANH, W, Inches(0.05), LAB, "ISLab dải màu")
    _hinh(sh, MSO_SHAPE.RECTANGLE, 0, THANH, Inches(1.58), Inches(0.05), CAM, "ISLab dải nhấn")
    lg_w = Inches(1.28)
    lg_h = Emu(int(lg_w * 351 / 805))
    chip_w, chip_h = lg_w + Inches(0.26), lg_h + Inches(0.20)
    chip_x, chip_y = W - Inches(0.34) - chip_w, (THANH - chip_h) // 2
    chip = _hinh(sh, MSO_SHAPE.ROUNDED_RECTANGLE, chip_x, chip_y, chip_w, chip_h, TRANG, "ISLab nền logo")
    chip.adjustments[0] = 0.10
    sh.add_picture(LOGO, chip_x + Inches(0.13), chip_y + Inches(0.10), width=lg_w).name = "ISLab logo"
    _hinh(sh, MSO_SHAPE.RECTANGLE, LE, Inches(7.04), RONG, Emu(6350), VIEN, "ISLab kẻ chân trang")
    tb, p = _chu_tinh(sh, LE, Inches(7.10), Inches(9.5), Inches(0.26), 10, MO, PP_ALIGN.LEFT,
                      "ISLab chân trang")
    for chu, dam in (("ISLab", True), ("  ·  Intelligent Systems Laboratory", False)):
        r = p.add_run()
        r.text = chu
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(10), dam, MO, FONT
    tb, p = _chu_tinh(sh, W - LE - Inches(1.0), Inches(7.10), Inches(1.0), Inches(0.26), 11, MO,
                      PP_ALIGN.RIGHT, "ISLab số trang")
    p._p.append(parse_xml(
        '<a:fld %s id="{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}" type="slidenum">'
        '<a:rPr lang="vi-VN" sz="1100"><a:solidFill><a:srgbClr val="%s"/></a:solidFill>'
        '<a:latin typeface="%s"/></a:rPr><a:t>‹#›</a:t></a:fld>' % (nsdecls("a"), _MAU["MO"], FONT)))
    for i, e in enumerate([e for e in tree if e not in truoc]):     # đưa xuống dưới các placeholder
        tree.remove(e)
        tree.insert(2 + i, e)

    # ---- placeholder gốc ------------------------------------------------------
    for p in m.placeholders:
        loai = p._element.find(".//" + qn("p:ph")).get("type")
        bp = p._element.txBody.find(qn("a:bodyPr"))
        for k in ("lIns", "tIns", "rIns", "bIns"):
            bp.set(k, "0")
        if loai == "title":
            _dat(p, LE, Inches(0.10), chip_x - LE - Inches(0.2), THANH - Inches(0.20))
            bp.set("anchor", "ctr")
        elif loai == "body":
            _dat(p, LE, TRAN, RONG, DAY - TRAN)
            bp.set("anchor", "t")
    return chip_x


def _layouts(prs, chip_x):
    ls = prs.slide_layouts
    for lo in list(ls)[7:]:
        ls.remove(lo)
    for lo in ls:
        for p in list(lo.placeholders):
            if p.placeholder_format.idx in (10, 11, 12):
                p._element.getparent().remove(p._element)
    rong_td = chip_x - LE - Inches(0.2)

    def dau_trang(lo):
        """Tiêu đề + dòng phụ nằm trong thanh tiêu đề."""
        _dat(_ph(lo, 0), LE, Inches(0.10), rong_td, Inches(0.54))
        _kieu_ph(_ph(lo, 0), 27, True, "FFFFFF", neo="b", goi_y="Tiêu đề trang")
        phu = _them_ph(lo, 13, "Dòng phụ", LE, Inches(0.66), rong_td, Inches(0.28))
        _kieu_ph(phu, 13, False, _MAU["PHU"], neo="t", goi_y="Dòng phụ: phạm vi, dữ liệu, điều kiện đo")

    # 0 — Bìa
    lo = ls[L_BIA]
    lo._element.cSld.set("name", "ISLab - Bìa")
    _nen_anh(lo, os.path.join(ASSETS, "nen_bia.png"))
    _dat(_ph(lo, 0), cm(0.66), cm(1.95), cm(8.5), cm(2.85))
    _kieu_ph(_ph(lo, 0), 34, True, "FFFFFF", neo="b", goi_y="Tên đề tài hoặc tên báo cáo")
    _dat(_ph(lo, 1), cm(0.66), cm(5.45), cm(8.5), cm(1.1))
    _kieu_ph(_ph(lo, 1), 18, False, _MAU["PHU"], goi_y="Loại báo cáo")
    tg = _them_ph(lo, 13, "Người trình bày", cm(0.66), cm(6.7), cm(8.5), cm(1.55))
    _kieu_ph(tg, 15, False, "FFFFFF", neo="b", goi_y="Người trình bày · Đơn vị · Thời gian")
    _hinh(SlideShapes(lo._element.cSld.spTree, lo), MSO_SHAPE.RECTANGLE,
          cm(0.66), cm(5.08), cm(1.6), cm(0.08), CAM, "ISLab gạch nhấn")

    # 1 — Nội dung
    lo = ls[L_NOI_DUNG]
    lo._element.cSld.set("name", "ISLab - Nội dung")
    dau_trang(lo)
    _dat(_ph(lo, 1), LE, TRAN, RONG, DAY - TRAN)

    # 2 — Chuyển mục
    lo = ls[L_MUC]
    lo._element.cSld.set("name", "ISLab - Chuyển mục")
    _nen_anh(lo, os.path.join(ASSETS, "nen_muc.png"))
    _dat(_ph(lo, 0), cm(6.1), cm(1.85), cm(8.9), cm(2.5))
    _kieu_ph(_ph(lo, 0), 34, True, _MAU["IS"], neo="b", goi_y="Tên phần")
    _dat(_ph(lo, 1), cm(6.1), cm(4.7), cm(8.9), cm(1.8))
    _kieu_ph(_ph(lo, 1), 18, False, _MAU["MO"], goi_y="Mô tả ngắn của phần này")
    so = _them_ph(lo, 13, "Số thứ tự phần", 0, cm(2.9), cm(5.2), cm(2.2))
    _kieu_ph(so, 88, True, "FFFFFF", canh="ctr", neo="ctr", goi_y="01")

    # 3 — Hai cột
    lo = ls[L_HAI_COT]
    lo._element.cSld.set("name", "ISLab - Hai cột")
    dau_trang(lo)
    khe = Inches(0.4)
    cot = (RONG - khe) // 2
    _dat(_ph(lo, 1), LE, TRAN, cot, DAY - TRAN)
    _dat(_ph(lo, 2), LE + cot + khe, TRAN, cot, DAY - TRAN)

    # 4 — Kết thúc
    lo = ls[L_KET]
    lo._element.cSld.set("name", "ISLab - Kết thúc")
    _xoa_ph(lo, (3, 4))
    _nen_anh(lo, os.path.join(ASSETS, "nen_bia.png"))
    _dat(_ph(lo, 0), cm(0.66), cm(1.95), cm(8.5), cm(2.85))
    _kieu_ph(_ph(lo, 0), 34, True, "FFFFFF", neo="b", goi_y="Cảm ơn quý thầy cô và các bạn đã lắng nghe")
    _dat(_ph(lo, 1), cm(0.66), cm(5.45), cm(8.5), cm(1.1))
    _kieu_ph(_ph(lo, 1), 18, False, _MAU["PHU"], goi_y="Hỏi và đáp")
    _dat(_ph(lo, 2), cm(0.66), cm(6.7), cm(8.5), cm(1.55))
    _kieu_ph(_ph(lo, 2), 15, False, _MAU["PHU"], neo="b", goi_y="Thông tin liên hệ")
    _hinh(SlideShapes(lo._element.cSld.spTree, lo), MSO_SHAPE.RECTANGLE,
          cm(0.66), cm(5.08), cm(1.6), cm(0.08), CAM, "ISLab gạch nhấn")

    # 5 — Chỉ tiêu đề (hình, bảng, sơ đồ tự đặt)
    lo = ls[L_TIEU_DE]
    lo._element.cSld.set("name", "ISLab - Chỉ tiêu đề")
    dau_trang(lo)

    # 6 — Trống (vẫn có thanh tiêu đề, logo, chân trang)
    ls[L_TRONG]._element.cSld.set("name", "ISLab - Trống")


def tao_bai(ten_ngan=None):
    """Tạo một bài trình chiếu mới đã có sẵn khung ISLab. `ten_ngan` in ở chân trang."""
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    chip_x = _master(prs)
    _layouts(prs, chip_x)
    if ten_ngan:
        for sp in prs.slide_master.shapes:
            if sp.name == "ISLab chân trang":
                r = sp.text_frame.paragraphs[0].add_run()
                r.text = "   |   " + ten_ngan
                r.font.size, r.font.color.rgb, r.font.name = Pt(10), MO, FONT
    return prs


# ============================================================ các loại trang ==
def _dien(ph, chu):
    """Điền chữ nhiều dòng vào placeholder, giữ kiểu của layout."""
    dong_chu = chu.split("\n") if isinstance(chu, str) else list(chu)
    tf = ph.text_frame
    tf.text = dong_chu[0]
    for d in dong_chu[1:]:
        tf.add_paragraph().text = d
    return tf


def _ghi_chu(slide, ghi_chu):
    if ghi_chu:
        slide.notes_slide.notes_text_frame.text = ghi_chu


def trang_bia(prs, tieu_de, phu="", tac_gia="", don_vi="", ngay="", ghi_chu=None):
    s = prs.slides.add_slide(prs.slide_layouts[L_BIA])
    _dien(s.shapes.title, tieu_de)
    _dien(_ph(s, 1), phu)
    tf = _ph(s, 13).text_frame
    tf.text = tac_gia
    if tf.paragraphs[0].runs:
        tf.paragraphs[0].runs[0].font.bold = True
        tf.paragraphs[0].runs[0].font.size = Pt(17)
    for d in (don_vi, ngay):
        if d:
            p = tf.add_paragraph()
            p.space_before = Pt(3)
            r = p.add_run()
            r.text = d
            r.font.color.rgb = PHU
    _ghi_chu(s, ghi_chu)
    return s


def trang_muc(prs, so, ten, mo_ta="", ghi_chu=None):
    s = prs.slides.add_slide(prs.slide_layouts[L_MUC])
    _dien(s.shapes.title, ten)
    _dien(_ph(s, 1), mo_ta)
    _dien(_ph(s, 13), so)
    _ghi_chu(s, ghi_chu)
    return s


def trang_ket(prs, loi="Cảm ơn quý thầy cô và các bạn đã lắng nghe", phu="Hỏi và đáp",
              lien_he="", ghi_chu=None):
    s = prs.slides.add_slide(prs.slide_layouts[L_KET])
    _dien(s.shapes.title, loi)
    _dien(_ph(s, 1), phu)
    _dien(_ph(s, 2), lien_he)
    _ghi_chu(s, ghi_chu)
    return s


def trang(prs, tieu_de, phu=None, kieu="tieu_de", ghi_chu=None):
    """Trang nội dung. kieu: "tieu_de" (tự đặt hình/bảng), "noi_dung" (một khung gạch
    đầu dòng), "hai_cot" (hai khung)."""
    lo = {"tieu_de": L_TIEU_DE, "noi_dung": L_NOI_DUNG, "hai_cot": L_HAI_COT}[kieu]
    s = prs.slides.add_slide(prs.slide_layouts[lo])
    s.shapes.title.text_frame.text = tieu_de
    if phu:
        _ph(s, 13).text_frame.text = phu
    else:                                   # không có dòng phụ: tiêu đề canh giữa thanh
        p13 = _ph(s, 13)
        p13._element.getparent().remove(p13._element)
        t = s.shapes.title
        t.top, t.height = Inches(0.10), THANH - Inches(0.20)
        t.left, t.width = LE, _ph(prs.slide_layouts[lo], 0).width
        t.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    _ghi_chu(s, ghi_chu)
    return s


# ================================================================ nội dung ====
def _chay(p, text, co=None, dam=None, mau=None):
    """Thêm chữ vào đoạn; **...** thành chữ đậm màu nhấn."""
    for phan in re.split(r"(\*\*.+?\*\*)", text):
        if not phan:
            continue
        nhan = phan.startswith("**")
        r = p.add_run()
        r.text = phan.strip("*") if nhan else phan
        r.font.name = FONT
        if co:
            r.font.size = Pt(co)
        if dam or nhan:
            r.font.bold = True
        if nhan:
            r.font.color.rgb = CAM
        elif mau is not None:
            r.font.color.rgb = mau


def gach(slide, muc, idx=1, co=None):
    """Điền gạch đầu dòng vào khung nội dung. Mỗi mục là chuỗi, hoặc (cấp, chuỗi)."""
    tf = _ph(slide, idx).text_frame
    dau = True
    for m in muc:
        cap, chu = m if isinstance(m, tuple) else (0, m)
        p = tf.paragraphs[0] if dau else tf.add_paragraph()
        dau = False
        p.level = cap
        _chay(p, chu, co=co)
    return tf


def hop(slide, x, y, w, h, fill=NHAT, line=None, bo=True):
    sp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if bo else MSO_SHAPE.RECTANGLE,
                                int(x), int(y), int(w), int(h))
    sp.fill.solid()
    sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line
        sp.line.width = Pt(0.75)
    _khong_bong(sp)
    if bo:
        sp.adjustments[0] = 0.06
    return sp


def chu(slide, x, y, w, h, text="", co=16, dam=False, mau=CHU, canh=PP_ALIGN.LEFT,
        neo=MSO_ANCHOR.TOP, gian=1.0):
    """Hộp chữ tự do; trả về text_frame."""
    tb = slide.shapes.add_textbox(int(x), int(y), int(w), int(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = neo
    p = tf.paragraphs[0]
    p.alignment, p.line_spacing = canh, gian
    if text:
        _chay(p, text, co=co, dam=dam, mau=mau)
    return tf


def them_dong(tf, text, co=16, dam=False, mau=CHU, truoc=4, canh=PP_ALIGN.LEFT, dau=None):
    p = tf.add_paragraph()
    p.alignment = canh
    p.space_before = Pt(truoc)
    if dau:
        r = p.add_run()
        r.text = dau + "  "
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(co), True, FONT, LAB
    _chay(p, text, co=co, dam=dam, mau=mau)
    return p


def the(slide, x, y, w, h, so, tieu_de, noi_dung="", co=14):
    """Thẻ có số thứ tự: vạch màu bên trái, số màu nhấn, tiêu đề xanh, nội dung."""
    hop(slide, x, y, w, h, fill=NHAT, bo=False)
    hop(slide, x, y, Inches(0.06), h, fill=IS, bo=False)
    tf = chu(slide, x + Inches(0.24), y, w - Inches(0.40), h, neo=MSO_ANCHOR.MIDDLE)
    p = tf.paragraphs[0]
    _chay(p, so + "   ", co=co + 3, dam=True, mau=CAM)
    _chay(p, tieu_de, co=co + 2, dam=True, mau=IS)
    if noi_dung:
        them_dong(tf, noi_dung, co=co, truoc=3)
    return tf


def dai_so_lieu(slide, y, muc, cao=Inches(1.05)):
    """Dải ô số liệu ngang: [(số, nhãn), ...]."""
    n = len(muc)
    khe = Inches(0.16)
    rong = (RONG - khe * (n - 1)) // n
    for i, (so, nhan) in enumerate(muc):
        x = LE + i * (rong + khe)
        hop(slide, x, y, rong, cao, fill=NHAT, line=VIEN)
        chu(slide, x, y + Inches(0.12), rong, Inches(0.46), so, co=25, dam=True, mau=CAM,
            canh=PP_ALIGN.CENTER)
        chu(slide, x + Inches(0.08), y + Inches(0.62), rong - Inches(0.16), Inches(0.36), nhan,
            co=11.5, mau=MO, canh=PP_ALIGN.CENTER)


def khoi(slide, x, y, w, h, tieu_de, dong_chu, mau=DAM, co=14):
    """Khối có thanh tên: dùng cho định nghĩa, cấu hình, danh sách thiết bị."""
    dau = Inches(0.44)
    hop(slide, x, y, w, h, fill=NHAT, bo=False)
    t = hop(slide, x, y, w, dau, fill=mau, bo=False)
    tf = t.text_frame
    tf.margin_left = Inches(0.16)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.LEFT
    _chay(tf.paragraphs[0], tieu_de, co=co + 1, dam=True, mau=TRANG)
    tf2 = chu(slide, x + Inches(0.18), y + dau + Inches(0.10), w - Inches(0.36),
              h - dau - Inches(0.16))
    for i, d in enumerate([dong_chu] if isinstance(dong_chu, str) else dong_chu):
        p = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        p.space_before = Pt(0 if i == 0 else 6)
        if not isinstance(dong_chu, str):
            r = p.add_run()
            r.text = "▪  "
            r.font.size, r.font.name, r.font.color.rgb = Pt(co), FONT, LAB
        _chay(p, d, co=co)
    return tf2


def bang(slide, x, y, w, h, du_lieu, rong_cot=None, co=13, cao_dau=Inches(0.44)):
    """Bảng: hàng đầu nền xanh đậm; ô bắt đầu bằng "@" được tô nền nhạt; **...** là chỗ nhấn."""
    hang, cot = len(du_lieu), len(du_lieu[0])
    g = slide.shapes.add_table(hang, cot, int(x), int(y), int(w), int(h)).table
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
            cell.margin_left = cell.margin_right = Inches(0.10)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.fill.solid()
            cell.fill.fore_color.rgb = DAM if i == 0 else (NHAT if o.startswith("@") else TRANG)
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
            _chay(p, o.lstrip("@"), co=co, dam=(i == 0), mau=TRANG if i == 0 else CHU)
    return g


def anh(slide, duong, x, y, w, h):
    """Chèn ảnh vừa khít khung (x, y, w, h), giữ tỷ lệ, canh giữa."""
    from PIL import Image
    iw, ih = Image.open(duong).size
    ti_le = min(w / iw, h / ih)
    nw, nh = int(iw * ti_le), int(ih * ti_le)
    return slide.shapes.add_picture(duong, int(x + (w - nw) // 2), int(y + (h - nh) // 2), nw, nh)


def chu_thich(slide, hinh, text, co=11):
    """Chú thích đặt ngay dưới một hình/khung."""
    return chu(slide, hinh.left, hinh.top + hinh.height + Inches(0.08), hinh.width, Inches(0.3),
               text, co=co, mau=MO, canh=PP_ALIGN.CENTER)


def nhan(slide, x, y, text, w=Inches(1.3), h=Inches(0.32)):
    """Nhãn nhỏ màu nhấn, ví dụ "Kết quả chính"."""
    sp = hop(slide, x, y, w, h, fill=CAM)
    sp.adjustments[0] = 0.25
    tf = sp.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.paragraphs[0].alignment = PP_ALIGN.CENTER
    _chay(tf.paragraphs[0], text, co=11, dam=True, mau=TRANG)
    return sp
