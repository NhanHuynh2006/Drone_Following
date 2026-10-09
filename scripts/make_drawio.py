#!/usr/bin/env python3
"""Sinh bộ sơ đồ .drawio cho SteadyDet-N **SteadyDet-N** (models/pfdet_nano_v17.py).

Mỗi sơ đồ = 1 file trong docs/drawio/, mở thẳng bằng https://app.diagrams.net
(File > Open From > Device). Sửa xong lưu lại vẫn là .drawio.

    python3 scripts/make_drawio.py

Mọi con số kênh/độ phân giải/GMACs lấy từ profile 'light' @640:
    base_c=24, neck_c=48, p2_ratio=0.5, head_ratio=0.75
    -> c1..c5 = 24/48/96/144/192 ; nc2=24 ; head_c=36 ; nc2_mid=18
    -> 1.171 M tham số, ~4.6 GFLOPs @640 (thop)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drawio_lib import Drawio, PAL

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs', 'drawio')
os.makedirs(OUT, exist_ok=True)


def save(d, fname):
    p = d.save(os.path.join(OUT, fname))
    print('  ✓', fname)
    return p


# ============================================================
# 01 — KIẾN TRÚC TỔNG THỂ SteadyDet-N
# ============================================================
def d01():
    d = Drawio('SteadyDet-N — kiến trúc tổng thể', 1760, 1300)
    d.title(60, 25, 1200, 'SteadyDet-N — kiến trúc tổng thể (profile light @640)',
            'models/pfdet_nano_v17.py · 1.171 M tham số · ~4.6 GFLOPs · 4 mức P2/P3/P4/P5 (stride 4/8/16/32) · đầu ra (B,5,H,W) = cls(1)+box(4)')

    cA, wA = 60, 250     # backbone
    cB, wB = 392, 168    # lateral 1x1
    cC, wC = 616, 128    # upsample + cộng
    cD, wD = 790, 196    # khối top-down
    cE, wE = 1054, 208   # LiteHead
    cF, wF = 1318, 226   # tensor ra

    yP2, yP3, yP4, yP5 = 288, 448, 608, 848

    d.group_frame(cA - 16, 118, wA + 32, 830, 'BACKBONE — dồn tính toán về P3–P5', 'backbone', fill='#F4FBF5')
    d.group_frame(cB - 16, 258, (cD + wD) - cB + 32, 690, 'NECK — FPN top-down một chiều, P2 đi kênh GIẢM', 'neck', fill='#FFFAF3')
    d.group_frame(cE - 16, 258, (cF + wF) - cE + 32, 690, 'HEAD — LiteHead dùng chung cho 4 mức', 'head', fill='#FBF4FD')

    inp = d.box(cA, 150, wA, 56, 'Ảnh vào\n1 × 3 × 640 × 640', 'input', fontsize=13)
    stem = d.box(cA, 232, wA, 56, 'EdgeContextStem  →  P1  s2\n24 ch · 320×320   ·  12.2 % compute', 'stem', fontsize=12)
    s2 = d.box(cA, yP2, wA, 60, 'stage_p2 = UIBBlock(24→48, s2)\nP2  s4 · 48 ch · 160×160  ·  10.0 %\nCHỈ 1 block (SteadyDet-W dùng 3)', 'backbone', fontsize=12)
    s3 = d.box(cA, yP3, wA, 60, 'stage_p3 = CSPUIBStage(48→96, n=2, extra_dw)\nP3  s8 · 96 ch · 80×80  ·  27.4 %', 'backbone', fontsize=12)
    s4 = d.box(cA, yP4, wA, 60, 'stage_p4 = CSPUIBStage(96→144, n=2, extra_dw)\nP4  s16 · 144 ch · 40×40  ·  17.8 %', 'backbone', fontsize=12)
    s5 = d.box(cA, yP5 - 100, wA, 60, 'stage_p5 = CSPUIBStage(144→192, n=2)\nP5  s32 · 192 ch · 20×20  ·  8.4 %', 'backbone', fontsize=12)
    attn = d.box(cA, yP5, wA, 60, 'AreaAttentionBlock(192, area=3, heads=2)\nCHỈ ở P5 — 20×20 nên rẻ  ·  5.6 %', 'attn', fontsize=12)

    d.edge(inp, stem); d.edge(stem, s2); d.edge(s2, s3); d.edge(s3, s4); d.edge(s4, s5); d.edge(s5, attn)

    l2 = d.box(cB, yP2 + 6, wB, 48, 'lat_p2  1×1\n48 → 24 ch', 'neck', fontsize=12)
    l3 = d.box(cB, yP3 + 6, wB, 48, 'lat_p3  1×1\n96 → 48 ch', 'neck', fontsize=12)
    l4 = d.box(cB, yP4 + 6, wB, 48, 'lat_p4  1×1\n144 → 48 ch', 'neck', fontsize=12)
    l5 = d.box(cB, yP5 + 6, wB, 48, 'lat_p5  1×1\n192 → 48 ch', 'neck', fontsize=12)
    for src, dst in ((s2, l2), (s3, l3), (s4, l4), (attn, l5)):
        d.edge(src, dst, exit_=(1, 0.5), entry=(0, 0.5), color=PAL['neck'][1])

    a4 = d.ellipse(cC + 24, yP4 + 4, 80, 52, '⊕\nup ×2', 'op', fontsize=13)
    a3 = d.ellipse(cC + 24, yP3 + 4, 80, 52, '⊕\nup ×2', 'op', fontsize=13)
    a2 = d.ellipse(cC + 24, yP2 + 4, 80, 52, '⊕\nup ×2', 'op', fontsize=13)
    p32 = d.box(cC, yP3 + 96, wC, 46, 'p3_to_p2  1×1\n48 → 24 ch', 'neck', fontsize=11)

    n5 = d.box(cD, yP5, wD, 48, 'reduce_p5  1×1  (n5)\n48 ch · 20×20', 'neck', fontsize=12)
    n4 = d.box(cD, yP4, wD, 60, 'td_p4 = UIBBlock(48→48)\nn4 · 48 ch · 40×40', 'neck', fontsize=12)
    n3 = d.box(cD, yP3, wD, 60, 'td_p3 = UIBBlock(48→48)\nn3 · 48 ch · 80×80', 'neck', fontsize=12)
    n2 = d.box(cD, yP2, wD, 60, 'td_p2 = UIBBlock(24→24)\nn2 · 24 ch · 160×160', 'neck', fontsize=12)

    d.edge(l5, n5, color=PAL['neck'][1])
    d.edge(n5, a4, 'n5', color=PAL['neck'][1], exit_=(0.5, 0), entry=(0.5, 1))
    d.edge(l4, a4, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(a4, n4, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(n4, a3, 'n4', color=PAL['neck'][1], exit_=(0.5, 0), entry=(0.5, 1))
    d.edge(l3, a3, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(a3, n3, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(n3, p32, 'n3', color=PAL['neck'][1], exit_=(0.5, 1), entry=(1, 0.5))
    d.edge(p32, a2, color=PAL['neck'][1], exit_=(0, 0.5), entry=(0.5, 1))
    d.edge(l2, a2, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(a2, n2, color=PAL['neck'][1], exit_=(1, 0.5), entry=(0, 0.5))

    heads = []
    for y, txt, out in (
        (yP2, 'LiteHead(24, mid=18)\n0.9 % compute · 0.8 K tham số', '(B, 5, 160, 160)\nstride 4 — người TÍ XÍU'),
        (yP3, 'LiteHead(48, mid=36)\n0.7 % · 2.5 K', '(B, 5, 80, 80)\nstride 8'),
        (yP4, 'LiteHead(48, mid=36)\n0.2 % · 2.5 K', '(B, 5, 40, 40)\nstride 16'),
        (yP5, 'LiteHead(48, mid=36)\n0.04 % · 2.5 K', '(B, 5, 20, 20)\nstride 32 — người TO/GẦN')):
        h = d.box(cE, y, wE, 56, txt, 'head', fontsize=12)
        o = d.box(cF, y + 2, wF, 52, out, 'out', fontsize=12)
        d.edge(h, o, color=PAL['head'][1])
        heads.append(h)
    for n, h in zip((n2, n3, n4, n5), heads):
        d.edge(n, h, color=PAL['head'][1], exit_=(1, 0.5), entry=(0, 0.5))

    d.text(cA, 980, 700, 200,
           'GHI CHÚ THIẾT KẾ (điểm khác SteadyDet-W)\n'
           '• P2 (160×160) là mức đắt nhất về không gian → SteadyDet-N để nó MỎNG: 1 UIBBlock + kênh giảm 48→24.\n'
           '• Toàn bộ 4 head chỉ tốn 1.8 % compute; riêng head P2 của SteadyDet-W (decoupled 5-conv) đã là ~1.8 GMACs.\n'
           '• Attention chỉ đặt ở P5 (20×20) — chi phí attention tỉ lệ với số ô, đặt ở P3 sẽ đắt gấp 16×.\n'
           '• Kết quả: 12.9 GFLOPs (SteadyDet-W) → 4.6 GFLOPs (SteadyDet-N) mà VẪN giữ nhánh P2.', fontsize=13)

    d.legend(1330, 980, [('input', 'ảnh vào'), ('stem', 'stem'), ('backbone', 'backbone'),
                         ('attn', 'attention'), ('neck', 'neck / FPN'), ('head', 'head'),
                         ('out', 'tensor ra')], w=220)
    return save(d, '01_kien_truc_tong_the.drawio')


# ============================================================
# 02 — EdgeContextStem
# ============================================================
def d02():
    d = Drawio('EdgeContextStem — stem 2 nhánh', 1180, 700)
    d.title(60, 25, 1000, 'EdgeContextStem — stem hai nhánh (P1, stride 2)',
            'models/pfdet_nano_v15.py:218 (SteadyDet-N dùng lại) · 3 → 24 ch · 640² → 320² · 12.2 % compute của cả mạng')

    x0 = 70
    inp = d.box(x0, 130, 190, 56, 'Ảnh 3 × 640 × 640', 'input')

    d.group_frame(320, 105, 470, 130, 'Nhánh CỤC BỘ — giữ nét biên', 'stem', fill='#EDFAFB')
    a1 = d.box(340, 140, 200, 56, 'ConvBN 3×3, s2\n3 → 12 ch · 320²', 'stem')
    a2 = d.box(570, 140, 200, 56, 'ConvBN 3×3, s1\n12 → 12 ch · 320²', 'stem')

    d.group_frame(320, 285, 470, 130, 'Nhánh NGỮ CẢNH — thấy vùng rộng, chống nhiễu', 'stem', fill='#EDFAFB')
    b1 = d.box(340, 320, 200, 56, 'AvgPool 2×2, s2\n3 ch · 320²', 'stem')
    b2 = d.box(570, 320, 200, 56, 'ConvBN 3×3, s1\n3 → 12 ch · 320²', 'stem')

    cat = d.ellipse(840, 210, 90, 60, 'concat\n24 ch', 'op')
    fuse = d.box(980, 212, 150, 56, 'ConvBN 1×1\n24 → 24 ch', 'stem')

    d.edge(inp, a1, exit_=(1, 0.35), entry=(0, 0.5))
    d.edge(inp, b1, exit_=(1, 0.65), entry=(0, 0.5))
    d.edge(a1, a2); d.edge(b1, b2)
    d.edge(a2, cat, exit_=(1, 0.5), entry=(0, 0.35))
    d.edge(b2, cat, exit_=(1, 0.5), entry=(0, 0.65))
    d.edge(cat, fuse)
    out = d.box(980, 320, 150, 50, 'P1: 24 × 320²', 'out')
    d.edge(fuse, out)

    d.text(70, 470, 1040, 180,
           'TẠI SAO 2 NHÁNH?\n'
           '• Người trên ảnh drone bay 3–5 m rất nhỏ; nếu stem chỉ có conv stride-2 thì chi tiết biên mất ngay ở lớp đầu.\n'
           '• Nhánh cục bộ giữ nét cạnh (2 conv 3×3 liên tiếp); nhánh ngữ cảnh lấy trung bình trước rồi mới conv → khử nhiễu cảm biến,\n'
           '  và chính nhánh này là thứ giúp mạng ít nhạy với NHOÈ do rung (đóng góp chính của bài).\n'
           '• Mỗi nhánh chỉ 12 kênh (out_c // 2) rồi mới concat → tổng chi phí bằng một stem 24 kênh thường.', fontsize=13)
    return save(d, '02_edge_context_stem.drawio')


# ============================================================
# 03 — UIBBlock
# ============================================================
def d03():
    d = Drawio('UIBBlock — khối cơ bản', 1320, 760)
    d.title(60, 25, 1100, 'UIBBlock — Universal Inverted Bottleneck (khối cơ bản của SteadyDet-N)',
            'MobileNetV4, ECCV 2024 · models/pfdet_nano_v15.py:177 · expand_ratio = 4 · dùng ở stage_p2, trong CSPUIBStage và ở td_p2/p3/p4')

    y = 150
    inp = d.box(70, y, 170, 56, 'x\nin_c × H × W', 'input')
    b1 = d.box(280, y, 190, 56, 'DWConv 3×3, s=stride\nin_c → in_c (groups=in_c)', 'backbone')
    b2 = d.box(510, y, 190, 56, 'PWConv 1×1\nin_c → 4·in_c  (nở)', 'backbone')
    b3 = d.box(740, y, 200, 56, 'FactDW 5×5  [tuỳ chọn]\n(1×5) rồi (5×1), depthwise', 'backbone', dashed=1)
    b4 = d.box(980, y, 190, 56, 'PWConv 1×1\n4·in_c → out_c  (nén)', 'backbone')
    add = d.ellipse(1200, y + 4, 60, 48, '⊕', 'op', fontsize=18)

    d.edge(inp, b1); d.edge(b1, b2); d.edge(b2, b3, 'BN+SiLU'); d.edge(b3, b4); d.edge(b4, add)

    sk1 = d.box(280, y + 170, 380, 56, 'stride = 1 và in_c = out_c\n→ nối tắt THẲNG (identity), không tốn gì', 'good', fontsize=12)
    sk2 = d.box(700, y + 170, 470, 56, 'ngược lại → nhánh chiếu:  [AvgPool(stride) nếu s>1] + Conv 1×1 + BN', 'note', fontsize=12)
    d.edge(inp, sk1, exit_=(0.5, 1), entry=(0, 0.5), dashed=1, color='#78909C')
    d.edge(sk1, sk2, dashed=1, color='#78909C')
    d.edge(sk2, add, exit_=(1, 0.5), entry=(0.5, 1), dashed=1, color='#78909C')

    d.text(70, 420, 1180, 260,
           'ĐỌC KHỐI NÀY THẾ NÀO\n'
           '• "Inverted bottleneck": nở 48 → 192 kênh ở GIỮA rồi nén về 48. Lọc depthwise (3×3, 5×5) RẺ nên chạy được\n'
           '  ở chỗ nhiều kênh; hai conv 1×1 để nở và nén mới là phần ĐẮT — 89 % chi phí của khối (đo ở stage_p3).\n'
           '• DWConv 3×3 groups=in_c: mỗi kênh một bộ lọc riêng → chi phí bằng 1/in_c so với conv thường.\n'
           '• FactDW 5×5 tách thành (1×5) rồi (5×1): trường nhìn 5×5 nhưng chi phí chỉ 2/5. Bật ở block CUỐI của stage_p3/p4\n'
           '  (extra_dw=True) — nơi cần nhìn rộng để phân biệt người với vệt nền.\n'
           '• Toàn bộ Conv đều bias=False + BatchNorm → lúc export gộp được BN vào Conv (reparameterize) cho INT8 Hailo.', fontsize=13)
    return save(d, '03_uib_block.drawio')


# ============================================================
# 04 — CSPUIBStage
# ============================================================
def d04():
    d = Drawio('CSPUIBStage — stage của backbone', 1400, 780)
    d.title(60, 25, 1150, 'CSPUIBStage — một stage của backbone SteadyDet-N (P3 / P4 / P5)',
            'CSP (YOLOv5) + UIB · models/pfdet_nano_v15.py:309 · SteadyDet-N dùng n_blocks = 2 cho cả 3 stage')

    inp = d.box(70, 220, 170, 60, 'vào\nin_c × H × W', 'input')
    down = d.box(280, 220, 210, 60, 'UIBBlock(in_c → out_c, s=2)\nGIẢM một nửa độ phân giải', 'backbone')
    d.edge(inp, down)

    sp = d.ellipse(530, 226, 70, 48, 'chia', 'op', fontsize=13)
    d.edge(down, sp)

    d.group_frame(640, 100, 420, 140, 'Nhánh A — học đặc trưng', 'backbone', fill='#F4FBF5')
    a0 = d.box(660, 138, 170, 52, 'ConvBN 1×1\nout_c → out_c/2', 'backbone', fontsize=12)
    a1 = d.box(860, 138, 180, 52, 'UIBBlock ×2\n(block cuối có FactDW 5×5)', 'backbone', fontsize=12)
    d.edge(a0, a1)

    d.group_frame(640, 290, 420, 120, 'Nhánh B — đường tắt', 'note', fill='#FAFAFA')
    b0 = d.box(660, 326, 170, 52, 'ConvBN 1×1\nout_c → out_c/2', 'note', fontsize=12)

    d.edge(sp, a0, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(sp, b0, exit_=(1, 0.5), entry=(0, 0.5))

    cat = d.ellipse(1100, 226, 80, 52, 'concat\nout_c', 'op', fontsize=12)
    d.edge(a1, cat, exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(b0, cat, exit_=(1, 0.5), entry=(0, 0.7))
    fu = d.box(1210, 224, 150, 56, 'ConvBN 1×1\nfuse out_c', 'backbone')
    d.edge(cat, fu)

    d.text(70, 460, 1260, 260,
           'VÌ SAO CHIA ĐÔI KÊNH?\n'
           '• Chỉ MỘT NỬA số kênh đi qua chuỗi UIBBlock nặng → tiết kiệm gần một nửa FLOPs của stage mà số kênh ra không đổi.\n'
           '• Nhánh B là đường tắt xuyên suốt stage → đường lan gradient ngắn, tránh "sập" đặc trưng khi xếp nhiều block.\n'
           '• Chi phí thực đo được ở SteadyDet-N (profile light) @640:  P3 27.4 %  ·  P4 17.8 %  ·  P5 8.4 % tổng compute.\n'
           '  Càng xuống sâu, độ phân giải càng nhỏ → tăng kênh (96 → 144 → 192) gần như miễn phí. Đó là lý do SteadyDet-N\n'
           '  dồn chiều sâu xuống P3–P5 và để P2 chỉ đúng 1 block.', fontsize=13)
    return save(d, '04_csp_uib_stage.drawio')


# ============================================================
# 05 — AreaAttention ở P5
# ============================================================
def d05():
    d = Drawio('AreaAttentionBlock tại P5', 1440, 800)
    d.title(60, 25, 1150, 'AreaAttentionBlock — tự chú ý theo cửa sổ, CHỈ đặt ở P5',
            'kiểu Swin (ICCV 2021) · models/pfdet_nano_v15.py:237 · c=192, area=3, num_heads=2 · vào/ra đều 192 × 20 × 20')

    y = 150
    inp = d.box(70, y, 170, 60, 'P5\n192 × 20 × 20', 'input')
    pad = d.box(270, y, 180, 60, 'pad cho chia hết 3\n20 → 21', 'attn', fontsize=12)
    part = d.box(480, y, 200, 60, 'cắt cửa sổ 3×3\n(B·49) × 9 token × 192', 'attn', fontsize=12)
    d.edge(inp, pad); d.edge(pad, part)

    d.group_frame(710, 105, 560, 155, 'Trong MỖI cửa sổ 3×3 (9 token)', 'attn', fill='#FDF0F5')
    q = d.box(730, y, 150, 56, 'LayerNorm\nqkv 1 lớp Linear', 'attn', fontsize=12)
    at = d.box(900, y, 170, 56, 'softmax(QKᵀ/√d)·V\n2 head × 96 chiều', 'attn', fontsize=12)
    pr = d.box(1090, y, 160, 56, 'Linear proj\n+ nối tắt', 'attn', fontsize=12)
    d.edge(part, q); d.edge(q, at); d.edge(at, pr)

    ffn = d.box(900, y + 190, 170, 56, 'LayerNorm → FFN\n192→384→192, GELU', 'attn', fontsize=12)
    d.edge(pr, ffn, exit_=(0.5, 1), entry=(1, 0.5))
    unp = d.box(480, y + 190, 200, 56, 'ghép cửa sổ về\n192 × 21 × 21', 'attn', fontsize=12)
    d.edge(ffn, unp, exit_=(0, 0.5), entry=(1, 0.5))
    crop = d.box(270, y + 190, 180, 56, 'cắt bỏ pad\n→ 20 × 20', 'attn', fontsize=12)
    d.edge(unp, crop)
    out = d.box(70, y + 190, 170, 56, 'ra P5\n192 × 20 × 20', 'out')
    d.edge(crop, out)

    d.text(70, 460, 1300, 300,
           'HAI QUYẾT ĐỊNH ĐÁNG CHÚ Ý\n'
           '1) CHỈ ĐẶT Ở P5. Chi phí attention tăng theo SỐ Ô của bản đồ đặc trưng. Ở 20×20 nó chỉ chiếm 5.6 % compute\n'
           '   của cả mạng; nếu đặt cùng khối này ở P3 (80×80) thì số ô gấp 16 lần — một mình nó sẽ nuốt hết ngân sách.\n'
           '2) CỬA SỔ 3×3, KHÔNG PHẢI TOÀN CỤC. Attention toàn cục trên 400 token tốn 400² phép so khớp; chia thành 49 cửa sổ\n'
           '   9 token chỉ còn 49 × 9² ≈ 4 000 → rẻ hơn ~40 lần, mà ở stride 32 thì một cửa sổ 3×3 đã phủ 96×96 pixel ảnh gốc,\n'
           '   quá đủ ngữ cảnh cho một người.\n\n'
           'LƯU Ý DEPLOY: bản HEF nhẹ nhất (c512) từng BỎ khối này để vừa Hailo; đo trên 92 khung bay thật, việc bỏ nó\n'
           'làm F1 tụt khoảng 0.046. Muốn lấy lại mà vẫn INT8 được thì thay bằng ConvAttnBlock (xem PROJECT.md).', fontsize=13)
    return save(d, '05_area_attention_p5.drawio')


# ============================================================
# 06 — Neck top-down
# ============================================================
def d06():
    d = Drawio('Neck FPN top-down của SteadyDet-N', 1420, 900)
    d.title(60, 25, 1150, 'Neck SteadyDet-N — FPN top-down MỘT chiều, nhánh P2 đi kênh giảm',
            'models/pfdet_nano_v17.py:131-143 (_feats) · SteadyDet-N BỎ BiFPN 2 chiều của SteadyDet-W, chỉ giữ đường đi xuống · toàn neck = 16.8 % compute')

    xl, xa, xn = 90, 430, 700
    yP5, yP4, yP3, yP2 = 700, 540, 380, 200

    for y, src, ch, res in ((yP5, 'p5 (sau attention)', '192', '20×20'),
                            (yP4, 'p4', '144', '40×40'),
                            (yP3, 'p3', '96', '80×80'),
                            (yP2, 'p2', '48', '160×160')):
        d.box(xl - 30, y, 130, 50, f'{src}\n{ch} ch · {res}', 'backbone', fontsize=11)

    l5 = d.box(xl + 130, yP5, 180, 50, 'lat_p5 1×1\n192 → 48', 'neck', fontsize=12)
    l4 = d.box(xl + 130, yP4, 180, 50, 'lat_p4 1×1\n144 → 48', 'neck', fontsize=12)
    l3 = d.box(xl + 130, yP3, 180, 50, 'lat_p3 1×1\n96 → 48', 'neck', fontsize=12)
    l2 = d.box(xl + 130, yP2, 180, 50, 'lat_p2 1×1\n48 → 24  ← KÊNH GIẢM', 'neck', fontsize=12)

    n5 = d.box(xn, yP5, 200, 50, 'reduce_p5 1×1 → n5\n48 ch · 20×20', 'neck', fontsize=12)
    a4 = d.ellipse(xa + 60, yP4 - 2, 80, 54, '⊕\nup×2', 'op', fontsize=12)
    n4 = d.box(xn, yP4, 200, 50, 'td_p4 UIBBlock → n4\n48 ch · 40×40', 'neck', fontsize=12)
    a3 = d.ellipse(xa + 60, yP3 - 2, 80, 54, '⊕\nup×2', 'op', fontsize=12)
    n3 = d.box(xn, yP3, 200, 50, 'td_p3 UIBBlock → n3\n48 ch · 80×80', 'neck', fontsize=12)
    p32 = d.box(xa + 20, yP3 - 90, 160, 46, 'p3_to_p2 1×1\n48 → 24', 'neck', fontsize=11)
    a2 = d.ellipse(xa + 60, yP2 - 2, 80, 54, '⊕\nup×2', 'op', fontsize=12)
    n2 = d.box(xn, yP2, 200, 50, 'td_p2 UIBBlock → n2\n24 ch · 160×160', 'neck', fontsize=12)

    c = PAL['neck'][1]
    d.edge(l5, n5, color=c)
    d.edge(n5, a4, color=c, exit_=(0.5, 0), entry=(0.5, 1))
    d.edge(l4, a4, color=c); d.edge(a4, n4, color=c)
    d.edge(n4, a3, color=c, exit_=(0.5, 0), entry=(0.5, 1))
    d.edge(l3, a3, color=c); d.edge(a3, n3, color=c)
    d.edge(n3, p32, color=c, exit_=(0.5, 0), entry=(1, 0.5))
    d.edge(p32, a2, color=c, exit_=(0, 0.5), entry=(0.5, 1))
    d.edge(l2, a2, color=c); d.edge(a2, n2, color=c)

    for y, lbl in ((yP5, 'head_p5'), (yP4, 'head_p4'), (yP3, 'head_p3'), (yP2, 'head_p2')):
        d.box(xn + 240, y, 130, 50, lbl, 'head', fontsize=12)

    d.text(90, 60, 1200, 60, '', fontsize=1)
    d.text(90, 790, 1250, 200,
           'BA ĐIỀU KHIẾN NECK NÀY RẺ\n'
           '• Chiếu kênh TRƯỚC khi cộng: mọi mức đều về 48 ch (P2 về 24 ch) nên phép cộng và block top-down luôn chạy ở kênh thấp.\n'
           '• Chỉ ĐI XUỐNG. SteadyDet-W dùng BiFPN có thêm đường đi lên và trọng số hợp nhất học được; ở mức nano, đường đi lên phải\n'
           '  chạy lại toàn bộ ở 160×160 — quá đắt so với phần AP nó mang lại. SteadyDet-N cắt hẳn.\n'
           '• Nhánh P2 dùng nc2 = neck_c // 2 = 24 ch. Vì P2 có 160×160 = 25 600 ô, mỗi kênh cắt đi ở đây tiết kiệm gấp 4 lần so với P3.', fontsize=13)
    return save(d, '06_neck_topdown.drawio')


# ============================================================
# 07 — LiteHead
# ============================================================
def d07():
    d = Drawio('LiteHead vs head decoupled 5-conv', 1460, 880)
    d.title(60, 25, 1200, 'LiteHead — head của SteadyDet-N, và cái nó thay thế',
            'models/pfdet_nano_v17.py:31 · depthwise-separable, tách cls/box, KHÔNG DFL · đây là chỗ SteadyDet-N tiết kiệm nhiều nhất')

    d.group_frame(60, 110, 660, 300, 'SteadyDet-N — LiteHead (đang dùng)', 'good', dashed=0, fill='#F1F9F2')
    i1 = d.box(85, 175, 130, 52, 'đặc trưng\nin_c × H × W', 'input', fontsize=11)
    h1 = d.box(240, 175, 150, 52, 'DWConv 3×3\ngroups=in_c', 'head', fontsize=11)
    h2 = d.box(410, 175, 150, 52, 'PWConv 1×1\nin_c → mid_c', 'head', fontsize=11)
    hc = d.box(580, 140, 120, 46, 'Conv 1×1\n→ cls (1 ch)', 'head', fontsize=11)
    hb = d.box(580, 220, 120, 46, 'Conv 1×1\n→ box (4 ch)', 'head', fontsize=11)
    d.edge(i1, h1); d.edge(h1, h2, 'BN+SiLU')
    d.edge(h2, hc, exit_=(1, 0.5), entry=(0, 0.5)); d.edge(h2, hb, exit_=(1, 0.5), entry=(0, 0.5))
    d.text(85, 300, 610, 100,
           'mid_c = 0.75 · in_c   ·   khởi tạo cls.bias = −5.5 (ít báo động giả lúc mới train)\n'
           'CHI PHÍ ở P2 (160×160, 24 ch):  0.019 GMACs  ·  0.8 K tham số  =  0.9 % cả mạng', fontsize=12, color='#1B5E20')

    d.group_frame(760, 110, 640, 300, 'SteadyDet-W — DecoupledHead 5 conv (đã bỏ ở SteadyDet-N)', 'bad', dashed=1, fill='#FDF1F2')
    j1 = d.box(785, 175, 120, 52, 'đặc trưng', 'input', fontsize=11)
    j2 = d.box(925, 175, 140, 52, 'RepConv 3×3\nchia sẻ', 'bad', fontsize=11)
    j3 = d.box(1090, 130, 140, 46, 'RepConv 3×3\n→ Conv 1×1 cls', 'bad', fontsize=11)
    j4 = d.box(1090, 215, 140, 46, 'RepConv 3×3\n→ Conv 1×1 box', 'bad', fontsize=11)
    d.edge(j1, j2, color=PAL['bad'][1])
    d.edge(j2, j3, color=PAL['bad'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(j2, j4, color=PAL['bad'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.text(785, 300, 590, 100,
           'Ba conv 3×3 ĐẦY ĐỦ (không depthwise) ở 48 kênh, chạy tại 160×160.\n'
           'CHI PHÍ ở P2:  ~1.8 GMACs  =  28 % toàn bộ mạng SteadyDet-W — chỉ riêng một cái head.', fontsize=12, color='#B71C1C')

    # thanh so sánh
    d.text(60, 450, 600, 30, 'Chi phí head tại P2 (160×160)', fontsize=15, color='#102A43', bold=True)
    d.box(60, 490, 60, 40, '', 'good', rounded=0)
    d.text(130, 498, 400, 26, 'SteadyDet-N LiteHead — 0.019 GMACs', fontsize=13, color='#1B5E20')
    d.box(60, 545, 1120, 40, '', 'bad', rounded=0)
    d.text(60, 596, 700, 26, 'SteadyDet-W DecoupledHead — ~1.8 GMACs   (gần 95× đắt hơn, cùng một vị trí)', fontsize=13, color='#B71C1C')

    d.text(60, 660, 1330, 200,
           'TẠI SAO ĐỔI ĐƯỢC MÀ KHÔNG MẤT ĐỘ CHÍNH XÁC ĐÁNG KỂ\n'
           '• Bài toán chỉ có 1 lớp (person). Head nhiều lớp cần dung lượng để tách lớp; ở đây không cần — thứ head phải làm\n'
           '  chỉ là "có người / không" và 4 số hộp. Tách nhánh cls/box vẫn giữ (tránh hai mục tiêu tranh nhau một bộ trọng số).\n'
           '• Bỏ DFL (phân phối rời rạc cho toạ độ, kiểu YOLOv8): DFL cần 16 bin × 4 cạnh = 64 kênh ra thay vì 4, và phải\n'
           '  softmax + tích phân lúc decode → tốn cả FLOPs lẫn thời gian hậu xử lý, lại khó lượng tử hoá INT8. SteadyDet-N hồi quy thẳng.\n'
           '• Cùng một LiteHead dùng cho cả 4 mức, chỉ khác số kênh vào → hợp đồng đầu ra (B,5,H,W) đồng nhất, decode một hàm duy nhất.', fontsize=13)
    return save(d, '07_litehead.drawio')


# ============================================================
# 08 — Ngân sách tính toán
# ============================================================
def d08():
    d = Drawio('Ngân sách tính toán SteadyDet-N', 1420, 880)
    d.title(60, 25, 1200, 'Compute đi đâu trong SteadyDet-N (light @640)',
            'Tính giải tích theo đúng định nghĩa module · tổng ~4.6 GFLOPs, 1.171 M tham số · cột dài = tốn nhiều')

    rows = [
        ('stage_p3  CSPUIB ×2  ·  80×80', 27.4, 93.9, 'backbone'),
        ('stage_p4  CSPUIB ×2  ·  40×40', 17.8, 240.4, 'backbone'),
        ('neck  (lateral + top-down)', 16.8, 69.8, 'neck'),
        ('EdgeContextStem  ·  320×320', 12.2, 2.6, 'stem'),
        ('stage_p2  UIBBlock  ·  160×160', 10.0, 8.7, 'backbone'),
        ('stage_p5  CSPUIB ×2  ·  20×20', 8.4, 450.7, 'backbone'),
        ('AreaAttention  ·  20×20', 5.6, 296.3, 'attn'),
        ('head_p2  LiteHead  ·  160×160', 0.9, 0.8, 'head'),
        ('head_p3  LiteHead  ·  80×80', 0.7, 2.5, 'head'),
        ('head_p4  LiteHead  ·  40×40', 0.2, 2.5, 'head'),
        ('head_p5  LiteHead  ·  20×20', 0.04, 2.5, 'head'),
    ]
    x0, y0, bar_x, scale, rh = 70, 120, 430, 26.0, 52
    d.text(x0, y0 - 34, 340, 26, 'khối', fontsize=13, color='#627D98', bold=True)
    d.text(bar_x, y0 - 34, 300, 26, '% compute', fontsize=13, color='#627D98', bold=True)
    d.text(1180, y0 - 34, 200, 26, 'tham số (K)', fontsize=13, color='#627D98', bold=True)
    for i, (name, pct, par, kind) in enumerate(rows):
        y = y0 + i * rh
        d.text(x0, y + 8, 350, 26, name, fontsize=13, color='#37474F')
        w = max(6, pct * scale)
        d.box(bar_x, y, w, 30, '', kind, rounded=0)
        d.text(bar_x + w + 10, y + 4, 120, 24, f'{pct:g} %', fontsize=13, color='#37474F', bold=True)
        pw = max(4, par / 460.0 * 150)
        d.box(1180, y + 6, pw, 18, '', 'note', rounded=0)
        d.text(1180 + pw + 8, y + 4, 120, 22, f'{par:.0f}', fontsize=11, color='#78909C')

    d.text(70, 720, 1300, 200,
           'BA ĐIỀU ĐỌC RA TỪ BẢNG NÀY\n'
           '• TOÀN BỘ 4 head cộng lại chỉ 1.8 % compute. Head KHÔNG còn là chỗ tốn tiền — đó chính là điều SteadyDet-N sửa so với SteadyDet-W,\n'
           '  nơi riêng head P2 đã chiếm 28 %.\n'
           '• Tham số và compute đi NGƯỢC nhau: stage_p5 nhiều tham số nhất (450 K) nhưng chỉ 8.4 % compute vì chạy ở 20×20;\n'
           '  stem chỉ 2.6 K tham số mà tốn 12.2 % vì chạy ở 320×320. Muốn giảm FLOPs thì cắt ở nơi ĐỘ PHÂN GIẢI cao, không phải nơi nhiều tham số.\n'
           '• Nhánh P2 (stage_p2 + head_p2 + phần neck ở 160×160) tổng cộng ~13 % — cái giá phải trả để nhìn thấy người tí xíu. Ở SteadyDet-W nó là ~45 %.', fontsize=13)
    return save(d, '08_ngan_sach_tinh_toan.drawio')


# ============================================================
# 09 — Các quyết định tối ưu
# ============================================================
def d09():
    d = Drawio('Các quyết định tối ưu trong SteadyDet-N', 1500, 900)
    d.title(60, 25, 1250, 'Bốn quyết định làm nên SteadyDet-N — từ 12.9 xuống 4.6 GFLOPs mà vẫn giữ nhánh P2',
            'Mỗi ô: VẤN ĐỀ đo được → CÁCH SỬA trong models/pfdet_nano_v17.py → GIÁ PHẢI TRẢ')

    items = [
        ('1 · Head quá đắt ở độ phân giải cao',
         'Head decoupled 5 conv 3×3 đầy đủ chạy tại 160×160:\nmột mình 1.8 GMACs = 28 % toàn mạng.',
         'LiteHead: DWConv 3×3 + PWConv 1×1 dùng chung,\nrồi 2 conv 1×1 tách cls / box. Bỏ DFL.',
         'Head còn 0.019 GMACs ở P2 (−95×).\nGiá: dung lượng head nhỏ hơn — chấp nhận được vì chỉ 1 lớp.'),
        ('2 · Nhánh P2 nuốt ngân sách',
         'P2 có 160×160 = 25 600 ô. SteadyDet-W chạy stage P2 đầy đủ\n3 block ở 48 kênh + neck 2 chiều đi qua đó.',
         'stage_p2 = ĐÚNG 1 UIBBlock. Neck ở P2 hạ xuống\nnc2 = 24 kênh (p2_ratio 0.5).',
         'Cả nhánh P2 còn ~13 % compute.\nGiá: ít dung lượng cho người tí xíu — bù bằng gán nhãn k_tiny=6 + NWD.'),
        ('3 · Attention đặt sai chỗ thì phá ngân sách',
         'Chi phí attention tỉ lệ số ô. Đặt ở P3 (80×80)\nđắt gấp 16 lần so với P5.',
         'Chỉ AreaAttentionBlock ở P5 (20×20), cửa sổ 3×3,\n2 head. Các mức khác thuần conv.',
         'Attention chỉ 5.6 % compute.\nGiá: bản HEF bỏ khối này mất ~0.046 F1 trên bay thật.'),
        ('4 · Phải gộp được BN, phải INT8 được',
         'Hailo-8L chạy INT8. Conv+BN rời và nhánh rẽ lạ\nlàm parser gãy hoặc lượng tử hoá kém.',
         'Mọi Conv bias=False + BN ngay sau → reparameterize()\ngộp BN vào Conv trước khi export ONNX.',
         '@640 đo được 56.9 FPS trên Pi5+Hailo-8L.\nINT8 gần như không mất gì: F1 0.880 vs FP32 0.836.'),
    ]
    for i, (title, prob, fix, cost) in enumerate(items):
        y = 120 + i * 180
        d.group_frame(60, y, 1380, 160, title, 'note', dashed=0, fill='#FCFCFD')
        d.box(80, y + 42, 400, 100, prob, 'bad', fontsize=12, align='left',
              extra='spacingLeft=10;verticalAlign=top;spacingTop=8;')
        d.box(510, y + 42, 400, 100, fix, 'good', fontsize=12, align='left',
              extra='spacingLeft=10;verticalAlign=top;spacingTop=8;')
        d.box(940, y + 42, 480, 100, cost, 'deploy', fontsize=12, align='left',
              extra='spacingLeft=10;verticalAlign=top;spacingTop=8;')
        d.text(80, y + 24, 400, 20, 'VẤN ĐỀ', fontsize=11, color='#B71C1C', bold=True)
        d.text(510, y + 24, 400, 20, 'CÁCH SỬA', fontsize=11, color='#1B5E20', bold=True)
        d.text(940, y + 24, 400, 20, 'KẾT QUẢ / GIÁ PHẢI TRẢ', fontsize=11, color='#311B92', bold=True)

    d.text(60, 848, 1380, 40,
           'Hợp đồng đầu ra GIỮ NGUYÊN qua cả 4 thay đổi: (B, 5, H, W) = cls(1) + box(4), strides [4, 8, 16, 32]'
           ' → dùng lại nguyên loss và decode cũ.', fontsize=13, color='#455A64', bold=True)
    return save(d, '09_toi_uu_steadydet.drawio')


# ============================================================
# 10 — Gán nhãn + hàm mất mát
# ============================================================
def d10():
    d = Drawio('Gán nhãn và hàm mất mát của SteadyDet-N', 1520, 980)
    d.title(60, 25, 1250, 'Gán nhãn & hàm mất mát — vì sao SteadyDet-N nhìn được người tí xíu',
            'utils/losses_v15.py (SteadyDet-N dùng chung contract head nên dùng chung loss) · TAL + STAL + QFL + NWD/CIoU + ASL + ProgLoss')

    pred = d.box(70, 130, 200, 60, 'Dự đoán 4 mức\n(B,5,H,W) × 4', 'head', fontsize=12)
    gt = d.box(70, 230, 200, 60, 'Nhãn thật\nhộp người (xywh)', 'input', fontsize=12)

    d.group_frame(320, 105, 620, 320, 'BƯỚC 1 — chọn ô nào chịu trách nhiệm cho mỗi người (TAL + STAL)', 'train', fill='#FFFEF5')
    a1 = d.box(345, 145, 265, 60, 'Chọn mức theo cỡ hộp\nscale_ranges (0-48 / 24-112 / 80-200 / 160+)\ncó CHỒNG LẤN → một người có thể ở 2 mức', 'train', fontsize=11)
    a2 = d.box(345, 225, 265, 60, 'Ô ứng viên quanh tâm hộp\nbán kính 3 ô nếu hộp tí xíu, 2 ô nếu thường', 'train', fontsize=11)
    a3 = d.box(640, 145, 275, 60, 'Điểm khớp = cls^1.0 · IoU^8.0 · thưởng STAL\nthưởng = 1 + 2·exp(−diện tích / 576)', 'train', fontsize=11)
    a4 = d.box(640, 225, 275, 60, 'Lấy top-k:  k = 6 nếu hộp < 576 px²,  k = 3 nếu thường\nk giảm dần về 3/2 theo epoch (ProgLoss)', 'train', fontsize=11)
    a5 = d.box(345, 315, 570, 60, 'BẢO HIỂM cho hộp SIÊU nhỏ (< 64 px²): nếu chưa đủ 4 ô thì NỚI bán kính thêm 1 rồi chọn lại\n→ không người nào bị bỏ trắng, đây là thứ giữ được recall ở VisDrone', 'train', fontsize=11)
    d.edge(pred, a1, color=PAL['train'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(gt, a1, color=PAL['train'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(a1, a2, color=PAL['train'][1]); d.edge(a2, a4, color=PAL['train'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(a1, a3, color=PAL['train'][1], exit_=(1, 0.5), entry=(0, 0.5)); d.edge(a3, a4, color=PAL['train'][1])

    d.group_frame(320, 460, 620, 300, 'BƯỚC 2 — hai nhánh mất mát', 'note', dashed=0, fill='#FCFCFD')
    l1 = d.box(345, 500, 270, 110,
               'PHÂN LOẠI — Quality Focal Loss (β=2)\n\n'
               'Đích KHÔNG phải 0/1 mà là IoU của hộp dự đoán\n'
               '→ ô nào khớp khít hơn thì bị ép tin cậy cao hơn.\n'
               'Chia cho số ô dương, không chia H·W (nếu chia\nH·W thì gradient loãng ~550 lần).',
               'head', fontsize=11, align='left', extra='spacingLeft=10;verticalAlign=top;spacingTop=8;')
    l2 = d.box(645, 500, 270, 110,
               'HỘP — trộn NWD với CIoU theo DIỆN TÍCH\n\n'
               'w = exp(−diện tích / 1024)\n'
               'loss = w · NWD + (1−w) · CIoU\n'
               'Cùng lệch 3 px: người 6×6 còn IoU 0.33,\nngười 60×60 vẫn 0.90. NWD phạt đều theo pixel.',
               'head', fontsize=11, align='left', extra='spacingLeft=10;verticalAlign=top;spacingTop=8;')
    l3 = d.box(345, 630, 570, 60,
               'Nhân thêm trọng số theo diện tích (ASL):  √(0.004 · 640² / diện tích), kẹp trong [1, 4]\n'
               '→ người 10×10 px "tính điểm" gấp 4 lần, 20×20 gấp 2, từ 40×40 trở lên bằng 1', 'head', fontsize=11)
    l4 = d.box(345, 705, 570, 44,
               'ProgLoss: đầu train obj×2 / box÷2 (học "có người ở đâu" trước), cuối train về đúng 8.0 / 4.0 (học hộp cho khít)', 'train', fontsize=11)
    d.edge(a5, l1, color=PAL['head'][1], exit_=(0.3, 1), entry=(0.5, 0))
    d.edge(a5, l2, color=PAL['head'][1], exit_=(0.7, 1), entry=(0.5, 0))
    d.edge(l1, l3, color=PAL['head'][1], exit_=(0.5, 1), entry=(0.3, 0))
    d.edge(l2, l3, color=PAL['head'][1], exit_=(0.5, 1), entry=(0.7, 0))
    d.edge(l3, l4, color=PAL['head'][1])

    tot = d.box(1000, 600, 420, 80, 'loss = 8.0 · L_cls  +  4.0 · L_box\n(trọng số chạy theo lịch ProgLoss)', 'out', fontsize=14)
    d.edge(l4, tot, color=PAL['out'][1], exit_=(1, 0.5), entry=(0, 0.5))

    d.text(1000, 130, 430, 420,
           'VÌ SAO PHẢI LÀM VẬY\n\n'
           'VisDrone bay cao → phần lớn người chỉ vài chục pixel.\n'
           'Với hộp cỡ đó, gán nhãn kiểu chuẩn (top-1 anchor, loss\nIoU) hỏng theo hai cách:\n\n'
           '• Quá ít ô dương → tín hiệu học gần như không có,\n'
           '  model học luôn "không có ai" cho chắc.\n'
           '  → sửa bằng k_tiny = 6 và bảo hiểm 4 ô.\n\n'
           '• IoU phạt hộp nhỏ nặng hơn hẳn: cùng lệch 3 pixel, người\n'
           '  6×6 còn IoU 0.33 trong khi người 60×60 vẫn 0.90.\n'
           '  → sửa bằng NWD (khoảng cách Wasserstein giữa hai\n'
           '    Gauss), mượt theo pixel chứ không theo tỉ lệ.\n\n'
           'Ba cơ chế này KHÔNG tốn gì lúc chạy thật — chúng chỉ\ntồn tại trong lúc train.', fontsize=13)
    return save(d, '10_gan_nhan_va_loss.drawio')


# ============================================================
# 11 — Giải mã đầu ra
# ============================================================
def d11():
    d = Drawio('Giải mã đầu ra của SteadyDet-N', 1440, 820)
    d.title(60, 25, 1200, 'Từ tensor ra tới hộp người — đường giải mã',
            'box_ops_np.py:78 decode_predictions_np · chạy một hàm duy nhất cho cả 4 mức vì hợp đồng head giống nhau')

    x = 70
    boxes = [
        ('(B, 5, H, W)\nmỗi mức một tensor', 'out'),
        ('kênh 0 → sigmoid\n= điểm tin cậy', 'head'),
        ('lọc điểm ≥ conf\n(deploy: 0.25)', 'head'),
        ('giữ top-500 mỗi mức\ntrước NMS', 'head'),
        ('giải toạ độ:\ncx = (x + σ(t₀) − 0.5)·s\ncy = (y + σ(t₁) − 0.5)·s\nw = exp(t₂)·s,  h = exp(t₃)·s', 'head'),
        ('gộp 4 mức\n+ NMS IoU 0.45', 'out'),
        ('lọc rác deploy:\ndiện tích ≥ 200 px²\ntỉ lệ cạnh ≤ 6', 'deploy'),
        ('hộp người cuối', 'input'),
    ]
    prev = None
    for i, (txt, kind) in enumerate(boxes):
        col, row = i % 4, i // 4
        bx = x + col * 340
        by = 150 + row * 220
        cur = d.box(bx, by, 260, 110, txt, kind, fontsize=12)
        if prev:
            if col == 0:
                d.edge(prev, cur, exit_=(0.5, 1), entry=(0.5, 0))
            else:
                d.edge(prev, cur)
        prev = cur

    d.text(70, 600, 1300, 200,
           'BA CHI TIẾT QUAN TRỌNG\n'
           '• Không có DFL, không có anchor box. Toạ độ tâm là độ lệch trong ô (sigmoid, kẹp trong ±0.5 ô); cạnh là exp(t)·stride,\n'
           '  clamp t trong [−4, 4] để INT8 không tràn. Decode là số học thuần → chạy được bằng NumPy trên CPU Pi 5 trong ~1 ms.\n'
           '• conf ngưỡng KHÁC NHAU theo từng bản HEF. Bản đang deploy: 0.25 cho F1 0.880; bản _fair cũ là 0.30. Đổi HEF là phải đo lại.\n'
           '• Hai bộ lọc cuối (min_box_px, max_aspect) CHỈ có ở đường deploy, không có lúc đánh giá paper — vì INT8 làm hộp kém khít\n'
           '  (AP75 rớt ~30 %) nên vài hộp dẹt lọt lưới NMS.', fontsize=13)
    return save(d, '11_giai_ma_dau_ra.drawio')


# ============================================================
# 12 — NMS-free dual head
# ============================================================
def d12():
    d = Drawio('Nhánh NMS-free của SteadyDet-N', 1400, 820)
    d.title(60, 25, 1150, 'SteadyDet-N NMS-free — hai head song song trên CÙNG đặc trưng',
            'models/pfdet_nano_v17.py:113-119 & forward_dual · bật bằng nms_free=True (configs/train_config_v17_nms_free.yaml)')

    feats = d.box(80, 300, 230, 70, 'Đặc trưng neck\nn2, n3, n4, n5', 'neck', fontsize=13)

    d.group_frame(390, 130, 480, 200, 'Head o2m — một người ↔ NHIỀU ô', 'head', fill='#FBF4FD')
    o2m = d.box(415, 175, 200, 60, 'head_p2..p5\nLiteHead ×4', 'head', fontsize=12)
    o2m2 = d.box(650, 175, 195, 60, 'k = 6 / 3 ô dương\nmỗi người', 'head', fontsize=12)
    d.text(415, 250, 430, 60, 'Học nhanh, tín hiệu dày → đây là head dùng khi TRAIN\nvà cũng là head mặc định lúc suy luận (cần NMS).', fontsize=12, color='#4A148C')
    d.edge(o2m, o2m2)

    d.group_frame(390, 380, 480, 200, 'Head o2o — một người ↔ ĐÚNG 1 ô', 'good', fill='#F1F9F2')
    o2o = d.box(415, 425, 200, 60, 'head_p2..p5_o2o\nLiteHead ×4 (bản sao)', 'good', fontsize=12)
    o2o2 = d.box(650, 425, 195, 60, 'k = 1 ô dương\nstal_k_min = 1', 'good', fontsize=12)
    d.text(415, 500, 430, 60, 'Không sinh hộp trùng → suy luận KHÔNG cần NMS.\nKhó train hơn vì tín hiệu thưa.', fontsize=12, color='#1B5E20')
    d.edge(o2o, o2o2)

    d.edge(feats, o2m, exit_=(1, 0.5), entry=(0, 0.5), color=PAL['head'][1])
    d.edge(feats, o2o, exit_=(1, 0.5), entry=(0, 0.5), color=PAL['good'][1])

    tr = d.box(950, 175, 380, 60, 'LÚC TRAIN — forward_dual(x)\nloss = loss_o2m + λ · loss_o2o  (λ = 1.0)', 'train', fontsize=12)
    ev = d.box(950, 425, 380, 60, 'LÚC EVAL nms_free=True\nchỉ chạy head o2o, bỏ hẳn NMS', 'deploy', fontsize=12)
    d.edge(o2m2, tr, color=PAL['train'][1]); d.edge(o2o2, ev, color=PAL['deploy'][1])
    d.edge(o2m2, ev, dashed=1, color='#B0BEC5')

    d.text(80, 620, 1250, 180,
           'KẾT QUẢ ĐO ĐƯỢC (VisDrone-val, 3 seed)\n'
           '• SteadyDet-N thường:      AP sạch 0.1892  ·  retain khi rung 16.8 ± 0.7 %\n'
           '• SteadyDet-N NMS-free:  AP sạch 0.1954  ·  retain 23.4 ± 0.8 %   → head o2o KHÔNG làm tụt AP, còn nhỉnh hơn một chút.\n'
           '• Gradient của head o2o đi ngược vào CHUNG backbone/neck với o2m → nó hoạt động như một ràng buộc phụ,\n'
           '  ép đặc trưng phải đủ sắc để một ô duy nhất tự tin. Đây là lý do khả năng chịu rung tăng dù không hề train rung.\n'
           '• Giá: thêm 4 LiteHead lúc train (rẻ, tổng < 2 % compute) và tăng thời gian train; lúc deploy chỉ giữ một trong hai.', fontsize=13)
    return save(d, '12_nms_free_dual_head.drawio')


# ============================================================
# 13 — Huấn luyện kháng rung
# ============================================================
def d13():
    d = Drawio('Huấn luyện kháng rung', 1480, 900)
    d.title(60, 25, 1250, 'Vibration-aware training — đóng góp chính, KHÔNG đụng vào kiến trúc',
            'vib_consistency.py · chỉ tồn tại lúc train → chi phí suy luận trên Pi 5 = 0')

    b = d.box(70, 150, 200, 60, 'Batch ảnh sạch\n(B, 3, 640, 640)', 'input', fontsize=12)
    sp = d.ellipse(320, 156, 90, 50, 'chia\n50 / 50', 'op', fontsize=12)
    d.edge(b, sp)

    clean = d.box(470, 120, 230, 56, '50 % giữ nguyên', 'input', fontsize=12)
    d.group_frame(455, 195, 700, 175, 'corrupt_batch() — mô phỏng rung, GIỮ NGUYÊN toạ độ hộp', 'train', fill='#FFFEF5')
    c1 = d.box(475, 235, 200, 52, 'Nhoè chuyển động\nvạch dài L, góc ngẫu nhiên', 'train', fontsize=11)
    c2 = d.box(700, 235, 190, 52, 'Nhiễu cảm biến\nGauss σ', 'train', fontsize=11)
    c3 = d.box(915, 235, 220, 52, 'Dao động độ sáng\n±br', 'train', fontsize=11)
    d.text(475, 300, 660, 60,
           'sev1: L=5, σ=0.02, br=0.10   ·   sev2: L=9, σ=0.04, br=0.20   ·   sev3: L=15, σ=0.06, br=0.30\n'
           'Lúc train severity lấy NGẪU NHIÊN 1–3; lúc benchmark cố định để so được.', fontsize=11, color='#F57F17')
    d.edge(sp, clean, exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(sp, c1, exit_=(1, 0.5), entry=(0, 0.5), color=PAL['train'][1])
    d.edge(c1, c2, color=PAL['train'][1]); d.edge(c2, c3, color=PAL['train'][1])

    net = d.box(1210, 180, 200, 80, 'SteadyDet-N\n(một model duy nhất)', 'backbone', fontsize=13)
    d.edge(clean, net, color=PAL['backbone'][1], exit_=(1, 0.5), entry=(0, 0.3))
    d.edge(c3, net, color=PAL['backbone'][1], exit_=(1, 0.5), entry=(0, 0.7))
    lo = d.box(1210, 300, 200, 60, 'Loss phát hiện\nBÌNH THƯỜNG', 'head', fontsize=12)
    d.edge(net, lo)

    # so sánh 2 cách
    d.group_frame(60, 430, 680, 250, 'CÁCH A — augmentation  (vib.mode: aug)  ✓ ĐANG DÙNG', 'good', dashed=0, fill='#F1F9F2')
    d.text(80, 470, 640, 200,
           '1 forward. Thay 50 % mỗi batch bằng ảnh đã làm rung, rồi tính\nloss phát hiện y như thường.\n\n'
           '   retain 36.2 ± 0.7 %   (baseline 16.8 ± 0.7 %)\n'
           '   thời gian train  1.0×\n\n'
           'Đơn giản hơn, rẻ hơn, VÀ tốt hơn.', fontsize=13, color='#1B5E20')

    d.group_frame(770, 430, 660, 250, 'CÁCH B — consistency  (VibConsistencyLoss)  ✗ KẾT QUẢ ÂM', 'bad', dashed=1, fill='#FDF1F2')
    d.text(790, 470, 620, 200,
           '2 forward (sạch + rung), ép dự đoán trên ảnh rung khớp dự đoán\ntrên ảnh sạch (ảnh sạch làm mốc, detach).\nKL trên cls + smooth-L1 trên box + MSE trên đặc trưng neck.\n\n'
           '   retain 30.0 ± 0.6 %       thời gian train  1.8×\n\n'
           'THUA augmentation. Đây là một trong hai kết quả âm được báo cáo\nthẳng trong bài — phần có giá trị nhất.', fontsize=13, color='#B71C1C')

    d.text(60, 710, 1370, 180,
           'HAI ĐỐI CHỨNG KHIẾN KẾT LUẬN ĐỨNG VỮNG\n'
           '• "Train thêm 120 epoch mà KHÔNG rung" → 16.9 ± 0.7 %, đúng bằng baseline. Vậy phần tăng KHÔNG phải do train lâu hơn.\n'
           '• Áp đúng công thức lên YOLO11n → 21.9 → 21.5 % (from scratch), 20.1 → 20.1 % (pretrained): KHÔNG chuyển được sang kiến trúc khác.\n'
           '  Đã bác 3 giả thuyết (COCO-pretrain, cỡ vật thể, headroom); nguyên nhân còn lại là kiến trúc hoặc loss — bài viết thẳng là chưa cô lập được.\n'
           '  Chính điều này làm phép so sánh với YOLO trở nên CÔNG BẰNG: kỹ thuật không nâng được mọi model.', fontsize=13)
    return save(d, '13_huan_luyen_khang_rung.drawio')


# ============================================================
# 14 — Đưa model lên Hailo
# ============================================================
def d14():
    d = Drawio('Đường đưa SteadyDet-N lên Hailo-8L', 1500, 800)
    d.title(60, 25, 1250, 'Từ checkpoint SteadyDet-N tới NPU — chuỗi export và điều làm nó gãy',
            'export_onnx_hailo.py → compile_hef.py · Raspberry Pi 5 + Hailo-8L, INT8')

    steps = [
        ('best.pt\ncheckpoint EMA', 'input'),
        ('reparameterize()\ngộp BN vào Conv', 'deploy'),
        ('torch.onnx.export\nopset 11, static 1×3×640×640\ndynamo = False (BẮT BUỘC)', 'deploy'),
        ('onnxsim\nconstant folding', 'deploy'),
        ('Hailo DFC parse\n→ .har', 'deploy'),
        ('Lượng tử hoá INT8\ncalib 75 % VisDrone + 25 % COCO', 'deploy'),
        ('Biên dịch\n→ .hef (4 context)', 'deploy'),
        ('HailoRT trên Pi 5\n17.6 ms/khung · 56.9 FPS', 'good'),
    ]
    prev = None
    for i, (txt, kind) in enumerate(steps):
        col, row = i % 4, i // 4
        bx, by = 70 + col * 355, 140 + row * 200
        cur = d.box(bx, by, 275, 95, txt, kind, fontsize=12)
        if prev:
            d.edge(prev, cur, exit_=(0.5, 1) if col == 0 else (1, 0.5),
                   entry=(0.5, 0) if col == 0 else (0, 0.5))
        prev = cur

    d.text(70, 560, 1380, 240,
           'BỐN CÁI BẪY ĐÃ THỰC SỰ GÃY VÀ ĐÃ SỬA (đo 2026-08-27)\n'
           '1. dynamo=False. Từ torch 2.6, torch.onnx.export mặc định dùng bộ export mới; nó BỎ thuộc tính kernel_shape của node Conv\n'
           '   (hợp lệ theo chuẩn ONNX) nhưng parser Hailo đọc thẳng thuộc tính đó → IndexError. Bộ TorchScript cũ giữ đủ.\n'
           '2. KHÔNG dynamic_axes. HEF là static shape, batch = 1. Kích thước export phải đúng kích thước chạy thật (640).\n'
           '3. Calib phải TRỘN ĐÚNG TỈ LỆ dữ liệu train (75 % VisDrone + 25 % COCO person) và KHÔNG được lấy từ tập chấm điểm.\n'
           '4. Ngưỡng conf phải ĐO LẠI cho từng bản HEF. Bản hiện tại 0.25 → F1 0.880; bản cũ 0.30. Chép ngưỡng cũ sang là sai.\n\n'
           'ĐIỀU BẤT NGỜ ĐO ĐƯỢC: nhanh hơn YOLO26n 1.70× trên CÙNG con chip (56.9 vs 37.5 FPS) — biến quyết định là số CONTEXT\n'
           'mà trình biên dịch chia ra (4 so với 6), KHÔNG phải GFLOPs. Trên NPU, cách model được chia khối quan trọng hơn số phép tính.', fontsize=13)
    return save(d, '14_export_hailo.drawio')


# ============================================================
# 15 — Kết quả
# ============================================================
def d15():
    d = Drawio('Kết quả — mCP và retain', 1520, 900)
    d.title(60, 25, 1250, 'Kết quả chính — trục đo là mCP (độ chính xác KHI ĐANG RUNG), không phải AP sạch',
            'VisDrone-val 548 ảnh @640, person-only, một giao thức, 3 seed · runs/paper_queue/UNIFIED.json')

    rows = [
        ('SteadyDet-W + vib-aware  (1.00 M · 12.9 G)', 0.0874, 39.4, 'good'),
        ('RT-DETR-l  (32.81 M · 108 G)', 0.0641, 25.6, 'note'),
        ('SteadyDet-N + vib-aware  (1.17 M · 4.6 G)', 0.0617, 36.2, 'good'),
        ('YOLO11-n  (2.59 M · 6.4 G)', 0.0557, 20.1, 'note'),
        ('YOLOv10-n  (2.71 M · 8.4 G)', 0.0538, 19.7, 'note'),
        ('YOLOv8-n  (3.01 M · 8.2 G)', 0.0522, 19.5, 'note'),
        ('YOLO26-n  (2.50 M · 5.8 G)', 0.0507, 19.0, 'note'),
        ('SteadyDet-N + consistency  (1.17 M · 4.6 G)', 0.0524, 30.0, 'bad'),
        ('SteadyDet-N baseline, không xử lý rung', 0.0318, 16.8, 'bad'),
    ]
    x0, y0, bx, rh = 70, 130, 520, 54
    d.text(x0, y0 - 32, 420, 24, 'model  (tham số · FLOPs)', fontsize=13, color='#627D98', bold=True)
    d.text(bx, y0 - 32, 300, 24, 'mCP  — AP trung bình khi rung sev1-3', fontsize=13, color='#627D98', bold=True)
    d.text(1230, y0 - 32, 260, 24, 'retain %  (giữ được bao nhiêu AP)', fontsize=13, color='#627D98', bold=True)
    for i, (name, mcp, ret, kind) in enumerate(rows):
        y = y0 + i * rh
        d.text(x0, y + 6, 440, 24, name, fontsize=13, color='#37474F', bold=(kind == 'good'))
        w = mcp / 0.09 * 480
        d.box(bx, y, w, 30, '', kind, rounded=0)
        d.text(bx + w + 10, y + 3, 110, 24, f'{mcp:.4f}', fontsize=12, color='#37474F', bold=True)
        rw = ret / 40.0 * 170
        d.box(1230, y + 6, rw, 18, '', kind, rounded=0)
        d.text(1230 + rw + 8, y + 3, 90, 22, f'{ret:.1f} %', fontsize=11, color='#78909C')

    d.text(70, 640, 1400, 250,
           'ĐỌC BẢNG NÀY CHO ĐÚNG — và những gì KHÔNG được claim\n'
           '• Trên AP SẠCH, SteadyDet-N THUA cả 4 YOLO nano (0.1708 vs 0.2675–0.2768). Không được claim kiến trúc trên trục đó.\n'
           '• Trên mCP, SteadyDet-N + vib-aware THẮNG cả 4 YOLO nano, với 45 % tham số và 72 % compute của con mạnh nhất.\n'
           '• So với RT-DETR-l (28× tham số): thắng ở sev2 (5.4σ), nhưng THUA ở mCP trung bình (0.0617 < 0.0641),\n'
           '  và ở sev3 chênh chỉ 0.64σ = trong nhiễu → KHÔNG được claim.\n'
           '• "Không nửa nào đủ một mình": chỉ kiến trúc → mCP 0.0318 (bét bảng); chỉ kỹ thuật rung áp lên YOLO11n → 0.0542\n'
           '  (không hơn YOLO11n thường 0.0557); CẢ HAI → 0.0617. Đóng góp là kiến trúc + chế độ train tính chung.\n'
           '• Bay thật (92 khung, Pi5 + Hailo, INT8): F1 = 0.880 @ conf 0.25, recall 0.837.', fontsize=13)
    return save(d, '15_ket_qua_mcp.drawio')


# ============================================================
# 16 — Vòng điều khiển (chỉ nói sơ)
# ============================================================
def d16():
    d = Drawio('Vòng bám 30 Hz (tóm tắt)', 1420, 620)
    d.title(60, 25, 1200, 'Model nằm ở đâu trong hệ thống bay — bản tóm tắt',
            'Phần điều khiển chi tiết xem FOLLOW_CONTROL.md · ở đây chỉ để thấy detector là mắt xích nào')

    chain = [
        ('Camera USB\n640×480 @ 30 Hz', 'input'),
        ('SteadyDet-N\ntrên Hailo-8L\n17.6 ms', 'backbone'),
        ('OC-SORT\ncấp ID cho người', 'op'),
        ('Chọn & khoá mục tiêu\nhộp to nhất ≈ gần nhất', 'op'),
        ('PID → vx, vy, vz, yaw', 'op'),
        ('SafetyManager\nkẹp tốc độ / độ cao / cự ly', 'bad'),
        ('Pixhawk · ArduPilot\nchế độ GUIDED, 30 Hz', 'deploy'),
    ]
    prev = None
    for i, (txt, kind) in enumerate(chain):
        col, row = i % 4, i // 4
        bx, by = 70 + col * 335, 150 + row * 170
        cur = d.box(bx, by, 255, 85, txt, kind, fontsize=12)
        if prev:
            d.edge(prev, cur, exit_=(0.5, 1) if col == 0 else (1, 0.5),
                   entry=(0.5, 0) if col == 0 else (0, 0.5))
        prev = cur

    d.box(1080, 320, 255, 85, 'Nhật ký bay CSV\nkèm rung thật (vibe_x/y/z)', 'note', fontsize=12)

    d.text(70, 460, 1300, 150,
           'Vì sao 30 Hz: detector sinh thông tin mới 30 lần/giây, nên mỗi lệnh gửi xuống mang đúng một phát hiện mới.\n'
           '10 Hz thì thêm 33 ms trễ vô ích; 50 Hz thì 40 % lệnh không mang tin gì mới. Khứ hồi lệnh đo được p50 4.75 ms / p99 7.73 ms.\n\n'
           'Ngân sách một vòng = 33.3 ms. Detector chiếm 17.6 ms → còn dư hơn một nửa cho tracking, PID và MAVLink.\n'
           'Đây là toàn bộ lý do phải ép model xuống 4.6 GFLOPs.', fontsize=13)
    return save(d, '16_vong_dieu_khien_tom_tat.drawio')


ALL = [d01, d02, d03, d04, d05, d06, d07, d08, d09, d10, d11, d12, d13, d14, d15, d16]

if __name__ == '__main__':
    print('Sinh sơ đồ .drawio vào', OUT)
    for f in ALL:
        f()
    print('\nXong. Mở bằng https://app.diagrams.net > File > Open From > Device.')


# ============================================================
# 17 — Lưu đồ vòng điều khiển 30 Hz
# ============================================================
def d17():
    d = Drawio('Lưu đồ vòng điều khiển 30 Hz', 1500, 1320)
    d.title(60, 25, 1250, 'Lưu đồ vòng bám — một lượt lặp của follow_drone/main.py',
            'Chạy ở 30 Hz (command_rate_hz), ngân sách 33.3 ms mỗi vòng · ArduPilot Copter, chế độ GUIDED')

    DIA = 'shape=rhombus;'
    x0, xm, xr = 120, 520, 950
    y = 120

    start = d.ellipse(xm + 40, y, 160, 50, 'bắt đầu vòng', 'input', fontsize=13)
    f1 = d.box(xm, y + 90, 240, 52, 'Lấy khung mới nhất từ CameraThread\n(hàng đợi 1 khung, không xếp hàng cũ)', 'input', fontsize=11)
    f2 = d.box(xm, y + 172, 240, 52, 'PersonDetectorHailo.infer()\nHailo-8L · 17.6 ms', 'backbone', fontsize=11)
    f3 = d.box(xm, y + 254, 240, 52, 'OC-SORT cập nhật track\nmax_age 45 · min_hits 3', 'op', fontsize=11)
    q1 = d.box(xm - 20, y + 336, 280, 76, 'TargetSelector\ncòn thấy mục tiêu đã khoá?', 'op', fontsize=12, extra=DIA)
    d.edge(start, f1); d.edge(f1, f2); d.edge(f2, f3); d.edge(f3, q1)

    # nhánh CÓ
    f4 = d.box(xm, y + 440, 240, 52, 'Ước lượng cự ly\ntừ chiều cao hộp (pinhole) + Kalman', 'op', fontsize=11)
    d.edge(q1, f4, 'CÒN', color='#2E7D32')

    d.group_frame(xm - 30, y + 512, 300, 210, 'VisualServo — 3 vòng PID ĐỘC LẬP', 'op', fill='#FAFCFF')
    p1 = d.box(xm - 10, y + 548, 260, 48, 'ForwardPID: sai số cự ly → vx\nKp .5  Ki .05  Kd .1  ·  v ≤ 3 m/s', 'op', fontsize=11)
    p2 = d.box(xm - 10, y + 604, 260, 48, 'YawPID: lệch tâm hộp theo x → yaw_rate\nKp 2.0  Ki 0  Kd .3  ·  ≤ 1.5 rad/s', 'op', fontsize=11)
    p3 = d.box(xm - 10, y + 660, 260, 48, 'VerticalPID: sai số độ cao AGL → vz\nKp .8  Ki .1  Kd .2  ·  v ≤ 2 m/s', 'op', fontsize=11)
    d.edge(f4, p1)
    d.text(xm + 270, y + 596, 240, 60, 'vy = 0 — bám ngang bằng CÁCH XOAY,\nkhông trượt ngang. Ít trục hơn = ít\ndao động, và người luôn ở giữa khung.', fontsize=11, color='#546E7A')

    saf = d.box(xm - 10, y + 748, 260, 56, 'SafetyManager.clamp_velocity()\nkẹp v ngang ≤5, yaw ≤1 rad/s,\ncự ly 3–15 m, độ cao 2–25 m AGL', 'bad', fontsize=11)
    d.edge(p3, saf)
    snd = d.box(xm - 10, y + 830, 260, 52, 'SET_POSITION_TARGET_LOCAL_NED\nvx, vy, vz, yaw_rate (khung thân)', 'deploy', fontsize=11)
    d.edge(saf, snd)
    log = d.box(xm + 300, y + 830, 250, 52, 'Ghi 1 dòng CSV\nkèm vibe_x/y/z, n_det, best_conf', 'note', fontsize=11)
    d.edge(snd, log, dashed=1, color='#90A4AE')
    slp = d.box(xm - 10, y + 912, 260, 48, 'ngủ bù cho đủ 1/30 s', 'input', fontsize=12)
    d.edge(snd, slp)
    d.edge(slp, f1, 'vòng tiếp theo', color='#455A64', exit_=(0, 0.5), entry=(0, 0.5),
           waypoints=[[x0 + 40, y + 936], [x0 + 40, y + 116]])

    # nhánh MẤT
    d.group_frame(xr - 60, y + 320, 420, 420, 'THANG MẤT MỤC TIÊU — dừng ở HOVER, KHÔNG cướp quyền bay', 'bad', fill='#FDF6F6')
    m1 = d.box(xr - 40, y + 360, 380, 52, 'COAST  ·  0 – 3 s\nbay tiếp theo dự đoán Kalman của track cũ', 'bad', fontsize=11)
    m2 = d.box(xr - 40, y + 428, 380, 52, 'HOVER  ·  sau 3 s\ngửi vận tốc 0 = đứng yên tại chỗ', 'bad', fontsize=11)
    m3 = d.box(xr - 40, y + 496, 380, 76,
               'Trong 5 s tiếp theo CHỈ nhận lại đúng người cũ\n(so vị trí + cỡ hộp — OC-SORT cấp ID MỚI khi người\nquay lại, không có re-ID ngoại hình)', 'bad', fontsize=11)
    m4 = d.box(xr - 40, y + 588, 380, 52, 'Sau đó mới chấp nhận người khác\n(auto_relock)', 'bad', fontsize=11)
    m5 = d.box(xr - 40, y + 656, 380, 60,
               'Ở HOVER MÃI cho tới khi người lái can thiệp.\nKHÔNG tự chuyển LOITER/RTL — chủ ý bỏ.', 'note', fontsize=11)
    d.edge(q1, m1, 'MẤT', color=PAL['bad'][1], exit_=(1, 0.5), entry=(0, 0.5))
    d.edge(m1, m2, color=PAL['bad'][1]); d.edge(m2, m3, color=PAL['bad'][1])
    d.edge(m3, m4, color=PAL['bad'][1]); d.edge(m4, m5, color=PAL['bad'][1])
    d.edge(m2, snd, 'vẫn gửi lệnh (v = 0)', color=PAL['bad'][1], dashed=1,
           exit_=(0, 0.5), entry=(1, 0.5), waypoints=[[xr - 90, y + 856]])

    d.text(120, 1180, 1300, 120,
           'VÌ SAO THANG DỪNG Ở HOVER: bản cũ leo tới LOITER rồi RTL bằng cách gọi thẳng set_mode() — tức là phần mềm CƯỚP chế độ bay\n'
           'khỏi tay người lái, và RTL của ArduPilot còn tự leo lên RTL_ALT. Nay vòng bám KHÔNG bao giờ gọi set_mode: mất người thì đứng yên,\n'
           'quyền quyết định trả về cho người lái. LOITER/RTL trong mã chỉ còn để đọc được log cũ.', fontsize=13)
    return save(d, '17_luu_do_vong_dieu_khien.drawio')


# ============================================================
# 18 — Chia luồng / tiến trình
# ============================================================
def d18():
    d = Drawio('Chia luồng và tiến trình khi bay', 1560, 1000)
    d.title(60, 25, 1300, 'Chia luồng — 2 tiến trình, 5 luồng, và ai chạy ở tần số nào',
            'Trên Raspberry Pi 5. Ràng buộc gốc: /dev/ttyACM0 CHỈ một tiến trình mở được, và NPU Hailo CHỈ một tiến trình dùng được.')

    # tiến trình 1
    d.group_frame(70, 120, 380, 300, 'TIẾN TRÌNH 1 — mavmux.py   ( ./run.sh mux )', 'deploy', fill='#F3F0FB')
    mx = d.box(95, 165, 330, 60, 'Giữ cổng /dev/ttyACM0\nMAVLink v2 · 94–100 gói/s · 0 lỗi', 'deploy', fontsize=12)
    mx2 = d.box(95, 245, 330, 60, 'Phát lại ra UDP cho nhiều client\n(đã test 2 client × 100 gói/s)', 'deploy', fontsize=12)
    d.edge(mx, mx2, color=PAL['deploy'][1])
    d.text(95, 320, 330, 80,
           'Tồn tại CHỈ vì cổng USB độc quyền: không có mux thì\nvòng bám và bảng giám sát không thể cùng nói chuyện\nvới mạch bay.', fontsize=11, color='#4527A0')

    # tiến trình 2
    d.group_frame(500, 120, 1000, 620, 'TIẾN TRÌNH 2 — follow_drone/main.py', 'backbone', fill='#F4FBF5')

    threads = [
        ('LUỒNG 1 · CameraThread', 'follow/camera.py:54 (daemon)', '30 Hz',
         'Đọc camera liên tục, giữ ĐÚNG 1 khung mới nhất.\nCAP_PROP_BUFFERSIZE = 1 và tắt auto-exposure —\nhai lỗi này từng ghìm e2e ở 15.4 FPS.', 'input'),
        ('LUỒNG 2 · MAVLink _recv_loop', 'follow/mavlink_client.py:104 (daemon)', 'theo gói đến',
         'Nhận telemetry, cập nhật state dùng chung.\nTốc độ đặt cho FCU: ATTITUDE / LOCAL_POSITION_NED /\nVFR_HUD 10 Hz · GPS · SYS_STATUS 5 Hz ·\nVIBRATION / EKF / BATTERY 2 Hz · HOME 0.2 Hz.', 'op'),
        ('LUỒNG 3 · heartbeat', 'follow/mavlink_client.py:106 (daemon)', '1 Hz',
         'Nhịp tim bắt buộc của MAVLink. Ngừng gửi là FCU\ncoi như mất liên lạc.', 'op'),
        ('LUỒNG 4 · Dashboard HTTP', 'follow/dashboard.py:157 (ThreadingHTTPServer)', 'theo yêu cầu',
         'Bảng giám sát cổng 8080, DÙNG LẠI khung hình và\nphát hiện có sẵn — không tốn thêm lượt suy luận.\nNhúng thẳng vào main.py vì NPU chỉ 1 tiến trình dùng được.', 'note'),
        ('LUỒNG CHÍNH · vòng bám', 'follow_drone/main.py', '30 Hz',
         'Detect → track → chọn mục tiêu → 3 PID → kẹp an toàn\n→ gửi SET_POSITION_TARGET → ghi log → ngủ bù.\nNgân sách 33.3 ms, detector chiếm 17.6 ms.', 'backbone'),
    ]
    for i, (name, src, hz, body, kind) in enumerate(threads):
        y = 165 + i * 112
        d.box(525, y, 300, 96, f'{name}\n{src}', kind, fontsize=11)
        d.box(845, y + 24, 90, 48, hz, 'train', fontsize=13)
        d.box(955, y, 520, 96, body, 'note', fontsize=11, align='left',
              extra='spacingLeft=12;verticalAlign=middle;')

    fcu = d.box(180, 620, 240, 70, 'Pixhawk 6C\nArduPilot Copter 4.7.0', 'deploy', fontsize=13)
    d.edge(mx2, fcu, color=PAL['deploy'][1], exit_=(0.5, 1), entry=(0.5, 0))
    d.edge(fcu, mx, 'telemetry', color=PAL['deploy'][1], dashed=1,
           exit_=(0, 0.5), entry=(0, 0.5), waypoints=[[130, 655], [130, 195]])
    d.text(70, 720, 380, 60, 'USB /dev/ttyACM0 · MAVLink v2', fontsize=12, color='#4527A0')

    d.edge(mx2, threads and d.box(525, 760, 300, 1, '', 'note', fontsize=1), '', color='#FFFFFF', arrow='none')

    d.text(500, 770, 1000, 200,
           'BA RÀNG BUỘC PHẦN CỨNG ĐÃ ĐỊNH HÌNH CÁCH CHIA NÀY\n'
           '1. NPU Hailo chỉ MỘT tiến trình dùng được → hailo_gcs.py và follow_drone KHÔNG chạy song song. Vì vậy bảng giám sát\n'
           '   phải là một LUỒNG bên trong main.py chứ không phải một tiến trình riêng, và nó dùng lại kết quả suy luận sẵn có.\n'
           '2. /dev/ttyACM0 chỉ MỘT tiến trình mở được → sinh ra mavmux.py.\n'
           '3. GIL của Python không cản ở đây vì mọi luồng đều chờ I/O (camera, socket) hoặc chờ NPU — phần tính toán nặng\n'
           '   nằm ngoài Python. Ngoài bãi chỉ cần đúng 2 lệnh:  ./run.sh mux   và   python main.py', fontsize=13)
    return save(d, '18_chia_luong_tien_trinh.drawio')


ALL = ALL + [d17, d18]
