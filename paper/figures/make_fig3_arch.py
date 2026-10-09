"""
Figure 3. SteadyDet-N. Rows are pyramid levels; the vertical axis is
resolution. Annotations mark the three decisions that make a stride-4 level
affordable.  Structure read from models/pfdet_nano_v17.py (light profile:
base_c=24, neck_c=48, p2_ratio=0.5, head_ratio=0.75).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, RED, GREEN, DARK = "#1f5fa8", "#c0392b", "#2e7d52", "#2b2b2b"

# VE DUNG CO IN (sua 2026-07-27b): ban truoc ve 11 inch roi in ra ~10 cm -> bop con
# 45-47% -> chu 7pt thanh 3.2pt tren giay. QUY TAC: be rong PDF (inch) x 2.54 phai
# XAP XI be rong in (cm). Dung ve to roi bop nho.
fig, ax = plt.subplots(figsize=(5.6, 2.95))
ax.set_xlim(0, 16.2); ax.set_ylim(-0.75, 6.55); ax.axis("off")

def box(x, y, w, h, text, fc="white", ec=DARK, fs=8.2, lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            zorder=4, linespacing=1.25)

def arrow(x1, y1, x2, y2, color=DARK, lw=1.0, ls="-"):
    """Mui nhui THANG (khong con tham so rad -> khong the ve cong nham)."""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), shrinkA=0, shrinkB=0,
                                 arrowstyle="-|>", mutation_scale=9,
                                 color=color, lw=lw, ls=ls, zorder=2,
                                 joinstyle="miter", capstyle="butt"))


def seg(pts, color=DARK, lw=1.0):
    """Noi nhieu diem bang cac doan NGANG/DOC; chi doan cuoi co mui nhui."""
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        last = (i == len(pts) - 2)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), shrinkA=0, shrinkB=0,
            arrowstyle="-|>" if last else "-", mutation_scale=9,
            color=color, lw=lw, zorder=2, joinstyle="miter", capstyle="butt"))

rows = {"P2": 4.30, "P3": 3.05, "P4": 1.80, "P5": 0.55}
res  = {"P2": r"$160^2$", "P3": r"$80^2$", "P4": r"$40^2$", "P5": r"$20^2$"}
strd = {"P2": "s4", "P3": "s8", "P4": "s16", "P5": "s32"}
h = 0.86

# level labels
for k, y in rows.items():
    ax.text(0.02, y + h/2, f"{k}\n{strd[k]} {res[k]}", fontsize=6.6, va="center",
            ha="left", color="#5a6472", linespacing=1.3)

# input + stem
box(1.30, 2.42, 1.25, 1.05, "input\nRGB", fc="#eef2f7", fs=6.8)
box(2.78, 2.42, 1.62, 1.05, "Edge\nContext\nStem s2", fc="#eef2f7", fs=6.8)
arrow(2.55, 2.95, 2.78, 2.95)

# backbone column
bx, bw = 4.66, 2.34
box(bx, rows["P2"], bw, h, "UIB $\\times$1\n48c", fc="#fdeeec", ec=RED, fs=6.8, lw=1.3)
box(bx, rows["P3"], bw, h, "CSP-UIB $\\times$2\n96c", fc="#eef2f7", fs=6.8)
box(bx, rows["P4"], bw, h, "CSP-UIB $\\times$2\n144c", fc="#eef2f7", fs=6.8)
box(bx, rows["P5"], bw, h, "CSP-UIB $\\times$2\n192c", fc="#eef2f7", fs=6.8)
# Stem -> P2: len thang tu dinh Stem roi re ngang vao P2 (khong cheo, khong cong)
SX = 2.78 + 1.62 / 2                     # tam ngang cua khoi Stem
seg([(SX, 2.42 + 1.05), (SX, rows["P2"] + h/2), (bx, rows["P2"] + h/2)])
for a, b in (("P2","P3"), ("P3","P4"), ("P4","P5")):
    arrow(bx + bw/2, rows[a], bx + bw/2, rows[b] + h)

# attention at P5 only
box(7.22, rows["P5"], 1.62, h, "window\nattention", fc="#e9f4ed", ec=GREEN, fs=6.5)
arrow(bx + bw, rows["P5"] + h/2, 7.22, rows["P5"] + h/2)

# neck
nx, nw = 9.22, 1.98
box(nx, rows["P2"], nw, h, "top-down\n24c", fc="#fdeeec", ec=RED, fs=6.8, lw=1.3)
box(nx, rows["P3"], nw, h, "top-down\n48c", fc="white", fs=6.8)
box(nx, rows["P4"], nw, h, "top-down\n48c", fc="white", fs=6.8)
box(nx, rows["P5"], nw, h, "reduce\n48c", fc="white", fs=6.8)

arrow(bx + bw, rows["P2"] + h/2, nx, rows["P2"] + h/2)
arrow(bx + bw, rows["P3"] + h/2, nx, rows["P3"] + h/2)
arrow(bx + bw, rows["P4"] + h/2, nx, rows["P4"] + h/2)
arrow(8.84, rows["P5"] + h/2, nx, rows["P5"] + h/2)
for a, b in (("P5","P4"), ("P4","P3"), ("P3","P2")):
    arrow(nx + nw/2, rows[a] + h, nx + nw/2, rows[b], color=BLUE, lw=1.1)
# (nhan "upsample + fuse" da chuyen vao caption: moi vi tri trong hinh deu dung
#  vao mui ten ngang hoac vao o neck. Mui ten xanh doc duoc mot minh.)

# heads
hx, hw = 12.05, 1.92
box(hx, rows["P2"], hw, h, "LiteHead", fc="#fdeeec", ec=RED, fs=6.8, lw=1.3)
for k in ("P3", "P4", "P5"):
    box(hx, rows[k], hw, h, "LiteHead", fc="white", fs=6.8)
for k in rows:
    arrow(nx + nw, rows[k] + h/2, hx, rows[k] + h/2)
    ax.text(hx + hw + 0.16, rows[k] + h/2, "cls 1\nbox 4", fontsize=6.5,
            va="center", color="#5a6472", linespacing=1.2)

# annotations: the three decisions
ax.add_patch(Rectangle((4.52, rows["P2"] - 0.13), 9.58, h + 0.26, fill=False,
                       ec=RED, lw=1.0, ls=(0, (4, 3)), zorder=1))
ax.text(0.15, 6.16, "$P_2$ (dashed) is kept thin: one block, half the neck "
        "width, a separable head.", fontsize=7.8, color=RED)
ax.text(0.15, 5.72, "That is what makes stride 4 affordable: "
        r"$12.9 \rightarrow 4.6$ GFLOPs at $640^2$.",
        fontsize=7.8, color="#3d4550")

ax.text(7.87, -0.46, "window attention only at the coarsest level, where the "
        "token count is small", fontsize=7.2, color=GREEN, ha="center")

fig.savefig("fig3_arch.pdf", bbox_inches="tight", dpi=300)
print("wrote fig3_arch.pdf")
