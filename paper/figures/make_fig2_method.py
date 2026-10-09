"""
Figure 2 (method). Doi khung 2026-07-26.

  TREN  = phuong phap CHINH: rung mot phan batch roi tinh loss detect binh thuong.
  DUOI  = BIEN THE consistency (2 forward + stop-gradient), TON 1.8x va TE HON
          (30.0 vs 36.2 retain) -> bao cao nhu ket qua am co kiem soat.

Dinh tuyen: toan bo doan thang ngang/doc. Khong dung arc3/rad.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, ORANGE, DARK, GOLD = "#2a78d6", "#eb6834", "#2b2b2b", "#9a7209"
GREY, PALE = "#6b7480", "#9db8d8"

# VE DUNG CO IN (sua 2026-07-27b): ban truoc ve 11 inch roi in ra ~10 cm -> bop con
# 45-47% -> chu 7pt thanh 3.2pt tren giay. QUY TAC: be rong PDF (inch) x 2.54 phai
# XAP XI be rong in (cm). Dung ve to roi bop nho.
fig, ax = plt.subplots(figsize=(5.6, 2.35))
ax.set_xlim(0, 16.8)
ax.set_ylim(-0.55, 6.80)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=DARK, fs=9, lw=1.1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.3)


def seg(pts, color=DARK, lw=1.2, head=True):
    """Noi cac diem bang doan THANG; chi doan cuoi co mui nhui."""
    for i in range(len(pts) - 1):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        last = (i == len(pts) - 2)
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), shrinkA=0, shrinkB=0,
            arrowstyle="-|>" if (last and head) else "-",
            mutation_scale=11, color=color, lw=lw, zorder=2,
            joinstyle="miter", capstyle="butt"))


H = 0.86
# ================= TREN: phuong phap chinh =================
YA = 4.78
CA = YA + H / 2
ax.add_patch(Rectangle((0.25, YA - 0.62), 16.3, H + 1.62, fc="#f2f7fc",
                       ec=BLUE, lw=1.2, ls="-", zorder=0, alpha=0.55))
ax.text(0.45, YA + H + 0.34, "Vibration-aware training",
        fontsize=8.4, color=BLUE, fontweight="bold", zorder=1, va="bottom")

box(0.75, YA, 1.30, H, "batch\n$x$", fc="#eef2f7", fs=7.4)
box(2.85, YA, 2.85, H, r"$\mathcal{T}$  vibration" + "\n" + r"w.p. $p=0.5$",
    fc="#fdefe8", ec=ORANGE, fs=6.8)
box(6.05, YA, 1.90, H, r"$f_\theta$", fc="#eaf1fa", ec=BLUE, fs=10.0)
box(8.90, YA, 2.30, H, "prediction", fs=7.0)
box(12.05, YA, 4.05, H, r"$\mathcal{L}_{\det}$ vs. labels $y$", fs=7.6)

seg([(2.05, CA), (2.85, CA)])
seg([(5.70, CA), (6.05, CA)], color=ORANGE)
seg([(7.95, CA), (8.90, CA)])
seg([(11.20, CA), (12.05, CA)])
ax.text(3.05, YA - 0.42, r"labels unchanged: $\mathcal{T}$ preserves box coordinates",
        fontsize=6.8, color=GREY, style="italic")

# ================= DUOI: bien the consistency =================
YT, YB = 2.05, 0.35
CT, CB = YT + H / 2, YB + H / 2
YM = 1.62
ax.add_patch(Rectangle((0.25, YB - 0.30), 16.3, (YT + H) - YB + 1.24,
                       fc="#f7f7f8", ec="#b9bec6", lw=1.0, ls=(0, (5, 3)),
                       zorder=0))
ax.text(0.45, YT + H + 0.30, "Consistency variant  (also tried; worse, and "
        r"$1.8\times$ the cost)", fontsize=8.0, color="#5a6472",
        fontweight="bold", zorder=1, va="bottom")

box(0.75, YM - 0.43, 1.30, 0.86, "batch\n$x$", fc="#f0f1f3", fs=7.4)
box(2.85, YB, 1.65, H, r"$\mathcal{T}$", fc="#fbf1ec", ec="#e0a184", fs=8.5)
box(5.30, YT, 1.55, H, r"$f_\theta$", fc="#eef3fa", ec=PALE, fs=9.2)
box(5.30, YB, 1.55, H, r"$f_\theta$", fc="#eef3fa", ec=PALE, fs=9.2)
box(7.75, YT, 2.05, H, "clean\npred.", fs=7.2)
box(7.75, YB, 2.05, H, "corrupted", fs=6.5)
box(10.45, YM - 0.33, 0.95, 0.66, "sg", fc="#fdf7e6", ec=GOLD, fs=7.8)
box(12.05, YT, 4.05, H, r"$\mathcal{L}_{\det}$ vs. $y$", fs=7.6)
box(12.05, YB, 4.05, H, r"$\mathcal{L}_{\mathrm{cons}}$" + "\n" + "cls + box + feat",
    fc="#eef3fa", ec=PALE, fs=7.0)

XS = 2.45
seg([(2.05, YM), (XS, YM), (XS, CT), (5.30, CT)], color="#7b828c")
seg([(2.05, YM), (XS, YM), (XS, CB), (2.85, CB)], color="#e0a184")
ax.plot([XS], [YM], marker="o", ms=3.2, color="#7b828c", zorder=5)
seg([(4.50, CB), (5.30, CB)], color="#e0a184")
seg([(6.85, CT), (7.75, CT)], color="#7b828c")
seg([(6.85, CB), (7.75, CB)], color="#e0a184")
seg([(9.80, CT), (12.05, CT)], color="#7b828c")
seg([(8.68, YT), (8.68, YM), (10.45, YM)], color="#7b828c")
seg([(11.40, YM), (11.75, YM), (11.75, CB + 0.24), (12.05, CB + 0.24)], color=GOLD)
seg([(9.80, CB), (12.05, CB)], color=PALE, lw=1.3)
ax.add_patch(FancyArrowPatch((6.08, YT - 0.02), (6.08, YB + H + 0.02),
                             shrinkA=0, shrinkB=0, arrowstyle="<->",
                             mutation_scale=9, color=PALE, lw=1.0,
                             ls=(0, (2, 2)), zorder=2))
ax.text(11.05, YM + 0.46, "no gradient", fontsize=6.2, color=GOLD, ha="center")

ax.text(8.4, -0.44, "Both exist only during training; either way the exported "
        "network is the unmodified baseline.",
        fontsize=7.2, ha="center", color="#3d4550", style="italic")

fig.savefig("fig2_method.pdf", bbox_inches="tight", dpi=300)
print("wrote fig2_method.pdf")
