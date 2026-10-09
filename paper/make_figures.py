"""
make_figures.py — sinh 3 hình cho bài báo, xuat ra figures/*.pdf (vector).
Chay:  python3 make_figures.py
So lieu lay tu runs/paper_queue/SUMMARY.txt va RESULTS_ROBUSTNESS.md
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
})

INK = "#1a1a1a"
BASE = "#b0b0b0"
OURS = "#c1121f"
ACC = "#2a6f97"
LIGHT = "#e8e8e8"

# =====================================================================
# FIGURE 1 — TEASER: "what did you do"
# =====================================================================
sev = [0, 1, 2, 3]
v17_base = [0.1892, 0.0803, 0.0111, 0.0057]
v17_vib = [0.1746, 0.1070, 0.0432, 0.0074]
v15_base = [0.235, 0.122, 0.033, 0.003]
v15_vib = [0.221, 0.145, 0.071, 0.011]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.35),
                               gridspec_kw={"width_ratios": [1.35, 1]})

ax1.axvspan(1.5, 3.4, color=LIGHT, zorder=0)
ax1.text(2.45, 0.207, "detector effectively blind", ha="center",
         va="center", fontsize=7, color="#666666", style="italic")

ax1.plot(sev, v17_base, "o--", color=BASE, ms=4, lw=1.4,
         label="baseline", zorder=3)
ax1.plot(sev, v17_vib, "o-", color=OURS, ms=4, lw=1.8,
         label="+ vibration-consistency (ours)", zorder=4)

ax1.annotate("", xy=(2, v17_vib[2]), xytext=(2, v17_base[2]),
             arrowprops=dict(arrowstyle="<->", color=ACC, lw=1.0))
ax1.text(2.1, 0.027, r"$3.9\times$", color=ACC, fontsize=8, fontweight="bold")

ax1.set_xticks(sev)
ax1.set_xticklabels(["clean", "sev1", "sev2", "sev3"])
ax1.set_xlabel("vibration severity")
ax1.set_ylabel(r"AP$_{50:95}$")
ax1.set_xlim(-0.25, 3.4)
ax1.set_ylim(0, 0.225)
ax1.legend(frameon=False, fontsize=7, loc="upper right")
ax1.spines[["top", "right"]].set_visible(False)
ax1.set_title("(a) accuracy collapses under vibration", fontsize=8, pad=6)

labels = ["v17", "v15"]
base_r = [17.1, 22.4]
ours_r = [30.1, 34.3]
x = np.arange(2)
w = 0.33
b1 = ax2.bar(x - w/2, base_r, w, color=BASE, label="baseline")
b2 = ax2.bar(x + w/2, ours_r, w, color=OURS, label="ours")
for b in list(b1) + list(b2):
    ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.9,
             f"{b.get_height():.1f}", ha="center", fontsize=7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels)
ax2.set_ylabel("retained AP (\\%)")
ax2.set_ylim(0, 42)
ax2.legend(frameon=False, fontsize=7, loc="upper left")
ax2.spines[["top", "right"]].set_visible(False)
ax2.set_title("(b) same gain on two architectures", fontsize=8, pad=6)

fig.tight_layout()
fig.savefig("figures/fig1_teaser.pdf", bbox_inches="tight")
plt.close(fig)
print("figures/fig1_teaser.pdf")

# =====================================================================
# FIGURE 2 — METHOD: "how did you do it"
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 2.6))
ax.set_xlim(0, 10.6)
ax.set_ylim(0, 4.0)
ax.axis("off")


def box(x, y, w, h, text, fc="white", ec=INK, ls="-", fs=7.5, lw=0.9, tc=INK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.045,rounding_size=0.09",
                       fc=fc, ec=ec, ls=ls, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, color=tc)


def arrow(x1, y1, x2, y2, color=INK, ls="-", lw=1.0, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=9, color=color, ls=ls, lw=lw,
                                 zorder=2, shrinkA=1, shrinkB=1,
                                 connectionstyle=f"arc3,rad={rad}"))


# --- training-only region ---
ax.add_patch(Rectangle((0.12, 0.15), 8.1, 3.55, fc="#fafafa", ec="#bbbbbb",
                       ls=(0, (4, 3)), lw=0.9, zorder=0))
ax.text(0.26, 3.52, "training graph only", fontsize=7, style="italic",
        color="#777777")

# --- single input, then split ---
box(0.32, 1.75, 1.05, 0.6, "batch $x$")
arrow(1.37, 2.05, 1.85, 2.85, rad=-0.25)          # up to clean branch
arrow(1.37, 2.05, 1.85, 1.15, rad=0.25)           # down to vibration

# --- clean (teacher) branch ---
box(1.95, 2.60, 1.55, 0.6, "$f_\theta$", fc="#eef3f7", ec=ACC)
arrow(3.50, 2.90, 4.15, 2.90)
box(4.15, 2.60, 1.95, 0.6, "clean prediction", fc="#eef3f7", ec=ACC)
ax.text(5.12, 3.32, "detached (stop-grad)", fontsize=6.6, color=ACC,
        ha="center")

# --- corrupted (student) branch ---
box(1.95, 0.85, 0.75, 0.6, "$\mathcal{T}$", fc="#fdeeee", ec=OURS, fs=9)
ax.text(2.33, 0.62, "vibration", fontsize=6.4, ha="center", color=OURS)
arrow(2.70, 1.15, 3.05, 1.15)
box(3.05, 0.85, 1.55, 0.6, "$f_\theta$  (same $\theta$)", fc="#fdeeee", ec=OURS,
    fs=7.0)
arrow(4.60, 1.15, 5.05, 1.15)
box(5.05, 0.85, 1.75, 0.6, "corrupted pred.", fc="#fdeeee", ec=OURS)

# --- losses ---
box(7.05, 2.60, 1.0, 0.6, "$\mathcal{L}_{\det}$", fc="white", fs=8.5)
arrow(6.10, 2.90, 7.05, 2.90)

box(7.05, 1.35, 1.0, 0.85, "$\mathcal{L}_{\mathrm{cons}}$",
    fc="#fff8e6", ec="#b8860b", fs=8.5)
arrow(6.10, 2.62, 7.10, 2.24, color="#b8860b")
arrow(6.80, 1.30, 7.20, 1.34, color="#b8860b")
ax.text(7.55, 0.95,
        "$\mathcal{L}_{\mathrm{cls}}$ soft, $T{=}2$\n"
        "$\mathcal{L}_{\mathrm{box}}$ conf-gated\n"
        "$\mathcal{L}_{\mathrm{feat}}$ neck",
        fontsize=6.6, va="top", ha="center", color="#8a6508")

# --- deployment ---
ax.add_patch(Rectangle((8.55, 0.15), 1.9, 3.55, fc="#f2f7f2", ec="#4a7c59",
                       lw=0.9, zorder=0))
ax.text(9.50, 3.52, "at inference", fontsize=7, style="italic",
        color="#4a7c59", ha="center")
box(8.95, 1.95, 1.05, 0.7, "$f_\theta$", fc="white", ec="#4a7c59", fs=9)
ax.text(9.50, 1.72,
        "one forward pass\nidentical graph\nidentical latency",
        fontsize=6.7, ha="center", va="top", color="#4a7c59")

fig.savefig("figures/fig2_method.pdf", bbox_inches="tight")
plt.close(fig)
print("figures/fig2_method.pdf")

# =====================================================================
# FIGURE 3 — ARCHITECTURE
# =====================================================================
fig, ax = plt.subplots(figsize=(6.6, 2.85))
ax.set_xlim(0, 11.4)
ax.set_ylim(0, 5.05)
ax.axis("off")

rows = [
    ("$P_2$", 3.45, "s4,  $160^2$", "48", "#fdeeee", OURS, "1$\\times$ UIB"),
    ("$P_3$", 2.52, "s8,  $80^2$", "96", "white", INK, "CSP $\\times$2"),
    ("$P_4$", 1.59, "s16,  $40^2$", "144", "white", INK, "CSP $\\times$2"),
    ("$P_5$", 0.66, "s32,  $20^2$", "192", "white", INK,
     "CSP $\\times$2 $+$ attn."),
]

# stem
box(1.35, 4.20, 1.65, 0.5, "dual-path stem", fc="#eef3f7", ec=ACC, fs=7.2)
ax.text(0.62, 4.45, "input\n$640^2$", ha="center", va="center", fontsize=7.2)
arrow(1.05, 4.45, 1.35, 4.45)
arrow(2.17, 4.20, 2.17, 3.95)

for name, y, res, ch, fc, ec, blab in rows:
    ax.text(0.30, y + 0.32, name, fontsize=8, fontweight="bold", color=ec)
    ax.text(0.30, y + 0.08, res, fontsize=6.2, color="#666")
    box(1.35, y, 2.45, 0.5, blab, fc=fc, ec=ec, fs=7.2)
    ax.text(3.92, y + 0.25, ch + "c", fontsize=6.4, color="#666", va="center")
    box(4.70, y, 1.65, 0.5, "$1{\\times}1$ lat. $+$ UIB", fc=fc, ec=ec, fs=6.8)
    box(7.20, y, 1.45, 0.5, "LiteHead", fc=fc, ec=ec, fs=7.2)
    arrow(3.80, y + 0.25, 4.70, y + 0.25)
    arrow(6.35, y + 0.25, 7.20, y + 0.25)
    arrow(8.65, y + 0.25, 9.25, y + 0.25)
    ax.text(9.34, y + 0.25, "cls $+$ box", fontsize=6.8, va="center")

for y0, y1 in [(3.45, 2.52), (2.52, 1.59), (1.59, 0.66)]:
    arrow(2.17, y0, 2.17, y1 + 0.5)
for y0, y1 in [(0.66, 1.59), (1.59, 2.52), (2.52, 3.45)]:
    arrow(5.52, y0 + 0.5, 5.52, y1, color=ACC, ls="--")

ax.text(2.17, 4.02, "backbone", fontsize=6.8, ha="center", color="#666",
        va="bottom")
ax.text(5.52, 4.02, "top-down neck", fontsize=6.8, ha="center", color=ACC,
        va="bottom")
ax.text(7.92, 4.02, "heads", fontsize=6.8, ha="center", color="#666",
        va="bottom")

ax.add_patch(FancyBboxPatch((3.55, 4.28), 7.4, 0.4,
             boxstyle="round,pad=0.04,rounding_size=0.06",
             fc="#fdeeee", ec=OURS, lw=0.8, zorder=1))
ax.text(7.25, 4.48,
        "$P_2$ retained but made cheap: one block, half neck width, "
        "separable head",
        fontsize=7.0, ha="center", va="center", color=OURS, zorder=2)

fig.savefig("figures/fig3_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print("figures/fig3_architecture.pdf")
