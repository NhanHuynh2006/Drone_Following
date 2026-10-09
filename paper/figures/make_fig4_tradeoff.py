"""
Figure 4. Clean accuracy (truc x) vs robustness giu duoc (truc y), moi model mot diem.

Day la hinh mang LAP LUAN TRUNG TAM cua bai bao, thay cho vai doan van:
  - Cac baseline nam ben PHAI (clean AP cao) nhung DUOI (retain thap) -> tien bo ve
    accuracy khong mua duoc robustness. retain KHONG phai ham don dieu cua clean AP.
  - Mui nhui = tac dung cua vibration-aware training: day model LEN CAO (robust hon) va
    hoi sang TRAI (mat it clean AP). Doc gia thay ngay ty gia trao doi.
  - Hai mui nhui song song tren hai kien truc khac nhau = bang chung generality.
  - Hai kim cuong CHONG LEN NHAU = cung che do do ap len YOLO11-n thi khong nhuc nhich.

VE DUNG CO IN (sua 2026-07-27b, sau khi compile that):
  Ban truoc ve o 6.8 inch roi in ra 10.2 cm -> bop con 59% -> moi chu con 4.3-5.5pt
  tren giay, khong doc noi va chong len nhau. Gio ve o 5.2 inch, in o 0.95\\textwidth
  (11.6 cm) -> ti le ~88%, chu 8.5pt thanh ~7.5pt tren giay. Doc duoc.
  QUY TAC: figsize (inch) x 2.54 phai XAP XI be rong in (cm). Dung ve to roi bop nho.

Bon con YOLO nano gio dan NHAN CHUNG mot lan thay vi bon nhan rieng: thong diep cua
hinh la "chung no dinh mot cuc", dan ten tung con la lam nguoc lai thong diep do.

So doc TU DOC tu runs/paper_queue/UNIFIED.json.
Chay: python make_fig4_tradeoff.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.abspath(os.path.join(HERE, "..", "..", "runs", "paper_queue", "UNIFIED.json"))
OUT = os.path.join(HERE, "fig4_tradeoff.pdf")

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#2b2b2b", "#6b7480"

M = json.load(open(JSON))["models"]
xy = lambda n: (M[n]["clean"]["AP"], M[n]["agg"]["retain_pct"]["mean"])
err = lambda n: M[n]["agg"]["retain_pct"]["std"]

YOLOS = [k for k in ("yolo11n", "yolov10n", "yolov8n", "yolo26n") if k in M]
# (baseline, +vib, nhan, marker, lech y cua nhan, canh le)
# SteadyDet-W dat nhan sang TRAI: ben phai la duong dan cua ghi chu YOLO11-n di qua.
PAIRS = [
    ("v17-baseline", "v17-vib-augonly", "SteadyDet-N", "s", 1.6, "left"),
    ("v15-baseline", "v15-augonly",     "SteadyDet-W", "^", -1.5, "right"),
]
NEG = [("v17-vibcons", "s"), ("v15-vibcons", "^")]

# ve o gan dung co in -> khong bi bop nho -> chu doc duoc
fig, ax = plt.subplots(figsize=(5.2, 3.0))
F_LAB, F_LEG, F_AX = 8.5, 7.8, 9.5

# ---- 4 con YOLO nano: ve rieng, dan nhan CHUNG mot lan ----
for k in YOLOS:
    x, y = xy(k)
    ax.errorbar(x, y, yerr=err(k), fmt="o", ms=5.5, mfc="white", mec=INK,
                mew=1.2, ecolor=MUTED, elinewidth=0.8, capsize=1.8, zorder=3)
if YOLOS:
    xs = [xy(k)[0] for k in YOLOS]
    ys = [xy(k)[1] for k in YOLOS]
    ax.annotate("4 YOLO nano\ngenerations",
                xy=(max(xs), min(ys) - 0.6), xytext=(0.288, 15.4),
                fontsize=F_LAB, color=INK, ha="right", va="center", linespacing=1.15,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2,
                                shrinkB=3, alpha=0.65))

# ---- RT-DETR: dan ten rieng vi no la paradigm khac ----
if "rtdetr-l" in M:
    x, y = xy("rtdetr-l")
    ax.errorbar(x, y, yerr=err("rtdetr-l"), fmt="o", ms=5.5, mfc="white", mec=INK,
                mew=1.2, ecolor=MUTED, elinewidth=0.8, capsize=1.8, zorder=3)
    ax.annotate("RT-DETR-l\n(32.8M)", (x + 0.004, y + 0.4), fontsize=F_LAB,
                color=INK, ha="left", va="center", linespacing=1.15)

# ---- model cua ta: mui nhui baseline -> +vib-aware ----
for kb, kv, lab, mk, ldy, ha in PAIRS:
    if kb not in M or kv not in M:
        continue
    x0, y0 = xy(kb)
    x1, y1 = xy(kv)
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.5,
                                shrinkA=6, shrinkB=7, alpha=0.9), zorder=2)
    ax.errorbar(x0, y0, yerr=err(kb), fmt=mk, ms=6.2, mfc="white", mec=ORANGE,
                mew=1.6, ecolor=ORANGE, elinewidth=0.8, capsize=1.8, zorder=4)
    ax.errorbar(x1, y1, yerr=err(kv), fmt=mk, ms=6.2, color=BLUE,
                ecolor=BLUE, elinewidth=0.8, capsize=1.8, zorder=4)
    dx = 0.003 if ha == "left" else -0.004
    ax.annotate(lab, (x0 + dx, y0 - ldy), fontsize=F_LAB, color=ORANGE,
                ha=ha, va="center", fontweight="bold")

# ---- ablation AM: consistency, mo hon, nam duoi dau mui nhui ----
for k, mk in NEG:
    if k not in M:
        continue
    x, y = xy(k)
    ax.errorbar(x, y, yerr=err(k), fmt=mk, ms=5.4, mfc="#cfdcf0", mec=BLUE,
                mew=1.1, ecolor="#9db8d8", elinewidth=0.7, capsize=1.8, zorder=3)

# ---- KET QUA AM: YOLO11-n truoc va sau cung che do -> hai diem CHONG NHAU ----
if "yolo11n-scratch" in M and "yolo11n-scratch-aug" in M:
    x0, y0 = xy("yolo11n-scratch")
    x1, y1 = xy("yolo11n-scratch-aug")
    ax.errorbar([x0, x1], [y0, y1],
                yerr=[err("yolo11n-scratch"), err("yolo11n-scratch-aug")],
                fmt="D", ms=4.8, mfc="white", mec=MUTED, mew=1.1,
                ecolor=MUTED, elinewidth=0.8, capsize=1.8, ls="none", zorder=3)
    ax.annotate("YOLO11-n, same regime:\nbefore and after coincide",
                xy=(min(x0, x1), min(y0, y1) - 0.7), xytext=(0.1985, 12.7),
                fontsize=F_LAB, color=MUTED, ha="left", va="center", linespacing=1.2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7, shrinkA=2,
                                shrinkB=4, alpha=0.65))

ax.set_xlabel("clean AP$_{50:95}$  (higher is better)", fontsize=F_AX)
ax.set_ylabel("robustness retained (%)", fontsize=F_AX)
ax.tick_params(labelsize=F_AX - 1.0)
ax.grid(alpha=0.2, lw=0.5)
ax.set_xlim(0.163, 0.293)
ax.set_ylim(10.5, 43.5)          # noi tren cho chu giai, noi duoi cho 2 ghi chu
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# chu giai o goc TREN-PHAI (vung trong nhat), ma hoa bang HINH DANG lan mau
h = [plt.Line2D([], [], ls="none", marker="o", ms=5.5, mfc="white", mec=INK,
                mew=1.2, label="published detector"),
     plt.Line2D([], [], ls="none", marker="s", ms=6.2, mfc="white", mec=ORANGE,
                mew=1.6, label="ours, baseline"),
     plt.Line2D([], [], ls="none", marker="s", ms=6.2, color=BLUE,
                label="ours, + vibration-aware"),
     plt.Line2D([], [], ls="none", marker="s", ms=5.4, mfc="#cfdcf0", mec=BLUE,
                mew=1.1, label="ours, consistency variant")]
ax.legend(handles=h, fontsize=F_LEG, frameon=False, loc="upper right",
          labelspacing=0.35, borderaxespad=0.3, handletextpad=0.5)

fig.savefig(OUT, bbox_inches="tight", dpi=300)
print("wrote", OUT)
for k in YOLOS + ["rtdetr-l"] + [x for p in PAIRS for x in p[:2]]:
    if k in M:
        x, y = xy(k)
        print(f"   {k:20s} clean AP={x:.4f}  retain={y:.1f} +/- {err(k):.1f}")
