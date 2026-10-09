"""
Figure 1 (teaser).
  Trai : mot frame BAY THAT (co nguoi ro rang, kem box GT) duoi 4 muc VisDrone-Shake.
  Phai : phan tram AP GIU DUOC theo muc rung, cho MOI ho detector.

Vi sao truc phai la "% giu duoc" chu khong phai AP tuyet doi: clean AP cua cac model
lech nhau nhieu (0.19 den 0.28), ve AP tuyet doi thi doc gia thay YOLO nam tren va
mat hoan toan thong diep "ai cung sup". Chuan hoa ve 100% tai clean lam moi duong
xuat phat cung mot cho -> so sanh duoc do DOC cua su sup do, dung la dai luong bai bao noi.

So do TU DOC tu runs/paper_queue/UNIFIED.json -> hinh khong bao gio lech voi bang.
Corruption tai dung _PRESETS cua vib_consistency.corrupt_batch.

Chay:  python make_fig1_teaser.py
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
JSON = os.path.join(ROOT, "runs", "paper_queue", "UNIFIED.json")

# frame bay that co nguoi TO nhat trong data/flight_val (box 204x248 px)
SRC = os.path.join(ROOT, "data", "flight_val", "images", "flight_0001.jpg")
LBL = os.path.join(ROOT, "data", "flight_val", "labels", "flight_0001.txt")
FALLBACK = os.path.join(HERE, "sample_frame.jpg")
OUT = "fig1_teaser.pdf"

BLUE, ORANGE, DARK = "#2a78d6", "#eb6834", "#2b2b2b"
GREY, BAND = "#8a8f98", "#d9dbde"

PRESETS = {1: (5, 0.02, 0.10), 2: (9, 0.04, 0.20), 3: (15, 0.06, 0.30)}
RNG = np.random.default_rng(0)


# ------------------------------------------------------------------ corruption
def motion_blur_kernel(length, angle_deg):
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2.0
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    for t in range(length):
        x = int(round(c + (t - c) * dx))
        y = int(round(c + (t - c) * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    return k / k.sum()


def corrupt(img, severity, angle=35.0):
    L, ns, br = PRESETS[severity]
    k = motion_blur_kernel(L, angle)
    pad = k.shape[0] // 2
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(img)
    for dy in range(k.shape[0]):
        for dx in range(k.shape[1]):
            w = k[dy, dx]
            if w:
                out += w * p[dy:dy + img.shape[0], dx:dx + img.shape[1]]
    out = out + RNG.normal(0, ns, out.shape)
    return np.clip(out * (1.0 + br * 0.6), 0, 1)


# ------------------------------------------------------------------ so lieu
M = json.load(open(JSON))["models"]


def retain_curve(name):
    """[100, %sev1, %sev2, %sev3] — phan tram AP giu duoc so voi clean."""
    r = M[name]
    c = r["clean"]["AP"]
    return [100.0] + [100.0 * r["agg"][f"sev{s}"]["mean"] / c for s in (1, 2, 3)]


YOLOS = [n for n in ("yolov8n", "yolo11n", "yolo26n", "yolov10n") if n in M]
sev = [0, 1, 2, 3]

# ------------------------------------------------------------------ ve
# VE DUNG CO IN (sua 2026-07-27b): ban truoc ve 11 inch roi in ra 10.7 cm -> bop
# con 48% -> chu 7.4pt thanh 3.6pt tren giay, khong doc noi. Gio ve ~5.6 inch de in
# gan 1:1. QUY TAC: be rong PDF (inch) x 2.54 phai XAP XI be rong in (cm).
fig = plt.figure(figsize=(5.8, 2.32))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.80, 1.22], wspace=0.34)

# ---- trai: dai anh 4 muc rung, co nguoi + box GT ----
gl = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.04)
path = SRC if os.path.exists(SRC) else FALLBACK
im = Image.open(path).convert("RGB")
W, H = im.size
gt = None
if os.path.exists(LBL):
    p = open(LBL).read().split()
    if len(p) >= 5:
        gt = [float(v) for v in p[1:5]]           # cx,cy,w,h chuan hoa
# crop VUONG theo chieu cao, canh giua nguoi -> nguoi chiem ~1/3 be ngang
side = min(W, H)
cx = gt[0] * W if gt else W / 2
left = int(max(0, min(W - side, cx - side / 2)))
im = im.crop((left, 0, left + side, side)).resize((380, 380), Image.LANCZOS)
img = np.asarray(im).astype(np.float32) / 255.0
if gt:
    bx = (gt[0] * W - left) / side * 380
    by = gt[1] * H / side * 380
    bw, bh = gt[2] * W / side * 380, gt[3] * H / side * 380

for i, t in enumerate(["clean", "sev1", "sev2", "sev3"]):
    ax = fig.add_subplot(gl[i])
    ax.imshow(img if i == 0 else corrupt(img, i))
    if gt:
        ax.add_patch(plt.Rectangle((bx - bw / 2, by - bh / 2), bw, bh,
                                   fill=False, ec="#19c37d", lw=1.5))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(t, fontsize=8, pad=2)
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("0.35")

# ---- phai: duong giu duoc AP ----
ax = fig.add_subplot(gs[1])
if YOLOS:
    cur = np.array([retain_curve(n) for n in YOLOS])
    ax.fill_between(sev, cur.min(0), cur.max(0), color=BAND, zorder=1)
    ax.plot(sev, cur.mean(0), color=GREY, lw=1.3, ls=(0, (4, 2)), zorder=2,
            marker="^", ms=4.0, label="YOLO-n ($%d$ models)" % len(YOLOS))
if "rtdetr-l" in M:
    ax.plot(sev, retain_curve("rtdetr-l"), color=DARK, lw=1.3, ls=(0, (1, 1.6)),
            marker="D", ms=3.6, zorder=3, label="RT-DETR-l (32.8M)")
ax.plot(sev, retain_curve("v17-baseline"), color=ORANGE, lw=2.0, ls="-",
        marker="o", ms=5.0, zorder=4, label="SteadyDet-N")
# DOI KHUNG 2026-07-26: duong CHINH gio la augmentation (36.2%), consistency (30.0%)
# ve mo hon vi da thanh ablation AM.
if "v17-vibcons" in M:
    ax.plot(sev, retain_curve("v17-vibcons"), color="#9db8d8", lw=1.4, ls=(0, (2, 1.4)),
            marker="s", ms=3.6, zorder=4, label="  + consistency")
ax.plot(sev, retain_curve("v17-vib-augonly"), color=BLUE, lw=2.2, ls=(0, (5, 1.6)),
        marker="s", ms=5.2, zorder=6, label="  + vib.-aware")

b2 = retain_curve("v17-baseline")[2]
v2 = retain_curve("v17-vib-augonly")[2]
ax.annotate("", xy=(2, v2), xytext=(2, b2),
            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.1,
                            shrinkA=1.5, shrinkB=1.5))
# Ghi thang GIA TRI, khong ghi ti so: truc y la retention chuan hoa nen ti so o day
# (22.5/6.5 = 3.5x) KHAC ti so AP tuyet doi ma bai bao trich (0.0392/0.0123 = 3.2x).
# Hai con so khac nhau cho cung mot viec se bi reviewer hoi -> chi ghi 2 gia tri.
ax.annotate("%.1f%% $\\rightarrow$ %.1f%%" % (b2, v2),
            xy=(2.04, (b2 + v2) / 2), xytext=(2.52, 56),
            color=BLUE, fontsize=7.6, fontweight="bold", va="center", ha="center",
            arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.8, alpha=0.75,
                            shrinkA=3, shrinkB=2),
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=BLUE, lw=0.7, alpha=0.95))

ax.set_xticks(sev)
ax.set_xticklabels(["clean", "sev1", "sev2", "sev3"], fontsize=7.6)
ax.set_ylabel("% of clean AP retained", fontsize=8.0)
ax.set_xlabel("vibration severity", fontsize=8.0)
ax.set_ylim(0, 105)
ax.set_xlim(-0.15, 3.30)   # cho nhan "6% -> 30%" o x=2.12 khong bi cat
ax.tick_params(labelsize=7.6)
ax.grid(alpha=0.22, lw=0.5)
ax.legend(fontsize=6.4, frameon=False, loc="upper right",
          handlelength=1.5, borderaxespad=0.0, labelspacing=0.22,
          handletextpad=0.35, bbox_to_anchor=(1.02, 1.04))
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

fig.savefig(OUT, bbox_inches="tight", dpi=300)
print("wrote", OUT, "| frame:", os.path.basename(path))
for n in ["v17-baseline", "v17-vibcons", "v17-vib-augonly"] + YOLOS + (["rtdetr-l"] if "rtdetr-l" in M else []):
    print(f"   {n:14s} retain curve = {[round(v,1) for v in retain_curve(n)]}")
