"""
Figure 4. Detections on real flight footage (2026-07-21, approx. 6 m altitude),
frames sampled at 5 s, 25 s and 40 s of an autonomous following run.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

files = [("demo_t5s.jpg", "t = 5 s"), ("demo_t25s.jpg", "t = 25 s"),
         ("demo_t40s.jpg", "t = 40 s")]

fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.95))
for ax, (f, cap) in zip(axes, files):
    ax.imshow(Image.open(f).convert("RGB"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(cap, fontsize=9, pad=3)
    for s in ax.spines.values():
        s.set_linewidth(0.6); s.set_color("0.35")
plt.subplots_adjust(wspace=0.04)
fig.savefig("fig4_qualitative.pdf", bbox_inches="tight", dpi=220)
print("wrote fig4_qualitative.pdf")
