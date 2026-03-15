"""
PFDet-Nano v5 Architecture Diagram Generator
Creates a publication-quality architecture diagram.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(-1, 21)
ax.set_ylim(-1, 13)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')

# ── Color scheme ──
C_INPUT  = '#E8F5E9'   # light green
C_CONV   = '#BBDEFB'   # light blue
C_IR     = '#C8E6C9'   # green
C_LAT    = '#FFF9C4'   # yellow
C_FPN    = '#FFCCBC'   # orange
C_PAN    = '#E1BEE7'   # purple
C_HEAD   = '#FFCDD2'   # red
C_OUT    = '#F5F5F5'   # gray
BORDER   = '#37474F'

def draw_block(x, y, w, h, text, color, fontsize=7, bold=False, sub_text=None):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor=BORDER, linewidth=1.2)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2 + (0.12 if sub_text else 0), text,
            ha='center', va='center', fontsize=fontsize, fontweight=weight, color='#212121')
    if sub_text:
        ax.text(x + w/2, y + h/2 - 0.18, sub_text,
                ha='center', va='center', fontsize=5.5, color='#616161', style='italic')

def draw_arrow(x1, y1, x2, y2, color='#546E7A', style='->', lw=1.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def draw_curved_arrow(x1, y1, x2, y2, color='#546E7A', connectionstyle="arc3,rad=0.3"):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.2,
                              connectionstyle=connectionstyle))

# ═══════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════
ax.text(10, 12.5, 'PFDet-Nano v5 Architecture', ha='center', va='center',
        fontsize=18, fontweight='bold', color='#1A237E')
ax.text(10, 12.0, 'Lightweight Anchor-Free Person Detector  |  ~900K params  |  Input: 416×416',
        ha='center', va='center', fontsize=9, color='#455A64')

# ═══════════════════════════════════════════════════
# BACKBONE (left column, bottom to top)
# ═══════════════════════════════════════════════════
bw, bh = 2.2, 0.7  # block width, height
bx = 0.5            # backbone x position

# Section label
ax.text(bx + bw/2, 11.2, '── BACKBONE ──', ha='center', fontsize=9,
        fontweight='bold', color='#1B5E20')

# Input
draw_block(bx, 10.2, bw, bh, 'Input', C_INPUT, 8, True, '416×416×3')

# Stem
draw_block(bx, 9.1, bw, bh, 'Stem', C_CONV, 8, True, 'Conv3×3/2 → Conv3×3')
ax.text(bx + bw + 0.15, 9.1 + bh/2, '208² × 16', fontsize=6, color='#757575', va='center')
draw_arrow(bx + bw/2, 10.2, bx + bw/2, 9.1 + bh)

# Stage 1
draw_block(bx, 8.0, bw, bh, 'Stage 1', C_IR, 8, True, 'Conv/2 + IR×1')
ax.text(bx + bw + 0.15, 8.0 + bh/2, '104² × 32', fontsize=6, color='#757575', va='center')
draw_arrow(bx + bw/2, 9.1, bx + bw/2, 8.0 + bh)

# Stage 2 (P3)
draw_block(bx, 6.9, bw, bh, 'Stage 2', C_IR, 8, True, 'Conv/2 + IR×2')
ax.text(bx + bw + 0.15, 6.9 + bh/2, '52² × 64', fontsize=6, color='#757575', va='center')
ax.text(bx - 0.15, 6.9 + bh/2, 'P3', fontsize=8, color='#E65100', fontweight='bold',
        ha='right', va='center')
draw_arrow(bx + bw/2, 8.0, bx + bw/2, 6.9 + bh)

# Stage 3 (P4)
draw_block(bx, 5.8, bw, bh, 'Stage 3', C_IR, 8, True, 'Conv/2 + IR×2')
ax.text(bx + bw + 0.15, 5.8 + bh/2, '26² × 128', fontsize=6, color='#757575', va='center')
ax.text(bx - 0.15, 5.8 + bh/2, 'P4', fontsize=8, color='#E65100', fontweight='bold',
        ha='right', va='center')
draw_arrow(bx + bw/2, 6.9, bx + bw/2, 5.8 + bh)

# Stage 4 (P5)
draw_block(bx, 4.7, bw, bh, 'Stage 4', C_IR, 8, True, 'Conv/2 + IR×2')
ax.text(bx + bw + 0.15, 4.7 + bh/2, '13² × 192', fontsize=6, color='#757575', va='center')
ax.text(bx - 0.15, 4.7 + bh/2, 'P5', fontsize=8, color='#E65100', fontweight='bold',
        ha='right', va='center')
draw_arrow(bx + bw/2, 5.8, bx + bw/2, 4.7 + bh)

# ═══════════════════════════════════════════════════
# InvertedResidual detail box
# ═══════════════════════════════════════════════════
irx, iry = 0.2, 2.5
irw, irh = 3.0, 1.8
box = FancyBboxPatch((irx, iry), irw, irh, boxstyle="round,pad=0.1",
                     facecolor='#F1F8E9', edgecolor='#33691E', linewidth=1.5, linestyle='--')
ax.add_patch(box)
ax.text(irx + irw/2, iry + irh - 0.15, 'InvertedResidual (IR)', ha='center',
        fontsize=8, fontweight='bold', color='#33691E')

# IR sub-blocks
sub_w, sub_h = 0.75, 0.4
sx = irx + 0.2
draw_block(sx, iry + 0.85, sub_w, sub_h, 'Expand', '#DCEDC8', 6, sub_text='1×1')
draw_block(sx + sub_w + 0.15, iry + 0.85, sub_w + 0.1, sub_h, 'DW Conv', '#DCEDC8', 6, sub_text='3×3')
draw_block(sx + 2*(sub_w + 0.15) - 0.05, iry + 0.85, sub_w, sub_h, 'Project', '#DCEDC8', 6, sub_text='1×1')

draw_arrow(sx + sub_w, iry + 0.85 + sub_h/2, sx + sub_w + 0.15, iry + 0.85 + sub_h/2)
draw_arrow(sx + 2*sub_w + 0.15 + 0.1, iry + 0.85 + sub_h/2, sx + 2*(sub_w + 0.15) - 0.05, iry + 0.85 + sub_h/2)

# Residual connection
ax.text(irx + irw/2, iry + 0.3, '+ skip connection (residual)', ha='center',
        fontsize=6.5, color='#558B2F', style='italic')

# ═══════════════════════════════════════════════════
# FPN - Top-Down (middle column)
# ═══════════════════════════════════════════════════
fx = 5.0
ax.text(fx + bw/2, 11.2, '── FPN (Top-Down) ──', ha='center', fontsize=9,
        fontweight='bold', color='#BF360C')

# Lat5
draw_block(fx, 4.7, bw, bh, 'Lat5', C_LAT, 8, True, 'Conv 1×1: 192→64')
draw_arrow(bx + bw, 4.7 + bh/2, fx, 4.7 + bh/2, color='#E65100')

# Lat4
draw_block(fx, 5.8, bw, bh, 'Lat4', C_LAT, 8, True, 'Conv 1×1: 128→64')
draw_arrow(bx + bw, 5.8 + bh/2, fx, 5.8 + bh/2, color='#E65100')

# Lat3
draw_block(fx, 6.9, bw, bh, 'Lat3', C_LAT, 8, True, 'Conv 1×1: 64→64')
draw_arrow(bx + bw, 6.9 + bh/2, fx, 6.9 + bh/2, color='#E65100')

# Upsample + Add → Smooth4 (P4_FPN)
draw_block(fx, 8.3, bw, 0.5, '↑ Upsample 2×', '#FFF3E0', 7)
draw_arrow(fx + bw/2, 4.7 + bh, fx + bw/2, 8.3, color='#E65100')  # from Lat5 up

draw_block(fx, 9.1, bw, bh, 'FPN Smooth4', C_FPN, 8, True, 'Add + Conv 3×3')
ax.text(fx + bw + 0.15, 9.1 + bh/2, 'P4_fpn: 26²×64', fontsize=6, color='#757575', va='center')
draw_arrow(fx + bw/2, 8.3 + 0.5, fx + bw/2, 9.1 + bh)
# Lat4 → Smooth4
draw_curved_arrow(fx + bw/2 + 0.3, 5.8 + bh, fx + bw/2 + 0.3, 9.1,
                  color='#E65100', connectionstyle="arc3,rad=-0.4")

# Upsample + Add → Smooth3 (P3_FPN)
draw_block(fx, 10.2, bw, bh, 'FPN Smooth3', C_FPN, 8, True, 'Add + Conv 3×3')
ax.text(fx + bw + 0.15, 10.2 + bh/2, 'P3_fpn: 52²×64', fontsize=6, color='#757575', va='center')
draw_arrow(fx + bw/2, 9.1, fx + bw/2, 10.2 + bh)
# Lat3 → Smooth3
draw_curved_arrow(fx + bw/2 - 0.3, 6.9 + bh, fx + bw/2 - 0.3, 10.2,
                  color='#E65100', connectionstyle="arc3,rad=0.5")

# ═══════════════════════════════════════════════════
# PAN - Bottom-Up (right-middle column)
# ═══════════════════════════════════════════════════
px = 9.5
ax.text(px + bw/2, 11.2, '── PAN (Bottom-Up) ──', ha='center', fontsize=9,
        fontweight='bold', color='#6A1B9A')

# P3_fpn output → goes to head directly
# P3_fpn also goes down via pan_down3

# PAN Down3 (stride 2: P3→P4)
draw_block(px, 9.1, bw, bh, 'PAN Down3', C_PAN, 8, True, 'Conv 3×3/2')
draw_arrow(fx + bw, 10.2 + bh/2, px, 9.1 + bh/2 + 0.2, color='#6A1B9A')

# PAN Smooth4
draw_block(px, 7.8, bw, bh, 'PAN Smooth4', C_PAN, 8, True, 'Add + Conv 3×3')
ax.text(px + bw + 0.15, 7.8 + bh/2, 'P4_pan: 26²×64', fontsize=6, color='#757575', va='center')
draw_arrow(px + bw/2, 9.1, px + bw/2, 7.8 + bh)
# P4_fpn → PAN Smooth4
draw_curved_arrow(fx + bw, 9.1 + bh/2 - 0.15, px, 7.8 + bh/2,
                  color='#6A1B9A', connectionstyle="arc3,rad=0.3")

# PAN Down4 (stride 2: P4→P5)
draw_block(px, 6.5, bw, bh, 'PAN Down4', C_PAN, 8, True, 'Conv 3×3/2')
draw_arrow(px + bw/2, 7.8, px + bw/2, 6.5 + bh)

# PAN Smooth5
draw_block(px, 5.2, bw, bh, 'PAN Smooth5', C_PAN, 8, True, 'Add + Conv 3×3')
ax.text(px + bw + 0.15, 5.2 + bh/2, 'P5_pan: 13²×64', fontsize=6, color='#757575', va='center')
draw_arrow(px + bw/2, 6.5, px + bw/2, 5.2 + bh)
# Lat5 → PAN Smooth5
draw_curved_arrow(fx + bw, 4.7 + bh/2, px, 5.2 + bh/2,
                  color='#6A1B9A', connectionstyle="arc3,rad=-0.2")

# ═══════════════════════════════════════════════════
# DETECTION HEADS (right column)
# ═══════════════════════════════════════════════════
hx = 14.0
hw = 3.5
ax.text(hx + hw/2, 11.2, '── Detection Heads ──', ha='center', fontsize=9,
        fontweight='bold', color='#B71C1C')

# Head P3 (large objects... wait, P3 is stride 8 = small stride = detect large-medium)
# Actually stride 8 detects small objects, stride 32 detects large objects
draw_block(hx, 9.8, hw, 0.9, 'Head P3 (stride 8)', C_HEAD, 8, True,
           'Conv3×3 → Conv3×3 → Conv1×1(5ch)')
ax.text(hx + hw + 0.15, 9.8 + 0.45, 'Small objects\n52×52×5', fontsize=6, color='#757575', va='center')
draw_arrow(fx + bw, 10.55, hx, 10.25, color='#B71C1C')

# Head P4
draw_block(hx, 7.8, hw, 0.9, 'Head P4 (stride 16)', C_HEAD, 8, True,
           'Conv3×3 → Conv3×3 → Conv1×1(5ch)')
ax.text(hx + hw + 0.15, 7.8 + 0.45, 'Medium objects\n26×26×5', fontsize=6, color='#757575', va='center')
draw_arrow(px + bw, 7.8 + bh/2, hx, 8.25, color='#B71C1C')

# Head P5
draw_block(hx, 5.8, hw, 0.9, 'Head P5 (stride 32)', C_HEAD, 8, True,
           'Conv3×3 → Conv3×3 → Conv1×1(5ch)')
ax.text(hx + hw + 0.15, 5.8 + 0.45, 'Large objects\n13×13×5', fontsize=6, color='#757575', va='center')
draw_arrow(px + bw, 5.2 + bh/2, hx, 6.25, color='#B71C1C')

# ═══════════════════════════════════════════════════
# OUTPUT format
# ═══════════════════════════════════════════════════
ox = 14.0
ow = 5.5
draw_block(ox, 4.2, ow, 1.2, 'Output per cell (5 channels)', C_OUT, 9, True)
ax.text(ox + ow/2, 4.7, 'ch0: objectness  |  ch1-2: Δx, Δy (offset)  |  ch3-4: log(w), log(h)',
        ha='center', va='center', fontsize=7, color='#424242')
# arrows from heads to output
draw_arrow(hx + hw/2 - 0.5, 5.8, ox + ow/2 - 0.5, 4.2 + 1.2, color='#757575', lw=0.8)
draw_arrow(hx + hw/2, 5.8, ox + ow/2, 4.2 + 1.2, color='#757575', lw=0.8)
draw_arrow(hx + hw/2 + 0.5, 5.8, ox + ow/2 + 0.5, 4.2 + 1.2, color='#757575', lw=0.8)

# Decode box
draw_block(ox, 2.8, ow, 1.1, 'Decode → NMS → Bounding Boxes + Foot Point', '#E3F2FD', 8, True)
ax.text(ox + ow/2, 3.15, 'cx = (σ(Δx)·2 − 0.5 + col) × stride / img_size\n'
        'cy = (σ(Δy)·2 − 0.5 + row) × stride / img_size\n'
        'w = exp(log_w) × stride / img_size',
        ha='center', va='center', fontsize=6, color='#37474F', family='monospace')
draw_arrow(ox + ow/2, 4.2, ox + ow/2, 2.8 + 1.1, color='#757575')

# ═══════════════════════════════════════════════════
# Legend
# ═══════════════════════════════════════════════════
lx, ly = 4.5, 2.0
ax.text(lx, ly + 1.0, 'Legend:', fontsize=8, fontweight='bold', color='#212121')
legend_items = [
    (C_CONV, 'Standard Conv + BN + SiLU'),
    (C_IR, 'InvertedResidual Block'),
    (C_LAT, 'Lateral Conv (1×1)'),
    (C_FPN, 'FPN Top-Down Path'),
    (C_PAN, 'PAN Bottom-Up Path'),
    (C_HEAD, 'Detection Head'),
]
for i, (color, label) in enumerate(legend_items):
    y = ly + 0.6 - i * 0.35
    box = FancyBboxPatch((lx, y), 0.4, 0.25, boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor=BORDER, linewidth=0.8)
    ax.add_patch(box)
    ax.text(lx + 0.55, y + 0.125, label, fontsize=6.5, va='center', color='#424242')

# Model stats
sx_stat, sy_stat = 8.0, 2.0
ax.text(sx_stat, sy_stat + 1.0, 'Model Stats:', fontsize=8, fontweight='bold', color='#212121')
stats = [
    'Parameters: ~900K (0.9M)',
    'Backbone: MobileNetV2-style IR blocks',
    'Neck: FPN + PAN (bidirectional)',
    'Activation: SiLU (Swish)',
    'Assignment: Multi-positive (3 cells/GT)',
    'Loss: BCE (obj) + CIoU + L1 (box)',
]
for i, s in enumerate(stats):
    ax.text(sx_stat, sy_stat + 0.6 - i * 0.3, f'• {s}', fontsize=6.5, color='#424242')

plt.tight_layout()
plt.savefig('runs/train_v5/pfdet_nano_v5_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('runs/train_v5/pfdet_nano_v5_architecture.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: runs/train_v5/pfdet_nano_v5_architecture.png")
print("Saved: runs/train_v5/pfdet_nano_v5_architecture.pdf")
