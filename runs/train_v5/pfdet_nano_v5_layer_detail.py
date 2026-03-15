"""
PFDet-Nano v5 - Chi tiết từng lớp (Layer-by-Layer Detail)
Tạo bảng chi tiết từng layer với input/output shape, số params, và giải thích.
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Tính params cho từng layer
def conv_params(c_in, c_out, k, groups=1, bias=False):
    """Số params của Conv2d"""
    p = (k * k * c_in * c_out) // groups
    if bias:
        p += c_out
    return p

def bn_params(c):
    """Số params của BatchNorm2d (weight + bias, không tính running stats)"""
    return c * 2

def convbn_params(c_in, c_out, k, groups=1):
    return conv_params(c_in, c_out, k, groups) + bn_params(c_out)

def ir_params(c, expand=2):
    """InvertedResidual params"""
    mid = c * expand
    p = convbn_params(c, mid, 1)       # expand
    p += convbn_params(mid, mid, 3, mid)  # dw
    p += convbn_params(mid, c, 1)      # project
    return p

def head_params(c_in, c_out=5):
    p = convbn_params(c_in, c_in, 3)   # conv1
    p += convbn_params(c_in, c_in, 3)  # conv2
    p += conv_params(c_in, c_out, 1, bias=True)  # final (has bias)
    return p


# ═══════════════════════════════════════════════════
# Build layer table data
# ═══════════════════════════════════════════════════

layers = []

def add(section, name, operation, in_shape, out_shape, params, note):
    layers.append({
        'section': section,
        'name': name,
        'operation': operation,
        'in_shape': in_shape,
        'out_shape': out_shape,
        'params': params,
        'note': note,
    })

# ── BACKBONE ──
add('BACKBONE', 'Stem.conv1', 'Conv2d 3×3/2 + BN + SiLU', '416×416×3', '208×208×16',
    convbn_params(3, 16, 3), 'Giảm resolution 1/2, trích xuất edge/color cơ bản')
add('BACKBONE', 'Stem.conv2', 'Conv2d 3×3/1 + BN + SiLU', '208×208×16', '208×208×16',
    convbn_params(16, 16, 3), 'Tăng depth cho stem, giữ nguyên resolution')

add('BACKBONE', 'Stage1.conv', 'Conv2d 3×3/2 + BN + SiLU', '208×208×16', '104×104×32',
    convbn_params(16, 32, 3), 'Downsample 1/4, tăng channels lên 32')
add('BACKBONE', 'Stage1.IR', 'InvertedResidual (expand=2)', '104×104×32', '104×104×32',
    ir_params(32), '32→64→64→32 + skip. Học texture, simple patterns')

add('BACKBONE', 'Stage2.conv', 'Conv2d 3×3/2 + BN + SiLU', '104×104×32', '52×52×64',
    convbn_params(32, 64, 3), 'Downsample 1/8 (stride 8). Output = P3')
add('BACKBONE', 'Stage2.IR×2', '2× InvertedResidual (expand=2)', '52×52×64', '52×52×64',
    ir_params(64) * 2, '64→128→128→64. Học body parts, limb patterns')

add('BACKBONE', 'Stage3.conv', 'Conv2d 3×3/2 + BN + SiLU', '52×52×64', '26×26×128',
    convbn_params(64, 128, 3), 'Downsample 1/16 (stride 16). Output = P4')
add('BACKBONE', 'Stage3.IR×2', '2× InvertedResidual (expand=2)', '26×26×128', '26×26×128',
    ir_params(128) * 2, '128→256→256→128. Học full body, pose patterns')

add('BACKBONE', 'Stage4.conv', 'Conv2d 3×3/2 + BN + SiLU', '26×26×128', '13×13×192',
    convbn_params(128, 192, 3), 'Downsample 1/32 (stride 32). Output = P5')
add('BACKBONE', 'Stage4.IR×2', '2× InvertedResidual (expand=2)', '13×13×192', '13×13×192',
    ir_params(192) * 2, '192→384→384→192. Học context, scene-level features')

# ── FPN (Top-Down) ──
add('FPN', 'Lat5', 'Conv2d 1×1 + BN + SiLU', '13×13×192', '13×13×64',
    convbn_params(192, 64, 1), 'Giảm channels P5 về 64 (neck channel)')
add('FPN', 'Lat4', 'Conv2d 1×1 + BN + SiLU', '26×26×128', '26×26×64',
    convbn_params(128, 64, 1), 'Giảm channels P4 về 64')
add('FPN', 'Lat3', 'Conv2d 1×1 + BN + SiLU', '52×52×64', '52×52×64',
    convbn_params(64, 64, 1), 'Giữ channels P3 = 64')

add('FPN', 'Upsample P5→P4', 'F.interpolate nearest 2×', '13×13×64', '26×26×64',
    0, 'Phóng to feature map P5 lên kích thước P4 (không có params)')
add('FPN', 'FPN Smooth4', 'Add + Conv2d 3×3 + BN + SiLU', '26×26×64', '26×26×64',
    convbn_params(64, 64, 3), 'Lat4 + Upsample(Lat5). Kết hợp semantic (P5) + spatial (P4)')

add('FPN', 'Upsample P4→P3', 'F.interpolate nearest 2×', '26×26×64', '52×52×64',
    0, 'Phóng to P4_fpn lên kích thước P3')
add('FPN', 'FPN Smooth3', 'Add + Conv2d 3×3 + BN + SiLU', '52×52×64', '52×52×64',
    convbn_params(64, 64, 3), 'Lat3 + Upsample(P4_fpn). Output = P3_fpn (chi tiết nhất)')

# ── PAN (Bottom-Up) ──
add('PAN', 'PAN Down3', 'Conv2d 3×3/2 + BN + SiLU', '52×52×64', '26×26×64',
    convbn_params(64, 64, 3), 'Downsample P3_fpn, truyền spatial detail xuống P4')
add('PAN', 'PAN Smooth4', 'Add + Conv2d 3×3 + BN + SiLU', '26×26×64', '26×26×64',
    convbn_params(64, 64, 3), 'P4_fpn + Down(P3_fpn). Output = P4_pan')

add('PAN', 'PAN Down4', 'Conv2d 3×3/2 + BN + SiLU', '26×26×64', '13×13×64',
    convbn_params(64, 64, 3), 'Downsample P4_pan, truyền spatial+semantic xuống P5')
add('PAN', 'PAN Smooth5', 'Add + Conv2d 3×3 + BN + SiLU', '13×13×64', '13×13×64',
    convbn_params(64, 64, 3), 'Lat5 + Down(P4_pan). Output = P5_pan')

# ── HEADS ──
add('HEAD', 'Head P3 - Conv1', 'Conv2d 3×3 + BN + SiLU', '52×52×64', '52×52×64',
    convbn_params(64, 64, 3), 'Feature refinement cho small objects')
add('HEAD', 'Head P3 - Conv2', 'Conv2d 3×3 + BN + SiLU', '52×52×64', '52×52×64',
    convbn_params(64, 64, 3), 'Thêm depth, tăng khả năng phân biệt')
add('HEAD', 'Head P3 - Pred', 'Conv2d 1×1 (bias)', '52×52×64', '52×52×5',
    conv_params(64, 5, 1, bias=True), 'Output 5ch: obj, dx, dy, lw, lh. Bias obj=-4.0')

add('HEAD', 'Head P4 - Conv1', 'Conv2d 3×3 + BN + SiLU', '26×26×64', '26×26×64',
    convbn_params(64, 64, 3), 'Feature refinement cho medium objects')
add('HEAD', 'Head P4 - Conv2', 'Conv2d 3×3 + BN + SiLU', '26×26×64', '26×26×64',
    convbn_params(64, 64, 3), 'Thêm depth')
add('HEAD', 'Head P4 - Pred', 'Conv2d 1×1 (bias)', '26×26×64', '26×26×5',
    conv_params(64, 5, 1, bias=True), 'Output 5ch. Mỗi cell quản lý vùng 16×16 px')

add('HEAD', 'Head P5 - Conv1', 'Conv2d 3×3 + BN + SiLU', '13×13×64', '13×13×64',
    convbn_params(64, 64, 3), 'Feature refinement cho large objects')
add('HEAD', 'Head P5 - Conv2', 'Conv2d 3×3 + BN + SiLU', '13×13×64', '13×13×64',
    convbn_params(64, 64, 3), 'Thêm depth')
add('HEAD', 'Head P5 - Pred', 'Conv2d 1×1 (bias)', '13×13×64', '13×13×5',
    conv_params(64, 5, 1, bias=True), 'Output 5ch. Mỗi cell quản lý vùng 32×32 px')

# ═══════════════════════════════════════════════════
# RENDER TABLE
# ═══════════════════════════════════════════════════

total_params = sum(l['params'] for l in layers)

# Section colors
section_colors = {
    'BACKBONE': '#E8F5E9',
    'FPN': '#FFF3E0',
    'PAN': '#F3E5F5',
    'HEAD': '#FFEBEE',
}
section_header_colors = {
    'BACKBONE': '#2E7D32',
    'FPN': '#E65100',
    'PAN': '#6A1B9A',
    'HEAD': '#B71C1C',
}

fig, ax = plt.subplots(figsize=(24, 22))
ax.axis('off')
fig.patch.set_facecolor('white')

# Title
ax.text(0.5, 0.98, 'PFDet-Nano v5 — Chi Tiết Từng Lớp (Layer-by-Layer)',
        transform=ax.transAxes, ha='center', fontsize=20, fontweight='bold', color='#1A237E')
ax.text(0.5, 0.965, f'Tổng tham số: {total_params:,} ({total_params/1e6:.3f}M)  |  Input: 416×416×3  |  Output: 3 scale × 5 channels',
        transform=ax.transAxes, ha='center', fontsize=11, color='#455A64')

# Table
col_widths = [0.04, 0.12, 0.20, 0.09, 0.09, 0.07, 0.30]
col_labels = ['#', 'Tên Layer', 'Phép Toán', 'Input Shape', 'Output Shape', 'Params', 'Giải Thích']
col_x = [0.035]
for w in col_widths[:-1]:
    col_x.append(col_x[-1] + w)

row_height = 0.023
start_y = 0.94

# Header
y = start_y
for j, (label, cx) in enumerate(zip(col_labels, col_x)):
    ax.text(cx + col_widths[j]/2, y, label, transform=ax.transAxes,
            ha='center', va='center', fontsize=9, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#37474F', edgecolor='none'))

y -= row_height * 1.3

current_section = None
row_num = 0

for i, layer in enumerate(layers):
    # Section header
    if layer['section'] != current_section:
        current_section = layer['section']
        section_params = sum(l['params'] for l in layers if l['section'] == current_section)

        # Section separator
        y -= row_height * 0.3
        ax.plot([0.03, 0.97], [y + row_height * 0.6, y + row_height * 0.6],
               transform=ax.transAxes, color=section_header_colors[current_section],
               linewidth=2, alpha=0.8)

        section_label = {
            'BACKBONE': f'BACKBONE — Trích xuất đặc trưng ({section_params:,} params)',
            'FPN': f'FPN (Top-Down) — Truyền semantic features xuống ({section_params:,} params)',
            'PAN': f'PAN (Bottom-Up) — Truyền spatial features lên ({section_params:,} params)',
            'HEAD': f'DETECTION HEADS — Dự đoán obj + box ({section_params:,} params)',
        }
        ax.text(0.5, y, section_label[current_section], transform=ax.transAxes,
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=section_header_colors[current_section])
        y -= row_height * 1.2

    row_num += 1
    bg_color = section_colors[layer['section']]
    if row_num % 2 == 0:
        # Slightly darker for alternating rows
        bg_alpha = 0.6
    else:
        bg_alpha = 0.3

    # Row background
    ax.fill_between([0.03, 0.97], y - row_height * 0.4, y + row_height * 0.5,
                    facecolor=bg_color, alpha=bg_alpha, transform=ax.transAxes)

    # Row data
    row_data = [
        str(row_num),
        layer['name'],
        layer['operation'],
        layer['in_shape'],
        layer['out_shape'],
        f"{layer['params']:,}" if layer['params'] > 0 else '0',
        layer['note'],
    ]

    aligns = ['center', 'left', 'left', 'center', 'center', 'right', 'left']
    fontsizes = [7, 7.5, 7, 7, 7, 7, 6.5]

    for j, (val, cx, align, fs) in enumerate(zip(row_data, col_x, aligns, fontsizes)):
        x_pos = cx + (col_widths[j]/2 if align == 'center' else 0.005 if align == 'left' else col_widths[j] - 0.005)
        ax.text(x_pos, y, val, transform=ax.transAxes,
                ha=align, va='center', fontsize=fs, color='#212121',
                fontweight='bold' if j == 1 else 'normal')

    y -= row_height

# ═══════════════════════════════════════════════════
# SUMMARY at bottom
# ═══════════════════════════════════════════════════
y -= row_height * 0.5

# Total line
ax.plot([0.03, 0.97], [y + row_height * 0.3, y + row_height * 0.3],
       transform=ax.transAxes, color='#1A237E', linewidth=2)

ax.text(0.5, y - row_height * 0.3,
        f'TỔNG: {total_params:,} tham số ({total_params/1e6:.3f}M)  |  '
        f'Backbone: {sum(l["params"] for l in layers if l["section"]=="BACKBONE"):,}  |  '
        f'FPN: {sum(l["params"] for l in layers if l["section"]=="FPN"):,}  |  '
        f'PAN: {sum(l["params"] for l in layers if l["section"]=="PAN"):,}  |  '
        f'Heads: {sum(l["params"] for l in layers if l["section"]=="HEAD"):,}',
        transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold', color='#1A237E')

# Explanation box
y -= row_height * 2.5
explanations = [
    'Giải thích các thuật ngữ:',
    '',
    '• Conv2d k×k/s: Tích chập 2D với kernel kích thước k×k, stride s. Trích xuất đặc trưng cục bộ từ vùng k×k pixel.',
    '• BN (BatchNorm): Chuẩn hóa output theo batch, giúp training ổn định và hội tụ nhanh hơn.',
    '• SiLU (Swish): Hàm kích hoạt phi tuyến f(x) = x × σ(x). Hiệu quả hơn ReLU cho detection.',
    '• InvertedResidual: Expand channels → Depthwise Conv → Project channels + Skip connection. Hiệu quả tham số cao (ít params, nhiều features).',
    '• FPN (Feature Pyramid Network): Truyền thông tin ngữ nghĩa (semantic) từ lớp sâu (P5) lên lớp nông (P3) qua upsample + add.',
    '• PAN (Path Aggregation Network): Truyền thông tin không gian (spatial) từ lớp nông (P3) xuống lớp sâu (P5) qua downsample + add.',
    '• Stride 8/16/32: Mỗi cell ở P3 "nhìn" vùng 8×8px, P4 nhìn 16×16px, P5 nhìn 32×32px. Stride nhỏ → detect vật nhỏ.',
    '• Output 5 channels: [objectness, Δx, Δy, log(w), log(h)] — xác suất có người + offset + kích thước bounding box.',
    '• Foot Point: Tính từ bottom-center của bounding box: foot_x = (x1+x2)/2, foot_y = y2. Không cần thêm channel.',
]

for i, line in enumerate(explanations):
    ax.text(0.04, y - i * row_height * 0.85, line, transform=ax.transAxes,
            fontsize=7.5 if i == 0 else 7,
            fontweight='bold' if i == 0 else 'normal',
            color='#1A237E' if i == 0 else '#37474F')

plt.tight_layout(pad=0.5)
plt.savefig('runs/train_v5/pfdet_nano_v5_layer_detail.png', dpi=180, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('runs/train_v5/pfdet_nano_v5_layer_detail.pdf', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("Saved: runs/train_v5/pfdet_nano_v5_layer_detail.png")
print("Saved: runs/train_v5/pfdet_nano_v5_layer_detail.pdf")
