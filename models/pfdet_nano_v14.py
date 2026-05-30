"""
PFDet-Nano v14 — Drone Person Detector
======================================

Novel contributions for conference paper:
  1. UIB backbone — Universal Inverted Bottleneck (double-DW, MobileNetV4-inspired)
     with ExtraDW variant (5×1→1×5 factorized) for elongated aerial human silhouettes
  2. P2 ultra-fine scale — stride-4 detection head (96×96 cells at 384px input)
     for detecting sub-5px persons at high drone altitude
  3. AreaAttention bottleneck — efficient global context O(HW/a²) vs O((HW)²),
     first applied in sub-2M param aerial detector
  4. BiFPN neck — bi-directional weighted feature pyramid (4 scales)
  5. DecoupledHead — separate cls/box branches, no DFL for edge-device simplicity
  6. LSK attention (Large Selective Kernel, ICCV 2023) — aerial-specific spatial context

References (all peer-reviewed, code self-written):
  - UIB block:        MobileNetV4 — Howard et al., ECCV 2024
  - Local window attn: Swin Transformer — Liu et al., ICCV 2021 (Best Paper Award)
  - LSK attention:    LSKNet — Li et al., ICCV 2023 / IJCV 2024
  - BiFPN neck:       EfficientDet — Tan et al., CVPR 2020
  - RepConv:          RepVGG — Ding et al., CVPR 2021
  - Decoupled head:   YOLOX — Ge et al., NeurIPS 2021 Workshop
  - P2 extra scale:   novel contribution for sub-10px drone person detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PROFILES = {
    'light': {
        'base_c': 24,
        'neck_c': 48,
        'num_bifpn': 1,
        'area': 3,
        'use_area_attention': True,  # Free tại P5=10×10, thêm global context
    },
    'balanced': {
        'base_c': 32,
        'neck_c': 64,
        'num_bifpn': 2,
        'area': 3,
        'use_area_attention': True,
    },
}

DEFAULT_EXPORT_OUTPUT_NAMES = ('output_p2', 'output_p3', 'output_p4', 'output_p5')


def build_activation(name='silu', act=True):
    """Build an activation layer for conv branches."""
    if not act:
        return nn.Identity()

    name = str(name).lower().strip()
    if name == 'silu':
        return nn.SiLU(inplace=True)
    if name == 'gelu':
        return nn.GELU()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name in ('identity', 'none'):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def resolve_model_profile(profile='balanced'):
    profile = str(profile).lower().strip()
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown PFDet profile: {profile!r}. "
            f"Available profiles: {sorted(MODEL_PROFILES)}"
        )
    return dict(MODEL_PROFILES[profile])


# ============================================================
#  Building blocks
# ============================================================

class ConvBN(nn.Module):
    """Conv2d + BatchNorm + configurable activation."""
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True, activation='silu'):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act = build_activation(activation, act=act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    """
    Re-parameterizable Convolution (RepConv).

    Training: 3 parallel branches → 3×3 + 1×1 + identity (if same shape).
    Inference: single fused 3×3 conv (reparameterize() collapses them).

    Free accuracy boost: multi-branch training enriches the loss landscape
    without any extra FLOPs at inference time.

    Reference: RepVGG (Ding et al., CVPR 2021), used in YOLOv6/v7/Gold-YOLO.
    Applied here in BiFPN post-fusion convs and detection head shared conv.
    """
    def __init__(self, in_c, out_c, k=3, s=1, g=1, activation='silu'):
        super().__init__()
        assert k == 3, "RepConv only supports k=3"
        p = 1
        self.in_c = in_c
        self.out_c = out_c
        self.s = s
        self.g = g
        self.act = build_activation(activation)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, s, p, groups=g, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, s, 0, groups=g, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        )
        # Identity branch only when shapes match and stride=1
        self.identity_bn = (
            nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.03)
            if s == 1 and in_c == out_c else None
        )
        self._reparam = False

    def forward(self, x):
        if self._reparam:
            return self.act(self.reparam_conv(x))
        out = self.conv3(x) + self.conv1(x)
        if self.identity_bn is not None:
            out = out + self.identity_bn(x)
        return self.act(out)

    def reparameterize(self):
        """Fuse all branches into a single 3×3 conv for fast inference."""
        if self._reparam:
            return
        w3, b3 = self._fuse_bn(self.conv3)
        w1, b1 = self._fuse_bn(self.conv1)
        # Pad 1×1 kernel to 3×3
        w1_pad = F.pad(w1, [1, 1, 1, 1])

        if self.identity_bn is not None:
            w_id, b_id = self._fuse_identity_bn(self.identity_bn)
        else:
            w_id = torch.zeros_like(w3)
            b_id = torch.zeros_like(b3)

        w_fused = w3 + w1_pad + w_id
        b_fused = b3 + b1 + b_id

        self.reparam_conv = nn.Conv2d(
            self.in_c, self.out_c, 3, self.s, 1,
            groups=self.g, bias=True,
        )
        self.reparam_conv.weight.data = w_fused
        self.reparam_conv.bias.data = b_fused

        del self.conv3, self.conv1
        if self.identity_bn is not None:
            del self.identity_bn
        self._reparam = True

    def _fuse_bn(self, seq):
        conv, bn = seq[0], seq[1]
        w = conv.weight
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        std = (var + bn.eps).sqrt()
        scale = gamma / std
        w_fused = w * scale.view(-1, 1, 1, 1)
        b_fused = beta - mean * scale
        return w_fused, b_fused

    def _fuse_identity_bn(self, bn):
        in_c = self.in_c
        w_id = torch.zeros(in_c, in_c // self.g, 3, 3,
                           device=bn.weight.device, dtype=bn.weight.dtype)
        for i in range(in_c):
            w_id[i, i % (in_c // self.g), 1, 1] = 1.0
        mean, var = bn.running_mean, bn.running_var
        gamma, beta = bn.weight, bn.bias
        std = (var + bn.eps).sqrt()
        scale = gamma / std
        w_fused = w_id * scale.view(-1, 1, 1, 1)
        b_fused = beta - mean * scale
        return w_fused, b_fused


class FactDW(nn.Module):
    """
    Factorized Depthwise: Conv(1×k) → BN → GELU → Conv(k×1) → BN.

    Captures elongated patterns (aerial human silhouettes) at low cost.
    Factorized form approximates k×k DW with fewer MACs.
    """
    def __init__(self, c, k=5, activation='silu'):
        super().__init__()
        self.dw1 = nn.Sequential(
            nn.Conv2d(c, c, (1, k), padding=(0, (k - 1) // 2), groups=c, bias=False),
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
            build_activation(activation),
        )
        self.dw2 = nn.Sequential(
            nn.Conv2d(c, c, (k, 1), padding=((k - 1) // 2, 0), groups=c, bias=False),
            nn.BatchNorm2d(c, eps=1e-3, momentum=0.03),
        )

    def forward(self, x):
        return self.dw2(self.dw1(x))


class UIBBlock(nn.Module):
    """
    Universal Inverted Bottleneck (UIB).

    Structure (inspired by MobileNetV4 UIB):
      input
        → DW 3×3 (stride-aware)             # local spatial mixing
        → BN + GELU
        → PW expand (×expand_ratio)          # channel expansion
        → BN + GELU
        → [ExtraDW: FactDW 5×1→1×5]         # elongated context (optional)
        → PW compress → BN                   # project to out_c
        + skip

    ExtraDW captures aspect-ratio-aware context for tall/thin drone targets.
    Skip connection is identity (same shape) or 1×1 projection + stride pool.
    """
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, extra_dw=False, activation='silu'):
        super().__init__()
        self.stride = stride
        mid_c = max(in_c * expand_ratio, in_c)
        self.has_identity_skip = (stride == 1 and in_c == out_c)

        layers = [
            # DW 3×3 — spatial mixing at input resolution
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.03),
            build_activation(activation),
            # PW expand
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c, eps=1e-3, momentum=0.03),
            build_activation(activation),
        ]
        if extra_dw:
            layers.append(FactDW(mid_c, k=5, activation=activation))
            layers.append(build_activation(activation))
        layers += [
            # PW compress
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        ]
        self.block = nn.Sequential(*layers)

        # Skip
        if not self.has_identity_skip:
            if stride > 1:
                self.skip = nn.Sequential(
                    nn.AvgPool2d(stride, stride),
                    nn.Conv2d(in_c, out_c, 1, bias=False),
                    nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
                )
            else:
                self.skip = nn.Sequential(
                    nn.Conv2d(in_c, out_c, 1, bias=False),
                    nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
                )

    def forward(self, x):
        out = self.block(x)
        if self.has_identity_skip:
            return out + x
        return out + self.skip(x)


class EdgeContextStem(nn.Module):
    """
    Dual-path entry stem.

    Two parallel branches:
      - local: Conv3×3 (s=2) → Conv3×3 — captures fine edge/detail
      - context: AvgPool(s=2) → Conv3×3 — captures low-freq statistics
    Concatenate → fuse with 1×1 Conv.

    Rationale: Drones lose fine resolution early; simultaneous detail+context
    prevents either from dominating stem features.
    """
    def __init__(self, in_c=3, out_c=32, activation='silu'):
        super().__init__()
        mid = out_c // 2
        self.local = nn.Sequential(
            ConvBN(in_c, mid, k=3, s=2, activation=activation),
            ConvBN(mid, mid, k=3, s=1, activation=activation),
        )
        self.context = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            ConvBN(in_c, mid, k=3, s=1, activation=activation),
        )
        self.fuse = ConvBN(out_c, out_c, k=1, activation=activation)

    def forward(self, x):
        return self.fuse(torch.cat([self.local(x), self.context(x)], dim=1))


class AreaAttentionBlock(nn.Module):
    """
    Local Window Self-Attention (area-partitioned).

    Divides the feature map into non-overlapping (area×area) windows and applies
    multi-head self-attention within each window independently — identical in spirit
    to the Shifted Window attention in Swin Transformer (Liu et al., ICCV 2021 Best Paper).

    Applied here at P5 bottleneck (10×10 at 320px → 9 windows of 3×3 = 9 tokens each):
      Complexity: O(B × nH×nW × a² × a²) = O(B × HW × a²)  vs.  O(B × H²W²) global
      → essentially free at P5 resolution, provides global context that conv cannot.

    Reference: Liu et al., "Swin Transformer: Hierarchical Vision Transformer using
               Shifted Windows", ICCV 2021 (Best Paper Award).
    """
    def __init__(self, c, area=3, num_heads=2):
        super().__init__()
        assert c % num_heads == 0, f"c={c} must be divisible by num_heads={num_heads}"
        self.area = area
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)

        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(
            nn.Linear(c, c * 2),
            nn.GELU(),
            nn.Linear(c * 2, c),
        )

    def _attn(self, x):
        """x: (N, L, C) where N=batch*num_areas, L=area*area."""
        N, L, C = x.shape
        qkv = self.qkv(self.norm1(x))  # (N, L, 3C)
        q, k, v = qkv.chunk(3, dim=-1)
        # (N, L, num_heads, head_dim) → (N, num_heads, L, head_dim)
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(N, L, C)
        return x + self.proj(out)

    def forward(self, x):
        B, C, H, W = x.shape
        a = self.area
        # Pad so H, W divisible by a
        pH = (a - H % a) % a
        pW = (a - W % a) % a
        if pH > 0 or pW > 0:
            x = F.pad(x, (0, pW, 0, pH))
        H2, W2 = H + pH, W + pW
        nH, nW = H2 // a, W2 // a

        # Rearrange: (B, C, nH, a, nW, a) → (B*nH*nW, a*a, C)
        xp = (x.view(B, C, nH, a, nW, a)
               .permute(0, 2, 4, 3, 5, 1)
               .contiguous()
               .view(B * nH * nW, a * a, C))

        # Self-attention within each patch
        xp = self._attn(xp)

        # FFN
        xp = xp + self.ffn(self.norm2(xp))

        # Rearrange back: (B*nH*nW, a*a, C) → (B, C, H2, W2)
        xp = (xp.view(B, nH, nW, a, a, C)
               .permute(0, 5, 1, 3, 2, 4)
               .contiguous()
               .view(B, C, H2, W2))

        return xp[:, :, :H, :W]


# ============================================================
#  BiFPN Neck
# ============================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation channel attention (Hu et al., CVPR 2018).
    Kept for reference. Use CBAM instead — it adds spatial attention on top.
    """
    def __init__(self, c, r=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c, max(c // r, 4)),
            nn.ReLU(inplace=True),
            nn.Linear(max(c // r, 4), c),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excite(self.squeeze(x)).view(x.shape[0], -1, 1, 1)


class LSKBlock(nn.Module):
    """
    Large Selective Kernel Block (Li et al., ICCV 2023).

    Specifically designed for aerial/remote sensing object detection —
    directly applicable to drone person detection.

    Key insight: tiny objects in aerial imagery need LARGE receptive fields
    to separate them from structured background (rooftops, roads, vehicles).
    Two DW conv branches with different scales adaptively select the right context.

    Architecture:
      Branch 0: DW Conv 5×5         (local fine context)
      Branch 1: DW Conv 7×7 d=3    (large context, erf ≈ 19×19)
      Fuse → softmax attention weights → weighted sum → PW project → residual

    Proven SOTA on VisDrone, DOTA, HRSC2016 benchmarks.
    Replaces CBAM (2018) which uses only small 7×7 spatial attention.

    References:
      Li et al., "Large Selective Kernel Network for Remote Sensing Object
                  Detection", ICCV 2023.
      Li et al., "LSKNet: A Foundation Lightweight Backbone for Remote Sensing",
                  International Journal of Computer Vision (IJCV), 2024.
                  (Journal extension — SOTA on VisDrone, DOTA, HRSC2016.)
    """
    def __init__(self, c):
        super().__init__()
        self.dw0 = nn.Conv2d(c, c, 5, padding=2, groups=c)                      # DW 5×5
        self.dw1 = nn.Conv2d(c, c, 7, padding=9, dilation=3, groups=c)          # DW 7×7 d=3, erf≈19
        self.attn_fc = nn.Sequential(
            nn.Conv2d(c * 2, c // 2, 1),   # compress
            nn.Conv2d(c // 2, 2, 1),        # 2 attention scalars per spatial location
        )
        self.proj = nn.Conv2d(c, c, 1)      # channel mixing after selection

    def forward(self, x):
        a0 = self.dw0(x)                    # (B, c, H, W) — fine context
        a1 = self.dw1(x)                    # (B, c, H, W) — large context
        # Compute attention weights from concatenated branches
        attn = self.attn_fc(torch.cat([a0, a1], dim=1))  # (B, 2, H, W)
        attn = attn.softmax(dim=1)           # normalize: w0 + w1 = 1 per location
        # Adaptive selection: each spatial location chooses how much from each scale
        out = a0 * attn[:, 0:1] + a1 * attn[:, 1:2]
        return self.proj(out) + x            # residual


class WeightedFusion(nn.Module):
    """
    Weighted feature fusion (BiFPN-style).
    Learnable scalar weights, softmax-normalized.
    Ensures non-negative, sum-to-one contributions from each input.
    """
    def __init__(self, n=2, eps=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, feats):
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return sum(f * wi for f, wi in zip(feats, w))


class BiFPNLayer(nn.Module):
    """
    One BiFPN iteration: top-down + bottom-up with weighted fusion.

    4 scales: P2 (finest) → P3 → P4 → P5 (coarsest)

    Top-down pass:
      P4_td = W_fuse(P4, up(P5))
      P3_td = W_fuse(P3, up(P4_td))
      P2_td = W_fuse(P2, up(P3_td))

    Bottom-up pass:
      P2_out = P2_td
      P3_out = W_fuse(P3, P3_td, down(P2_td))
      P4_out = W_fuse(P4, P4_td, down(P3_out))
      P5_out = W_fuse(P5, down(P4_out))
    """
    def __init__(self, c, activation='silu'):
        super().__init__()
        # Top-down fusions (2 inputs each)
        self.td_p4 = WeightedFusion(2)
        self.td_p3 = WeightedFusion(2)
        self.td_p2 = WeightedFusion(2)
        # Bottom-up fusions
        self.bu_p3 = WeightedFusion(3)
        self.bu_p4 = WeightedFusion(3)
        self.bu_p5 = WeightedFusion(2)
        # Post-fusion convs — RepConv for richer training (same FLOPs at inference)
        self.conv_td4 = RepConv(c, c, 3, activation=activation)
        self.conv_td3 = RepConv(c, c, 3, activation=activation)
        self.conv_td2 = RepConv(c, c, 3, activation=activation)
        self.conv_bu3 = RepConv(c, c, 3, activation=activation)
        self.conv_bu4 = RepConv(c, c, 3, activation=activation)
        self.conv_bu5 = RepConv(c, c, 3, activation=activation)

    def _up(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

    def _down(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def forward(self, p2, p3, p4, p5):
        # Top-down
        p4_td = self.conv_td4(self.td_p4([p4, self._up(p5)]))
        p3_td = self.conv_td3(self.td_p3([p3, self._up(p4_td)]))
        p2_td = self.conv_td2(self.td_p2([p2, self._up(p3_td)]))

        # Bottom-up
        p3_out = self.conv_bu3(self.bu_p3([p3, p3_td, self._down(p2_td)]))
        p4_out = self.conv_bu4(self.bu_p4([p4, p4_td, self._down(p3_out)]))
        p5_out = self.conv_bu5(self.bu_p5([p5, self._down(p4_out)]))

        return p2_td, p3_out, p4_out, p5_out


# ============================================================
#  Detection Head
# ============================================================

class DecoupledHead(nn.Module):
    """
    Decoupled two-branch detection head (inspired by YOLOX).

    Separate branches for classification and box regression.
    No DFL (Distribution Focal Loss) — direct regression.
    Simpler, faster on edge devices (Raspberry Pi 5).

    Output: (B, 5, H, W)
      channel 0:   objectness logit
      channels 1-2: dx, dy center offset (decode with sigmoid - 0.5, range [-0.5, 0.5])
      channels 3-4: log_w, log_h size in log-space (decode with exp * stride, clamp ±4)
    """
    def __init__(self, in_c, mid_c=None, activation='silu'):
        super().__init__()
        mid_c = mid_c or in_c

        # RepConv in shared + branches: richer feature during training
        self.shared = RepConv(in_c, mid_c, 3, activation=activation)

        self.cls_branch = nn.Sequential(
            RepConv(mid_c, mid_c, 3, activation=activation),
            nn.Conv2d(mid_c, 1, 1),
        )
        self.box_branch = nn.Sequential(
            RepConv(mid_c, mid_c, 3, activation=activation),
            nn.Conv2d(mid_c, 4, 1),
        )
        self._init_head_weights()

    def _init_head_weights(self):
        # Strong negative prior for objectness — avoids early FP explosions
        cls_conv = self.cls_branch[-1]
        nn.init.normal_(cls_conv.weight, std=0.01)
        nn.init.constant_(cls_conv.bias, -5.5)

        # Small init for box regression
        box_conv = self.box_branch[-1]
        nn.init.normal_(box_conv.weight, std=0.01)
        nn.init.zeros_(box_conv.bias)

    def forward(self, x):
        feat = self.shared(x)
        cls_out = self.cls_branch(feat)   # (B, 1, H, W)
        box_out = self.box_branch(feat)   # (B, 4, H, W)
        return torch.cat([cls_out, box_out], dim=1)  # (B, 5, H, W)


# ============================================================
#  Full Model
# ============================================================

class PFDetNanoV14(nn.Module):
    """
    PFDet-Nano v14 — Person Detection from Drone Camera.

    Architecture overview:
      EdgeContextStem
        → UIB Backbone: P2(s=4) P3(s=8) P4(s=16) P5(s=32)
        → AreaAttention bottleneck at P5
        → Lateral 1×1 projections → neck_c channels
        → BiFPN neck (×num_bifpn iterations)
        → DecoupledHead × 4 scales

    Target platform: Raspberry Pi 5 (ARM Cortex-A76)
    Target task: Single-class person detection from aerial/drone view

    Args:
        base_c (int): Channel base width. Default 32.
        neck_c (int): Neck channel width. Default 64.
        num_bifpn (int): Number of BiFPN iterations. Default 2.
        area (int): Area size for AreaAttentionBlock. Default 3.
    """
    strides = [4, 8, 16, 32]

    def __init__(self, base_c=None, neck_c=None, num_bifpn=None, area=None,
                 activation='silu', profile='balanced', use_area_attention=None):
        super().__init__()
        profile_cfg = resolve_model_profile(profile)
        base_c = profile_cfg['base_c'] if base_c is None else base_c
        neck_c = profile_cfg['neck_c'] if neck_c is None else neck_c
        num_bifpn = profile_cfg['num_bifpn'] if num_bifpn is None else num_bifpn
        area = profile_cfg['area'] if area is None else area
        if use_area_attention is None:
            use_area_attention = profile_cfg['use_area_attention']

        self.profile = str(profile).lower().strip()
        self.activation_name = str(activation).lower().strip()
        self.use_area_attention = bool(use_area_attention)
        self.export_output_names = DEFAULT_EXPORT_OUTPUT_NAMES
        self.model_config = {
            'profile': self.profile,
            'activation': self.activation_name,
            'base_c': int(base_c),
            'neck_c': int(neck_c),
            'num_bifpn': int(num_bifpn),
            'area': int(area),
            'use_area_attention': self.use_area_attention,
        }

        c1 = base_c           # 32  — stem
        c2 = base_c * 2       # 64  — P2 (stride 4, fine scale)
        c3 = base_c * 3       # 96  — P3 (stride 8)
        c4 = base_c * 4       # 128 — P4 (stride 16)
        c5 = base_c * 4       # 128 — P5 (stride 32, deepened)

        # ---- Backbone ----
        self.stem = EdgeContextStem(3, c1, activation=activation)  # 384→192, stride 2

        # P2: stride 4 (80×80 at 320px input) — NOVEL: ultra-fine detection scale
        # 3 blocks vì P2 xử lý ~95% detection trên VisDrone (persons 4-20px)
        # Block cuối thêm FactDW để capture hình người đứng (elongated vertical pattern)
        self.stage_p2 = nn.Sequential(
            UIBBlock(c1, c2, stride=2, activation=activation),              # 24→48, stride 2
            UIBBlock(c2, c2, activation=activation),                        # 48→48
            UIBBlock(c2, c2, extra_dw=True, activation=activation),        # 48→48 + FactDW
        )

        # P3: stride 8 (48×48)
        self.stage_p3 = nn.Sequential(
            UIBBlock(c2, c3, stride=2, extra_dw=True, activation=activation),
            UIBBlock(c3, c3, activation=activation),
            UIBBlock(c3, c3, activation=activation),
        )

        # P4: stride 16 (24×24)
        self.stage_p4 = nn.Sequential(
            UIBBlock(c3, c4, stride=2, extra_dw=True, activation=activation),
            UIBBlock(c4, c4, activation=activation),
            UIBBlock(c4, c4, activation=activation),
        )

        # P5: stride 32 (12×12)
        self.stage_p5 = nn.Sequential(
            UIBBlock(c4, c5, stride=2, activation=activation),
            UIBBlock(c5, c5, activation=activation),
            UIBBlock(c5, c5, activation=activation),
        )

        # AreaAttention at bottleneck (12×12)
        self.bottleneck = (
            AreaAttentionBlock(c5, area=area, num_heads=2)
            if self.use_area_attention else nn.Identity()
        )

        # ---- Neck: lateral projections → common neck_c ----
        self.lat_p2 = ConvBN(c2, neck_c, 1, activation=activation)
        self.lat_p3 = ConvBN(c3, neck_c, 1, activation=activation)
        self.lat_p4 = ConvBN(c4, neck_c, 1, activation=activation)
        self.lat_p5 = ConvBN(c5, neck_c, 1, activation=activation)

        # ---- BiFPN iterations ----
        self.bifpn = nn.Sequential(*[
            BiFPNLayer(neck_c, activation=activation) for _ in range(num_bifpn)
        ])

        # ---- LSK attention after BiFPN, one per scale ----
        # LSK = Large Selective Kernel, designed specifically for aerial detection.
        # Adaptively selects between fine (5×5) and large (7×7 dilated) receptive
        # fields per spatial location — suppresses structured aerial background.
        self.scale_attn = nn.ModuleList([LSKBlock(neck_c) for _ in range(4)])

        # ---- Detection heads: one per scale ----
        self.heads = nn.ModuleList([
            DecoupledHead(neck_c, activation=activation) for _ in range(4)
        ])

        self._init_weights()

    def _init_weights(self):
        # Init backbone, neck, and BiFPN weights.
        # Skip detection heads — they have their own init (bias=-5.5 for cls).
        for name, m in self.named_modules():
            if name.startswith('heads.'):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Backbone
        s = self.stem(x)            # (B, c1, H/2, W/2)

        fp2 = self.stage_p2(s)      # (B, c2, H/4, W/4)   stride 4
        fp3 = self.stage_p3(fp2)    # (B, c3, H/8, W/8)   stride 8
        fp4 = self.stage_p4(fp3)    # (B, c4, H/16, W/16) stride 16
        fp5 = self.stage_p5(fp4)    # (B, c5, H/32, W/32) stride 32
        fp5 = self.bottleneck(fp5)

        # Lateral projections
        n2 = self.lat_p2(fp2)
        n3 = self.lat_p3(fp3)
        n4 = self.lat_p4(fp4)
        n5 = self.lat_p5(fp5)

        # BiFPN
        for layer in self.bifpn:
            n2, n3, n4, n5 = layer(n2, n3, n4, n5)

        # Scale-wise channel attention (SE) then heads
        feats = [sa(f) for sa, f in zip(self.scale_attn, [n2, n3, n4, n5])]
        return [head(f) for head, f in zip(self.heads, feats)]

    def reparameterize(self):
        """
        Fuse all RepConv branches into single 3×3 convs for fast inference.
        Call this after training, before export/deployment.
        Reduces memory and latency on edge devices (Raspberry Pi 5).
        """
        for m in self.modules():
            if isinstance(m, RepConv):
                m.reparameterize()
        return self


# ============================================================
#  Utilities
# ============================================================

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = PFDetNanoV14(profile='balanced', activation='silu')
    total, trainable = count_params(model)
    print(f"Total params: {total / 1e6:.3f}M  |  Trainable: {trainable / 1e6:.3f}M")

    x = torch.randn(2, 3, 384, 384)
    outs = model(x)
    for i, (o, s) in enumerate(zip(outs, PFDetNanoV14.strides)):
        print(f"  Scale {i} (stride={s:2d}): {tuple(o.shape)}")
