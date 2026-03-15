"""
PFDet-Nano: Person & Foot Detection for UAV Follow-Me
=====================================================
Lightweight anchor-free one-class detector optimized for Jetson Nano.
Architecture: CSP-Lite backbone + Lightweight BiFPN neck + Decoupled head

Design principles:
  - Depthwise separable convolutions everywhere to minimize FLOPs
  - Channel Shuffle for better feature mixing at low cost
  - Lightweight attention (Coordinate Attention) for spatial awareness
  - Multi-scale detection at stride 8/16/32
  - Decoupled classification and regression heads
  - Output: objectness, bbox(cx,cy,w,h), foot_point(fx,fy) per cell

Output encoding per cell (7 channels):
  ch0: objectness logit
  ch1: delta_x  (sigmoid -> offset from cell left)
  ch2: delta_y  (sigmoid -> offset from cell top)
  ch3: log_w    (exp -> width  relative to image)
  ch4: log_h    (exp -> height relative to image)
  ch5: foot_x   (sigmoid -> normalized image coord)
  ch6: foot_y   (sigmoid -> normalized image coord)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
#  Basic building blocks
# ============================================================

def autopad(k, p=None):
    """Auto-pad to 'same' for given kernel size."""
    return k // 2 if p is None else p


class ConvBnAct(nn.Module):
    """Standard Conv + BatchNorm + SiLU block."""
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Separable Convolution: DW(k) + PW(1)."""
    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBnAct(c_in, c_in, k, s, g=c_in, act=act)
        self.pw = ConvBnAct(c_in, c_out, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


def channel_shuffle(x, groups=2):
    """Shuffle channels for better information flow."""
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


# ============================================================
#  Coordinate Attention (lightweight spatial attention)
# ============================================================

class CoordAttention(nn.Module):
    """
    Coordinate Attention mechanism.
    Captures long-range spatial dependencies along both axes
    with minimal overhead (two 1D poolings + small FC).
    """
    def __init__(self, c, reduction=16):
        super().__init__()
        mid = max(8, c // reduction)
        self.fc_shared = nn.Sequential(
            nn.Conv2d(c, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )
        self.fc_h = nn.Conv2d(mid, c, 1, bias=True)
        self.fc_w = nn.Conv2d(mid, c, 1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        # Pool along W-axis -> (B,C,H,1) and H-axis -> (B,C,1,W)
        avg_h = x.mean(dim=3, keepdim=True)              # (B,C,H,1)
        avg_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (B,C,W,1)

        # Concat along spatial dim and pass through shared FC
        cat = torch.cat([avg_h, avg_w], dim=2)            # (B,C,H+W,1)
        cat = self.fc_shared(cat)                          # (B,mid,H+W,1)

        split_h, split_w = cat.split([H, W], dim=2)
        attn_h = self.fc_h(split_h).sigmoid()              # (B,C,H,1)
        attn_w = self.fc_w(split_w.permute(0, 1, 3, 2)).sigmoid()  # (B,C,1,W)

        return x * attn_h * attn_w


# ============================================================
#  CSP-Lite Block (backbone building block)
# ============================================================

class ShuffleBottleneck(nn.Module):
    """Lightweight bottleneck with channel shuffle."""
    def __init__(self, c, expansion=0.5):
        super().__init__()
        mid = int(c * expansion)
        self.cv1 = ConvBnAct(c, mid, 1, 1)
        self.cv2 = DWConv(mid, c, 3, 1)

    def forward(self, x):
        return x + channel_shuffle(self.cv2(self.cv1(x)))


class CSPLite(nn.Module):
    """
    CSP-style block: split channels, apply N bottlenecks to one half,
    then concat + fuse. Much more efficient than full C2f.
    """
    def __init__(self, c_in, c_out, n=2, use_attn=False):
        super().__init__()
        mid = c_out // 2
        self.cv1 = ConvBnAct(c_in, c_out, 1, 1)  # channel align
        self.blocks = nn.Sequential(*[ShuffleBottleneck(mid) for _ in range(n)])
        self.cv2 = ConvBnAct(c_out, c_out, 1, 1)  # final fuse
        self.attn = CoordAttention(c_out) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        a, b = x.chunk(2, dim=1)
        b = self.blocks(b)
        out = torch.cat([a, b], dim=1)
        out = self.cv2(out)
        return self.attn(out)


# ============================================================
#  BiFPN Neck (lightweight version)
# ============================================================

class FastFusion(nn.Module):
    """Weighted feature fusion with learnable weights (BiFPN-style)."""
    def __init__(self, n_inputs):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, inputs):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = sum(wi * xi for wi, xi in zip(w, inputs))
        return out


class BiFPNLayer(nn.Module):
    """Single BiFPN layer: top-down + bottom-up with weighted fusion."""
    def __init__(self, c):
        super().__init__()
        # Top-down fusions
        self.fuse_p4_td = FastFusion(2)
        self.fuse_p3_out = FastFusion(2)
        # Bottom-up fusions
        self.fuse_p4_out = FastFusion(3)
        self.fuse_p5_out = FastFusion(2)
        # Depthwise conv after each fusion
        self.conv_p4_td = DWConv(c, c, 3, 1)
        self.conv_p3_out = DWConv(c, c, 3, 1)
        self.conv_p4_out = DWConv(c, c, 3, 1)
        self.conv_p5_out = DWConv(c, c, 3, 1)

    def forward(self, p3, p4, p5):
        # --- Top-down ---
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.conv_p4_td(self.fuse_p4_td([p4, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_out = self.conv_p3_out(self.fuse_p3_out([p3, p4_up]))

        # --- Bottom-up ---
        p3_down = F.max_pool2d(p3_out, 2)
        p4_out = self.conv_p4_out(self.fuse_p4_out([p4, p4_td, p3_down]))

        p4_down = F.max_pool2d(p4_out, 2)
        p5_out = self.conv_p5_out(self.fuse_p5_out([p5, p4_down]))

        return p3_out, p4_out, p5_out


# ============================================================
#  Decoupled Detection Head
# ============================================================

class DecoupledHead(nn.Module):
    """
    Decoupled head: separate branches for classification and regression.
    This is a key insight from YOLOX - decoupling improves accuracy significantly.

    Output: 7 channels per cell
      [0]   objectness logit
      [1:5] box regression (dx, dy, log_w, log_h)
      [5:7] foot point (fx, fy)
    """
    def __init__(self, c):
        super().__init__()
        # Classification branch
        self.cls_stem = nn.Sequential(
            DWConv(c, c, 3, 1),
            DWConv(c, c, 3, 1),
        )
        self.cls_pred = nn.Conv2d(c, 1, 1, bias=True)

        # Regression branch (box + foot)
        self.reg_stem = nn.Sequential(
            DWConv(c, c, 3, 1),
            DWConv(c, c, 3, 1),
        )
        self.box_pred = nn.Conv2d(c, 4, 1, bias=True)
        self.foot_pred = nn.Conv2d(c, 2, 1, bias=True)

        # Initialize objectness bias for rare positive (following FCOS/YOLOX)
        nn.init.constant_(self.cls_pred.bias, -4.6)  # ~1% positive rate

    def forward(self, x):
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)
        obj = self.cls_pred(cls_feat)       # (B,1,H,W)
        box = self.box_pred(reg_feat)       # (B,4,H,W)
        foot = self.foot_pred(reg_feat)     # (B,2,H,W)
        return torch.cat([obj, box, foot], dim=1)  # (B,7,H,W)


# ============================================================
#  Full Model: PFDet-Nano
# ============================================================

class PFDetNano(nn.Module):
    """
    Person & Foot Detector - Nano version for Jetson Nano.

    Architecture:
      Backbone: Lightweight CSP with Coordinate Attention
      Neck: 2x BiFPN layers
      Head: Decoupled cls/reg heads at 3 scales

    Args:
        base_c: base channel width (default=24, use 32 for higher accuracy)
        num_bifpn: number of BiFPN repetitions (default=2)
    """
    def __init__(self, base_c=24, num_bifpn=2):
        super().__init__()
        c1 = base_c          # 24
        c2 = base_c * 2      # 48
        c3 = base_c * 4      # 96
        c4 = base_c * 6      # 144
        c5 = base_c * 8      # 192

        # --- Backbone ---
        self.stem = nn.Sequential(
            ConvBnAct(3, c1, 3, 2),           # /2
            ConvBnAct(c1, c1, 3, 1),
        )
        self.stage2 = nn.Sequential(
            ConvBnAct(c1, c2, 3, 2),           # /4
            CSPLite(c2, c2, n=1),
        )
        self.stage3 = nn.Sequential(
            ConvBnAct(c2, c3, 3, 2),           # /8   -> P3
            CSPLite(c3, c3, n=2, use_attn=True),
        )
        self.stage4 = nn.Sequential(
            ConvBnAct(c3, c4, 3, 2),           # /16  -> P4
            CSPLite(c4, c4, n=2, use_attn=True),
        )
        self.stage5 = nn.Sequential(
            ConvBnAct(c4, c5, 3, 2),           # /32  -> P5
            CSPLite(c5, c5, n=1, use_attn=True),
        )

        # --- Lateral convolutions (align channels for neck) ---
        neck_c = c3  # Use c3 as neck channel (96 for base_c=24)
        self.lat_p3 = ConvBnAct(c3, neck_c, 1, 1)
        self.lat_p4 = ConvBnAct(c4, neck_c, 1, 1)
        self.lat_p5 = ConvBnAct(c5, neck_c, 1, 1)

        # --- BiFPN Neck ---
        self.bifpn = nn.ModuleList([BiFPNLayer(neck_c) for _ in range(num_bifpn)])

        # --- Detection Heads (one per scale) ---
        self.head_p3 = DecoupledHead(neck_c)  # stride 8
        self.head_p4 = DecoupledHead(neck_c)  # stride 16
        self.head_p5 = DecoupledHead(neck_c)  # stride 32

        self.strides = [8, 16, 32]
        self._init_weights()
        self._init_head_biases()

    def _init_weights(self):
        """Initialize weights with best practices."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_head_biases(self):
        """Restore task-specific head initialization after generic Conv init."""
        for head in [self.head_p3, self.head_p4, self.head_p5]:
            nn.init.normal_(head.cls_pred.weight, mean=0.0, std=0.01)
            nn.init.constant_(head.cls_pred.bias, -4.6)
            nn.init.normal_(head.box_pred.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.box_pred.bias)
            nn.init.normal_(head.foot_pred.weight, mean=0.0, std=0.01)
            nn.init.zeros_(head.foot_pred.bias)

    def forward(self, x):
        # Backbone
        x = self.stem(x)        # /2
        x = self.stage2(x)      # /4
        p3 = self.stage3(x)     # /8
        p4 = self.stage4(p3)    # /16
        p5 = self.stage5(p4)    # /32

        # Lateral alignment
        p3 = self.lat_p3(p3)
        p4 = self.lat_p4(p4)
        p5 = self.lat_p5(p5)

        # BiFPN neck
        for bifpn_layer in self.bifpn:
            p3, p4, p5 = bifpn_layer(p3, p4, p5)

        # Detection heads
        out_p3 = self.head_p3(p3)
        out_p4 = self.head_p4(p4)
        out_p5 = self.head_p5(p5)

        return [out_p3, out_p4, out_p5]


# ============================================================
#  Utility: count parameters and FLOPs estimate
# ============================================================

def count_params(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, img_size=416):
    """Rough FLOPs estimate using thop if available, else manual."""
    try:
        from thop import profile
        x = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except ImportError:
        return None, None


# ============================================================
#  Main: test shapes and param count
# ============================================================

if __name__ == "__main__":
    for base_c in [24, 32]:
        print(f"\n{'='*50}")
        print(f"PFDet-Nano (base_c={base_c})")
        print(f"{'='*50}")

        model = PFDetNano(base_c=base_c)
        total, trainable = count_params(model)
        print(f"Parameters: {total/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)")

        x = torch.randn(1, 3, 416, 416)
        outs = model(x)
        for i, o in enumerate(outs):
            print(f"  Scale {i} (stride {model.strides[i]}): {tuple(o.shape)}")

        flops, _ = estimate_flops(model, 416)
        if flops:
            print(f"FLOPs: {flops/1e9:.2f}G")
