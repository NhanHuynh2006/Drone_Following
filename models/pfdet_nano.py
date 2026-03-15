"""
PFDet-Nano v5: Lightweight Person Detector for Raspberry Pi 5
=============================================================
Anchor-free single-class detector with FPN + PAN neck.

Output per cell (5 channels):
  ch0: objectness logit
  ch1: delta_x  (sigmoid*2 - 0.5 -> offset from cell, range [-0.5, 1.5])
  ch2: delta_y  (sigmoid*2 - 0.5 -> offset from cell, range [-0.5, 1.5])
  ch3: log_w    (exp -> width in stride units)
  ch4: log_h    (exp -> height in stride units)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBN(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, k // 2, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class InvertedResidual(nn.Module):
    """MobileNetV2-style inverted residual: expand -> DW -> project."""
    def __init__(self, c, expand=2):
        super().__init__()
        mid = c * expand
        self.expand = ConvBN(c, mid, 1)
        self.dw = ConvBN(mid, mid, 3, g=mid)
        self.project = ConvBN(mid, c, 1, act=False)

    def forward(self, x):
        return x + self.project(self.dw(self.expand(x)))


class PFDetNano(nn.Module):
    """
    Person detector with FPN + PAN neck.
    base_c=32 -> ~900K params (fine for RPi5).
    """
    def __init__(self, base_c=32, **_kwargs):
        super().__init__()
        c2 = base_c       # 32
        c3 = base_c * 2   # 64
        c4 = base_c * 4   # 128
        c5 = base_c * 6   # 192

        # Backbone
        self.stem = nn.Sequential(ConvBN(3, 16, 3, 2), ConvBN(16, 16, 3, 1))
        self.stage1 = nn.Sequential(ConvBN(16, c2, 3, 2), InvertedResidual(c2))          # /4
        self.stage2 = nn.Sequential(ConvBN(c2, c3, 3, 2), InvertedResidual(c3), InvertedResidual(c3))  # /8 P3
        self.stage3 = nn.Sequential(ConvBN(c3, c4, 3, 2), InvertedResidual(c4), InvertedResidual(c4))  # /16 P4
        self.stage4 = nn.Sequential(ConvBN(c4, c5, 3, 2), InvertedResidual(c5), InvertedResidual(c5))  # /32 P5

        # FPN neck (top-down)
        neck_c = c3  # 64
        self.lat3 = ConvBN(c3, neck_c, 1)
        self.lat4 = ConvBN(c4, neck_c, 1)
        self.lat5 = ConvBN(c5, neck_c, 1)
        self.fpn_smooth4 = ConvBN(neck_c, neck_c, 3)
        self.fpn_smooth3 = ConvBN(neck_c, neck_c, 3)

        # PAN neck (bottom-up)
        self.pan_down3 = ConvBN(neck_c, neck_c, 3, s=2)  # P3 -> P4
        self.pan_smooth4 = ConvBN(neck_c, neck_c, 3)
        self.pan_down4 = ConvBN(neck_c, neck_c, 3, s=2)  # P4 -> P5
        self.pan_smooth5 = ConvBN(neck_c, neck_c, 3)

        # Per-scale detection heads (5ch: obj, dx, dy, lw, lh)
        self.head_p3 = self._make_head(neck_c)
        self.head_p4 = self._make_head(neck_c)
        self.head_p5 = self._make_head(neck_c)

        self.strides = [8, 16, 32]
        self._init_weights()

    def _make_head(self, c):
        return nn.Sequential(
            ConvBN(c, c, 3),
            ConvBN(c, c, 3),
            nn.Conv2d(c, 5, 1, bias=True),
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Init head objectness bias to low confidence
        for head in [self.head_p3, self.head_p4, self.head_p5]:
            pred_conv = head[-1]
            nn.init.constant_(pred_conv.bias[0], -4.0)  # sigmoid(-4) ~ 0.018

    def forward(self, x):
        # Backbone
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        # FPN top-down
        p5_lat = self.lat5(p5)
        p4_lat = self.lat4(p4)
        p3_lat = self.lat3(p3)

        p4_fpn = self.fpn_smooth4(p4_lat + F.interpolate(p5_lat, size=p4_lat.shape[2:], mode='nearest'))
        p3_fpn = self.fpn_smooth3(p3_lat + F.interpolate(p4_fpn, size=p3_lat.shape[2:], mode='nearest'))

        # PAN bottom-up
        p4_pan = self.pan_smooth4(p4_fpn + self.pan_down3(p3_fpn))
        p5_pan = self.pan_smooth5(p5_lat + self.pan_down4(p4_pan))

        return [self.head_p3(p3_fpn), self.head_p4(p4_pan), self.head_p5(p5_pan)]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, img_size=320):
    try:
        from thop import profile
        x = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except ImportError:
        return None, None


if __name__ == "__main__":
    model = PFDetNano(base_c=32)
    total, _ = count_params(model)
    print(f"Params: {total:,} ({total/1e6:.3f}M)")
    x = torch.randn(1, 3, 416, 416)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"  Scale {i} (stride {model.strides[i]}): {tuple(o.shape)}")
