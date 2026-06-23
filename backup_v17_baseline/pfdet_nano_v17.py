"""
PFDet-Nano v17 — FLOPs-efficient cho drone tiny-person (Jetson Nano B01, TensorRT FP16).
=========================================================================================

Vấn đề v15: params nhỏ (1.0M) NHƯNG FLOPs cao (12.9 GFLOPs@640) vì xử lý P2 (160×160) quá nặng
(head 5-conv + stage P2 đầy đủ). Head P2 một mình = 1.82 GMACs (28% tổng).

v17 GIỮ P2 cho người tí xíu nhưng làm RẺ:
  1) P2 path mỏng: chỉ 1 block nhẹ, KÊNH GIẢM (neck_c//2) -> head ở 160×160 rẻ ~4×.
  2) LiteHead: depthwise-sep (DW3×3 + PW) thay decoupled 5-conv -> rẻ mọi scale, vẫn tách cls/box.
  3) Backbone nặng dồn về P3–P5 (spatial nhỏ -> rẻ). AreaAttention chỉ ở P5.
  4) Reparameterizable (conv+bn fuse) cho TensorRT FP16.

Contract head GIỮ NGUYÊN v15: out (B,5,H,W) = cls(1)+box(4), strides [4,8,16,32]
-> plug thẳng PFDetLossV15 / decode / train_v3 hiện có. (NMS-free là phase sau.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pfdet_nano_v15 import (
    ConvBN, UIBBlock, CSPUIBStage, EdgeContextStem,
    AreaAttentionBlock, resolve_model_profile, count_params,
)


# ============================================================
#  LiteHead — depthwise-sep, tách cls/box, RẺ (thay decoupled 5-conv)
# ============================================================
class LiteHead(nn.Module):
    """shared DWConv3×3 + PW -> cls 1×1 (1ch) + box 1×1 (4ch). reg trực tiếp (DFL-free như YOLO26)."""

    def __init__(self, in_c, mid_c=None, activation='silu'):
        super().__init__()
        mid_c = mid_c or in_c
        act = nn.SiLU(inplace=True) if activation == 'silu' else nn.ReLU(inplace=True)
        # shared: depthwise 3×3 + pointwise -> mid_c
        self.shared = nn.Sequential(
            nn.Conv2d(in_c, in_c, 3, 1, 1, groups=in_c, bias=False), nn.BatchNorm2d(in_c), act,
            nn.Conv2d(in_c, mid_c, 1, bias=False), nn.BatchNorm2d(mid_c), act,
        )
        self.cls = nn.Conv2d(mid_c, 1, 1)
        self.box = nn.Conv2d(mid_c, 4, 1)
        # init cls bias âm (ít FP lúc đầu — bài học v12/v13)
        nn.init.constant_(self.cls.bias, -5.5)

    def forward(self, x):
        x = self.shared(x)
        return torch.cat([self.cls(x), self.box(x)], dim=1)   # (B,5,H,W)


# ============================================================
#  PFDet-Nano v17
# ============================================================
class PFDetNanoV17(nn.Module):
    strides = [4, 8, 16, 32]

    def __init__(self, base_c=None, neck_c=None, num_bifpn=None, area=None,
                 activation='silu', profile='balanced', use_area_attention=None,
                 p2_ratio=0.5, head_ratio=0.75):
        super().__init__()
        cfg = resolve_model_profile(profile)
        base_c = cfg['base_c'] if base_c is None else base_c
        neck_c = cfg['neck_c'] if neck_c is None else neck_c
        num_bifpn = cfg['num_bifpn'] if num_bifpn is None else num_bifpn
        area = cfg['area'] if area is None else area
        if use_area_attention is None:
            use_area_attention = cfg['use_area_attention']
        self.profile = str(profile).lower().strip()
        self.activation_name = activation
        # P2 dùng kênh GIẢM để rẻ ở 160×160 ; P3–P5 dùng neck_c
        nc2 = max(16, int(neck_c * p2_ratio))
        head_c = max(16, int(neck_c * head_ratio))
        self.model_config = {'profile': self.profile, 'activation': activation,
                             'base_c': int(base_c), 'neck_c': int(neck_c),
                             'num_bifpn': int(num_bifpn), 'area': int(area),
                             'use_area_attention': bool(use_area_attention),
                             'p2_ratio': p2_ratio, 'head_ratio': head_ratio}

        c1, c2, c3, c4, c5 = base_c, base_c*2, base_c*4, base_c*6, base_c*8

        # ---- Backbone: P2 MỎNG, compute dồn P3–P5 ----
        self.stem = EdgeContextStem(3, c1, activation=activation)          # P1 s2
        self.stage_p2 = UIBBlock(c1, c2, stride=2, activation=activation)   # P2 s4 — CHỈ 1 block (v15 dùng 3)
        self.stage_p3 = CSPUIBStage(c2, c3, n_blocks=2, stride=2, extra_dw=True, activation=activation)
        self.stage_p4 = CSPUIBStage(c3, c4, n_blocks=2, stride=2, extra_dw=True, activation=activation)
        self.stage_p5 = CSPUIBStage(c4, c5, n_blocks=2, stride=2, activation=activation)
        self.bottleneck_p5 = AreaAttentionBlock(c5, area=area, num_heads=2) if use_area_attention else nn.Identity()

        # ---- Neck: FPN top-down nhẹ. P2 ở kênh GIẢM (nc2) ----
        self.lat_p2 = ConvBN(c2, nc2, 1, activation=activation)
        self.lat_p3 = ConvBN(c3, neck_c, 1, activation=activation)
        self.lat_p4 = ConvBN(c4, neck_c, 1, activation=activation)
        self.lat_p5 = ConvBN(c5, neck_c, 1, activation=activation)
        # top-down: P5->P4->P3->P2 (cộng sau khi chiếu kênh)
        self.reduce_p5 = ConvBN(neck_c, neck_c, 1, activation=activation)
        self.td_p4 = UIBBlock(neck_c, neck_c, activation=activation)
        self.td_p3 = UIBBlock(neck_c, neck_c, activation=activation)
        self.p3_to_p2 = ConvBN(neck_c, nc2, 1, activation=activation)       # chiếu về kênh P2 nhỏ
        self.td_p2 = UIBBlock(nc2, nc2, activation=activation)

        # ---- Heads: LiteHead. P2 head ở kênh nhỏ -> rẻ ----
        self.head_p2 = LiteHead(nc2, mid_c=max(16, int(nc2 * head_ratio)), activation=activation)
        self.head_p3 = LiteHead(neck_c, mid_c=head_c, activation=activation)
        self.head_p4 = LiteHead(neck_c, mid_c=head_c, activation=activation)
        self.head_p5 = LiteHead(neck_c, mid_c=head_c, activation=activation)

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if 'head_' in name and ('.cls' in name or '.box' in name):
                continue   # giữ init head (cls bias -5.5)
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        s = self.stem(x)
        p2 = self.stage_p2(s)
        p3 = self.stage_p3(p2)
        p4 = self.stage_p4(p3)
        p5 = self.stage_p5(p4)
        p5 = self.bottleneck_p5(p5)

        # lateral
        l2, l3, l4, l5 = self.lat_p2(p2), self.lat_p3(p3), self.lat_p4(p4), self.lat_p5(p5)
        # top-down FPN
        n5 = self.reduce_p5(l5)
        n4 = self.td_p4(l4 + F.interpolate(n5, size=l4.shape[-2:], mode='nearest'))
        n3 = self.td_p3(l3 + F.interpolate(n4, size=l3.shape[-2:], mode='nearest'))
        n2 = self.td_p2(l2 + F.interpolate(self.p3_to_p2(n3), size=l2.shape[-2:], mode='nearest'))

        return [self.head_p2(n2), self.head_p3(n3), self.head_p4(n4), self.head_p5(n5)]

    @torch.no_grad()
    def reparameterize(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize') and m is not self:
                m.reparameterize()
        return self


if __name__ == "__main__":
    import copy, thop
    for prof in ['light']:
        m = PFDetNanoV17(profile=prof).eval()
        tot, _ = count_params(m)
        x = torch.randn(1, 3, 640, 640)
        outs = m(x)
        fl, _ = thop.profile(copy.deepcopy(m), inputs=(x,), verbose=False)
        print(f"[{prof}] {tot/1e6:.3f}M params | {fl/1e9:.2f} GMACs@640 | out {[tuple(o.shape) for o in outs]}")
