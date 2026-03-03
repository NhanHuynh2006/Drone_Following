"""
PFDet-Nano: Person & Foot Detection for UAV Follow-Me.
Lightweight anchor-free detector with improved head initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):
    return k // 2 if p is None else p


class ConvBnAct(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, act=True):
        super().__init__()
        self.dw = ConvBnAct(c_in, c_in, k, s, g=c_in, act=act)
        self.pw = ConvBnAct(c_in, c_out, 1, 1, act=act)

    def forward(self, x):
        return self.pw(self.dw(x))


def channel_shuffle(x, groups=2):
    B, C, H, W = x.shape
    x = x.view(B, groups, C // groups, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class CoordAttention(nn.Module):
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
        avg_h = x.mean(dim=3, keepdim=True)
        avg_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        cat = torch.cat([avg_h, avg_w], dim=2)
        cat = self.fc_shared(cat)
        split_h, split_w = cat.split([H, W], dim=2)
        attn_h = self.fc_h(split_h).sigmoid()
        attn_w = self.fc_w(split_w.permute(0, 1, 3, 2)).sigmoid()
        return x * attn_h * attn_w


class ShuffleBottleneck(nn.Module):
    def __init__(self, c, expansion=0.5):
        super().__init__()
        mid = max(8, int(c * expansion))
        self.cv1 = ConvBnAct(c, mid, 1, 1)
        self.cv2 = DWConv(mid, c, 3, 1)

    def forward(self, x):
        return x + channel_shuffle(self.cv2(self.cv1(x)))


class CSPLite(nn.Module):
    def __init__(self, c_in, c_out, n=2, use_attn=False):
        super().__init__()
        mid = c_out // 2
        self.cv1 = ConvBnAct(c_in, c_out, 1, 1)
        self.blocks = nn.Sequential(*[ShuffleBottleneck(mid) for _ in range(n)])
        self.cv2 = ConvBnAct(c_out, c_out, 1, 1)
        self.attn = CoordAttention(c_out) if use_attn else nn.Identity()

    def forward(self, x):
        x = self.cv1(x)
        a, b = x.chunk(2, dim=1)
        b = self.blocks(b)
        out = torch.cat([a, b], dim=1)
        out = self.cv2(out)
        return self.attn(out)


class FastFusion(nn.Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n_inputs, dtype=torch.float32))
        self.eps = 1e-4

    def forward(self, inputs):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return sum(wi * xi for wi, xi in zip(w, inputs))


class BiFPNLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.fuse_p4_td = FastFusion(2)
        self.fuse_p3_out = FastFusion(2)
        self.fuse_p4_out = FastFusion(3)
        self.fuse_p5_out = FastFusion(2)
        self.conv_p4_td = DWConv(c, c, 3, 1)
        self.conv_p3_out = DWConv(c, c, 3, 1)
        self.conv_p4_out = DWConv(c, c, 3, 1)
        self.conv_p5_out = DWConv(c, c, 3, 1)

    def forward(self, p3, p4, p5):
        p5_up = F.interpolate(p5, size=p4.shape[-2:], mode='nearest')
        p4_td = self.conv_p4_td(self.fuse_p4_td([p4, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest')
        p3_out = self.conv_p3_out(self.fuse_p3_out([p3, p4_up]))

        p3_down = F.max_pool2d(p3_out, 2)
        p4_out = self.conv_p4_out(self.fuse_p4_out([p4, p4_td, p3_down]))

        p4_down = F.max_pool2d(p4_out, 2)
        p5_out = self.conv_p5_out(self.fuse_p5_out([p5, p4_down]))
        return p3_out, p4_out, p5_out


class DecoupledHead(nn.Module):
    def __init__(self, c, obj_prior_prob=0.01):
        super().__init__()
        self.cls_stem = nn.Sequential(DWConv(c, c, 3, 1), DWConv(c, c, 3, 1))
        self.reg_stem = nn.Sequential(DWConv(c, c, 3, 1), DWConv(c, c, 3, 1))
        self.cls_pred = nn.Conv2d(c, 1, 1, bias=True)
        self.box_pred = nn.Conv2d(c, 4, 1, bias=True)
        self.foot_pred = nn.Conv2d(c, 2, 1, bias=True)
        self.obj_prior_prob = obj_prior_prob

    def init_output_biases(self):
        prior_bias = -torch.log(torch.tensor((1.0 - self.obj_prior_prob) / self.obj_prior_prob))
        nn.init.constant_(self.cls_pred.bias, float(prior_bias))
        nn.init.normal_(self.cls_pred.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.box_pred.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.foot_pred.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.box_pred.bias)
        nn.init.zeros_(self.foot_pred.bias)

    def forward(self, x):
        cls_feat = self.cls_stem(x)
        reg_feat = self.reg_stem(x)
        obj = self.cls_pred(cls_feat)
        box = self.box_pred(reg_feat)
        foot = self.foot_pred(reg_feat)
        return torch.cat([obj, box, foot], dim=1)


class PFDetNano(nn.Module):
    def __init__(self, base_c=24, num_bifpn=2):
        super().__init__()
        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 4
        c4 = base_c * 6
        c5 = base_c * 8

        self.stem = nn.Sequential(
            ConvBnAct(3, c1, 3, 2),
            ConvBnAct(c1, c1, 3, 1),
        )
        self.stage2 = nn.Sequential(
            ConvBnAct(c1, c2, 3, 2),
            CSPLite(c2, c2, n=1),
        )
        self.stage3 = nn.Sequential(
            ConvBnAct(c2, c3, 3, 2),
            CSPLite(c3, c3, n=2, use_attn=True),
        )
        self.stage4 = nn.Sequential(
            ConvBnAct(c3, c4, 3, 2),
            CSPLite(c4, c4, n=2, use_attn=True),
        )
        self.stage5 = nn.Sequential(
            ConvBnAct(c4, c5, 3, 2),
            CSPLite(c5, c5, n=1, use_attn=True),
        )

        neck_c = c3
        self.lat_p3 = ConvBnAct(c3, neck_c, 1, 1)
        self.lat_p4 = ConvBnAct(c4, neck_c, 1, 1)
        self.lat_p5 = ConvBnAct(c5, neck_c, 1, 1)
        self.bifpn = nn.ModuleList([BiFPNLayer(neck_c) for _ in range(num_bifpn)])
        self.head_p3 = DecoupledHead(neck_c)
        self.head_p4 = DecoupledHead(neck_c)
        self.head_p5 = DecoupledHead(neck_c)
        self.strides = [8, 16, 32]
        self._init_weights()
        self._init_head_priors()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                m.eps = 1e-3
                m.momentum = 0.03

    def _init_head_priors(self):
        for head in (self.head_p3, self.head_p4, self.head_p5):
            head.init_output_biases()

    def forward(self, x):
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.stage5(p4)
        p3 = self.lat_p3(p3)
        p4 = self.lat_p4(p4)
        p5 = self.lat_p5(p5)
        for bifpn_layer in self.bifpn:
            p3, p4, p5 = bifpn_layer(p3, p4, p5)
        return [self.head_p3(p3), self.head_p4(p4), self.head_p5(p5)]


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, img_size=416):
    try:
        from thop import profile
        x = torch.randn(1, 3, img_size, img_size)
        flops, params = profile(model, inputs=(x,), verbose=False)
        return flops, params
    except ImportError:
        return None, None


if __name__ == '__main__':
    for base_c in [24, 32]:
        model = PFDetNano(base_c=base_c)
        total, trainable = count_params(model)
        print(f'PFDet-Nano (base_c={base_c}) -> {total/1e6:.2f}M params')
        x = torch.randn(1, 3, 416, 416)
        outs = model(x)
        for i, out in enumerate(outs):
            print(i, out.shape)
