"""
PFDet-Nano v16 — Stability-First Hybrid for Drone Person Detection
==================================================================

Design goal:
  Keep the tiny-person-friendly spatial prior of v14 in the early backbone,
  while avoiding the most aggressive backbone changes introduced in v15.

Backbone policy:
  1. P2 stays plain UIB stack  -> no backbone LSK at the finest scale
  2. P3/P4 revert to full UIB stacks -> preserve dense local detail
  3. P5 uses a light CSPUIB stage -> keep deeper semantics efficient
  4. AreaAttention only at P5 -> keep global context where it is cheapest

This file is standalone and does not import model code from v14/v15.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


MODEL_PROFILES = {
    "light": {
        "base_c": 24,
        "neck_c": 48,
        "num_bifpn": 1,
        "area": 3,
        "use_area_attention": True,
    },
    "balanced": {
        "base_c": 32,
        "neck_c": 64,
        "num_bifpn": 2,
        "area": 3,
        "use_area_attention": True,
    },
}

DEFAULT_EXPORT_OUTPUT_NAMES = ("output_p2", "output_p3", "output_p4", "output_p5")


def build_activation(name="silu", act=True):
    if not act:
        return nn.Identity()
    name = str(name).lower().strip()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("identity", "none"):
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


def resolve_model_profile(profile="balanced"):
    profile = str(profile).lower().strip()
    if profile not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown PFDet profile: {profile!r}. Available: {sorted(MODEL_PROFILES)}"
        )
    return dict(MODEL_PROFILES[profile])


class ConvBN(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True, activation="silu"):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03)
        self.act = build_activation(activation, act=act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class RepConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, g=1, activation="silu"):
        super().__init__()
        assert k == 3
        self.in_c, self.out_c, self.s, self.g = in_c, out_c, s, g
        self.act = build_activation(activation)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, s, 1, groups=g, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, s, 0, groups=g, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        )
        self.identity_bn = (
            nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.03)
            if s == 1 and in_c == out_c
            else None
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
        if self._reparam:
            return
        w3, b3 = self._fuse_bn(self.conv3)
        w1, b1 = self._fuse_bn(self.conv1)
        w1_pad = F.pad(w1, [1, 1, 1, 1])
        if self.identity_bn is not None:
            w_id, b_id = self._fuse_identity_bn(self.identity_bn)
        else:
            w_id = torch.zeros_like(w3)
            b_id = torch.zeros_like(b3)
        w_fused = w3 + w1_pad + w_id
        b_fused = b3 + b1 + b_id
        self.reparam_conv = nn.Conv2d(
            self.in_c, self.out_c, 3, self.s, 1, groups=self.g, bias=True
        )
        self.reparam_conv.weight.data = w_fused
        self.reparam_conv.bias.data = b_fused
        del self.conv3, self.conv1
        if self.identity_bn is not None:
            del self.identity_bn
        self._reparam = True

    def _fuse_bn(self, seq):
        conv, bn = seq[0], seq[1]
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return conv.weight * scale.view(-1, 1, 1, 1), bn.bias - bn.running_mean * scale

    def _fuse_identity_bn(self, bn):
        in_c = self.in_c
        w_id = torch.zeros(
            in_c, in_c // self.g, 3, 3, device=bn.weight.device, dtype=bn.weight.dtype
        )
        for i in range(in_c):
            w_id[i, i % (in_c // self.g), 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        scale = bn.weight / std
        return w_id * scale.view(-1, 1, 1, 1), bn.bias - bn.running_mean * scale


class FactDW(nn.Module):
    def __init__(self, c, k=5, activation="silu"):
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
    def __init__(self, in_c, out_c, stride=1, expand_ratio=4, extra_dw=False, activation="silu"):
        super().__init__()
        mid_c = max(in_c * expand_ratio, in_c)
        self.has_identity_skip = stride == 1 and in_c == out_c
        layers = [
            nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False),
            nn.BatchNorm2d(in_c, eps=1e-3, momentum=0.03),
            build_activation(activation),
            nn.Conv2d(in_c, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c, eps=1e-3, momentum=0.03),
            build_activation(activation),
        ]
        if extra_dw:
            layers.append(FactDW(mid_c, k=5, activation=activation))
            layers.append(build_activation(activation))
        layers += [
            nn.Conv2d(mid_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c, eps=1e-3, momentum=0.03),
        ]
        self.block = nn.Sequential(*layers)
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
        return out + x if self.has_identity_skip else out + self.skip(x)


class UIBStack(nn.Sequential):
    def __init__(self, in_c, out_c, n_blocks=3, stride=2, extra_dw_last=False, activation="silu"):
        blocks = []
        for i in range(n_blocks):
            block_in = in_c if i == 0 else out_c
            block_stride = stride if i == 0 else 1
            use_extra = extra_dw_last and i == n_blocks - 1
            blocks.append(
                UIBBlock(
                    block_in,
                    out_c,
                    stride=block_stride,
                    extra_dw=use_extra,
                    activation=activation,
                )
            )
        super().__init__(*blocks)


class EdgeContextStem(nn.Module):
    def __init__(self, in_c=3, out_c=32, activation="silu"):
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
    def __init__(self, c, area=3, num_heads=2):
        super().__init__()
        assert c % num_heads == 0
        self.area = area
        self.num_heads = num_heads
        self.head_dim = c // num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(c)
        self.qkv = nn.Linear(c, c * 3, bias=False)
        self.proj = nn.Linear(c, c, bias=False)
        self.norm2 = nn.LayerNorm(c)
        self.ffn = nn.Sequential(nn.Linear(c, c * 2), nn.GELU(), nn.Linear(c * 2, c))

    def _attn(self, x):
        n, l, c = x.shape
        qkv = self.qkv(self.norm1(x))
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(n, l, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        out = (attn.softmax(dim=-1) @ v).transpose(1, 2).contiguous().view(n, l, c)
        return x + self.proj(out)

    def forward(self, x):
        b, c, h, w = x.shape
        a = self.area
        p_h = (a - h % a) % a
        p_w = (a - w % a) % a
        if p_h > 0 or p_w > 0:
            x = F.pad(x, (0, p_w, 0, p_h))
        h2, w2 = h + p_h, w + p_w
        n_h, n_w = h2 // a, w2 // a
        xp = (
            x.view(b, c, n_h, a, n_w, a)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .view(b * n_h * n_w, a * a, c)
        )
        xp = self._attn(xp)
        xp = xp + self.ffn(self.norm2(xp))
        xp = (
            xp.view(b, n_h, n_w, a, a, c)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(b, c, h2, w2)
        )
        return xp[:, :, :h, :w]


class LSKBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw0 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.dw1 = nn.Conv2d(c, c, 7, padding=9, dilation=3, groups=c)
        self.attn_fc = nn.Sequential(
            nn.Conv2d(c * 2, c // 2, 1),
            nn.Conv2d(c // 2, 2, 1),
        )
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        a0, a1 = self.dw0(x), self.dw1(x)
        attn = self.attn_fc(torch.cat([a0, a1], dim=1)).softmax(dim=1)
        return self.proj(a0 * attn[:, 0:1] + a1 * attn[:, 1:2]) + x


class CSPUIBStage(nn.Module):
    def __init__(self, in_c, out_c, n_blocks, stride=2, extra_dw=False, activation="silu"):
        super().__init__()
        mid = out_c // 2
        self.down = UIBBlock(in_c, out_c, stride=stride, activation=activation)
        self.split_a = ConvBN(out_c, mid, 1, activation=activation)
        self.split_b = ConvBN(out_c, mid, 1, activation=activation)

        blocks = []
        for i in range(n_blocks):
            use_extra = extra_dw and i == n_blocks - 1
            blocks.append(UIBBlock(mid, mid, extra_dw=use_extra, activation=activation))
        self.blocks = nn.Sequential(*blocks)
        self.fuse = ConvBN(out_c, out_c, 1, activation=activation)

    def forward(self, x):
        x = self.down(x)
        a = self.blocks(self.split_a(x))
        b = self.split_b(x)
        return self.fuse(torch.cat([a, b], dim=1))


class WeightedFusion(nn.Module):
    def __init__(self, n=2, eps=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, feats):
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        return sum(f * wi for f, wi in zip(feats, w))


class BiFPNLayer(nn.Module):
    def __init__(self, c, activation="silu"):
        super().__init__()
        self.td_p4 = WeightedFusion(2)
        self.td_p3 = WeightedFusion(2)
        self.td_p2 = WeightedFusion(2)
        self.bu_p3 = WeightedFusion(3)
        self.bu_p4 = WeightedFusion(3)
        self.bu_p5 = WeightedFusion(2)
        self.conv_td4 = RepConv(c, c, 3, activation=activation)
        self.conv_td3 = RepConv(c, c, 3, activation=activation)
        self.conv_td2 = RepConv(c, c, 3, activation=activation)
        self.conv_bu3 = RepConv(c, c, 3, activation=activation)
        self.conv_bu4 = RepConv(c, c, 3, activation=activation)
        self.conv_bu5 = RepConv(c, c, 3, activation=activation)

    def _up(self, x):
        return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

    def _down(self, x):
        return F.max_pool2d(x, kernel_size=2, stride=2)

    def forward(self, p2, p3, p4, p5):
        p4_td = self.conv_td4(self.td_p4([p4, self._up(p5)]))
        p3_td = self.conv_td3(self.td_p3([p3, self._up(p4_td)]))
        p2_td = self.conv_td2(self.td_p2([p2, self._up(p3_td)]))
        p3_out = self.conv_bu3(self.bu_p3([p3, p3_td, self._down(p2_td)]))
        p4_out = self.conv_bu4(self.bu_p4([p4, p4_td, self._down(p3_out)]))
        p5_out = self.conv_bu5(self.bu_p5([p5, self._down(p4_out)]))
        return p2_td, p3_out, p4_out, p5_out


class DecoupledHead(nn.Module):
    def __init__(self, in_c, mid_c=None, activation="silu"):
        super().__init__()
        mid_c = mid_c or in_c
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
        cls_conv = self.cls_branch[-1]
        nn.init.normal_(cls_conv.weight, std=0.01)
        nn.init.constant_(cls_conv.bias, -5.5)
        box_conv = self.box_branch[-1]
        nn.init.normal_(box_conv.weight, std=0.01)
        nn.init.zeros_(box_conv.bias)

    def forward(self, x):
        feat = self.shared(x)
        return torch.cat([self.cls_branch(feat), self.box_branch(feat)], dim=1)


class PFDetNanoV16(nn.Module):
    """
    v16 = stable early backbone + lighter deep semantic stage.

    Compared with v15:
      - removes P2 backbone LSK
      - removes AreaAttention at P4
      - restores plain UIB stacks at P3/P4
      - keeps only a light CSP stage at P5
    """

    strides = [4, 8, 16, 32]

    def __init__(
        self,
        base_c=None,
        neck_c=None,
        num_bifpn=None,
        area=None,
        activation="silu",
        profile="balanced",
        use_area_attention=None,
    ):
        super().__init__()
        profile_cfg = resolve_model_profile(profile)
        base_c = profile_cfg["base_c"] if base_c is None else base_c
        neck_c = profile_cfg["neck_c"] if neck_c is None else neck_c
        num_bifpn = profile_cfg["num_bifpn"] if num_bifpn is None else num_bifpn
        area = profile_cfg["area"] if area is None else area
        if use_area_attention is None:
            use_area_attention = profile_cfg["use_area_attention"]

        self.profile = str(profile).lower().strip()
        self.activation_name = str(activation).lower().strip()
        self.use_area_attention = bool(use_area_attention)
        self.export_output_names = DEFAULT_EXPORT_OUTPUT_NAMES
        self.model_config = {
            "profile": self.profile,
            "activation": self.activation_name,
            "base_c": int(base_c),
            "neck_c": int(neck_c),
            "num_bifpn": int(num_bifpn),
            "area": int(area),
            "use_area_attention": self.use_area_attention,
        }

        c1 = base_c
        c2 = base_c * 2
        c3 = base_c * 3
        c4 = base_c * 4
        c5 = base_c * 4

        self.stem = EdgeContextStem(3, c1, activation=activation)

        self.stage_p2 = UIBStack(c1, c2, n_blocks=3, stride=2, extra_dw_last=True, activation=activation)
        self.stage_p3 = UIBStack(c2, c3, n_blocks=3, stride=2, extra_dw_last=False, activation=activation)
        self.stage_p4 = UIBStack(c3, c4, n_blocks=3, stride=2, extra_dw_last=False, activation=activation)
        self.stage_p5 = CSPUIBStage(c4, c5, n_blocks=2, stride=2, extra_dw=False, activation=activation)

        self.bottleneck_p5 = (
            AreaAttentionBlock(c5, area=area, num_heads=2)
            if self.use_area_attention
            else nn.Identity()
        )

        self.lat_p2 = ConvBN(c2, neck_c, 1, activation=activation)
        self.lat_p3 = ConvBN(c3, neck_c, 1, activation=activation)
        self.lat_p4 = ConvBN(c4, neck_c, 1, activation=activation)
        self.lat_p5 = ConvBN(c5, neck_c, 1, activation=activation)

        self.bifpn = nn.ModuleList(
            [BiFPNLayer(neck_c, activation=activation) for _ in range(num_bifpn)]
        )
        self.scale_attn = nn.ModuleList([LSKBlock(neck_c) for _ in range(4)])
        self.heads = nn.ModuleList([DecoupledHead(neck_c, activation=activation) for _ in range(4)])

        self._init_weights()

    def _init_weights(self):
        for name, m in self.named_modules():
            if name.startswith("heads."):
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        s = self.stem(x)

        fp2 = self.stage_p2(s)
        fp3 = self.stage_p3(fp2)
        fp4 = self.stage_p4(fp3)
        fp5 = self.stage_p5(fp4)
        fp5 = self.bottleneck_p5(fp5)

        n2 = self.lat_p2(fp2)
        n3 = self.lat_p3(fp3)
        n4 = self.lat_p4(fp4)
        n5 = self.lat_p5(fp5)

        for layer in self.bifpn:
            n2, n3, n4, n5 = layer(n2, n3, n4, n5)

        feats = [sa(f) for sa, f in zip(self.scale_attn, [n2, n3, n4, n5])]
        return [head(f) for head, f in zip(self.heads, feats)]

    def reparameterize(self):
        for m in self.modules():
            if isinstance(m, RepConv):
                m.reparameterize()
        return self


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    for profile in ["light", "balanced"]:
        model = PFDetNanoV16(profile=profile)
        total, _ = count_params(model)
        x = torch.randn(1, 3, 640, 640)
        outs = model(x)
        print(f"[{profile}] {total / 1e6:.2f}M params")
        for i, (o, s) in enumerate(zip(outs, PFDetNanoV16.strides)):
            print(f"  P{i+2} (stride={s}): {tuple(o.shape)}")
