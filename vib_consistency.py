"""
vib_consistency.py — ĐÓNG GÓP CHÍNH: detector kháng rung/nhoè khi bay drone
============================================================================

Hai phần:
  1) corrupt_batch(): mô phỏng rung drone = motion blur có hướng (camera shake khi phơi sáng)
     + nhiễu cảm biến + dao động sáng. GIỮ NGUYÊN toạ độ box (blur/noise/bright không dời vật)
     -> dùng được cho cả train (random) lẫn benchmark (severity cố định).
  2) VibConsistencyLoss(): ép dự đoán trên ảnh RUNG khớp dự đoán trên ảnh SẠCH (sạch = anchor,
     detach). Dạy model "nhìn xuyên" nhoè. TRAIN-TIME ONLY -> inference/Pi không đổi.

Cơ chế giống KD (hai forward + match) nhưng cùng MỘT model, cùng channel -> không cần adapter.
"""

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
#  Mô phỏng rung drone (box-preserving)
# ============================================================
def motion_blur_kernel(length, angle_deg, device, dtype):
    """Kernel nhoè chuyển động: 1 vạch dài `length` theo góc `angle_deg`, chuẩn hoá tổng=1."""
    length = max(3, int(length) | 1)  # lẻ
    k = torch.zeros(length, length, device=device, dtype=dtype)
    c = (length - 1) / 2.0
    a = math.radians(angle_deg)
    dx, dy = math.cos(a), math.sin(a)
    for t in range(length):
        x = int(round(c + (t - c) * dx))
        y = int(round(c + (t - c) * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    s = k.sum()
    return k / s if s > 0 else k


_PRESETS = {1: (5, 0.02, 0.10), 2: (9, 0.04, 0.20), 3: (15, 0.06, 0.30)}  # (blur_len, noise_std, bright)


def corrupt_batch(imgs, severity=None, angle=None):
    """
    imgs: (B,3,H,W). severity: 1/2/3 ; None -> random 1..3 (dùng khi train).
    angle: None -> random hướng rung (thực tế). Trả ảnh đã "rung", cùng shape, cùng range.
    """
    if severity is None:
        severity = random.choice([1, 2, 3])
    L, ns, br = _PRESETS.get(int(severity), _PRESETS[2])
    ang = random.uniform(0, 180) if angle is None else angle

    k = motion_blur_kernel(L, ang, imgs.device, imgs.dtype)        # (L,L)
    Lk = k.shape[0]
    w = k.view(1, 1, Lk, Lk).repeat(imgs.shape[1], 1, 1, 1)        # depthwise
    out = F.conv2d(imgs, w, padding=Lk // 2, groups=imgs.shape[1])

    if ns > 0:
        out = out + torch.randn_like(out) * ns
    if br > 0:
        out = out * (1.0 + random.uniform(-br, br))
    return out


# ============================================================
#  Consistency loss: ảnh rung -> khớp ảnh sạch (anchor detach)
# ============================================================
class VibConsistencyLoss(nn.Module):
    def __init__(self, temperature=2.0, conf_thr=0.1,
                 w_cls=1.0, w_box=1.0, w_feat=0.5, use_feature=True):
        super().__init__()
        self.T = float(temperature)
        self.conf_thr = float(conf_thr)
        self.w_cls, self.w_box, self.w_feat = w_cls, w_box, w_feat
        self.use_feature = use_feature

    def forward(self, vib_outs, clean_outs, vib_feats=None, clean_feats=None):
        T, eps = self.T, 1e-6
        kc = vib_outs[0].new_zeros(())
        kb = vib_outs[0].new_zeros(())
        for vo, co in zip(vib_outs, clean_outs):
            v_cls, v_box = vo[:, 0:1], vo[:, 1:5]
            c_cls, c_box = co[:, 0:1].detach(), co[:, 1:5].detach()
            c_prob = torch.sigmoid(c_cls / T)
            kc = kc + F.binary_cross_entropy_with_logits(v_cls / T, c_prob) * (T * T)
            mask = (torch.sigmoid(c_cls) > self.conf_thr).float()
            kb = kb + (F.smooth_l1_loss(v_box, c_box, reduction="none") * mask).sum() / (mask.sum() * 4 + eps)
        n = len(vib_outs)
        kc, kb = kc / n, kb / n
        loss = self.w_cls * kc + self.w_box * kb
        logs = {"vib_cls": float(kc.detach()), "vib_box": float(kb.detach())}
        if self.use_feature and vib_feats is not None and clean_feats is not None:
            kf = vib_outs[0].new_zeros(())
            for vf, cf in zip(vib_feats, clean_feats):
                kf = kf + F.mse_loss(vf, cf.detach())  # cùng model -> cùng channel, khỏi adapter
            kf = kf / len(vib_feats)
            loss = loss + self.w_feat * kf
            logs["vib_feat"] = float(kf.detach())
        logs["vib_total"] = float(loss.detach())
        return loss, logs


# ============================================================
#  Test
# ============================================================
if __name__ == "__main__":
    from aux_train import build_student
    torch.manual_seed(0)

    print("=== Test 1: corrupt_batch giữ shape, khác ảnh gốc ===")
    x = torch.rand(1, 3, 320, 320)
    for sev in [1, 2, 3]:
        xc = corrupt_batch(x, severity=sev)
        diff = (xc - x).abs().mean().item()
        print(f"  severity {sev}: shape {tuple(xc.shape)}  |Δ|mean={diff:.4f}")
        assert xc.shape == x.shape and diff > 0

    print("\n=== Test 2: consistency loss + backward (cùng 1 model) ===")
    model = build_student("light").train()
    clean = torch.randn(1, 3, 320, 320)
    vib = corrupt_batch(clean, severity=2)
    c_out, c_aux = model(clean, return_aux=True)
    v_out, v_aux = model(vib, return_aux=True)
    vibloss = VibConsistencyLoss(use_feature=True)
    loss, logs = vibloss(v_out, c_out, v_aux["neck_feats"], c_aux["neck_feats"])
    print("  logs:", {k: round(v, 4) for k, v in logs.items()})
    loss.backward()
    g = model.stem.local[0].conv.weight.grad
    assert torch.isfinite(loss) and g is not None and g.norm() > 0
    print(f"  backward OK, grad backbone norm={g.norm().item():.4e}")
    print("\n✅ VIB_CONSISTENCY PASS")
