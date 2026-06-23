"""
distillation.py — Knowledge Distillation cho PFDet-Nano
========================================================

Teacher = PFDet-Nano BALANCED (2.0M, bạn tự train)  ->  Student = LIGHT (1.0M).

Vì teacher & student CÙNG kiến trúc nên:
  - Output KD: head outputs cả hai đều (B, 5, H, W) mỗi scale -> so trực tiếp, KHÔNG cần adapter.
  - Feature KD (tuỳ chọn): neck feature lệch channel (teacher 64 vs student 48)
    -> dùng adapter 1x1 chiếu student -> teacher channels.

KD chỉ chạy lúc TRAIN. Teacher được freeze (eval + no_grad). Model deploy vẫn là student 1M.

LƯU Ý: distillation ở đây là kỹ thuật PHỤ TRỢ tiêu chuẩn (mượn), KHÔNG phải đóng góp.
Đóng góp của bạn là density_aux. Giữ KD ở dạng chuẩn để khỏi làm loãng câu chuyện paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_teacher(model_ctor, ckpt_path, profile="balanced", device="cuda", **ctor_kwargs):
    """
    Tạo teacher, load weights, freeze hoàn toàn.
    model_ctor: class model gốc, vd PFDetNanoV15 (KHÔNG cần density head cho teacher).
    """
    teacher = model_ctor(profile=profile, **ctor_kwargs)
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("ema", state.get("model", state.get("state_dict", state)))
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    if missing:
        print(f"[teacher] missing keys: {len(missing)} (ok nếu chỉ là head aux)")
    if unexpected:
        print(f"[teacher] unexpected keys: {len(unexpected)}")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


class DistillationLoss(nn.Module):
    """
    Tính KD loss giữa student và teacher.

      total_kd = w_cls * KD_cls + w_box * KD_box + w_feat * KD_feat

    - KD_cls : student khớp xác suất "mềm" của teacher (BCE với soft target + temperature).
    - KD_box : chỉ distill box ở vùng teacher tự tin (prob > conf_thr) -> tránh nhiễu nền.
    - KD_feat: MSE giữa neck feature (student qua adapter) và teacher feature (detach).

    Adapters là module CÓ THAM SỐ -> nhớ thêm vào optimizer (xem hướng dẫn).
    """

    def __init__(
        self,
        student_neck_c=48,
        teacher_neck_c=64,
        n_scales=4,
        use_feature_kd=True,
        temperature=2.0,
        conf_thr=0.1,
        w_cls=1.0,
        w_box=1.0,
        w_feat=0.5,
    ):
        super().__init__()
        self.T = float(temperature)
        self.conf_thr = float(conf_thr)
        self.w_cls, self.w_box, self.w_feat = w_cls, w_box, w_feat
        self.use_feature_kd = use_feature_kd
        if use_feature_kd:
            self.adapters = nn.ModuleList(
                [nn.Conv2d(student_neck_c, teacher_neck_c, 1, bias=False) for _ in range(n_scales)]
            )
        else:
            self.adapters = None

    def forward(self, student_outs, teacher_outs, student_feats=None, teacher_feats=None):
        T, eps = self.T, 1e-6
        kd_cls = student_outs[0].new_zeros(())
        kd_box = student_outs[0].new_zeros(())

        for s_o, t_o in zip(student_outs, teacher_outs):
            s_cls, s_box = s_o[:, 0:1], s_o[:, 1:5]
            t_cls, t_box = t_o[:, 0:1].detach(), t_o[:, 1:5].detach()

            # --- cls: soft target từ teacher ---
            t_prob = torch.sigmoid(t_cls / T)
            kd_cls = kd_cls + F.binary_cross_entropy_with_logits(s_cls / T, t_prob) * (T * T)

            # --- box: chỉ vùng teacher tự tin ---
            mask = (torch.sigmoid(t_cls) > self.conf_thr).float()
            denom = mask.sum() * 4.0 + eps
            box_l = F.smooth_l1_loss(s_box, t_box, reduction="none") * mask
            kd_box = kd_box + box_l.sum() / denom

        n = len(student_outs)
        kd_cls, kd_box = kd_cls / n, kd_box / n

        loss = self.w_cls * kd_cls + self.w_box * kd_box
        logs = {"kd_cls": float(kd_cls.detach()), "kd_box": float(kd_box.detach())}

        # --- feature KD (tuỳ chọn) ---
        if self.use_feature_kd and student_feats is not None and teacher_feats is not None:
            kd_feat = student_outs[0].new_zeros(())
            for i, (sf, tf) in enumerate(zip(student_feats, teacher_feats)):
                sf_adapted = self.adapters[i](sf)
                if sf_adapted.shape[-2:] != tf.shape[-2:]:
                    sf_adapted = F.interpolate(sf_adapted, size=tf.shape[-2:],
                                               mode="bilinear", align_corners=False)
                kd_feat = kd_feat + F.mse_loss(sf_adapted, tf.detach())
            kd_feat = kd_feat / len(student_feats)
            loss = loss + self.w_feat * kd_feat
            logs["kd_feat"] = float(kd_feat.detach())

        logs["kd_total"] = float(loss.detach())
        return loss, logs


class TeacherFeatCapture:
    """
    Bắt neck feature của teacher gốc (PFDetNanoV15) qua forward hook trên scale_attn,
    KHÔNG cần sửa code teacher. Dùng cho feature KD.

        cap = TeacherFeatCapture(teacher)
        with torch.no_grad():
            t_outs = teacher(x)
        t_feats = cap.feats   # list 4 tensor [P2,P3,P4,P5]
        ...
        cap.remove()          # gỡ hook khi xong
    """

    def __init__(self, teacher):
        self.feats = [None] * len(teacher.scale_attn)
        self._handles = []
        for i, m in enumerate(teacher.scale_attn):
            self._handles.append(m.register_forward_hook(self._mk(i)))

    def _mk(self, i):
        def hook(_m, _inp, out):
            self.feats[i] = out
        return hook

    def remove(self):
        for h in self._handles:
            h.remove()


# ============================================================
#  Sanity test
# ============================================================
if __name__ == "__main__":
    from models import PFDetNanoV15
    from density_aux import PFDetNanoV15Density

    torch.manual_seed(0)
    print("=== Test: teacher(balanced 2M) -> student(light 1M + density) ===")
    teacher = PFDetNanoV15(profile="balanced").eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = PFDetNanoV15Density(profile="light").train()

    print(f"teacher: {sum(p.numel() for p in teacher.parameters())/1e6:.2f}M  "
          f"student: {sum(p.numel() for p in student.parameters())/1e6:.2f}M")

    x = torch.randn(1, 3, 320, 320)
    cap = TeacherFeatCapture(teacher)
    with torch.no_grad():
        t_outs = teacher(x)
    t_feats = [f for f in cap.feats]
    cap.remove()

    s_outs, aux = student(x, return_aux=True)
    print("teacher out shapes :", [tuple(o.shape) for o in t_outs])
    print("student out shapes :", [tuple(o.shape) for o in s_outs])
    print("teacher feat ch    :", [f.shape[1] for f in t_feats], "(64)")
    print("student feat ch    :", [f.shape[1] for f in aux["neck_feats"]], "(48)")
    for to, so in zip(t_outs, s_outs):
        assert to.shape == so.shape  # output KD: không cần adapter

    kd = DistillationLoss(student_neck_c=48, teacher_neck_c=64, use_feature_kd=True)
    loss, logs = kd(s_outs, t_outs, student_feats=aux["neck_feats"], teacher_feats=t_feats)
    print("KD logs (output + feature):", {k: round(v, 4) for k, v in logs.items()})
    loss.backward()
    g = student.stem.local[0].conv.weight.grad
    ga = kd.adapters[0].weight.grad
    print(f"grad student backbone norm = {g.norm().item():.4e}")
    print(f"grad adapter[0] norm       = {ga.norm().item():.4e} (phải >0 -> nhớ add vào optimizer)")
    assert torch.isfinite(loss) and g.norm().item() > 0 and ga.norm().item() > 0
    print("\n✅ DISTILLATION TEST PASS (output KD + feature KD)")
