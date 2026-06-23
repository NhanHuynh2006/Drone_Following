"""
aux_train.py — Gộp density + distillation thành 1 objective cắm vào train_v3.py
================================================================================

Mục tiêu: chỉ sửa train_v3.py ở 4 chỗ (xem INTEGRATION_PATCH.md). Mọi thứ aux gói ở đây.

  - build_student(...)   : tạo student v15 + density head (thay model gốc khi chạy cấu hình density)
  - AuxObjective         : tính density loss + KD loss, gọi 1 dòng trong train step
  - .kd_param_group(...)  : param group adapter để thêm vào optimizer (chỉ khi bật KD)

Tất cả aux là TRAIN-TIME. Inference gọi model(x) -> giống hệt model gốc -> deploy Pi không đổi.
"""

import torch

from pfdet_nano_v15 import PFDetNanoV15
from density_aux import PFDetNanoV15Density, build_density_targets, density_loss
from distillation import DistillationLoss, load_teacher, TeacherFeatCapture


def build_student(profile="light", density_scale_idx=1, **kw):
    """Student = v15 + density head. density_scale_idx: 0=P2 1=P3(mặc định) 2=P4 3=P5."""
    return PFDetNanoV15Density(profile=profile, density_scale_idx=density_scale_idx, **kw)


def build_teacher(ckpt_path, profile="balanced", device="cuda"):
    """Teacher = PFDet balanced (cùng kiến trúc). Freeze sẵn."""
    return load_teacher(PFDetNanoV15, ckpt_path, profile=profile, device=device)


class AuxObjective:
    """
    Gọi: extra_loss, logs = aux(det_outs, aux_dict, labels_list, imgs, epoch)
    Cộng extra_loss vào loss detection của bạn rồi backward chung.
    """

    def __init__(
        self,
        img_size,
        density_stride,
        # density
        use_density=True,
        lambda_density=0.5,
        density_count_weight=0.1,
        density_scale=100.0,
        density_sigma_factor=0.2,
        # distillation
        use_kd=False,
        teacher=None,
        student_neck_c=48,
        teacher_neck_c=64,
        lambda_kd=1.0,
        kd_warmup_epochs=15,
        use_feature_kd=True,
        device="cuda",
    ):
        self.img_size = img_size
        self.density_stride = density_stride
        self.use_density = use_density
        self.lambda_density = lambda_density
        self.density_count_weight = density_count_weight
        self.density_scale = density_scale
        self.density_sigma_factor = density_sigma_factor

        self.use_kd = use_kd and teacher is not None
        self.teacher = teacher
        self.lambda_kd = lambda_kd
        self.kd_warmup_epochs = kd_warmup_epochs
        if self.use_kd:
            self.kd = DistillationLoss(
                student_neck_c=student_neck_c,
                teacher_neck_c=teacher_neck_c,
                use_feature_kd=use_feature_kd,
            ).to(device)
        else:
            self.kd = None

    def kd_param_group(self, lr, weight_decay):
        """Trả param group cho optimizer (chỉ gọi khi use_kd). Có use_muon=False để MuSGD hiểu;
        AdamW/SGD bỏ qua key thừa nên an toàn cho cả hai."""
        if self.kd is None:
            return None
        return {
            "params": list(self.kd.parameters()),
            "weight_decay": weight_decay,
            "use_muon": False,
            "lr": lr,
            "lr_ratio": 1.0,
        }

    def __call__(self, det_outs, aux_dict, labels_list, imgs, epoch):
        device = imgs.device
        # MULTI-SCALE: lấy size THỰC của batch hiện tại (img_size đổi mỗi vài batch)
        H, W = int(imgs.shape[-2]), int(imgs.shape[-1])
        extra = det_outs[0].new_zeros(())
        logs = {}

        if self.use_density:
            # labels_list[b] = (N,5) [cls,cx,cy,w,h] CHUẨN HOÁ ; person class = 0
            scale_vec = torch.tensor([W, H, W, H], dtype=torch.float32)  # cx,w×W ; cy,h×H
            boxes_px = []
            for l in labels_list:
                if len(l) > 0:
                    p = l[l[:, 0] == 0, 1:5].float() * scale_vec  # norm cxcywh -> pixel cxcywh
                else:
                    p = l[:, 1:5]
                boxes_px.append(p)
            gt_density = build_density_targets(
                boxes_px, img_size=(H, W), stride=self.density_stride,
                sigma_factor=self.density_sigma_factor, density_scale=self.density_scale,
                box_format="cxcywh", device=device,
            )
            dl = density_loss(aux_dict["density"], gt_density,
                              density_scale=self.density_scale, count_weight=self.density_count_weight)
            extra = extra + self.lambda_density * dl
            logs["density"] = float(dl.detach())

        if self.kd is not None and epoch >= self.kd_warmup_epochs:
            cap = TeacherFeatCapture(self.teacher)
            with torch.no_grad():
                t_outs = self.teacher(imgs)
            t_feats = list(cap.feats)
            cap.remove()
            kd_loss, kd_logs = self.kd(det_outs, t_outs,
                                       student_feats=aux_dict["neck_feats"], teacher_feats=t_feats)
            extra = extra + self.lambda_kd * kd_loss
            logs.update({f"kd_{k}": v for k, v in kd_logs.items()})

        return extra, logs


# ============================================================
#  Smoke test: ráp với CHÍNH PFDetLossV15 -> backward
# ============================================================
if __name__ == "__main__":
    from utils import PFDetLossV15

    torch.manual_seed(0)
    device = "cpu"
    img_size = 320
    strides = (4, 8, 16, 32)

    student = build_student(profile="light", density_scale_idx=1).to(device).train()
    criterion = PFDetLossV15(img_size=img_size, strides=strides, total_epochs=400)

    # batch giả: 2 ảnh, nhãn [cls,cx,cy,w,h] chuẩn hoá, person=0
    imgs = torch.randn(2, 3, img_size, img_size, device=device)
    labels_list = [
        torch.tensor([[0, 0.30, 0.30, 0.05, 0.08],
                      [0, 0.50, 0.55, 0.03, 0.04],
                      [0, 0.70, 0.40, 0.10, 0.12]], dtype=torch.float32),
        torch.tensor([[0, 0.45, 0.50, 0.04, 0.06]], dtype=torch.float32),
    ]

    print("=== Cấu hình B: detection + density ===")
    aux = AuxObjective(img_size=img_size, density_stride=student.density_stride,
                       use_density=True, lambda_density=0.5, use_kd=False, device=device)
    det_outs, aux_dict = student(imgs, return_aux=True)
    det_loss, det_logs = criterion(det_outs, labels_list, epoch=20)
    extra, logs = aux(det_outs, aux_dict, labels_list, imgs, epoch=20)
    total = det_loss + extra
    print(f"det_loss={det_loss.item():.4f}  aux={logs}  total={total.item():.4f}")
    total.backward()
    g = student.stem.local[0].conv.weight.grad
    assert torch.isfinite(total) and g is not None and g.norm() > 0
    print(f"backward OK, grad backbone norm={g.norm().item():.4e}")

    print("\n=== Cấu hình D: detection + density + KD (teacher balanced 2M) ===")
    teacher = PFDetNanoV15(profile="balanced").to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student2 = build_student(profile="light").to(device).train()
    aux2 = AuxObjective(img_size=img_size, density_stride=student2.density_stride,
                        use_density=True, lambda_density=0.5,
                        use_kd=True, teacher=teacher, lambda_kd=1.0,
                        kd_warmup_epochs=0, device=device)
    det_outs2, aux_dict2 = student2(imgs, return_aux=True)
    det_loss2, _ = criterion(det_outs2, labels_list, epoch=20)
    extra2, logs2 = aux2(det_outs2, aux_dict2, labels_list, imgs, epoch=20)
    total2 = det_loss2 + extra2
    print(f"det_loss={det_loss2.item():.4f}  aux_logs={ {k: round(v,4) for k,v in logs2.items()} }")
    total2.backward()
    print(f"adapter param group ví dụ: {list(aux2.kd_param_group(0.01, 5e-4).keys())}")
    assert torch.isfinite(total2)
    print("\n✅ AUX_TRAIN SMOKE TEST PASS (ráp đúng với PFDetLossV15 thật)")
