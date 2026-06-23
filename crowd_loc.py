"""
crowd_loc.py — Crowd-aware localization weighting (cơ chế 2 của DGS)
=====================================================================

Ý tưởng: ở vùng người CHEN CHÚC, box dễ chồng lấn và định vị lỏng là nơi AP75 sụp nhất
(đo thực tế: AP75=0.13). Ta tăng trọng số BOX LOSS cho các positive anchor thuộc GT nằm
trong vùng mật độ cao -> ép model định vị khít hơn đúng chỗ khó.

- Dùng GT density (số hàng xóm mềm quanh mỗi GT) -> KHÔNG tạo feedback loop (an toàn).
- Bổ sung cho ASL (vốn up-weight theo SIZE). Cái này up-weight theo MẬT ĐỘ.
- Train-time only, 0 chi phí inference.

Tích hợp vào utils/losses_v15.py: xem cuối file.
"""

import torch


def crowd_density_weights(gt_boxes_xywh, gi_index, img_size,
                          alpha=1.0, sigma_px=64.0, max_w=3.0, eps=1e-6):
    """
    gt_boxes_xywh : (G,4) tensor [cx,cy,w,h] CHUẨN HOÁ 0-1 (GT của 1 ảnh).
    gi_index      : (P,) long — GT index của từng positive anchor.
    img_size      : để quy tâm GT ra pixel khi tính khoảng cách.
    return        : (P,) tensor trọng số trong [1, max_w] cho từng positive.

    density_g = Σ_j exp(-||c_g - c_j||² / (2 σ²))  (kể cả j=g -> tối thiểu =1)
    weight_g  = clamp(1 + alpha * (density_g - 1) / (max_density - 1), 1, max_w)
    """
    device = gt_boxes_xywh.device
    G = gt_boxes_xywh.shape[0]
    if G == 0 or len(gi_index) == 0:
        return torch.ones(len(gi_index), device=device)

    centers = gt_boxes_xywh[:, :2] * img_size                 # (G,2) pixel
    d2 = torch.cdist(centers, centers) ** 2                    # (G,G)
    kernel = torch.exp(-d2 / (2.0 * sigma_px * sigma_px))      # (G,G)
    density = kernel.sum(dim=1)                                # (G,) >=1 (self=1)

    # chuẩn hoá về [0,1] theo ảnh: GT đông hàng xóm nhất -> 1
    dmax = density.max()
    dnorm = (density - 1.0) / (dmax - 1.0 + eps)               # GT đơn độc -> 0
    w_per_gt = (1.0 + alpha * dnorm).clamp(1.0, max_w)         # (G,)

    return w_per_gt[gi_index]


# ============================================================
#  Test: trọng số ở cụm đông > ở GT đơn độc
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    img_size = 640
    # 4 người chen chúc quanh (0.5,0.5) + 1 người đơn độc ở góc
    gt = torch.tensor([
        [0.50, 0.50, 0.03, 0.05],
        [0.51, 0.50, 0.03, 0.05],
        [0.50, 0.51, 0.03, 0.05],
        [0.52, 0.52, 0.03, 0.05],
        [0.90, 0.10, 0.03, 0.05],   # đơn độc
    ])
    gi = torch.arange(5)
    w = crowd_density_weights(gt, gi, img_size, alpha=1.0, sigma_px=64.0)
    print("trọng số localization theo GT:", [round(float(x), 3) for x in w])
    print(f"  4 người cụm: {[round(float(x),3) for x in w[:4]]} (phải > 1)")
    print(f"  người đơn độc: {round(float(w[4]),3)} (phải = 1.0)")
    assert (w[:4] > 1.0).all(), "vùng đông phải có trọng số > 1"
    assert abs(float(w[4]) - 1.0) < 1e-4, "GT đơn độc phải = 1.0"

    # quan trọng: differentiable-safe khi nhân vào loss (chỉ là hệ số, detach GT)
    box_loss_per = torch.rand(5, requires_grad=True)
    weighted = (box_loss_per * w.detach()).mean()
    weighted.backward()
    assert box_loss_per.grad is not None
    print("\n✅ CROWD_LOC OK — nhân vào box_loss_per được, gradient chảy bình thường.")
