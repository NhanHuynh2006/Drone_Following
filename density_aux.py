"""
density_aux.py — Density-guided auxiliary supervision for PFDet-Nano v15
========================================================================

ĐÓNG GÓP CHÍNH của paper. Thêm một head phụ dự đoán density map (mật độ người),
được huấn luyện song song với detector. Head này chỉ chạy lúc TRAIN — khi
inference gọi model(x) bình thường thì head bị bỏ qua hoàn toàn => 0 FPS cost.

Cơ chế:
  - Tap feature P3 (stride 8) sau neck -> DensityHead -> density map (B,1,H,W)
  - GT density sinh từ box bằng Gaussian splatting (mỗi box = 1 Gaussian, mass=1)
  - Loss = MSE(pred, gt)  (+ optional count loss)
  - Gradient backprop qua neck/backbone => regularize feature chung, tăng recall
    ở cảnh đông người chen chúc (VisDrone).

File này standalone, chỉ import model gốc pfdet_nano_v15.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import PFDetNanoV15


# ============================================================
#  Density head (chỉ dùng khi train)
# ============================================================
class DensityHead(nn.Module):
    """Small conv stack: neck feature -> 1-channel non-negative density map."""

    def __init__(self, in_c, hidden=None):
        super().__init__()
        hidden = hidden or in_c
        self.net = nn.Sequential(
            nn.Conv2d(in_c, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden // 2),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden // 2, 1, 1),
        )

    def forward(self, x):
        # softplus -> density luôn >= 0, mượt hơn ReLU (gradient không chết ở 0)
        return F.softplus(self.net(x))


# ============================================================
#  Model con: v15 + density head
# ============================================================
class PFDetNanoV15Density(PFDetNanoV15):
    """
    PFDet-Nano v15 với density head phụ.

    forward(x)                 -> giống hệt model gốc (list 4 head outputs). DÙNG KHI INFERENCE.
    forward(x, return_aux=True)-> (det_outs, aux) với aux = {'density':..., 'neck_feats':[...]}
                                  DÙNG KHI TRAIN (lấy density + feats cho distillation).

    density_scale_idx: lấy feature scale nào nuôi density head.
        0=P2(str4) 1=P3(str8, mặc định) 2=P4(str16) 3=P5(str32)
        P3 là chuẩn cho density map (cân bằng độ phân giải vs compute).
    """

    def __init__(self, *args, density_scale_idx=1, density_hidden=None, **kwargs):
        super().__init__(*args, **kwargs)
        neck_c = self.model_config["neck_c"]
        self.density_scale_idx = int(density_scale_idx)
        self.density_stride = self.strides[self.density_scale_idx]
        self.density_head = DensityHead(neck_c, density_hidden)

    def forward(self, x, return_aux=False):
        # ---- replicate base forward, capture neck feats ----
        s = self.stem(x)
        fp2 = self.stage_p2(s)
        fp2 = self.p2_backbone_attn(fp2)
        fp3 = self.stage_p3(fp2)
        fp4 = self.stage_p4(fp3)
        fp5 = self.stage_p5(fp4)
        fp4 = self.bottleneck_p4(fp4)
        fp5 = self.bottleneck_p5(fp5)

        n2 = self.lat_p2(fp2)
        n3 = self.lat_p3(fp3)
        n4 = self.lat_p4(fp4)
        n5 = self.lat_p5(fp5)
        for layer in self.bifpn:
            n2, n3, n4, n5 = layer(n2, n3, n4, n5)

        feats = [sa(f) for sa, f in zip(self.scale_attn, [n2, n3, n4, n5])]
        det_outs = [head(f) for head, f in zip(self.heads, feats)]

        if not return_aux:
            return det_outs

        density = self.density_head(feats[self.density_scale_idx])
        return det_outs, {"density": density, "neck_feats": feats}


# ============================================================
#  Sinh GT density bằng Gaussian splatting
# ============================================================
def _splat_gaussian(density, cx, cy, sigma, radius):
    """Cộng 1 Gaussian (chuẩn hoá mass=1) vào density tại (cx,cy) theo grid units."""
    H, W = density.shape
    x0 = max(0, int(math.floor(cx - radius)))
    x1 = min(W, int(math.ceil(cx + radius + 1)))
    y0 = max(0, int(math.floor(cy - radius)))
    y1 = min(H, int(math.ceil(cy + radius + 1)))
    if x1 <= x0 or y1 <= y0:
        return
    ys = torch.arange(y0, y1, device=density.device, dtype=density.dtype).view(-1, 1)
    xs = torch.arange(x0, x1, device=density.device, dtype=density.dtype).view(1, -1)
    g = torch.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2.0 * sigma * sigma))
    ssum = g.sum()
    if ssum > 0:
        density[y0:y1, x0:x1] += g / ssum  # mỗi box đóng góp tổng mass = 1


@torch.no_grad()
def build_density_targets(
    boxes_per_image,
    img_size,
    stride,
    sigma_factor=0.2,
    sigma_min=1.0,
    sigma_max=8.0,
    density_scale=100.0,
    box_format="cxcywh",
    device="cpu",
    dtype=torch.float32,
):
    """
    boxes_per_image : list độ dài B; mỗi phần tử là tensor (N_i, 4) toạ độ PIXEL tuyệt đối
                      ở kích thước img_size hiện tại (KHÔNG normalize 0-1).
                      box_format='cxcywh' (cx,cy,w,h) hoặc 'xyxy' (x1,y1,x2,y2).
    img_size        : int (vuông) hoặc (H, W) pixel của ảnh đầu vào.
    stride          : stride của scale nuôi density head (model.density_stride, vd 8).
    return          : tensor (B, 1, H_d, W_d) với H_d = H//stride.

    density_scale: nhân toàn map để giá trị đủ lớn -> MSE có gradient hữu ích.
                   Số người dự đoán = density.sum() / density_scale.
    """
    if isinstance(img_size, int):
        H, W = img_size, img_size
    else:
        H, W = img_size
    Hd, Wd = H // stride, W // stride

    B = len(boxes_per_image)
    out = torch.zeros(B, 1, Hd, Wd, device=device, dtype=dtype)

    for b, boxes in enumerate(boxes_per_image):
        if boxes is None or len(boxes) == 0:
            continue
        boxes = boxes.to(device=device, dtype=dtype)
        if box_format == "xyxy":
            x1, y1, x2, y2 = boxes.unbind(-1)
            cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
            bw, bh = (x2 - x1).clamp(min=1e-3), (y2 - y1).clamp(min=1e-3)
        elif box_format == "cxcywh":
            cx, cy, bw, bh = boxes.unbind(-1)
        else:
            raise ValueError(f"box_format không hỗ trợ: {box_format}")

        # đưa về grid units
        gx, gy = cx / stride, cy / stride
        gw, gh = (bw / stride).clamp(min=1e-3), (bh / stride).clamp(min=1e-3)
        sigmas = (sigma_factor * torch.sqrt(gw * gh)).clamp(sigma_min, sigma_max)

        dmap = out[b, 0]
        for i in range(boxes.shape[0]):
            sx, sy = float(gx[i]), float(gy[i])
            if sx < 0 or sy < 0 or sx >= Wd or sy >= Hd:
                continue
            sig = float(sigmas[i])
            _splat_gaussian(dmap, sx, sy, sig, radius=3.0 * sig)

    return out * density_scale


# ============================================================
#  Density loss
# ============================================================
def density_loss(pred_density, gt_density, density_scale=100.0, count_weight=0.0):
    """
    pred_density, gt_density: (B,1,H,W). gt đã nhân density_scale trong build_density_targets.
    Nếu kích thước lệch (do làm tròn stride) -> resize pred về gt.
    count_weight > 0: thêm L1 trên tổng count (số người) để ổn định.
    """
    if pred_density.shape[-2:] != gt_density.shape[-2:]:
        pred_density = F.interpolate(
            pred_density, size=gt_density.shape[-2:], mode="bilinear", align_corners=False
        )
    loss = F.mse_loss(pred_density, gt_density)
    if count_weight > 0:
        pred_count = pred_density.sum(dim=(1, 2, 3)) / density_scale
        gt_count = gt_density.sum(dim=(1, 2, 3)) / density_scale
        loss = loss + count_weight * F.l1_loss(pred_count, gt_count)
    return loss


# ============================================================
#  Sanity test
# ============================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== Test 1: model con forward (inference vs train) ===")
    model = PFDetNanoV15Density(profile="light", density_scale_idx=1)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.density_head.parameters())
    print(f"total params: {n_params/1e6:.3f}M | density_head: {n_head/1e6:.3f}M "
          f"({100*n_head/n_params:.1f}% — chỉ tốn lúc train)")
    print(f"density stride = {model.density_stride}")

    x = torch.randn(2, 3, 320, 320)
    model.eval()
    outs = model(x)  # inference mode
    assert isinstance(outs, list) and len(outs) == 4
    print("inference forward OK, head shapes:", [tuple(o.shape) for o in outs])

    model.train()
    det_outs, aux = model(x, return_aux=True)
    print("train forward OK, density shape:", tuple(aux["density"].shape))
    assert aux["density"].shape[0] == 2 and aux["density"].shape[1] == 1

    print("\n=== Test 2: build_density_targets (Gaussian splatting) ===")
    # ảnh 0: 3 người (1 tiny 8px, 1 vừa 30px, 1 to 80px) — đều nằm trong 320px ; ảnh 1: rỗng
    boxes0 = torch.tensor([
        [ 60.,  60.,  8.,  8.],
        [160., 120., 30., 30.],
        [250., 250., 80., 80.],
    ])
    boxes1 = torch.zeros(0, 4)
    gt = build_density_targets([boxes0, boxes1], img_size=320,
                               stride=model.density_stride, density_scale=100.0)
    print("gt density shape:", tuple(gt.shape))
    print(f"ảnh 0: count = sum/scale = {gt[0].sum().item()/100:.3f} (kỳ vọng ~3.0)")
    print(f"ảnh 1: count = {gt[1].sum().item()/100:.3f} (kỳ vọng 0.0)")
    assert abs(gt[0].sum().item() / 100 - 3.0) < 0.05, "tổng mass phải ~= số box"
    assert gt[1].sum().item() == 0.0

    print("\n=== Test 3: density_loss + backprop ===")
    det_outs, aux = model(x, return_aux=True)
    gt_train = build_density_targets([boxes0, boxes1], img_size=320,
                                     stride=model.density_stride, density_scale=100.0)
    loss = density_loss(aux["density"], gt_train, count_weight=0.1)
    print(f"density loss = {loss.item():.4f} (finite: {torch.isfinite(loss).item()})")
    loss.backward()
    # kiểm tra gradient lan tới backbone (regularize feature chung)
    g = model.stem.local[0].conv.weight.grad
    print(f"gradient tới stem.local: norm = {g.norm().item():.4e} "
          f"(>0 nghĩa là density signal regularize được backbone)")
    assert g is not None and g.norm().item() > 0
    print("\n✅ TẤT CẢ TEST PASS")
