"""
PFDet-Nano v14 Loss Function
============================

Loss components and their references (all peer-reviewed):

  QFL  — Quality Focal Loss
         Li et al., "Generalized Focal Loss", NeurIPS 2020.

  TAL  — Task-Aligned Label Assignment
         Feng et al., "TOOD: Task-aligned One-stage Object Detection", ICCV 2021.

  NWD  — Normalized Wasserstein Distance for tiny objects
         Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object
         Detection", ISPRS Journal of Photogrammetry and Remote Sensing, 2022.
         (Top journal — Q1 remote sensing. Specifically validated on VisDrone.)

  CIoU — Complete IoU regression loss
         Zheng et al., "Distance-IoU Loss: Faster and Better Learning for
         Bounding Box Regression", AAAI 2020.

  ASL  — Area Scale Loss weighting (up-weights tiny object gradient).
         Drone-specific design; no single canonical reference.
"""

import torch
import torch.nn.functional as F

from .box_ops import bbox_ciou, bbox_iou_xyxy

# ============================================================
#  QFL: Quality Focal Loss
# ============================================================

def quality_focal_loss(pred_logits, targets, beta=2.0):
    p = torch.sigmoid(pred_logits)
    focal_w = (targets - p).abs().pow(beta)
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
    return focal_w * bce

# ============================================================
#  NWD: Normalized Wasserstein Distance
# ============================================================

def nwd_loss(pred_xyxy, target_xyxy, c=0.04, eps=1e-7):
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(dim=1)
    t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(dim=1)

    p_cx = (p_x1 + p_x2) / 2;  p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2;  t_cy = (t_y1 + t_y2) / 2
    p_sw = (p_x2 - p_x1) / 2;  p_sh = (p_y2 - p_y1) / 2
    t_sw = (t_x2 - t_x1) / 2;  t_sh = (t_y2 - t_y1) / 2

    w2 = ((p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2) +
          (p_sw - t_sw).pow(2) + (p_sh - t_sh).pow(2))
    nwd = torch.exp(-w2.sqrt() / (c + eps))
    return 1.0 - nwd

# ============================================================
#  TAL: Task-Aligned Label Assignment (SOTA Modded)
# ============================================================

def _tal_assign_one_scale(raw_b, gt_boxes_xywh, stride, H, W, img_size,
                          k_tiny_eff, k_normal_eff, tiny_area_px,
                          search_r_tiny, search_r_normal, scale_range,
                          tal_alpha=1.0, tal_beta=6.0):
    device = raw_b.device
    G = len(gt_boxes_xywh)
    occupied = {}

    for gi in range(G):
        cx  = float(gt_boxes_xywh[gi, 0])
        cy  = float(gt_boxes_xywh[gi, 1])
        gw  = float(gt_boxes_xywh[gi, 2])
        gh  = float(gt_boxes_xywh[gi, 3])

        gw_px = gw * img_size
        gh_px = gh * img_size
        max_dim = max(gw_px, gh_px)

        # [SCALE LIMIT] Bỏ qua GT box nếu không thuộc nhiệm vụ của FPN scale này
        if max_dim < scale_range[0] or max_dim > scale_range[1]:
            continue

        area_px = gw_px * gh_px
        is_tiny = area_px < tiny_area_px
        k  = k_tiny_eff  if is_tiny else k_normal_eff
        sr = search_r_tiny if is_tiny else search_r_normal

        gt_x1 = cx - gw / 2;  gt_y1 = cy - gh / 2
        gt_x2 = cx + gw / 2;  gt_y2 = cy + gh / 2

        gx_f = cx * W;  gy_f = cy * H
        ci0  = min(max(int(gx_f), 0), W - 1)
        ri0  = min(max(int(gy_f), 0), H - 1)

        cand_r, cand_c = [], []
        for dr in range(-sr, sr + 1):
            for dc in range(-sr, sr + 1):
                r = ri0 + dr;  c = ci0 + dc
                if 0 <= r < H and 0 <= c < W:
                    cand_r.append(r);  cand_c.append(c)

        if not cand_r:
            continue

        rows_t = torch.tensor(cand_r, device=device, dtype=torch.long)
        cols_t = torch.tensor(cand_c, device=device, dtype=torch.long)
        M = len(cand_r)

        pred_scores = torch.sigmoid(raw_b[0, 0, rows_t, cols_t])

        dx = torch.sigmoid(raw_b[0, 1, rows_t, cols_t]) - 0.5
        dy = torch.sigmoid(raw_b[0, 2, rows_t, cols_t]) - 0.5
        lw = raw_b[0, 3, rows_t, cols_t].clamp(-4, 4)
        lh = raw_b[0, 4, rows_t, cols_t].clamp(-4, 4)

        p_cx_t = (cols_t.float() + dx) * stride / img_size
        p_cy_t = (rows_t.float() + dy) * stride / img_size
        p_w_t  = torch.exp(lw) * stride / img_size
        p_h_t  = torch.exp(lh) * stride / img_size

        p_x1_t = p_cx_t - p_w_t / 2;  p_y1_t = p_cy_t - p_h_t / 2
        p_x2_t = p_cx_t + p_w_t / 2;  p_y2_t = p_cy_t + p_h_t / 2

        gt_x1_t = torch.tensor(gt_x1, device=device)
        gt_y1_t = torch.tensor(gt_y1, device=device)
        gt_x2_t = torch.tensor(gt_x2, device=device)
        gt_y2_t = torch.tensor(gt_y2, device=device)

        ix1 = torch.maximum(p_x1_t, gt_x1_t)
        iy1 = torch.maximum(p_y1_t, gt_y1_t)
        ix2 = torch.minimum(p_x2_t, gt_x2_t)
        iy2 = torch.minimum(p_y2_t, gt_y2_t)
        inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        p_area_t = (p_w_t * p_h_t).clamp(min=0)
        gt_area  = gw * gh
        union = p_area_t + gt_area - inter + 1e-7
        iou = (inter / union).clamp(0.0, 1.0)

        alignment = pred_scores.pow(tal_alpha) * iou.pow(tal_beta)

        actual_k = min(k, M)
        if actual_k < M:
            _, topk_idx = alignment.topk(actual_k)
        else:
            topk_idx = torch.arange(M, device=device)

        for ii in topk_idx.tolist():
            r = cand_r[ii];  c = cand_c[ii]
            score = float(alignment[ii].item())
            key = (r, c)
            if key not in occupied or score > occupied[key][0]:
                occupied[key] = (score, gi)

    rows_out, cols_out, gi_out = [], [], []
    for (r, c), (_, gi) in occupied.items():
        rows_out.append(r);  cols_out.append(c);  gi_out.append(gi)

    return rows_out, cols_out, gi_out

# ============================================================
#  Main Loss Class
# ============================================================

class PFDetLossV14:
    def __init__(
        self, img_size=320, strides=(4, 8, 16, 32),
        obj_weight=8.0, box_weight=4.0, qfl_beta=2.0, hard_floor=0.0,
        obj_warmup_epochs=10, prog_epochs=250, asl_area_ref=0.004, asl_max=4.0,
        tiny_area_px=576.0, k_tiny=6, k_normal=3, search_r_tiny=3, search_r_normal=2,
        tal_alpha=1.0, tal_beta=6.0, nwd_area_thr=1024.0, nwd_c=0.04, total_epochs=250,
    ):
        self.img_size = img_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.qfl_beta = qfl_beta
        self.hard_floor = hard_floor
        self.obj_warmup_epochs = max(1, obj_warmup_epochs)
        self.prog_epochs = max(1, prog_epochs)
        self.asl_area_ref = asl_area_ref
        self.asl_max = asl_max
        self.tiny_area_px = tiny_area_px
        self.k_tiny = k_tiny
        self.k_normal = k_normal
        self.search_r_tiny = search_r_tiny
        self.search_r_normal = search_r_normal
        self.tal_alpha = tal_alpha
        self.tal_beta = tal_beta
        self.nwd_area_thr = nwd_area_thr
        self.nwd_c = nwd_c
        self.total_epochs = total_epochs

        # [SCALE LIMITS] Phân vùng chịu trách nhiệm FPN — căn chỉnh cho VisDrone tiny persons
        # VisDrone persons tại 320px thường từ 4-60px → P2 chiếm gần hết nếu ngưỡng quá cao.
        # Hạ ngưỡng dưới P3 xuống 24px và P5 xuống 160px để tăng training signal cho P3/P4/P5.
        self.scale_ranges = [
            (0,   48),      # P2 (Stride 4)  - tiny: 0 → 48px
            (24, 112),      # P3 (Stride 8)  - small: 24 → 112px  (overlap 24-48px với P2)
            (80, 200),      # P4 (Stride 16) - medium: 80 → 200px
            (160, 99999),   # P5 (Stride 32) - large: > 160px
        ]

    def _obj_target_schedule(self, iou, epoch):
        # warm_ratio giảm từ 1.0 → 0.0 trong obj_warmup_epochs đầu
        warm_ratio = max(0.0, 1.0 - epoch / max(1, self.obj_warmup_epochs))
        # hard_floor chỉ active trong giai đoạn đầu (decay → 0 sau 2*warmup_epochs)
        # Tránh dead zone khi IoU~0, nhưng KHÔNG giữ mãi để precision converge được
        floor = self.hard_floor * max(0.0, 1.0 - epoch / max(1, self.obj_warmup_epochs * 2))
        warm_ratio = max(warm_ratio, floor)
        return (iou * (1.0 - warm_ratio) + warm_ratio).clamp(0.0, 1.0)

    def _prog_k(self, epoch):
        ratio = min(1.0, epoch / self.prog_epochs)
        # Floor tối thiểu: k_tiny >= 3, k_normal >= 2 để tránh sparse gradient cuối train
        k_tiny_eff   = max(3, round(self.k_tiny   * (1.0 - ratio) + ratio))
        k_normal_eff = max(2, round(self.k_normal * (1.0 - ratio) + ratio))
        return k_tiny_eff, k_normal_eff

    def summary(self):
        return f"PFDetLossV14 (QFL + SoftNWD/CIoU + TAL_ScaleLimits, {len(self.strides)}-scale)"

    def __call__(self, preds, labels_list, epoch=0):
        device = preds[0].device
        B      = preds[0].shape[0]

        total_obj_loss  = torch.tensor(0.0, device=device)
        total_box_loss  = torch.tensor(0.0, device=device)
        total_iou       = 0.0
        total_iou_count = 0
        n_pos_total     = 0
        n_pos_per_scale = [0] * len(self.strides)

        k_tiny_eff, k_normal_eff = self._prog_k(epoch)

        for b in range(B):
            gt_labels     = labels_list[b]
            gt_boxes_xywh = (gt_labels[gt_labels[:, 0] == 0, 1:5]
                             if len(gt_labels) > 0 else gt_labels[:, 1:5])
            G = len(gt_boxes_xywh)

            for si, (pred, stride) in enumerate(zip(preds, self.strides)):
                raw_b = pred[b:b+1].float()
                _, _, H, W = raw_b.shape

                obj_logits_map  = raw_b[0, 0]
                obj_targets_map = torch.zeros(H, W, device=device, dtype=torch.float32)
                pos_mask_flat   = torch.zeros(H * W, device=device, dtype=torch.bool)

                if G == 0:
                    qfl = quality_focal_loss(obj_logits_map.view(-1), obj_targets_map.view(-1), self.qfl_beta)
                    total_obj_loss = total_obj_loss + qfl.mean()
                    continue

                with torch.no_grad():
                    rows, cols, gi_list = _tal_assign_one_scale(
                        raw_b, gt_boxes_xywh.to(device),
                        stride, H, W, self.img_size,
                        k_tiny_eff, k_normal_eff,
                        self.tiny_area_px,
                        self.search_r_tiny, self.search_r_normal,
                        scale_range=self.scale_ranges[si], # Áp dụng rào cản Scale
                        tal_alpha=self.tal_alpha,
                        tal_beta=self.tal_beta,
                    )

                n_pos = len(rows)
                n_pos_total += n_pos
                n_pos_per_scale[si] += n_pos

                if n_pos > 0:
                    rows_t = torch.tensor(rows, device=device, dtype=torch.long)
                    cols_t = torch.tensor(cols, device=device, dtype=torch.long)
                    gi_t   = torch.tensor(gi_list, device=device, dtype=torch.long)
                    gt_t   = gt_boxes_xywh.to(device)

                    dx = torch.sigmoid(raw_b[0, 1, rows_t, cols_t]) - 0.5
                    dy = torch.sigmoid(raw_b[0, 2, rows_t, cols_t]) - 0.5
                    lw = raw_b[0, 3, rows_t, cols_t].clamp(-4, 4)
                    lh = raw_b[0, 4, rows_t, cols_t].clamp(-4, 4)

                    p_cx = (cols_t.float() + dx) * stride / self.img_size
                    p_cy = (rows_t.float() + dy) * stride / self.img_size
                    p_w  = torch.exp(lw) * stride / self.img_size
                    p_h  = torch.exp(lh) * stride / self.img_size

                    pred_xyxy = torch.stack([
                        p_cx - p_w / 2, p_cy - p_h / 2,
                        p_cx + p_w / 2, p_cy + p_h / 2,
                    ], dim=1)

                    t_cx, t_cy = gt_t[gi_t, 0], gt_t[gi_t, 1]
                    t_w, t_h   = gt_t[gi_t, 2], gt_t[gi_t, 3]
                    gt_xyxy = torch.stack([
                        t_cx - t_w / 2, t_cy - t_h / 2,
                        t_cx + t_w / 2, t_cy + t_h / 2,
                    ], dim=1)

                    area_px  = t_w * self.img_size * t_h * self.img_size + 1e-7
                    
                    # [SOFT-BLEND NWD & CIoU] Hòa trộn mượt mà dựa trên diện tích
                    nwd_w = torch.exp(-area_px / self.nwd_area_thr)
                    ciou_w = 1.0 - nwd_w

                    l_nwd = nwd_loss(pred_xyxy, gt_xyxy, c=self.nwd_c)
                    l_ciou = bbox_ciou(pred_xyxy, gt_xyxy)
                    
                    box_loss_per = (nwd_w * l_nwd) + (ciou_w * l_ciou)

                    asl_w = (self.asl_area_ref * self.img_size ** 2 / area_px).sqrt().clamp(1.0, self.asl_max)
                    total_box_loss = total_box_loss + (asl_w * box_loss_per).mean()

                    with torch.no_grad():
                        iou = bbox_iou_xyxy(pred_xyxy.detach(), gt_xyxy.detach())
                        total_iou += float(iou.sum().item())
                        total_iou_count += int(iou.numel())

                    obj_targets_map[rows_t, cols_t] = self._obj_target_schedule(iou, epoch)
                    pos_mask_flat[rows_t * W + cols_t] = True

                # QFL Loss thuần túy (Đã loại bỏ OHEM để ổn định training)
                qfl = quality_focal_loss(obj_logits_map.view(-1), obj_targets_map.view(-1), self.qfl_beta)
                obj_loss_scale = (qfl[pos_mask_flat].sum() + qfl[~pos_mask_flat].sum()) / max(1, n_pos)
                total_obj_loss = total_obj_loss + obj_loss_scale

        n_pos_total    = max(1, n_pos_total)
        total_obj_loss = total_obj_loss / B
        total_box_loss = total_box_loss / B

        loss = self.obj_weight * total_obj_loss + self.box_weight * total_box_loss

        loss_dict = {
            'total': float(loss.item()),
            'obj':   float(total_obj_loss.item()),
            'box':   float(total_box_loss.item()),
            'iou':   total_iou / max(1, total_iou_count),
            'n_pos': n_pos_total / B,
        }
        for si, stride in enumerate(self.strides):
            loss_dict[f'pos_s{stride}'] = n_pos_per_scale[si] / B

        return loss, loss_dict