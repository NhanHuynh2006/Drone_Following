"""
PFDet-Nano v15 Loss Function
============================

Improvements over v14 (all from YOLOv26, arXiv 2509.25164):

  STAL — Small-Target-Aware Label Assignment (YOLOv26, 2025)
         Two mechanisms from the paper:
         1. Guaranteed minimum 4 anchors for objects < stal_min_area_px (64px²)
            → expands search radius if top-k gives fewer than 4 candidates
         2. Multiplicative alignment bonus for tiny objects:
            alignment *= (1 + γ × exp(-area_px / area_ref))
            → boosts ranking priority of tiny objects in top-k selection

  ProgLoss — Progressive Loss Balancing (YOLOv26, 2025)
         Dynamically shifts loss weight balance over training:
         - Early epochs: obj loss weight high, box loss weight low
           → model learns "is there a person" before "exactly where"
         - Late epochs: box weight increases to base, obj decreases to base
           → model focuses on precise localization
         Linear schedule (progress = epoch / total_epochs):
           obj_weight: base_obj * prog_factor → base_obj  (decreasing)
           box_weight: base_box / prog_factor → base_box  (increasing)

All other components unchanged from v14 (all peer-reviewed):
  QFL  — NeurIPS 2020
  NWD  — ISPRS Journal 2022
  CIoU — AAAI 2020
  TAL  — ICCV 2021 (base assignment, extended with STAL)
  ASL  — drone-specific area scale loss
"""

import math
import torch
import torch.nn.functional as F

from .box_ops import bbox_ciou, bbox_iou_xyxy


def quality_focal_loss(pred_logits, targets, beta=2.0):
    p = torch.sigmoid(pred_logits)
    focal_w = (targets - p).abs().pow(beta)
    bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
    return focal_w * bce


def nwd_loss(pred_xyxy, target_xyxy, c=0.04, eps=1e-7):
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(dim=1)
    t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(dim=1)
    p_cx = (p_x1 + p_x2) / 2;  p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2;  t_cy = (t_y1 + t_y2) / 2
    p_sw = (p_x2 - p_x1) / 2;  p_sh = (p_y2 - p_y1) / 2
    t_sw = (t_x2 - t_x1) / 2;  t_sh = (t_y2 - t_y1) / 2
    w2 = ((p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2) +
          (p_sw - t_sw).pow(2) + (p_sh - t_sh).pow(2))
    return 1.0 - torch.exp(-w2.sqrt() / (c + eps))


def _tal_assign_one_scale_v15(raw_b, gt_boxes_xywh, stride, H, W, img_size,
                               k_tiny_eff, k_normal_eff, tiny_area_px,
                               search_r_tiny, search_r_normal, scale_range,
                               tal_alpha=1.0, tal_beta=6.0,
                               stal_gamma=2.0, stal_area_ref=576.0,
                               stal_min_area_px=64.0, stal_k_min=4):
    """
    STAL-enhanced TAL assignment (YOLOv26, arXiv 2509.25164).

    Two STAL mechanisms:
    1. Minimum assignment guarantee: if a tiny object (<stal_min_area_px)
       gets fewer than stal_k_min anchors, expand search_r by 1 and retry.
    2. Alignment bonus: alignment *= (1 + γ·exp(-area_px/area_ref))
       Tiny objects score higher in top-k → less likely to be outcompeted.
    """
    device = raw_b.device
    G = len(gt_boxes_xywh)
    occupied = {}

    for gi in range(G):
        cx = float(gt_boxes_xywh[gi, 0])
        cy = float(gt_boxes_xywh[gi, 1])
        gw = float(gt_boxes_xywh[gi, 2])
        gh = float(gt_boxes_xywh[gi, 3])

        gw_px = gw * img_size
        gh_px = gh * img_size
        max_dim = max(gw_px, gh_px)

        if max_dim < scale_range[0] or max_dim > scale_range[1]:
            continue

        area_px  = gw_px * gh_px
        is_tiny  = area_px < tiny_area_px
        k        = k_tiny_eff  if is_tiny else k_normal_eff
        sr       = search_r_tiny if is_tiny else search_r_normal

        # STAL mechanism 1: guarantee min anchors for extremely tiny objects
        is_micro = area_px < stal_min_area_px
        k        = max(k, stal_k_min) if is_micro else k

        gt_x1 = cx - gw / 2;  gt_y1 = cy - gh / 2
        gt_x2 = cx + gw / 2;  gt_y2 = cy + gh / 2

        gx_f = cx * W;  gy_f = cy * H
        ci0  = min(max(int(gx_f), 0), W - 1)
        ri0  = min(max(int(gy_f), 0), H - 1)

        def _get_candidates(search_radius):
            cand_r, cand_c = [], []
            for dr in range(-search_radius, search_radius + 1):
                for dc in range(-search_radius, search_radius + 1):
                    r = ri0 + dr;  c = ci0 + dc
                    if 0 <= r < H and 0 <= c < W:
                        cand_r.append(r);  cand_c.append(c)
            return cand_r, cand_c

        cand_r, cand_c = _get_candidates(sr)
        if not cand_r:
            continue

        def _compute_alignment(cand_r, cand_c):
            rows_t = torch.tensor(cand_r, device=device, dtype=torch.long)
            cols_t = torch.tensor(cand_c, device=device, dtype=torch.long)

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

            ix1 = torch.maximum(p_x1_t, gt_x1_t); iy1 = torch.maximum(p_y1_t, gt_y1_t)
            ix2 = torch.minimum(p_x2_t, gt_x2_t); iy2 = torch.minimum(p_y2_t, gt_y2_t)
            inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
            union = (p_w_t * p_h_t).clamp(min=0) + gw * gh - inter + 1e-7
            iou = (inter / union).clamp(0.0, 1.0)

            # STAL mechanism 2: alignment bonus for tiny objects
            stal_bonus = 1.0 + stal_gamma * math.exp(-area_px / max(stal_area_ref, 1e-7))
            alignment = pred_scores.pow(tal_alpha) * iou.pow(tal_beta) * stal_bonus

            return rows_t, cols_t, alignment

        rows_t, cols_t, alignment = _compute_alignment(cand_r, cand_c)
        M = len(cand_r)
        actual_k = min(k, M)
        topk_idx = alignment.topk(actual_k)[1] if actual_k < M else torch.arange(M, device=device)

        # STAL mechanism 1: if micro object got fewer than stal_k_min, expand search
        if is_micro and actual_k < stal_k_min:
            cand_r2, cand_c2 = _get_candidates(sr + 2)
            if len(cand_r2) > len(cand_r):
                rows_t, cols_t, alignment = _compute_alignment(cand_r2, cand_c2)
                M2 = len(cand_r2)
                actual_k2 = min(max(stal_k_min, k), M2)
                topk_idx = alignment.topk(actual_k2)[1] if actual_k2 < M2 else torch.arange(M2, device=device)
                cand_r, cand_c = cand_r2, cand_c2

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


class PFDetLossV15:
    """
    v15 Loss: TAL + STAL + ProgLoss + QFL + NWD/CIoU + ASL.

    ProgLoss schedule (YOLOv26):
      obj_weight: obj_weight * prog_factor → obj_weight  (decreasing over epochs)
      box_weight: box_weight / prog_factor → box_weight  (increasing over epochs)
      progress = epoch / total_epochs, linear interpolation.
      Default prog_factor=2.0: obj starts 2× higher, box starts 2× lower.
    """
    def __init__(
        self, img_size=640, strides=(4, 8, 16, 32),
        obj_weight=8.0, box_weight=4.0, qfl_beta=2.0, hard_floor=0.0,
        obj_warmup_epochs=10, prog_epochs=400, asl_area_ref=0.004, asl_max=4.0,
        tiny_area_px=576.0, k_tiny=6, k_normal=3, search_r_tiny=3, search_r_normal=2,
        tal_alpha=1.0, tal_beta=8.0, nwd_area_thr=1024.0, nwd_c=0.019,
        stal_gamma=2.0, stal_area_ref=576.0, stal_min_area_px=64.0, stal_k_min=4,
        prog_loss_factor=2.0, total_epochs=400,
    ):
        self.img_size          = img_size
        self.strides           = strides
        self.obj_weight        = obj_weight
        self.box_weight        = box_weight
        self.qfl_beta          = qfl_beta
        self.hard_floor        = hard_floor
        self.obj_warmup_epochs = max(1, obj_warmup_epochs)
        self.prog_epochs       = max(1, prog_epochs)
        self.asl_area_ref      = asl_area_ref
        self.asl_max           = asl_max
        self.tiny_area_px      = tiny_area_px
        self.k_tiny            = k_tiny
        self.k_normal          = k_normal
        self.search_r_tiny     = search_r_tiny
        self.search_r_normal   = search_r_normal
        self.tal_alpha         = tal_alpha
        self.tal_beta          = tal_beta
        self.nwd_area_thr      = nwd_area_thr
        self.nwd_c             = nwd_c
        self.stal_gamma        = stal_gamma
        self.stal_area_ref     = stal_area_ref
        self.stal_min_area_px  = stal_min_area_px
        self.stal_k_min        = stal_k_min
        self.prog_loss_factor  = prog_loss_factor
        self.total_epochs      = total_epochs

        self.scale_ranges = [
            (0,    48),
            (24,  112),
            (80,  200),
            (160, 99999),
        ]

    def _prog_weights(self, epoch):
        """ProgLoss: progressive weight shift from cls-heavy to reg-heavy."""
        progress = min(1.0, epoch / max(1, self.total_epochs))
        f = self.prog_loss_factor
        # obj_weight: starts high (× f), linearly decreases to base
        obj_w = self.obj_weight * (f + (1.0 - f) * progress)
        # box_weight: starts low (/ f), linearly increases to base
        box_w = self.box_weight * (1.0 / f + (1.0 - 1.0 / f) * progress)
        return obj_w, box_w

    def _obj_target_schedule(self, iou, epoch):
        warm_ratio = max(0.0, 1.0 - epoch / max(1, self.obj_warmup_epochs))
        floor = self.hard_floor * max(0.0, 1.0 - epoch / max(1, self.obj_warmup_epochs * 2))
        warm_ratio = max(warm_ratio, floor)
        return (iou * (1.0 - warm_ratio) + warm_ratio).clamp(0.0, 1.0)

    def _prog_k(self, epoch):
        ratio = min(1.0, epoch / self.prog_epochs)
        return max(3, round(self.k_tiny * (1.0 - ratio) + ratio)), \
               max(2, round(self.k_normal * (1.0 - ratio) + ratio))

    def summary(self):
        return (f"PFDetLossV15 (QFL + NWD/CIoU + STAL[γ={self.stal_gamma}] "
                f"+ ProgLoss[f={self.prog_loss_factor}])")

    def __call__(self, preds, labels_list, epoch=0):
        device = preds[0].device
        B = preds[0].shape[0]

        obj_weight, box_weight = self._prog_weights(epoch)

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
                    qfl = quality_focal_loss(
                        obj_logits_map.view(-1), obj_targets_map.view(-1), self.qfl_beta)
                    # Empty frames are common in VisDrone; averaging keeps their
                    # pure-negative signal from overwhelming positive batches.
                    total_obj_loss = total_obj_loss + qfl.mean()
                    continue

                with torch.no_grad():
                    rows, cols, gi_list = _tal_assign_one_scale_v15(
                        raw_b, gt_boxes_xywh.to(device),
                        stride, H, W, self.img_size,
                        k_tiny_eff, k_normal_eff,
                        self.tiny_area_px,
                        self.search_r_tiny, self.search_r_normal,
                        scale_range=self.scale_ranges[si],
                        tal_alpha=self.tal_alpha,
                        tal_beta=self.tal_beta,
                        stal_gamma=self.stal_gamma,
                        stal_area_ref=self.stal_area_ref,
                        stal_min_area_px=self.stal_min_area_px,
                        stal_k_min=self.stal_k_min,
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
                        p_cx - p_w/2, p_cy - p_h/2, p_cx + p_w/2, p_cy + p_h/2
                    ], dim=1)

                    t_cx, t_cy = gt_t[gi_t, 0], gt_t[gi_t, 1]
                    t_w,  t_h  = gt_t[gi_t, 2], gt_t[gi_t, 3]
                    gt_xyxy = torch.stack([
                        t_cx - t_w/2, t_cy - t_h/2, t_cx + t_w/2, t_cy + t_h/2
                    ], dim=1)

                    area_px = t_w * self.img_size * t_h * self.img_size + 1e-7
                    nwd_w   = torch.exp(-area_px / self.nwd_area_thr)
                    l_nwd   = nwd_loss(pred_xyxy, gt_xyxy, c=self.nwd_c)
                    l_ciou  = bbox_ciou(pred_xyxy, gt_xyxy)
                    box_loss_per = nwd_w * l_nwd + (1.0 - nwd_w) * l_ciou

                    # ASL: up-weight tiny objects (match v14 formula: area_ref in normalized units → pixel units)
                    asl_w = (self.asl_area_ref * self.img_size ** 2 / area_px).sqrt().clamp(1.0, self.asl_max)
                    box_loss_per = box_loss_per * asl_w

                    with torch.no_grad():
                        ix1 = torch.maximum(pred_xyxy[:, 0], gt_xyxy[:, 0])
                        iy1 = torch.maximum(pred_xyxy[:, 1], gt_xyxy[:, 1])
                        ix2 = torch.minimum(pred_xyxy[:, 2], gt_xyxy[:, 2])
                        iy2 = torch.minimum(pred_xyxy[:, 3], gt_xyxy[:, 3])
                        inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
                        union = (p_w * p_h).clamp(0) + t_w * t_h - inter + 1e-7
                        iou_vals = (inter / union).clamp(0, 1)

                    iou_targets = self._obj_target_schedule(iou_vals, epoch)
                    obj_targets_map[rows_t, cols_t] = iou_targets
                    pos_mask_flat[rows_t * W + cols_t] = True
                    total_iou += iou_vals.sum().item()
                    total_iou_count += n_pos
                    total_box_loss = total_box_loss + box_loss_per.mean()

                # Normalize obj_loss by n_pos (same as v14) — not by H×W.
                # qfl.mean() = qfl.sum()/H×W dilutes gradient ~550× vs /n_pos.
                qfl = quality_focal_loss(
                    obj_logits_map.view(-1), obj_targets_map.view(-1), self.qfl_beta)
                obj_loss_scale = qfl.sum() / max(1, int(pos_mask_flat.sum()))
                total_obj_loss = total_obj_loss + obj_loss_scale

        obj_loss = total_obj_loss / B
        box_loss = total_box_loss / max(B, 1)
        loss = obj_weight * obj_loss + box_weight * box_loss

        mean_iou = total_iou / max(total_iou_count, 1)
        loss_dict = {
            'total': loss.item(), 'obj': obj_loss.item(), 'box': box_loss.item(),
            'iou': mean_iou, 'n_pos': n_pos_total / B,
            **{f'pos_s{self.strides[i]}': n_pos_per_scale[i] / B for i in range(len(self.strides))},
        }
        return loss, loss_dict
