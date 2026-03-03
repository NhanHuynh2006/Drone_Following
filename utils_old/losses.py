"""
Loss functions for PFDet-Nano.

Key improvements over original:
  1. Focal Loss for objectness -> handles extreme class imbalance
  2. CIoU Loss for box regression -> much better than L1
  3. SimOTA-Lite label assignment -> assigns multiple positive cells per object
  4. Proper encoding: exp(pred) for w/h instead of raw sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils.box_ops import xywh2xyxy, bbox_ciou, box_iou, make_grid, decode_predictions


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    FL(p) = -alpha * (1-p)^gamma * log(p)     for positive
    FL(p) = -(1-alpha) * p^gamma * log(1-p)   for negative

    Critical for UAV detection where >99% of cells are negative.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred_logit, target):
        """
        pred_logit: raw logit (before sigmoid)
        target: binary target (0 or 1)
        """
        p = torch.sigmoid(pred_logit)
        ce = F.binary_cross_entropy_with_logits(pred_logit, target, reduction='none')

        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        modulating = (1 - p_t) ** self.gamma

        loss = alpha_t * modulating * ce

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SimOTAAssigner:
    """
    Simplified OTA (Optimal Transport Assignment) for label assignment.

    Instead of assigning just 1 cell per object (which makes training very hard),
    SimOTA dynamically assigns multiple positive cells based on cost matrix.

    This is THE most important improvement for detection accuracy.

    Steps:
      1. For each GT, find candidate cells within a center region
      2. Compute cost = classification_cost + regression_cost
      3. Use top-k dynamic selection based on IoU
    """
    def __init__(self, center_radius=2.5, num_candidates=10):
        self.center_radius = center_radius
        self.num_candidates = num_candidates

    @torch.no_grad()
    def assign(self, pred_raw, gt_boxes, gt_foots, stride, img_size):
        """
        Assign targets for a single image at a single scale.

        Args:
            pred_raw: (7, H, W) raw predictions
            gt_boxes: (N, 4) normalized cxywh
            gt_foots: (N, 2) normalized foot xy
            stride: int
            img_size: int

        Returns:
            obj_target: (1, H, W) objectness target (soft label = IoU)
            box_target: (4, H, W) box targets
            foot_target: (2, H, W) foot targets
            pos_mask: (1, H, W) positive cell mask
        """
        device = pred_raw.device
        C, H, W = pred_raw.shape
        n_gt = gt_boxes.shape[0]

        obj_target = torch.zeros(1, H, W, device=device)
        box_target = torch.zeros(4, H, W, device=device)
        foot_target = torch.zeros(2, H, W, device=device)
        pos_mask = torch.zeros(1, H, W, device=device)

        if n_gt == 0:
            return obj_target, box_target, foot_target, pos_mask

        # Generate grid
        grid = make_grid(H, W, device=device)  # (H*W, 2)

        # Decode predictions for cost computation
        pred = pred_raw.reshape(C, -1).permute(1, 0)   # (H*W, 7)
        pred_obj_logit = pred[:, 0]                    # raw logits
        pred_obj = torch.sigmoid(pred_obj_logit)       # probabilities only if needed later

        pred_cx = (torch.sigmoid(pred[:, 1]) + grid[:, 0]) * stride / img_size
        pred_cy = (torch.sigmoid(pred[:, 2]) + grid[:, 1]) * stride / img_size
        pred_w = torch.exp(pred[:, 3].clamp(-5, 5)) * stride / img_size
        pred_h = torch.exp(pred[:, 4].clamp(-5, 5)) * stride / img_size

        pred_boxes_xyxy = xywh2xyxy(torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1))

        for gi in range(n_gt):
            gt_cx, gt_cy, gt_w, gt_h = gt_boxes[gi]
            gt_fx, gt_fy = gt_foots[gi]

            # Step 1: Find candidate cells in center region
            gt_cx_grid = gt_cx * img_size / stride
            gt_cy_grid = gt_cy * img_size / stride

            # Center region (cells near the GT center)
            dist_x = torch.abs(grid[:, 0] - gt_cx_grid)
            dist_y = torch.abs(grid[:, 1] - gt_cy_grid)
            in_center = (dist_x < self.center_radius) & (dist_y < self.center_radius)

            # Also add cells that the GT box covers
            gt_left = (gt_cx - gt_w / 2) * img_size / stride
            gt_right = (gt_cx + gt_w / 2) * img_size / stride
            gt_top = (gt_cy - gt_h / 2) * img_size / stride
            gt_bottom = (gt_cy + gt_h / 2) * img_size / stride

            in_box = (grid[:, 0] >= gt_left) & (grid[:, 0] < gt_right) & \
                     (grid[:, 1] >= gt_top) & (grid[:, 1] < gt_bottom)

            candidates = in_center | in_box
            cand_idx = torch.where(candidates)[0]

            if len(cand_idx) == 0:
                # Fallback: just use the nearest cell
                dist = (grid[:, 0] - gt_cx_grid) ** 2 + (grid[:, 1] - gt_cy_grid) ** 2
                cand_idx = dist.argmin().unsqueeze(0)

            # Step 2: Compute cost for candidates
            gt_xyxy = xywh2xyxy(gt_boxes[gi:gi+1])  # (1, 4)
            cand_pred_xyxy = pred_boxes_xyxy[cand_idx]  # (K, 4)

            pair_iou = box_iou(cand_pred_xyxy, gt_xyxy).squeeze(-1)  # (K,)
            iou_cost = -torch.log(pair_iou + 1e-7)

            cls_cost = F.binary_cross_entropy_with_logits(
                pred_obj_logit[cand_idx],
                torch.ones_like(pred_obj_logit[cand_idx]),
                reduction='none'
            )

            cost = cls_cost + 3.0 * iou_cost

            # Step 3: Dynamic k selection (based on IoU sum)
            n_pos = max(1, min(int(pair_iou.sum().item()), self.num_candidates, len(cand_idx)))

            _, topk_idx = cost.topk(n_pos, largest=False)
            pos_idx = cand_idx[topk_idx]

            # Assign targets
            for pidx in pos_idx:
                gj = pidx // W
                gi_x = pidx % W

                pos_mask[0, gj, gi_x] = 1.0
                # Soft objectness label = IoU (better than hard 1.0)
                cur_iou = pair_iou[topk_idx[0]].item() if len(topk_idx) > 0 else 1.0
                obj_target[0, gj, gi_x] = max(obj_target[0, gj, gi_x].item(), cur_iou)

                # Box target (offsets relative to grid cell)
                tx = gt_cx * img_size / stride - gi_x.float()
                ty = gt_cy * img_size / stride - gj.float()
                tw = torch.log(gt_w * img_size / stride + 1e-7)
                th = torch.log(gt_h * img_size / stride + 1e-7)

                box_target[0, gj, gi_x] = tx
                box_target[1, gj, gi_x] = ty
                box_target[2, gj, gi_x] = tw
                box_target[3, gj, gi_x] = th

                foot_target[0, gj, gi_x] = gt_fx
                foot_target[1, gj, gi_x] = gt_fy

        return obj_target, box_target, foot_target, pos_mask


class PFDetLoss(nn.Module):
    """
    Combined loss for PFDet-Nano.

    Components:
      1. Focal Loss for objectness (handles 99%+ negative cells)
      2. CIoU Loss for box regression (much better than L1/L2)
      3. Smooth L1 for foot point regression
      4. SimOTA assignment for multi-positive training

    All losses are properly normalized and weighted.
    """
    def __init__(self, img_size=416, strides=[8, 16, 32],
                 obj_weight=1.0, box_weight=5.0, foot_weight=1.0,
                 focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.img_size = img_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.box_weight = box_weight
        self.foot_weight = foot_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.assigner = SimOTAAssigner(center_radius=2.5, num_candidates=10)

    def forward(self, predictions, targets_list):
        """
        Args:
            predictions: list of 3 tensors [(B,7,H1,W1), (B,7,H2,W2), (B,7,H3,W3)]
            targets_list: list of B tensors, each (N_i, 7) with [cls, cx, cy, w, h, fx, fy]
                          all normalized to [0,1]

        Returns:
            total_loss, loss_dict
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        total_obj_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_foot_loss = torch.tensor(0.0, device=device)
        total_pos = 0

        for si, (pred, stride) in enumerate(zip(predictions, self.strides)):
            B, C, H, W = pred.shape

            for b in range(B):
                labels = targets_list[b].to(device)

                # Filter person class only (cls_id == 0)
                person_mask = labels[:, 0] == 0
                labels = labels[person_mask]

                if len(labels) == 0:
                    gt_boxes = torch.zeros(0, 4, device=device)
                    gt_foots = torch.zeros(0, 2, device=device)
                else:
                    gt_boxes = labels[:, 1:5]  # cx, cy, w, h
                    gt_foots = labels[:, 5:7]  # fx, fy

                    # Scale-aware filtering: only keep GTs appropriate for this scale
                    gt_size = torch.max(gt_boxes[:, 2], gt_boxes[:, 3]) * self.img_size
                    if si == 0:     # stride 8: small objects
                        scale_mask = gt_size < 128
                    elif si == 1:   # stride 16: medium objects
                        scale_mask = (gt_size >= 48) & (gt_size < 288)
                    else:           # stride 32: large objects
                        scale_mask = gt_size >= 128

                    gt_boxes = gt_boxes[scale_mask]
                    gt_foots = gt_foots[scale_mask]

                # SimOTA assignment
                obj_tgt, box_tgt, foot_tgt, pmask = self.assigner.assign(
                    pred[b], gt_boxes, gt_foots, stride, self.img_size
                )

                # --- Objectness loss (Focal Loss) ---
                obj_logit = pred[b, 0:1]  # (1, H, W)
                obj_loss = self.focal_loss(obj_logit, obj_tgt).mean()
                total_obj_loss = total_obj_loss + obj_loss

                # --- Box + Foot loss on positives only ---
                npos = pmask.sum().clamp(min=1.0)
                total_pos += npos.item()

                if npos > 0.5:
                    # Decode predicted boxes at positive locations
                    pred_sig = pred[b]  # (7, H, W)

                    # Extract positive predictions
                    pm = pmask[0] > 0.5  # (H, W) boolean

                    pred_dx = torch.sigmoid(pred_sig[1][pm])
                    pred_dy = torch.sigmoid(pred_sig[2][pm])
                    pred_lw = pred_sig[3][pm]
                    pred_lh = pred_sig[4][pm]

                    tgt_dx = box_tgt[0][pm]
                    tgt_dy = box_tgt[1][pm]
                    tgt_lw = box_tgt[2][pm]
                    tgt_lh = box_tgt[3][pm]

                    # Get grid positions for positive cells
                    pos_ij = torch.nonzero(pm, as_tuple=False)  # (npos, 2) [j, i]
                    pos_i = pos_ij[:, 1].float()
                    pos_j = pos_ij[:, 0].float()

                    # Reconstruct absolute boxes for CIoU
                    pred_cx = (pred_dx + pos_i) * stride / self.img_size
                    pred_cy = (pred_dy + pos_j) * stride / self.img_size
                    pred_w = torch.exp(pred_lw.clamp(-5, 5)) * stride / self.img_size
                    pred_h = torch.exp(pred_lh.clamp(-5, 5)) * stride / self.img_size

                    tgt_cx = (tgt_dx + pos_i) * stride / self.img_size
                    tgt_cy = (tgt_dy + pos_j) * stride / self.img_size
                    tgt_w = torch.exp(tgt_lw.clamp(-5, 5)) * stride / self.img_size
                    tgt_h = torch.exp(tgt_lh.clamp(-5, 5)) * stride / self.img_size

                    pred_xyxy = xywh2xyxy_flat(pred_cx, pred_cy, pred_w, pred_h)
                    tgt_xyxy = xywh2xyxy_flat(tgt_cx, tgt_cy, tgt_w, tgt_h)

                    ciou_loss = bbox_ciou(pred_xyxy, tgt_xyxy).mean()
                    total_box_loss = total_box_loss + ciou_loss

                    # Foot loss (Smooth L1)
                    pred_fx = torch.sigmoid(pred_sig[5][pm])
                    pred_fy = torch.sigmoid(pred_sig[6][pm])
                    tgt_fx = foot_tgt[0][pm]
                    tgt_fy = foot_tgt[1][pm]

                    foot_loss = (F.smooth_l1_loss(pred_fx, tgt_fx, beta=0.02) +
                                 F.smooth_l1_loss(pred_fy, tgt_fy, beta=0.02))
                    total_foot_loss = total_foot_loss + foot_loss

        # Normalize by batch * num_scales
        n_scale_batch = batch_size * len(self.strides)
        total_obj_loss = total_obj_loss / n_scale_batch
        total_box_loss = total_box_loss / max(1, n_scale_batch)
        total_foot_loss = total_foot_loss / max(1, n_scale_batch)

        total_loss = (self.obj_weight * total_obj_loss +
                      self.box_weight * total_box_loss +
                      self.foot_weight * total_foot_loss)

        loss_dict = {
            'total': total_loss.item(),
            'obj': total_obj_loss.item(),
            'box': total_box_loss.item(),
            'foot': total_foot_loss.item(),
            'n_pos': total_pos,
        }
        return total_loss, loss_dict


def xywh2xyxy_flat(cx, cy, w, h):
    """Convert flat tensors to (N, 4) xyxy."""
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
