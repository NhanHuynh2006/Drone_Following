"""
Loss functions for PFDet-Nano v5.

Multi-positive assignment (YOLOv5-style):
  Each GT gets 3 positive cells per scale (center + 2 nearest neighbors).
  Uses extended sigmoid decode: sigmoid(x)*2 - 0.5, range [-0.5, 1.5].

IoU-aware objectness:
  Positive cells get soft target = pred-GT IoU (detached), not hard 1.0.
  This teaches the model to output calibrated confidence scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.box_ops import bbox_ciou, box_iou


class PFDetLoss(nn.Module):
    def __init__(self, img_size=416, strides=(8, 16, 32),
                 obj_weight=1.0, box_weight=5.0, **_kwargs):
        super().__init__()
        self.img_size = img_size
        self.strides = strides
        self.obj_weight = obj_weight
        self.box_weight = box_weight

    def _get_positive_cells(self, gt_cx_g, gt_cy_g, H, W):
        """
        Multi-positive: for each GT, assign center cell + 2 nearest neighbors.
        Returns list of (row, col) tuples for each GT.
        """
        n_gt = gt_cx_g.shape[0]
        all_cells = []

        for gi in range(n_gt):
            gx = gt_cx_g[gi].item()
            gy = gt_cy_g[gi].item()
            col = int(gx)
            row = int(gy)
            col = max(0, min(W - 1, col))
            row = max(0, min(H - 1, row))

            cells = [(row, col)]  # center cell always

            # X neighbor: if offset < 0.5, assign left; else assign right
            offset_x = gx - col
            if offset_x < 0.5 and col > 0:
                cells.append((row, col - 1))
            elif offset_x >= 0.5 and col < W - 1:
                cells.append((row, col + 1))

            # Y neighbor: if offset < 0.5, assign up; else assign down
            offset_y = gy - row
            if offset_y < 0.5 and row > 0:
                cells.append((row - 1, col))
            elif offset_y >= 0.5 and row < H - 1:
                cells.append((row + 1, col))

            all_cells.append(cells)

        return all_cells

    def forward(self, predictions, targets_list):
        device = predictions[0].device
        B = predictions[0].shape[0]

        all_obj_logits = []
        all_obj_targets = []

        total_box = torch.tensor(0.0, device=device)
        total_pos = 0
        n_box_items = 0

        for pred, stride in zip(predictions, self.strides):
            _, _, H, W = pred.shape

            for b in range(B):
                labels = targets_list[b].to(device)
                person = labels[labels[:, 0] == 0] if len(labels) > 0 else labels

                gt_boxes = person[:, 1:5] if len(person) > 0 else torch.zeros(0, 4, device=device)
                n_gt = gt_boxes.shape[0]

                # Init targets
                obj_target = torch.zeros(H, W, device=device)
                pos_mask = torch.zeros(H, W, device=device, dtype=torch.bool)
                # Store GT index for each positive cell (for multi-positive,
                # if two GTs map to same cell, last one wins)
                gt_idx_map = torch.full((H, W), -1, device=device, dtype=torch.long)

                if n_gt > 0:
                    # GT center in grid coordinates
                    gt_cx_g = gt_boxes[:, 0] * self.img_size / stride
                    gt_cy_g = gt_boxes[:, 1] * self.img_size / stride

                    # Get positive cells for each GT
                    all_cells = self._get_positive_cells(gt_cx_g, gt_cy_g, H, W)

                    for gi in range(n_gt):
                        for (r, c) in all_cells[gi]:
                            pos_mask[r, c] = True
                            gt_idx_map[r, c] = gi

                # Collect objectness
                all_obj_logits.append(pred[b, 0].reshape(-1))

                n_pos = pos_mask.sum().item()
                total_pos += n_pos

                if n_pos > 0:
                    n_box_items += 1
                    pos_ij = torch.nonzero(pos_mask, as_tuple=False)
                    rows = pos_ij[:, 0]
                    cols = pos_ij[:, 1]
                    gi_for_pos = gt_idx_map[rows, cols]

                    # Decode predicted box (extended sigmoid)
                    p_cx = (torch.sigmoid(pred[b, 1][pos_mask]) * 2 - 0.5 + cols.float()) * stride / self.img_size
                    p_cy = (torch.sigmoid(pred[b, 2][pos_mask]) * 2 - 0.5 + rows.float()) * stride / self.img_size
                    p_w = torch.exp(pred[b, 3][pos_mask].clamp(-5, 5)) * stride / self.img_size
                    p_h = torch.exp(pred[b, 4][pos_mask].clamp(-5, 5)) * stride / self.img_size

                    # GT targets for each positive cell
                    t_cx = gt_boxes[gi_for_pos, 0]
                    t_cy = gt_boxes[gi_for_pos, 1]
                    t_w = gt_boxes[gi_for_pos, 2]
                    t_h = gt_boxes[gi_for_pos, 3]

                    pred_xyxy = torch.stack([p_cx - p_w/2, p_cy - p_h/2,
                                            p_cx + p_w/2, p_cy + p_h/2], dim=-1)
                    tgt_xyxy = torch.stack([t_cx - t_w/2, t_cy - t_h/2,
                                           t_cx + t_w/2, t_cy + t_h/2], dim=-1)

                    ciou_loss = bbox_ciou(pred_xyxy, tgt_xyxy).clamp(min=0)
                    total_box = total_box + ciou_loss.mean()

                    # IoU-aware objectness target (soft label)
                    with torch.no_grad():
                        # Compute IoU between pred and GT for each positive cell
                        inter_x1 = torch.max(pred_xyxy[:, 0], tgt_xyxy[:, 0])
                        inter_y1 = torch.max(pred_xyxy[:, 1], tgt_xyxy[:, 1])
                        inter_x2 = torch.min(pred_xyxy[:, 2], tgt_xyxy[:, 2])
                        inter_y2 = torch.min(pred_xyxy[:, 3], tgt_xyxy[:, 3])
                        inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
                        area_p = (p_w * p_h).clamp(min=1e-7)
                        area_t = (t_w * t_h).clamp(min=1e-7)
                        union = area_p + area_t - inter
                        iou = (inter / (union + 1e-7)).clamp(0, 1)

                        # Set objectness target = IoU (soft label)
                        for pi in range(len(rows)):
                            obj_target[rows[pi], cols[pi]] = iou[pi]

                all_obj_targets.append(obj_target.reshape(-1))

        # Objectness loss: BCE with sum/n_pos normalization
        all_obj_logits = torch.cat(all_obj_logits)
        all_obj_targets = torch.cat(all_obj_targets)

        obj_loss_sum = F.binary_cross_entropy_with_logits(
            all_obj_logits, all_obj_targets, reduction='sum')
        total_obj = obj_loss_sum / max(1.0, float(total_pos))

        n_box_items = max(1, n_box_items)
        total_box = total_box / n_box_items

        total_loss = self.obj_weight * total_obj + self.box_weight * total_box

        return total_loss, {
            'total': total_loss.item(),
            'obj': total_obj.item(),
            'box': total_box.item(),
            'n_pos': total_pos,
        }
