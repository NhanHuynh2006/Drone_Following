"""
Box operations for PFDet-Nano.
Includes: CIoU loss, NMS, box format conversions, grid generation.
"""

import torch
import torch.nn.functional as F
import math
import numpy as np


# ============================================================
#  Format Conversions
# ============================================================

def xywh2xyxy(boxes):
    """Convert (cx,cy,w,h) -> (x1,y1,x2,y2). Works with both tensors and numpy."""
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)
    else:
        cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=-1)


def xyxy2xywh(boxes):
    """Convert (x1,y1,x2,y2) -> (cx,cy,w,h)."""
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)
    else:
        x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        return np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=-1)


# ============================================================
#  IoU Computations (vectorized)
# ============================================================

def box_iou(box1, box2):
    """
    Compute IoU between two sets of boxes.
    box1: (N, 4) in xyxy format
    box2: (M, 4) in xyxy format
    Returns: (N, M) IoU matrix
    """
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])

    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter

    return inter / (union + 1e-7)


def bbox_ciou(pred_xyxy, target_xyxy):
    """
    Complete IoU loss (CIoU) between predicted and target boxes.
    Both inputs: (..., 4) in xyxy format.
    Returns: CIoU loss value (1 - CIoU), same shape as leading dims.
    """
    eps = 1e-7

    # Intersection
    inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Union
    w1 = pred_xyxy[..., 2] - pred_xyxy[..., 0]
    h1 = pred_xyxy[..., 3] - pred_xyxy[..., 1]
    w2 = target_xyxy[..., 2] - target_xyxy[..., 0]
    h2 = target_xyxy[..., 3] - target_xyxy[..., 1]
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter

    iou = inter / (union + eps)

    # Enclosing box
    enclose_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    enclose_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    enclose_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    enclose_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])

    # Distance term (center distance / diagonal of enclosing box)
    pred_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    pred_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    tgt_cx = (target_xyxy[..., 0] + target_xyxy[..., 2]) / 2
    tgt_cy = (target_xyxy[..., 1] + target_xyxy[..., 3]) / 2

    rho2 = (pred_cx - tgt_cx) ** 2 + (pred_cy - tgt_cy) ** 2
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps

    # Aspect ratio penalty
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou  # CIoU loss (lower is better)


# ============================================================
#  Grid Generation
# ============================================================

def make_grid(h, w, device='cpu', dtype=torch.float32):
    """
    Create grid of cell coordinates.
    Returns: (H*W, 2) tensor of (grid_x, grid_y) for each cell.
    """
    yv, xv = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                             torch.arange(w, device=device, dtype=dtype),
                             indexing='ij')
    return torch.stack([xv, yv], dim=-1).reshape(-1, 2)


# ============================================================
#  Decode predictions
# ============================================================

def decode_predictions(raw_pred, stride, img_size):
    """
    Decode raw model output to absolute coordinates.

    Args:
        raw_pred: (B, 7, H, W) raw output from detection head
        stride: spatial stride for this scale (8, 16, or 32)
        img_size: input image size (int)

    Returns:
        decoded: (B, H*W, 7) with:
            [0]   objectness score (after sigmoid)
            [1:3] center_xy in normalized [0,1] coords
            [2:5] wh in normalized [0,1] coords
            [5:7] foot_xy in normalized [0,1] coords
    """
    B, C, H, W = raw_pred.shape
    device = raw_pred.device

    # Flatten spatial dims: (B, 7, H, W) -> (B, 7, H*W) -> (B, H*W, 7)
    pred = raw_pred.reshape(B, C, -1).permute(0, 2, 1)

    # Grid offsets
    grid = make_grid(H, W, device=device)  # (H*W, 2)

    # Objectness
    obj = torch.sigmoid(pred[..., 0:1])

    # Box center: sigmoid(delta) + grid_offset, then normalize
    cx = (torch.sigmoid(pred[..., 1:2]) + grid[None, :, 0:1]) * stride / img_size
    cy = (torch.sigmoid(pred[..., 2:3]) + grid[None, :, 1:2]) * stride / img_size

    # Box size: exp(log_pred) * stride, then normalize, clamped for stability
    bw = torch.exp(pred[..., 3:4].clamp(-5, 5)) * stride / img_size
    bh = torch.exp(pred[..., 4:5].clamp(-5, 5)) * stride / img_size

    # Foot point (sigmoid to [0,1])
    fx = torch.sigmoid(pred[..., 5:6])
    fy = torch.sigmoid(pred[..., 6:7])

    decoded = torch.cat([obj, cx, cy, bw, bh, fx, fy], dim=-1)
    return decoded


def decode_predictions_np(raw_pred, stride, img_size):
    """
    Numpy version for inference (single image, no batch dim).
    raw_pred: (7, H, W) numpy array
    Returns: list of dicts with 'score', 'box' (xyxy normalized), 'foot' (xy normalized)
    """
    C, H, W = raw_pred.shape
    results = []

    obj_map = 1.0 / (1.0 + np.exp(-raw_pred[0]))  # sigmoid

    for j in range(H):
        for i in range(W):
            score = obj_map[j, i]
            if score < 0.01:  # very low threshold, NMS will filter
                continue

            dx = 1.0 / (1.0 + np.exp(-raw_pred[1, j, i]))
            dy = 1.0 / (1.0 + np.exp(-raw_pred[2, j, i]))

            cx = (i + dx) * stride / img_size
            cy = (j + dy) * stride / img_size

            lw = np.clip(raw_pred[3, j, i], -5, 5)
            lh = np.clip(raw_pred[4, j, i], -5, 5)
            bw = np.exp(lw) * stride / img_size
            bh = np.exp(lh) * stride / img_size

            fx = 1.0 / (1.0 + np.exp(-raw_pred[5, j, i]))
            fy = 1.0 / (1.0 + np.exp(-raw_pred[6, j, i]))

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            results.append({
                'score': float(score),
                'box': [float(x1), float(y1), float(x2), float(y2)],
                'foot': [float(fx), float(fy)],
                'cx': float(cx),
                'cy': float(cy),
            })

    return results


# ============================================================
#  NMS (Non-Maximum Suppression)
# ============================================================

def nms_numpy(detections, iou_threshold=0.5):
    """
    Standard NMS on a list of detection dicts.
    Each dict must have 'score' and 'box' (xyxy) keys.
    Returns filtered list.
    """
    if len(detections) == 0:
        return []

    boxes = np.array([d['box'] for d in detections], dtype=np.float32)
    scores = np.array([d['score'] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        idx = order[0]
        keep.append(idx)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[idx], x1[order[1:]])
        yy1 = np.maximum(y1[idx], y1[order[1:]])
        xx2 = np.minimum(x2[idx], x2[order[1:]])
        yy2 = np.minimum(y2[idx], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[idx] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return [detections[i] for i in keep]


def nms_torch(boxes, scores, iou_threshold=0.5):
    """
    Torch NMS.
    boxes: (N, 4) xyxy
    scores: (N,)
    Returns: indices to keep
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0].item()
        keep.append(idx)

        if order.numel() == 1:
            break

        xx1 = torch.max(x1[idx], x1[order[1:]])
        yy1 = torch.max(y1[idx], y1[order[1:]])
        xx2 = torch.min(x2[idx], x2[order[1:]])
        yy2 = torch.min(y2[idx], y2[order[1:]])

        inter = (xx2 - xx1).clamp(0) * (yy2 - yy1).clamp(0)
        union = areas[idx] + areas[order[1:]] - inter
        iou = inter / (union + 1e-7)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
