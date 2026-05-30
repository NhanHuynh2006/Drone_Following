"""
Box operations for PFDet-Nano.

Provides:
  - xywh2xyxy: convert normalized [cx, cy, w, h] → [x1, y1, x2, y2]
  - iou_numpy: IoU between two boxes (numpy)
  - nms_numpy: Non-maximum suppression (numpy)
  - decode_predictions_np: decode one raw scale output → list of dets
  - bbox_ciou: Complete IoU loss (PyTorch, differentiable)
"""

import math
import numpy as np
import torch


# ============================================================
#  Numpy utilities (used in inference / evaluation)
# ============================================================

def xywh2xyxy(box):
    """
    Convert [cx, cy, w, h] (normalized) → [x1, y1, x2, y2].
    Accepts a 1-D array-like of length 4.
    """
    cx, cy, w, h = box[0], box[1], box[2], box[3]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def _iou_single_np(b1, b2):
    """IoU between two [x1,y1,x2,y2] boxes (numpy scalars)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-7)


def nms_numpy(dets, iou_threshold=0.45):
    """
    DIoU-NMS over a list of detection dicts.

    Extends standard IoU-NMS with a center-distance penalty:
      DIoU = IoU - ρ²(center_a, center_b) / c²
    where c = diagonal of smallest enclosing box.

    Key advantage for VisDrone crowds:
      Two persons standing side-by-side have low center distance
      but high IoU overlap — standard NMS would suppress one.
      DIoU-NMS better preserves distinct detections in dense crowds.

    Reference: Zheng et al., "Distance-IoU Loss", AAAI 2020.
    """
    if len(dets) == 0:
        return []

    scores = np.array([d['score'] for d in dets], dtype=np.float32)
    boxes  = np.array([d['box']   for d in dets], dtype=np.float32)  # (N, 4)

    order = np.argsort(-scores)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    keep = []
    suppressed = np.zeros(len(dets), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(int(idx))

        remaining = order[~suppressed[order]]
        remaining = remaining[remaining != idx]
        if len(remaining) == 0:
            break

        ix1 = np.maximum(x1[idx], x1[remaining])
        iy1 = np.maximum(y1[idx], y1[remaining])
        ix2 = np.minimum(x2[idx], x2[remaining])
        iy2 = np.minimum(y2[idx], y2[remaining])
        inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
        union = areas[idx] + areas[remaining] - inter + 1e-7
        iou = inter / union

        # DIoU penalty: center distance / enclosing diagonal
        enclose_x1 = np.minimum(x1[idx], x1[remaining])
        enclose_y1 = np.minimum(y1[idx], y1[remaining])
        enclose_x2 = np.maximum(x2[idx], x2[remaining])
        enclose_y2 = np.maximum(y2[idx], y2[remaining])
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7
        rho2 = (cx[idx] - cx[remaining]) ** 2 + (cy[idx] - cy[remaining]) ** 2
        diou = iou - rho2 / c2

        suppressed[remaining[diou >= iou_threshold]] = True

    return [dets[k] for k in keep]


def decode_predictions_np(raw, stride, img_size, conf_thr=0.005, pre_nms_topk=500):
    """
    Decode one scale's raw output into detection dicts — fully vectorized.

    Args:
        raw:           numpy array (5, H, W)
        stride:        int — feature stride (4, 8, 16, 32)
        img_size:      int — input image size
        conf_thr:      float — score threshold
        pre_nms_topk:  int — keep at most this many dets per scale (by score)

    Returns:
        list of dicts: {'score': float, 'box': [x1, y1, x2, y2]}
    """
    _sig = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))

    C, H, W = raw.shape
    scores = _sig(raw[0])          # (H, W)

    mask = scores >= conf_thr
    if not mask.any():
        return []

    ys, xs = np.where(mask)        # filtered cell indices

    sc = scores[ys, xs]            # (N,)

    # Keep only top-k by score before NMS (avoids huge N² cost on noisy preds)
    if len(sc) > pre_nms_topk:
        topk_idx = np.argpartition(-sc, pre_nms_topk)[:pre_nms_topk]
        ys, xs, sc = ys[topk_idx], xs[topk_idx], sc[topk_idx]

    dx = _sig(raw[1, ys, xs]) - 0.5
    dy = _sig(raw[2, ys, xs]) - 0.5
    lw = np.clip(raw[3, ys, xs], -4.0, 4.0)
    lh = np.clip(raw[4, ys, xs], -4.0, 4.0)

    cx = (xs + dx) * (stride / img_size)
    cy = (ys + dy) * (stride / img_size)
    w  = np.exp(lw) * (stride / img_size)
    h  = np.exp(lh) * (stride / img_size)

    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5

    # Sort by score descending before returning
    order = np.argsort(-sc)
    return [
        {'score': float(sc[i]), 'box': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])]}
        for i in order
    ]


def _LEGACY_decode_predictions_np_loop(raw, stride, img_size, conf_thr=0.005):
    """Old Python-loop version — kept for reference, DO NOT USE in training."""
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))

    C, H, W = raw.shape
    dets = []
    scores = _sigmoid(raw[0])
    ys, xs = np.where(scores >= conf_thr)
    for j, i in zip(ys, xs):
        score = float(scores[j, i])
        dx = float(_sigmoid(raw[1, j, i])) - 0.5
        dy = float(_sigmoid(raw[2, j, i])) - 0.5
        cx = (i + dx) * stride / img_size
        cy = (j + dy) * stride / img_size
        lw = float(np.clip(raw[3, j, i], -4, 4))
        lh = float(np.clip(raw[4, j, i], -4, 4))
        w = math.exp(lw) * stride / img_size
        h = math.exp(lh) * stride / img_size

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        dets.append({'score': score, 'box': [x1, y1, x2, y2]})

    return dets


# ============================================================
#  PyTorch: CIoU (differentiable, used in loss)
# ============================================================

def bbox_ciou(pred_xyxy, target_xyxy, eps=1e-7):
    """
    Complete IoU loss between predicted and target boxes.

    CIoU = IoU - ρ²(b, b_gt)/c² - α·v
    where:
      ρ = center distance
      c = diagonal of smallest enclosing box
      v = aspect ratio consistency term
      α = v / (1 - IoU + v)

    Args:
        pred_xyxy:   (N, 4) tensor [x1, y1, x2, y2] normalized
        target_xyxy: (N, 4) tensor [x1, y1, x2, y2] normalized

    Returns:
        loss: (N,) tensor, CIoU loss = 1 - CIoU (range [0, 2])
    """
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(dim=1)
    t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(dim=1)

    # Intersection
    ix1 = torch.max(p_x1, t_x1)
    iy1 = torch.max(p_y1, t_y1)
    ix2 = torch.min(p_x2, t_x2)
    iy2 = torch.min(p_y2, t_y2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    # Areas
    p_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    t_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = p_area + t_area - inter + eps
    iou = inter / union

    # Enclosing box diagonal
    ex1 = torch.min(p_x1, t_x1)
    ey1 = torch.min(p_y1, t_y1)
    ex2 = torch.max(p_x2, t_x2)
    ey2 = torch.max(p_y2, t_y2)
    c2 = (ex2 - ex1).pow(2) + (ey2 - ey1).pow(2) + eps

    # Center distance squared
    p_cx = (p_x1 + p_x2) / 2
    p_cy = (p_y1 + p_y2) / 2
    t_cx = (t_x1 + t_x2) / 2
    t_cy = (t_y1 + t_y2) / 2
    rho2 = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    # Aspect ratio consistency
    p_w = (p_x2 - p_x1).clamp(min=eps)
    p_h = (p_y2 - p_y1).clamp(min=eps)
    t_w = (t_x2 - t_x1).clamp(min=eps)
    t_h = (t_y2 - t_y1).clamp(min=eps)
    v = (4.0 / (math.pi ** 2)) * (torch.atan(t_w / t_h) - torch.atan(p_w / p_h)).pow(2)

    with torch.no_grad():
        alpha = v / ((1 - iou) + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return (1.0 - ciou).clamp(min=0.0)


def bbox_iou_xyxy(pred_xyxy, target_xyxy, eps=1e-7):
    """
    Plain IoU between predicted and target boxes in xyxy format.

    Args:
        pred_xyxy:   (N, 4) tensor [x1, y1, x2, y2]
        target_xyxy: (N, 4) tensor [x1, y1, x2, y2]

    Returns:
        iou: (N,) tensor in [0, 1]
    """
    p_x1, p_y1, p_x2, p_y2 = pred_xyxy.unbind(dim=1)
    t_x1, t_y1, t_x2, t_y2 = target_xyxy.unbind(dim=1)

    ix1 = torch.max(p_x1, t_x1)
    iy1 = torch.max(p_y1, t_y1)
    ix2 = torch.min(p_x2, t_x2)
    iy2 = torch.min(p_y2, t_y2)
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)

    p_area = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    t_area = (t_x2 - t_x1).clamp(min=0) * (t_y2 - t_y1).clamp(min=0)
    union = p_area + t_area - inter + eps

    return (inter / union).clamp(0.0, 1.0)
