"""
Box operations for PFDet-Nano.
Includes: CIoU loss, NMS, box format conversions, grid generation.
"""

import math
import numpy as np
import torch


def xywh2xyxy(boxes):
    if isinstance(boxes, torch.Tensor):
        cx, cy, w, h = boxes.unbind(-1)
        return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=-1)


def xyxy2xywh(boxes):
    if isinstance(boxes, torch.Tensor):
        x1, y1, x2, y2 = boxes.unbind(-1)
        return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return np.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], axis=-1)


def box_iou(box1, box2):
    area1 = (box1[:, 2] - box1[:, 0]).clamp(min=0) * (box1[:, 3] - box1[:, 1]).clamp(min=0)
    area2 = (box2[:, 2] - box2[:, 0]).clamp(min=0) * (box2[:, 3] - box2[:, 1]).clamp(min=0)
    inter_x1 = torch.max(box1[:, None, 0], box2[None, :, 0])
    inter_y1 = torch.max(box1[:, None, 1], box2[None, :, 1])
    inter_x2 = torch.min(box1[:, None, 2], box2[None, :, 2])
    inter_y2 = torch.min(box1[:, None, 3], box2[None, :, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-7)


def bbox_ciou(pred_xyxy, target_xyxy):
    eps = 1e-7
    inter_x1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    inter_y1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    inter_x2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    inter_y2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    w1 = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=eps)
    h1 = (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=eps)
    w2 = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=eps)
    h2 = (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=eps)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    iou = inter / (union + eps)

    enclose_x1 = torch.min(pred_xyxy[..., 0], target_xyxy[..., 0])
    enclose_y1 = torch.min(pred_xyxy[..., 1], target_xyxy[..., 1])
    enclose_x2 = torch.max(pred_xyxy[..., 2], target_xyxy[..., 2])
    enclose_y2 = torch.max(pred_xyxy[..., 3], target_xyxy[..., 3])

    pred_cx = (pred_xyxy[..., 0] + pred_xyxy[..., 2]) / 2
    pred_cy = (pred_xyxy[..., 1] + pred_xyxy[..., 3]) / 2
    tgt_cx = (target_xyxy[..., 0] + target_xyxy[..., 2]) / 2
    tgt_cy = (target_xyxy[..., 1] + target_xyxy[..., 3]) / 2

    rho2 = (pred_cx - tgt_cx).pow(2) + (pred_cy - tgt_cy).pow(2)
    c2 = (enclose_x2 - enclose_x1).pow(2) + (enclose_y2 - enclose_y1).pow(2) + eps
    v = (4 / (math.pi ** 2)) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - rho2 / c2 - alpha * v
    return 1.0 - ciou


def make_grid(h, w, device='cpu', dtype=torch.float32):
    yv, xv = torch.meshgrid(
        torch.arange(h, device=device, dtype=dtype),
        torch.arange(w, device=device, dtype=dtype),
        indexing='ij'
    )
    return torch.stack([xv, yv], dim=-1).reshape(-1, 2)


def decode_predictions(raw_pred, stride, img_size):
    B, C, H, W = raw_pred.shape
    device = raw_pred.device
    pred = raw_pred.reshape(B, C, -1).permute(0, 2, 1)
    grid = make_grid(H, W, device=device, dtype=pred.dtype)
    obj = torch.sigmoid(pred[..., 0:1])
    cx = (torch.sigmoid(pred[..., 1:2]) + grid[None, :, 0:1]) * stride / img_size
    cy = (torch.sigmoid(pred[..., 2:3]) + grid[None, :, 1:2]) * stride / img_size
    bw = torch.exp(pred[..., 3:4].clamp(-5, 5)) * stride / img_size
    bh = torch.exp(pred[..., 4:5].clamp(-5, 5)) * stride / img_size
    fx = torch.sigmoid(pred[..., 5:6])
    fy = torch.sigmoid(pred[..., 6:7])
    return torch.cat([obj, cx, cy, bw, bh, fx, fy], dim=-1)


def decode_predictions_np(raw_pred, stride, img_size):
    C, H, W = raw_pred.shape
    results = []
    obj_map = 1.0 / (1.0 + np.exp(-raw_pred[0]))
    for j in range(H):
        for i in range(W):
            score = obj_map[j, i]
            if score < 0.01:
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
            results.append({'score': float(score), 'box': [float(x1), float(y1), float(x2), float(y2)], 'foot': [float(fx), float(fy)], 'cx': float(cx), 'cy': float(cy)})
    return results


def nms_numpy(detections, iou_threshold=0.5):
    if len(detections) == 0:
        return []
    boxes = np.array([d['box'] for d in detections], dtype=np.float32)
    scores = np.array([d['score'] for d in detections], dtype=np.float32)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
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
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return [detections[i] for i in keep]


def nms_torch(boxes, scores, iou_threshold=0.5):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
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
        order = order[1:][iou <= iou_threshold]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)
