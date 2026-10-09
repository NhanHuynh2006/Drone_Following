"""
box_ops_np.py — CHEP NGUYEN XI tu utils/box_ops.py, bo phan torch de chay tren Pi.
=================================================================================

Vi sao phai co file nay: AP cua HEF chi so sanh duoc voi AP cua FP32 khi CA HAI
dung DUNG MOT decode + MOT NMS. Duong eval chuan cua repo (train_v3.py:evaluate)
goi decode_predictions_np (co pre_nms_topk=500) + nms_numpy (DIoU-NMS).
Ban NMS rieng truoc day trong hailo_eval_val.py khac ca hai thu do
-> so sanh khong hop le, va cham hon ~40 lan vi thieu pre_nms_topk.

KHONG sua tay file nay. Neu utils/box_ops.py doi thi sinh lai.
"""
import numpy as np


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
