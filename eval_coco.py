"""
eval_coco.py — Đánh giá CHUẨN (pycocotools) cho PFDet-Nano
===========================================================

Thay cho hàm evaluate() hiện tại (chỉ cho 1 con AP@0.5). Cung cấp đầy đủ:
  - AP@0.5, AP@0.5:0.95 (COCO mAP)
  - AP_small / AP_medium / AP_large  <-- BẮT BUỘC cho câu chuyện tiny object
  - AR (average recall)
  - Phân tầng theo MẬT ĐỘ (số người/ảnh) -> chứng minh density head giúp cảnh đông

Cài: pip install pycocotools

Cách dùng (drop-in, tái dùng decode/nms CỦA BẠN):

    from eval_coco import evaluate_coco
    metrics = evaluate_coco(model, val_loader, device,
                            img_size=cfg['img_size'], strides=model.strides,
                            decode_fn=decode_predictions_np, nms_fn=nms_numpy,
                            conf_thr=0.01)
    print(metrics['overall'])      # dict AP50, AP, AP_s, AP_m, AP_l, AR
    print(metrics['by_crowd'])     # dict theo nhóm mật độ

Nhãn vào: list per-image tensor (N,5) = [class, cx, cy, w, h] CHUẨN HOÁ 0-1 (đúng format repo).
Box predict: dict {'box':[x1,y1,x2,y2] chuẩn hoá 0-1, 'score':float} (đúng output decode của bạn).
"""

import numpy as np
import torch

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    _HAS_COCO = True
except Exception:
    _HAS_COCO = False


# COCO area ranges (pixel^2) — small <32^2, medium 32^2..96^2, large >96^2
def _norm_cxcywh_to_xywh_px(boxes_norm, img_size):
    """[cx,cy,w,h] norm -> [x,y,w,h] pixel (COCO format)."""
    if len(boxes_norm) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    b = np.asarray(boxes_norm, dtype=np.float32) * img_size
    x = b[:, 0] - b[:, 2] / 2.0
    y = b[:, 1] - b[:, 3] / 2.0
    return np.stack([x, y, b[:, 2], b[:, 3]], axis=1)


def _norm_xyxy_to_xywh_px(box_norm, img_size):
    """[x1,y1,x2,y2] norm -> [x,y,w,h] pixel."""
    x1, y1, x2, y2 = [v * img_size for v in box_norm]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def coco_eval(pred_per_image, gt_per_image, img_size, verbose=False):
    """
    pred_per_image : list (len = #images); mỗi phần tử = list dict {'box':xyxy_norm,'score'}
    gt_per_image   : list (len = #images); mỗi phần tử = np.array (N,4) [cx,cy,w,h] norm (chỉ person)
    return: dict {AP, AP50, AP75, AP_s, AP_m, AP_l, AR_100, AR_s, AR_m, AR_l}
    """
    assert _HAS_COCO, "Cần: pip install pycocotools"

    images, annotations, results = [], [], []
    ann_id = 1
    for img_id, (preds, gts) in enumerate(zip(pred_per_image, gt_per_image), start=1):
        images.append({"id": img_id, "width": img_size, "height": img_size})
        gt_xywh = _norm_cxcywh_to_xywh_px(gts, img_size)
        for g in gt_xywh:
            w, h = float(g[2]), float(g[3])
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": 1,
                "bbox": [float(g[0]), float(g[1]), w, h],
                "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
        for d in preds:
            bx = _norm_xyxy_to_xywh_px(d["box"], img_size)
            results.append({
                "image_id": img_id, "category_id": 1,
                "bbox": [float(v) for v in bx], "score": float(d["score"]),
            })

    coco_gt = COCO()
    coco_gt.dataset = {"images": images, "annotations": annotations,
                       "categories": [{"id": 1, "name": "person"}]}
    coco_gt.createIndex()

    if len(results) == 0:
        keys = ["AP", "AP50", "AP75", "AP_s", "AP_m", "AP_l", "AR_100", "AR_s", "AR_m", "AR_l"]
        return {k: 0.0 for k in keys}

    coco_dt = coco_gt.loadRes(results)
    ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
    ev.evaluate(); ev.accumulate()
    if verbose:
        ev.summarize()
    else:
        ev.summarize()  # pycocotools in ra stats; muốn tắt thì redirect stdout
    s = ev.stats  # 12 giá trị chuẩn COCO
    return {
        "AP": float(s[0]), "AP50": float(s[1]), "AP75": float(s[2]),
        "AP_s": float(s[3]), "AP_m": float(s[4]), "AP_l": float(s[5]),
        "AR_100": float(s[8]), "AR_s": float(s[9]), "AR_m": float(s[10]), "AR_l": float(s[11]),
    }


@torch.no_grad()
def evaluate_coco(model, loader, device, img_size, strides, decode_fn, nms_fn,
                  conf_thr=0.01, nms_iou=0.5, max_det=300,
                  crowd_bins=((0, 10), (10, 30), (30, 1e9))):
    """
    Drop-in thay evaluate(). Tái dùng decode_fn (decode_predictions_np) và nms_fn (nms_numpy) của bạn.
    Trả về metrics tổng + phân tầng theo số người/ảnh (crowd_bins).
    """
    model.eval()
    pred_all, gt_all, ngt_all = [], [], []

    for imgs, labels_list, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)  # inference: density head KHÔNG chạy
        for b in range(imgs.shape[0]):
            gt = labels_list[b]
            gt_person = gt[gt[:, 0] == 0] if len(gt) > 0 else gt
            gt_boxes = gt_person[:, 1:5].cpu().numpy() if len(gt_person) > 0 else np.zeros((0, 4))

            dets = []
            for si, pred in enumerate(preds):
                raw = pred[b].cpu().numpy()
                dets.extend(decode_fn(raw, strides[si], img_size, conf_thr=conf_thr))
            dets = nms_fn(dets, iou_threshold=nms_iou)[:max_det]

            pred_all.append(dets)
            gt_all.append(gt_boxes)
            ngt_all.append(len(gt_boxes))

    out = {"overall": coco_eval(pred_all, gt_all, img_size)}

    # phân tầng theo mật độ
    by_crowd = {}
    ngt_arr = np.array(ngt_all)
    for lo, hi in crowd_bins:
        idx = np.where((ngt_arr >= lo) & (ngt_arr < hi))[0]
        label = f"{lo}-{int(hi) if hi < 1e8 else 'inf'} người"
        if len(idx) == 0:
            by_crowd[label] = {"n_images": 0}
            continue
        p = [pred_all[i] for i in idx]
        g = [gt_all[i] for i in idx]
        m = coco_eval(p, g, img_size)
        m["n_images"] = int(len(idx))
        by_crowd[label] = m
    out["by_crowd"] = by_crowd
    return out


# ============================================================
#  Self-test phần tính COCO metric (dữ liệu giả)
# ============================================================
if __name__ == "__main__":
    if not _HAS_COCO:
        print("pycocotools chưa cài -> pip install pycocotools"); raise SystemExit

    rng = np.random.default_rng(0)
    img_size = 640
    pred_all, gt_all = [], []
    for _ in range(20):
        n = rng.integers(1, 8)
        gts = []
        preds = []
        for _ in range(n):
            cx, cy = rng.uniform(0.1, 0.9, 2)
            w = h = rng.uniform(0.02, 0.15)  # nhiều box nhỏ -> rơi vào AP_small
            gts.append([cx, cy, w, h])
            # predict lệch nhẹ + score cao -> mô phỏng detector tốt
            jx, jy = rng.normal(0, 0.005, 2)
            x1, y1 = cx - w / 2 + jx, cy - h / 2 + jy
            x2, y2 = cx + w / 2 + jx, cy + h / 2 + jy
            preds.append({"box": [x1, y1, x2, y2], "score": float(rng.uniform(0.6, 0.95))})
        gt_all.append(np.array(gts, dtype=np.float32))
        pred_all.append(preds)

    print("=== Self-test coco_eval (detector gần khớp GT) ===")
    m = coco_eval(pred_all, gt_all, img_size, verbose=False)
    print({k: round(v, 4) for k, v in m.items()})
    assert m["AP50"] > 0.7, "detector khớp tốt thì AP50 phải cao"
    assert m["AP_s"] >= 0.0
    print("\n✅ EVAL_COCO METRIC LOGIC OK — AP_small/medium/large tính được.")
