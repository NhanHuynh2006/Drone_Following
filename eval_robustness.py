"""
eval_robustness.py — Benchmark "VisDrone-Shake": đo AP theo mức độ rung
=======================================================================

Tái dùng corrupt_batch (vib_consistency) + coco_eval (eval_coco). Cho ra đường cong
robustness: AP@.5:.95 ở severity 0 (sạch) / 1 / 2 / 3, và mCP (mean corruption performance).

Đây là BẰNG CHỨNG cho đóng góp: so model thường vs model train với vibration-consistency,
đường cong của model robust phải tụt CHẬM hơn khi severity tăng.

Chạy:
  python eval_robustness.py --weights ... --val-images ... --val-labels ... --img-size 640
"""

import argparse
import inspect
import numpy as np
import torch
from torch.utils.data import DataLoader

from vib_consistency import corrupt_batch
from eval_coco import coco_eval


@torch.no_grad()
def evaluate_robustness(model, loader, device, img_size, strides, decode_fn, nms_fn,
                        severities=(0, 1, 2, 3), conf_thr=0.01, nms_iou=0.5, max_det=300):
    model.eval()
    results = {}
    for sev in severities:
        pred_all, gt_all = [], []
        for imgs, labels_list, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            if sev > 0:
                imgs = corrupt_batch(imgs, severity=sev)   # rung
            preds = model(imgs)
            for b in range(imgs.shape[0]):
                gt = labels_list[b]
                gt_p = gt[gt[:, 0] == 0] if len(gt) > 0 else gt
                gt_boxes = gt_p[:, 1:5].cpu().numpy() if len(gt_p) > 0 else np.zeros((0, 4))
                dets = []
                for si, pred in enumerate(preds):
                    dets.extend(decode_fn(pred[b].cpu().numpy(), strides[si], img_size, conf_thr=conf_thr))
                dets = nms_fn(dets, iou_threshold=nms_iou)[:max_det]
                pred_all.append(dets)
                gt_all.append(gt_boxes)
        m = coco_eval(pred_all, gt_all, img_size)
        tag = "clean" if sev == 0 else f"sev{sev}"
        results[tag] = m
        print(f"[robust] {tag}: AP={m['AP']:.4f} AP50={m['AP50']:.4f} "
              f"AP75={m['AP75']:.4f} AP_s={m['AP_s']:.4f}")

    # mCP = trung bình AP trên các mức rung (>0). rPC = mCP / AP_clean (giữ được bao nhiêu %)
    corr = [results[f"sev{s}"]["AP"] for s in severities if s > 0]
    if corr and "clean" in results and results["clean"]["AP"] > 0:
        mcp = float(np.mean(corr))
        results["_summary"] = {"AP_clean": results["clean"]["AP"], "mCP": mcp,
                               "retain_%": 100.0 * mcp / results["clean"]["AP"]}
        print(f"[robust] mCP={mcp:.4f}  giữ được {results['_summary']['retain_%']:.1f}% so với sạch")
    return results


def _build_val(VisDronePersonDataset, val_images, val_labels, img_size):
    sig = set(inspect.signature(VisDronePersonDataset.__init__).parameters.keys())
    pool = {"images_dir": val_images, "img_dir": val_images, "labels_dir": val_labels,
            "label_dir": val_labels, "img_size": img_size, "augment": False, "train": False}
    return VisDronePersonDataset(**{k: v for k, v in pool.items() if k in sig})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--profile", default="light")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if "cpu" in args.device or torch.cuda.is_available() else "cpu")
    try:
        from models.pfdet_nano_v15 import PFDetNanoV15
        model = PFDetNanoV15(profile=args.profile)
    except Exception:
        from models import build_model
        model = build_model("v15", profile=args.profile)
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt.get("ema", ckpt.get("model", ckpt.get("state_dict", ckpt))) if isinstance(ckpt, dict) else ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    from datasets import VisDronePersonDataset, collate_fn
    from utils import decode_predictions_np, nms_numpy
    val_set = _build_val(VisDronePersonDataset, args.val_images, args.val_labels, args.img_size)
    loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)

    evaluate_robustness(model, loader, device, args.img_size, model.strides,
                        decode_predictions_np, nms_numpy)


if __name__ == "__main__":
    main()
