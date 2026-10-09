#!/usr/bin/env python3
"""
flight_eval_unified.py — do 6 dong cua bang "bay that" bang DUNG MOT code path.

Vi sao can file nay: cac so bay that trong PAPER_INPUTS.md duoc do rai rac o nhieu
thoi diem bang nhieu lenh khac nhau. Bang trong paper KHONG duoc tron nhu vay
(quy tac "mot giao thuc duy nhat"). File nay dung lai chinh predictor + coco_eval
cua paper_eval_unified.py, chi doi tap val sang footage bay.

Khong co rung nhan tao o day: nhoe la do may bay tao ra that. severity=0.

    python flight_eval_unified.py --out runs/paper_queue/FLIGHT.json
"""
import argparse
import json

import numpy as np
import torch

from eval_coco import coco_eval
from paper_eval_unified import PFDetPredictor, build_val_loader, gt_boxes_of

# (ten trong bang, checkpoint, version, nhom du lieu train, xu ly)
MODELS = [
    ("deploy-baseline",  "runs/train_v17_deploy/best.pt",           "v17", "VisDrone + COCO person", "none"),
    ("deploy-augonly",   "runs/train_v17_augonly_deploy/best.pt",   "v17", "VisDrone + COCO person", "vibration-aware"),
    ("deploy-vibcons",   "runs/train_v17_vib_deploy/best.pt",       "v17", "VisDrone + COCO person", "consistency"),
    ("light-baseline",   "runs/train_v17_light/best.pt",            "v17", "VisDrone only", "none"),
    ("light-augonly",    "runs/train_v17_vib_augonly/best.pt",      "v17", "VisDrone only", "vibration-aware"),
    ("light-vibcons",    "runs/train_v17_vib/best.pt",              "v17", "VisDrone only", "consistency"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", default="data/flight_val/images")
    ap.add_argument("--val-labels", default="data/flight_val/labels")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--profile", default="light")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000, help="so lan bootstrap tren frame (0=tat)")
    ap.add_argument("--out", default="runs/paper_queue/FLIGHT.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ds, loader = build_val_loader(args.val_images, args.val_labels,
                                  args.img_size, args.batch, args.workers)
    print(f"[data] {len(ds)} frame bay")

    out = {"val_images": args.val_images, "img_size": args.img_size,
           "n_boot": args.boot, "models": {}}
    preds = {}   # ten -> list per-frame detections, giu lai de bootstrap
    gts = None
    for name, ckpt, ver, data, treat in MODELS:
        pred = PFDetPredictor(ckpt, ver, args.profile, device, args.img_size)
        pred_all, gt_all = [], []
        with torch.no_grad():
            for imgs, labels_list, _ in loader:
                imgs = imgs.to(device, non_blocking=True)
                dets_batch = pred(imgs)
                for b in range(imgs.shape[0]):
                    pred_all.append(dets_batch[b])
                    gt_all.append(gt_boxes_of(labels_list[b]))
        gts = gt_all
        preds[name] = pred_all
        m = coco_eval(pred_all, gt_all, args.img_size)
        out["models"][name] = {"ckpt": ckpt, "train_data": data, "treatment": treat,
                               **{k: round(float(v), 4) for k, v in m.items()}}
        print(f"{name:18s} {data:24s} {treat:16s} "
              f"AP50={m['AP50']:.4f} AP={m['AP']:.4f} AP75={m['AP75']:.4f} AR100={m['AR_100']:.4f}")
        del pred
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------------
    # Bootstrap TREN FRAME. 92 frame la it; khong co khoang tin cay thi khong
    # the noi chenh lech giua hai xu ly la that hay la nhieu lay mau.
    # Lay mau lai CUNG bo frame cho moi model -> so sanh ghep cap (paired).
    # ---------------------------------------------------------------------
    if args.boot > 0:
        n = len(gts)
        rng = np.random.default_rng(0)
        names = list(preds)
        boot = {k: {"AP50": [], "AP": [], "AP75": []} for k in names}
        print(f"\n[boot] {args.boot} lan lay mau lai tren {n} frame (paired)...")
        for it in range(args.boot):
            idx = rng.integers(0, n, size=n)
            g = [gts[i] for i in idx]
            for k in names:
                p = [preds[k][i] for i in idx]
                m = coco_eval(p, g, args.img_size)
                for key in boot[k]:
                    boot[k][key].append(m[key])
        for k in names:
            out["models"][k]["boot_ci95"] = {
                key: [round(float(np.percentile(v, 2.5)), 4),
                      round(float(np.percentile(v, 97.5)), 4)]
                for key, v in boot[k].items()}
        # chenh lech ghep cap giua cac xu ly trong cung nhom du lieu
        out["paired_diff_ci95"] = {}
        for a, b in [("deploy-augonly", "deploy-baseline"),
                     ("deploy-vibcons", "deploy-baseline"),
                     ("deploy-vibcons", "deploy-augonly")]:
            d = {}
            for key in ("AP50", "AP", "AP75"):
                v = np.array(boot[a][key]) - np.array(boot[b][key])
                d[key] = {"mean": round(float(v.mean()), 4),
                          "ci95": [round(float(np.percentile(v, 2.5)), 4),
                                   round(float(np.percentile(v, 97.5)), 4)],
                          "p_gt0": round(float((v > 0).mean()), 3)}
            out["paired_diff_ci95"][f"{a} - {b}"] = d
            print(f"  {a} - {b}")
            for key, x in d.items():
                star = "  <- khoang KHONG chua 0" if x["ci95"][0] > 0 or x["ci95"][1] < 0 else ""
                print(f"     {key:5s} {x['mean']:+.4f}  CI95 [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]{star}")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
