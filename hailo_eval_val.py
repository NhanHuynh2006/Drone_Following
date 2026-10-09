"""
hailo_eval_val.py — CHẠY TRÊN PI 5. Đo ACCURACY của model HEF (INT8) trên VisDrone-val.
=======================================================================================

Vì sao cần: paper đang claim "186 FPS trên Hailo-8L" nhưng CHƯA có AP của chính model
chạy trên đó. Reviewer sẽ hỏi ngay, vì với detector tí hon thì INT8 PTQ từng làm AP
sập 0.18 -> 0.009 trong repo này. Con số FPS không kèm AP là con số vô nghĩa.

CÁCH DÙNG — 2 bước, để giữ ĐÚNG MỘT giao thức đo với cả bài báo:

  1) TRÊN PI (chỉ chạy model + dump dự đoán, KHÔNG tính AP ở đây):
       python hailo_eval_val.py --hef pfdet_v17_640_hailo8l.hef --img-size 640 \
           --val-images data/visdrone/val/images --out hef_preds_640.json
     (cần copy sẵn data/visdrone/val sang Pi, hoặc mount qua NFS/USB)

  2) COPY hef_preds_640.json về laptop rồi CHẤM ĐIỂM bằng chính code của paper:
       python hailo_eval_val.py --score hef_preds_640.json \
           --val-images data/visdrone/val/images --val-labels data/visdrone/val/labels

Bước 2 gọi `eval_coco.coco_eval` — CÙNG hàm mà mọi bảng trong paper dùng, nên số này
so trực tiếp được với FP32 trong UNIFIED.json. Đừng tự viết mAP riêng.

Ghi chú letterbox: script này letterbox y hệt loader của repo (đệm xám 114, giữ tỉ lệ),
và dump box ở toạ độ CHUẨN HOÁ trong khung vuông img_size — khớp với `coco_eval`.
"""

import argparse
import json
import os
import time

import numpy as np

STRIDES = (4, 8, 16, 32)


# ---------------------------------------------------------------- decode / nms
# DUNG THANG duong chuan cua repo (utils/box_ops.py, cai ma train_v3.py:evaluate goi):
#   decode_predictions_np(conf_thr, pre_nms_topk=500)  +  nms_numpy(iou_threshold=0.5) [DIoU]
# Chi khi CA HEF va FP32 dung cung 2 ham nay thi hieu AP moi la "do INT8", chu khong
# phai "do khac NMS". box_ops_np.py la ban chep numpy-only cua chinh utils/box_ops.py.
from box_ops_np import decode_predictions_np, nms_numpy


def decode_scale(raw, stride, img_size, conf_thr):
    return decode_predictions_np(raw, stride, img_size, conf_thr=conf_thr)


def nms(dets, iou_thr=0.5, max_det=300):
    return nms_numpy(dets, iou_threshold=iou_thr)[:max_det]


def letterbox(img, size):
    """Đệm xám 114 giữ tỉ lệ — GIỐNG datasets/visdrone_person.py."""
    import cv2
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


# ---------------------------------------------------------------- BƯỚC 1: trên Pi
def run_on_pi(args):
    import cv2
    from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                InputVStreamParams, OutputVStreamParams, InferVStreams,
                                FormatType)
    imgs = sorted(f for f in os.listdir(args.val_images)
                  if f.lower().endswith((".jpg", ".png", ".jpeg")))
    print(f"[hailo] {len(imgs)} ảnh | hef={args.hef} | size={args.img_size} "
          f"| conf={args.conf}")
    hef = HEF(args.hef)
    preds, lat = {}, []
    with VDevice() as dev:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = dev.configure(hef, cfg)[0]
        in_name = hef.get_input_vstream_infos()[0].name
        in_vp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_vp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        with InferVStreams(ng, in_vp, out_vp) as pipe, ng.activate():
            for i, name in enumerate(imgs):
                bgr = cv2.imread(os.path.join(args.val_images, name))
                if bgr is None:
                    continue
                canvas = letterbox(bgr, args.img_size)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                t0 = time.perf_counter()
                out = pipe.infer({in_name: rgb})
                lat.append((time.perf_counter() - t0) * 1000)
                outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2],
                              reverse=True)
                dets = []
                for si, o in enumerate(outs):
                    raw = np.transpose(o[0], (2, 0, 1))
                    dets += decode_scale(raw, STRIDES[si], args.img_size, args.conf)
                preds[name] = nms(dets, args.nms_iou, args.max_det)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(imgs)}  inference p50={np.median(lat):.2f}ms", flush=True)
    lat = np.array(lat)
    meta = {"hef": os.path.basename(args.hef), "img_size": args.img_size,
            "conf": args.conf, "nms_iou": args.nms_iou, "max_det": args.max_det,
            "n_images": len(preds),
            "infer_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                         "p95": float(np.percentile(lat, 95)),
                         "p99": float(np.percentile(lat, 99)),
                         "max": float(lat.max())}}
    json.dump({"meta": meta, "preds": preds}, open(args.out, "w"))
    print(f"\n[hailo] -> {args.out}")
    print(f"[hailo] inference only: p50={meta['infer_ms']['p50']:.2f}ms "
          f"p99={meta['infer_ms']['p99']:.2f}ms -> {1000/meta['infer_ms']['p50']:.0f} FPS (p50)")
    print("[hailo] Bước tiếp: copy file này về laptop rồi chạy --score để lấy AP.")


# ---------------------------------------------------------------- BƯỚC 2: chấm điểm
def score(args):
    from eval_coco import coco_eval          # CÙNG hàm mà cả paper dùng
    d = json.load(open(args.score))
    meta, preds = d["meta"], d["preds"]
    sz = meta["img_size"]
    import cv2
    pred_all, gt_all = [], []
    for name in sorted(preds):
        stem = os.path.splitext(name)[0]
        lp = os.path.join(args.val_labels, stem + ".txt")
        gt = []
        if os.path.exists(lp):
            for line in open(lp):
                p = line.split()
                if len(p) >= 5 and int(float(p[0])) == 0:
                    gt.append([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        gt = np.array(gt, np.float32) if gt else np.zeros((0, 4), np.float32)

        # GT trong file .txt chuan hoa theo ANH GOC (vd 1920x1080), con box du doan
        # chuan hoa theo CANVAS letterbox sz x sz. Anh khong vuong -> hai he nay LECH.
        # Phai dua GT qua dung phep letterbox da dung luc infer, neu khong AP bi dim oan.
        if len(gt):
            img = cv2.imread(os.path.join(args.val_images, name))
            if img is None:
                raise FileNotFoundError(
                    f"can anh goc de doi toa do GT: {os.path.join(args.val_images, name)}")
            h0, w0 = img.shape[:2]
            r = min(sz / h0, sz / w0)
            nh, nw = int(round(h0 * r)), int(round(w0 * r))
            top, left = (sz - nh) // 2, (sz - nw) // 2
            gt = np.stack([(gt[:, 0] * w0 * r + left) / sz,
                           (gt[:, 1] * h0 * r + top) / sz,
                           gt[:, 2] * w0 * r / sz,
                           gt[:, 3] * h0 * r / sz], axis=1).astype(np.float32)

        pred_all.append(preds[name])
        gt_all.append(gt)
    m = coco_eval(pred_all, gt_all, sz)
    print(f"\n=== HEF INT8: {meta['hef']} @ {sz} ({meta['n_images']} ảnh) ===")
    print({k: round(v, 4) for k, v in m.items()})
    print(f"latency inference: {meta['infer_ms']}")
    # ĐỐI CHỨNG FP32 phải là ĐÚNG checkpoint mà HEF này được build ra, chấm trên ĐÚNG
    # tập ảnh này. So với model khác trong UNIFIED.json thì hiệu số không còn là
    # "mất do quantise" nữa mà lẫn cả khác biệt giữa hai lần train.
    print("\nĐối chứng FP32 (chạy trên laptop, CÙNG checkpoint + CÙNG 548 ảnh + CÙNG decode/NMS):")
    print("  python run_eval_coco.py --weights <ckpt gốc của HEF này> \\")
    print("      --val-images data/visdrone/val/images --val-labels data/visdrone/val/labels \\")
    print("      --img-size %d --profile light --version v17 --conf 0.01" % sz)
    if args.fp32_ap:
        ref = float(args.fp32_ap)
        print(f"\n  FP32 AP={ref:.4f}  vs  INT8 AP={m['AP']:.4f}  "
              f"-> {100*(m['AP']-ref)/ref:+.1f}% do quantise")
    if args.fp32_ap50:
        ref = float(args.fp32_ap50)
        print(f"  FP32 AP50={ref:.4f}  vs  INT8 AP50={m['AP50']:.4f}  "
              f"-> {100*(m['AP50']-ref)/ref:+.1f}% do quantise")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="pfdet_v17_640_hailo8l.hef")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--val-images", default="data/visdrone/val/images")
    ap.add_argument("--val-labels", default="data/visdrone/val/labels")
    ap.add_argument("--conf", type=float, default=0.01, help="0.01 = giống cả paper")
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--out", default="hef_preds.json")
    ap.add_argument("--fp32-ap", default=None,
                    help="AP FP32 cua DUNG checkpoint nay, de in ra % rot do quantise")
    ap.add_argument("--fp32-ap50", default=None, help="AP50 FP32 tuong ung")
    ap.add_argument("--score", default=None,
                    help="chấm điểm file json đã dump (chạy trên laptop)")
    a = ap.parse_args()
    score(a) if a.score else run_on_pi(a)
