"""
test_accuracy_onnx.py — Test độ chính xác model ONNX trên VisDrone val (1 FILE, chạy ở đâu cũng được)
=====================================================================================================

Nạp 1 file .onnx (FP32 hoặc INT8) -> chạy hết val 548 ảnh -> in COCO AP (AP@.5:.95, AP50, AP75, AP_s,
breakdown theo mật độ). Dùng để kiểm chứng model deploy chính xác bằng PyTorch gốc.

Chạy trên MÁY NÀY:
  python test_accuracy_onnx.py --onnx deploy/pfdet_E_320.onnx --img-size 320 \
      --val-images data/visdrone/val/images --val-labels data/visdrone/val/labels

Copy sang Pi 5 (cần: onnxruntime, numpy, opencv, + file utils/eval_coco/datasets hoặc chỉ cần
decode_predictions_np + nms_numpy + coco_eval). Cùng lệnh, đổi đường dẫn.

So sánh: cùng img-size, AP của ONNX FP32 phải ≈ PyTorch; INT8 thường tụt nhẹ 1-3% AP (đánh đổi tốc độ).
"""

import os
import sys
import time
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import onnxruntime as ort
from utils import decode_predictions_np, nms_numpy
from eval_coco import coco_eval
from datasets import VisDronePersonDataset

STRIDES = [4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, help="file .onnx (FP32 hoặc INT8)")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--val-images", default="data/visdrone/val/images")
    ap.add_argument("--val-labels", default="data/visdrone/val/labels")
    ap.add_argument("--conf-thr", type=float, default=0.01)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--threads", type=int, default=0, help="0=mặc định ORT (Pi nên đặt =4)")
    args = ap.parse_args()

    so = ort.SessionOptions()
    if args.threads > 0:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, sess_options=so,
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    print(f"[onnx] {args.onnx}")
    print(f"  input={in_name}  outputs={out_names}  img_size={args.img_size}")

    ds = VisDronePersonDataset(args.val_images, args.val_labels,
                               img_size=args.img_size, augment=False)
    print(f"[data] {len(ds)} ảnh val @ {args.img_size}px")

    pred_all, gt_all, ngt_all = [], [], []
    t_infer = 0.0
    for i in range(len(ds)):
        img, labels, _ = ds[i]                       # img: (3,H,W) float 0-1
        x = img.unsqueeze(0).numpy().astype(np.float32)
        t0 = time.perf_counter()
        outs = sess.run(out_names, {in_name: x})     # list 4 raw head tensors
        t_infer += time.perf_counter() - t0

        dets = []
        for si, raw in enumerate(outs):
            dets.extend(decode_predictions_np(raw[0], STRIDES[si], args.img_size, conf_thr=args.conf_thr))
        dets = nms_numpy(dets, iou_threshold=args.nms_iou)[:args.max_det]

        gt = labels
        gt_p = gt[gt[:, 0] == 0] if len(gt) > 0 else gt
        gt_boxes = gt_p[:, 1:5].numpy() if len(gt_p) > 0 else np.zeros((0, 4))
        pred_all.append(dets); gt_all.append(gt_boxes); ngt_all.append(len(gt_boxes))

        if (i + 1) % 100 == 0:
            print(f"  ...{i+1}/{len(ds)}")

    fps = len(ds) / t_infer
    print(f"\n[speed] {1000*t_infer/len(ds):.1f} ms/ảnh  |  {fps:.1f} FPS (chỉ inference, CPU x86 máy này)")

    m = coco_eval(pred_all, gt_all, args.img_size, verbose=False)
    print("\n[COCO] overall:", {k: round(v, 4) for k, v in m.items()})
    ngt = np.array(ngt_all)
    for lo, hi in [(0, 10), (10, 30), (30, 1e9)]:
        idx = np.where((ngt >= lo) & (ngt < hi))[0]
        tag = f"{lo}-{int(hi) if hi < 1e8 else 'inf'} người (n={len(idx)})"
        if len(idx) == 0:
            print(f"[COCO] {tag}: (rỗng)"); continue
        mm = coco_eval([pred_all[i] for i in idx], [gt_all[i] for i in idx], args.img_size, verbose=False)
        print(f"[COCO] {tag}: AP50={mm['AP50']:.3f} AP={mm['AP']:.3f} AP_s={mm['AP_s']:.3f} AR_s={mm['AR_s']:.3f}")


if __name__ == "__main__":
    main()
