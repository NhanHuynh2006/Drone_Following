"""
FP32 tren laptop, CUNG 92 frame flight_val, CUNG decode/nms nhu duong HEF.
Muc dich: tach bach "mat do INT8" khoi "model von da yeu tren domain nay".
"""
import argparse, glob, os, sys, time
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flight_record import STRIDES, decode_scale, nms, letterbox
import onnxruntime as ort

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", default="pfdet_v17_hailo_coco_640.onnx")
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--dir", default="data/flight_val")
ap.add_argument("--floor", type=float, default=0.02)
ap.add_argument("--device", default="cuda:0")
a = ap.parse_args()

def iou(b, g):
    xx1, yy1 = max(b[0], g[0]), max(b[1], g[1])
    xx2, yy2 = min(b[2], g[2]), min(b[3], g[3])
    inter = max(0, xx2-xx1) * max(0, yy2-yy1)
    return inter / ((b[2]-b[0])*(b[3]-b[1]) + (g[2]-g[0])*(g[3]-g[1]) - inter + 1e-9)

sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
print(f"[onnx] {a.onnx}")

files = sorted(glob.glob(os.path.join(a.dir, "images", "*.jpg")))
per_img = []
if True:
    for f in files:
        img = cv2.imread(f); H0, W0 = img.shape[:2]
        canvas, r, top, left = letterbox(img, a.img_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb.transpose(2, 0, 1)[None]
        out = sess.run(None, {in_name: x})
        outs = sorted([o[0] for o in out], key=lambda o: o.shape[-1]*o.shape[-2], reverse=True)
        dets = []
        for si, o in enumerate(outs):
            dets += decode_scale(o, STRIDES[si], a.img_size, a.floor)
        dets = nms(dets, 0.5, max_det=300)
        preds = []
        for d in dets:
            x1 = (d["box"][0]*a.img_size - left)/r; y1 = (d["box"][1]*a.img_size - top)/r
            x2 = (d["box"][2]*a.img_size - left)/r; y2 = (d["box"][3]*a.img_size - top)/r
            preds.append([x1, y1, x2, y2, d["score"]])
        gt = []
        lp = os.path.join(a.dir, "labels", os.path.basename(f).rsplit(".",1)[0] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp):
                p = ln.split()
                if len(p) < 5: continue
                cx, cy, w, h = [float(v) for v in p[1:5]]
                gt.append([(cx-w/2)*W0, (cy-h/2)*H0, (cx+w/2)*W0, (cy+h/2)*H0])
        per_img.append((preds, gt))

n_gt = sum(len(g) for _, g in per_img)
print(f"\nFP32 | {len(files)} anh | {n_gt} nguoi that")
print(f"\n{'conf':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'recall':>8} {'prec':>8} {'F1':>8} {'FP/frame':>9}")
best = None
for c in (0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60):
    TP = FP = 0
    for preds, gt in per_img:
        p = sorted([x for x in preds if x[4] >= c], key=lambda x: -x[4])
        used = [False]*len(gt)
        for b in p:
            hit, bi = -1, 0.5
            for i, g in enumerate(gt):
                if used[i]: continue
                v = iou(b, g)
                if v >= bi: bi, hit = v, i
            if hit >= 0: used[hit] = True; TP += 1
            else: FP += 1
    FN = n_gt - TP
    rec, pre = TP/max(1,n_gt), TP/max(1,TP+FP)
    f1 = 2*rec*pre/max(1e-9, rec+pre)
    if best is None or f1 > best[1]: best = (c, f1)
    print(f"{c:>6.2f} {TP:>5} {FP:>5} {FN:>5} {rec:>8.3f} {pre:>8.3f} {f1:>8.3f} {FP/len(files):>9.2f}")
print(f"\n  -> conf tot nhat theo F1: {best[0]:.2f} (F1={best[1]:.3f})")
