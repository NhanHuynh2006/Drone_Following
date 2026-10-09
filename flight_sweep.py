"""
Quet nguong conf tren data/flight_val (92 frame BAY THAT ~6m, co nhan tay).
Tra loi cau hoi deploy: dat conf bao nhieu thi vua bat duoc nguoi vua khong nhieu?

Chay 1 lan inference cho moi anh (conf san 0.02), roi quet nguong OFFLINE.
"""
import argparse, glob, os, time
import cv2, numpy as np
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams,
                            HailoStreamInterface, InputVStreamParams,
                            OutputVStreamParams, FormatType)
from flight_record import STRIDES, decode_scale, nms, letterbox

ap = argparse.ArgumentParser()
ap.add_argument("--hef", default="models/pfdet_v17_640_hailo8l.hef")
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--dir", default="data/flight_val")
ap.add_argument("--floor", type=float, default=0.02, help="conf san khi decode")
ap.add_argument("--min-area-px", type=float, default=0.0, help="loc box nho hon (px^2 tren anh goc)")
ap.add_argument("--max-aspect", type=float, default=0.0, help="loc box qua det/qua bet (0=tat)")
a = ap.parse_args()

def iou(b, g):
    xx1, yy1 = max(b[0], g[0]), max(b[1], g[1])
    xx2, yy2 = min(b[2], g[2]), min(b[3], g[3])
    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
    ab = (b[2]-b[0])*(b[3]-b[1]); ag = (g[2]-g[0])*(g[3]-g[1])
    return inter / (ab + ag - inter + 1e-9)

files = sorted(glob.glob(os.path.join(a.dir, "images", "*.jpg")))
hef = HEF(a.hef)
per_img = []          # (list preds [x1,y1,x2,y2,score] px goc, list gt px goc)
lat = []
with VDevice() as dev:
    cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    ng = dev.configure(hef, cfg)[0]
    in_name = hef.get_input_vstream_infos()[0].name
    with InferVStreams(ng, InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                       OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)) as pipe, \
         ng.activate():
        for f in files:
            img = cv2.imread(f); H0, W0 = img.shape[:2]
            canvas, r, top, left = letterbox(img, a.img_size)
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
            t0 = time.perf_counter()
            out = pipe.infer({in_name: rgb})
            lat.append((time.perf_counter() - t0) * 1e3)
            outs = sorted(out.values(), key=lambda x: x.shape[1]*x.shape[2], reverse=True)
            dets = []
            for si, o in enumerate(outs):
                dets += decode_scale(np.transpose(o[0], (2,0,1)), STRIDES[si], a.img_size, a.floor)
            dets = nms(dets, 0.5, max_det=300)
            preds = []
            for d in dets:                                  # canvas-norm -> pixel anh goc
                x1 = (d["box"][0]*a.img_size - left)/r; y1 = (d["box"][1]*a.img_size - top)/r
                x2 = (d["box"][2]*a.img_size - left)/r; y2 = (d["box"][3]*a.img_size - top)/r
                w, h = x2-x1, y2-y1
                if a.min_area_px and w*h < a.min_area_px: continue
                if a.max_aspect and (max(w/(h+1e-9), h/(w+1e-9)) > a.max_aspect): continue
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
lat.sort()
print(f"\n{a.hef} | {len(files)} anh | {n_gt} nguoi that | inference p50 {lat[len(lat)//2]:.2f} ms")
if a.min_area_px or a.max_aspect:
    print(f"  loc: min_area={a.min_area_px}px^2  max_aspect={a.max_aspect}")
print(f"\n{'conf':>6} {'TP':>5} {'FP':>5} {'FN':>5} {'recall':>8} {'prec':>8} {'F1':>8} {'FP/frame':>9}")
best = None
for c in (0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60):
    TP = FP = 0
    for preds, gt in per_img:
        p = sorted([x for x in preds if x[4] >= c], key=lambda x: -x[4])
        used = [False]*len(gt)
        for b in p:
            hit = -1; bi = 0.5
            for i, g in enumerate(gt):
                if used[i]: continue
                v = iou(b, g)
                if v >= bi: bi, hit = v, i
            if hit >= 0: used[hit] = True; TP += 1
            else: FP += 1
    FN = n_gt - TP
    rec = TP/max(1, n_gt); pre = TP/max(1, TP+FP)
    f1 = 2*rec*pre/max(1e-9, rec+pre)
    if best is None or f1 > best[1]: best = (c, f1)
    print(f"{c:>6.2f} {TP:>5} {FP:>5} {FN:>5} {rec:>8.3f} {pre:>8.3f} {f1:>8.3f} {FP/len(files):>9.2f}")
print(f"\n  -> conf tot nhat theo F1: {best[0]:.2f} (F1={best[1]:.3f})")
