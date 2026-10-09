"""
fp32_dump.py — Dump du doan FP32 ra ĐÚNG định dạng của hailo_eval_val.py.

Mục đích: để chấm FP32 và INT8 bằng **CÙNG MỘT** hàm `score()`, cùng một phép
letterbox GT, cùng decode/NMS. PROJECT.md mục 8 đã ghi rõ: trộn hai harness đo
khác nhau cho chênh lệch cỡ bằng chênh lệch GIỮA CÁC MODEL, tức là báo cáo
"khác biệt do code chấm điểm" thành "khác biệt do lượng tử hoá".

    python fp32_dump.py --weights runs/train_v17_hailo_coco/best.pt --out fp32_preds.json
    python hailo_eval_val.py --score fp32_preds.json --val-labels data/visdrone/val/labels
"""
import argparse, glob, json, os, time
import numpy as np, cv2, torch
from models import build_model_from_checkpoint
from flight_record import STRIDES, decode_scale, nms, letterbox

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--img-size", type=int, default=640)
ap.add_argument("--val-images", default="data/visdrone/val/images")
ap.add_argument("--conf", type=float, default=0.01)
ap.add_argument("--nms-iou", type=float, default=0.5)
ap.add_argument("--max-det", type=int, default=300)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="fp32_preds.json")
a = ap.parse_args()

model, _, ver, mcfg = build_model_from_checkpoint(a.weights, device=a.device, use_ema=True)
print(f"[fp32] {a.weights} | {ver} | cfg={mcfg}")

imgs = sorted(glob.glob(os.path.join(a.val_images, "*.jpg")))
preds, lat = {}, []
with torch.no_grad():
    for i, f in enumerate(imgs):
        img = cv2.imread(f)
        canvas, _, _, _ = letterbox(img, a.img_size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(a.device)
        t0 = time.perf_counter()
        out = model(x)
        lat.append((time.perf_counter() - t0) * 1e3)
        outs = sorted([o.detach().float().cpu().numpy()[0] for o in out],
                      key=lambda o: o.shape[-1] * o.shape[-2], reverse=True)
        dets = []
        for si, o in enumerate(outs):
            dets += decode_scale(o, STRIDES[si], a.img_size, a.conf)
        preds[os.path.basename(f)] = nms(dets, a.nms_iou, a.max_det)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(imgs)}", flush=True)

lat = np.array(lat)
meta = {"hef": f"FP32:{os.path.basename(a.weights)}", "img_size": a.img_size,
        "conf": a.conf, "nms_iou": a.nms_iou, "max_det": a.max_det,
        "n_images": len(preds),
        "infer_ms": {"mean": float(lat.mean()), "p50": float(np.percentile(lat, 50)),
                     "p95": float(np.percentile(lat, 95)),
                     "p99": float(np.percentile(lat, 99)), "max": float(lat.max())}}
json.dump({"meta": meta, "preds": preds}, open(a.out, "w"))
print(f"[fp32] -> {a.out}  ({len(preds)} anh)")
