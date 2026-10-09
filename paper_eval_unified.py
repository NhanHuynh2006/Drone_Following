"""
paper_eval_unified.py — MỘT thước đo duy nhất cho MỌI model của paper
=====================================================================

Vá 2 lỗ hổng phương pháp mà reviewer sẽ bắt ngay:

  (1) **Giao thức lệch.** Số pfdet đo bằng `pycocotools` (eval_coco.coco_eval), số YOLO đo bằng
      `ultralytics.val()` → khác conf/max_det/định nghĩa area → KHÔNG so trực tiếp được.
      Script này chạy MỌI model (pfdet + ultralytics) trên CÙNG loader, CÙNG letterbox 640,
      CÙNG conf/NMS/max_det, rồi tính AP bằng CÙNG `coco_eval`.

  (2) **Eval rung không seed.** `corrupt_batch()` random hướng blur / nhiễu / độ sáng, nên mỗi lần
      chạy `eval_robustness.py` ra retain% khác nhau (đo được 17.0/17.1 cho baseline, 30.1/31.4/31.7
      cho vibration). Số 1 lần đo là 1 lần rút thăm, không phải kết quả. Ở đây mỗi severity được
      seed lại → tái lập được, và chạy nhiều seed → báo mean ± std.

Chạy (mặc định làm hết, ~1-2h trên RTX 4060):

    python paper_eval_unified.py --seeds 0 1 2 --out runs/paper_queue/UNIFIED.json

Chỉ 1 nhóm:
    python paper_eval_unified.py --only pfdet
    python paper_eval_unified.py --only yolo
"""

import argparse
import inspect
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval_coco import coco_eval
from vib_consistency import _PRESETS, corrupt_batch


def corruption_fingerprint():
    """Van tay cua tham so rung. Doi _PRESETS -> doi van tay -> moi so retention cu
    deu VO HIEU, vi retention phu thuoc truc tiep vao muc do rung."""
    return {str(k): list(v) for k, v in sorted(_PRESETS.items())}

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- model zoo
# (tên trong paper, đường dẫn ckpt, họ, checkpoint dùng best hay last)
PFDET_MODELS = [
    ("v17-baseline",     "runs/train_v17_light/best.pt",           "v17"),
    ("v17-vibcons",      "runs/train_v17_vib/best.pt",             "v17"),
    ("v17-nmsfree",      "runs/train_v17_nms_free/best.pt",        "v17"),
    # --- 2 ĐỐI CHỨNG (cùng schedule/lr/epoch/aug với train_config_v17_vib.yaml) ---
    # augonly: rung chỉ làm augmentation, KHÔNG consistency -> tách "thấy ảnh rung" khỏi "consistency"
    # ft_control: finetune 120 epoch KHÔNG rung gì -> tách tác dụng của "train thêm"
    ("v17-vib-augonly",  "runs/train_v17_vib_augonly/best.pt",     "v17"),
    ("v17-ft-control",   "runs/train_v17_ft_control/best.pt",      "v17"),
    # --- DOI KHUNG 2026-07-26: augment-only THANG consistency (36.2 vs 30.0 retain).
    # Phuong phap chinh cua bai gio la AUGMENTATION -> generality phai do lai cho NO.
    ("v15-augonly",      "runs/train_v15_augonly/best.pt",         "v15"),
    ("v17-augonly-deploy", "runs/train_v17_augonly_deploy/best.pt", "v17"),
    ("v15-baseline",     "runs/train_v15_light_musgd_v3/best.pt",  "v15"),
    # v15_vib_E: đo CẢ HAI checkpoint. Paper phải dùng CÙNG quy tắc chọn checkpoint với baseline
    # (= best theo clean val AP), không được lấy last.pt cho model của mình và best.pt cho baseline.
    ("v15-vibcons",      "runs/train_v15_vib_E/best.pt",           "v15"),
    ("v15-vibcons-last", "runs/train_v15_vib_E/last.pt",           "v15"),
    # nhánh deployment (+COCO): đo trên CHÍNH VisDrone-val để chứng minh thứ tự xếp hạng
    # trên benchmark ĐẢO CHIỀU so với trên footage bay thật (Section flight).
    ("v17-deploy",       "runs/train_v17_deploy/best.pt",          "v17"),
    ("v17-vib-deploy",   "runs/train_v17_vib_deploy/best.pt",      "v17"),
]

YOLO_MODELS = [
    ("yolov8n",  "runs/yolo/yolov8n/weights/best.pt"),
    ("yolo11n",  "runs/yolo/yolo11n/weights/best.pt"),
    ("yolo26n",  "runs/yolo/yolo26n/weights/best.pt"),
    ("yolov10n", "runs/yolo/yolov10n/weights/best.pt"),
    ("rtdetr-l", "runs/yolo/rtdetr-l/weights/best.pt"),
    # GENERALITY ngoài họ của mình: vib-consistency áp lên YOLO11n (train_yolo_vibcons.py).
    # So với dòng "yolo11n" ở trên -> chứng minh kỹ thuật không bó buộc vào detector của tác giả.
    ("yolo11n-vibcons", "runs/yolo/yolo11n_vibcons/weights/best.pt"),
    ("yolo11n-augonly", "runs/yolo/yolo11n_augonly/weights/best.pt"),
    # 2026-07-27: YOLO11n FROM SCRATCH (khong COCO-pretrain) + augmentation.
    # Tra loi: vi sao xu ly khong co tac dung tren yolo11n? Do COCO-pretrain da co san
    # dac trung khang nhoe, hay do kien truc? So cap scratch-baseline vs scratch-aug.
    ("yolo11n-scratch",     "runs/yolo/yolo11n_scratch/weights/best.pt"),
    ("yolo11n-scratch-aug", "runs/yolo/yolo11n_scratch_aug/weights/best.pt"),
    ("yolo11n-scratch-ft",  "runs/yolo/yolo11n_scratch_ft/weights/best.pt"),
]


# ---------------------------------------------------------------- data
def build_val_loader(val_images, val_labels, img_size, batch, workers):
    from datasets import VisDronePersonDataset, collate_fn
    sig = set(inspect.signature(VisDronePersonDataset.__init__).parameters.keys())
    pool = {"images_dir": val_images, "img_dir": val_images,
            "labels_dir": val_labels, "label_dir": val_labels,
            "img_size": img_size, "augment": False, "train": False}
    ds = VisDronePersonDataset(**{k: v for k, v in pool.items() if k in sig})
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=collate_fn, pin_memory=True)
    return ds, loader


def gt_boxes_of(labels_tensor):
    """(N,5)[cls,cx,cy,w,h] chuẩn hoá -> (N,4) cxcywh chuẩn hoá, chỉ class person."""
    if len(labels_tensor) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    g = labels_tensor[labels_tensor[:, 0] == 0]
    return g[:, 1:5].cpu().numpy() if len(g) > 0 else np.zeros((0, 4), dtype=np.float32)


# ---------------------------------------------------------------- predictors
class PFDetPredictor:
    """pfdet: decode numpy + nms_numpy của repo (y hệt eval_coco/eval_robustness)."""

    def __init__(self, ckpt_path, version, profile, device, img_size,
                 conf_thr=0.01, nms_iou=0.5, max_det=300):
        from models import build_model, normalize_model_version
        from utils import decode_predictions_np, nms_numpy
        self.decode_fn, self.nms_fn = decode_predictions_np, nms_numpy
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ver = version or (ckpt.get("model_version") if isinstance(ckpt, dict) else None) or "v15"
        model = build_model(normalize_model_version(ver), profile=profile)
        state = ckpt.get("ema", ckpt.get("model", ckpt.get("state_dict", ckpt))) \
            if isinstance(ckpt, dict) else ckpt
        if hasattr(state, "state_dict"):
            state = state.state_dict()
        model.load_state_dict(state, strict=False)
        self.model = model.to(device).eval()
        self.strides = model.strides
        self.device, self.img_size = device, img_size
        self.conf_thr, self.nms_iou, self.max_det = conf_thr, nms_iou, max_det
        self.params = sum(p.numel() for p in model.parameters())

    @torch.no_grad()
    def __call__(self, imgs):
        preds = self.model(imgs)
        out = []
        for b in range(imgs.shape[0]):
            dets = []
            for si, p in enumerate(preds):
                dets.extend(self.decode_fn(p[b].cpu().numpy(), self.strides[si],
                                           self.img_size, conf_thr=self.conf_thr))
            out.append(self.nms_fn(dets, iou_threshold=self.nms_iou)[:self.max_det])
        return out


class UltralyticsPredictor:
    """
    ultralytics: nhận THẲNG tensor (B,3,640,640) RGB 0-1 đã letterbox bởi loader của repo,
    nên không có bước tiền xử lý riêng nào -> đúng cùng ảnh đầu vào với pfdet.
    Trả box về dạng xyxy chuẩn hoá 0-1 + score, giống output decode của pfdet.
    """

    def __init__(self, ckpt_path, device, img_size, conf_thr=0.01, nms_iou=0.5, max_det=300):
        from ultralytics import YOLO
        self.m = YOLO(ckpt_path)
        self.m.to(device)
        self.device, self.img_size = device, img_size
        self.conf_thr, self.nms_iou, self.max_det = conf_thr, nms_iou, max_det
        self.params = sum(p.numel() for p in self.m.model.parameters())

    @torch.no_grad()
    def __call__(self, imgs):
        res = self.m.predict(imgs, conf=self.conf_thr, iou=self.nms_iou,
                             max_det=self.max_det, verbose=False, device=self.device)
        S = float(self.img_size)
        out = []
        for r in res:
            dets = []
            b = r.boxes
            if b is not None and len(b) > 0:
                xyxy = b.xyxy.cpu().numpy() / S           # -> chuẩn hoá 0-1
                conf = b.conf.cpu().numpy()
                for box, sc in zip(xyxy, conf):
                    dets.append({"box": [float(v) for v in box], "score": float(sc)})
            out.append(dets)
        return out


# ---------------------------------------------------------------- eval core
@torch.no_grad()
def eval_one_pass(predictor, loader, device, img_size, severity, seed):
    """1 lượt quét val ở 1 mức severity với seed CỐ ĐỊNH -> dict metric COCO."""
    if severity > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    pred_all, gt_all = [], []
    for imgs, labels_list, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        if severity > 0:
            imgs = corrupt_batch(imgs, severity=severity)
            imgs = imgs.clamp(0.0, 1.0)      # rung xong vẫn phải là ảnh hợp lệ
        dets_batch = predictor(imgs)
        for b in range(imgs.shape[0]):
            pred_all.append(dets_batch[b])
            gt_all.append(gt_boxes_of(labels_list[b]))
    return coco_eval(pred_all, gt_all, img_size)


def eval_model(name, predictor, loader, device, img_size, seeds, severities=(1, 2, 3)):
    """clean (1 lần, không random) + mỗi severity × mỗi seed. Trả mean/std của retain%."""
    t0 = time.time()
    rec = {"name": name, "params_M": round(predictor.params / 1e6, 3), "seeds": list(seeds)}

    clean = eval_one_pass(predictor, loader, device, img_size, severity=0, seed=0)
    rec["clean"] = {k: round(v, 4) for k, v in clean.items()}
    print(f"  [clean] AP={clean['AP']:.4f} AP50={clean['AP50']:.4f} "
          f"AP75={clean['AP75']:.4f} AP_s={clean['AP_s']:.4f}", flush=True)

    per_seed = {}
    for sd in seeds:
        sev_ap = {}
        for sev in severities:
            m = eval_one_pass(predictor, loader, device, img_size, severity=sev, seed=sd)
            sev_ap[f"sev{sev}"] = {k: round(v, 4) for k, v in m.items()}
            print(f"  [seed {sd}] sev{sev}: AP={m['AP']:.4f} AP50={m['AP50']:.4f}", flush=True)
        mcp = float(np.mean([sev_ap[f"sev{s}"]["AP"] for s in severities]))
        retain = 100.0 * mcp / clean["AP"] if clean["AP"] > 0 else 0.0
        sev_ap["mCP"] = round(mcp, 4)
        sev_ap["retain_pct"] = round(retain, 2)
        per_seed[str(sd)] = sev_ap
        print(f"  [seed {sd}] mCP={mcp:.4f} retain={retain:.2f}%", flush=True)

    rec["per_seed"] = per_seed
    # tổng hợp qua seed: mean ± std cho từng severity + retain
    agg = {}
    for sev in severities:
        vals = [per_seed[str(sd)][f"sev{sev}"]["AP"] for sd in seeds]
        agg[f"sev{sev}"] = {"mean": round(float(np.mean(vals)), 4),
                            "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 4)}
    rt = [per_seed[str(sd)]["retain_pct"] for sd in seeds]
    mc = [per_seed[str(sd)]["mCP"] for sd in seeds]
    agg["mCP"] = {"mean": round(float(np.mean(mc)), 4),
                  "std": round(float(np.std(mc, ddof=1)) if len(mc) > 1 else 0.0, 4)}
    agg["retain_pct"] = {"mean": round(float(np.mean(rt)), 2),
                         "std": round(float(np.std(rt, ddof=1)) if len(rt) > 1 else 0.0, 2)}
    rec["agg"] = agg
    rec["minutes"] = round((time.time() - t0) / 60, 1)
    print(f"  >>> {name}: clean AP={clean['AP']:.4f} | retain "
          f"{agg['retain_pct']['mean']:.1f} ± {agg['retain_pct']['std']:.1f}% "
          f"({rec['minutes']} phút)", flush=True)
    return rec


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-images", default="data/visdrone/val/images")
    ap.add_argument("--val-labels", default="data/visdrone/val/labels")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--profile", default="light")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--only", choices=["pfdet", "yolo", "all"], default="all")
    ap.add_argument("--out", default="runs/paper_queue/UNIFIED.json")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    ds, loader = build_val_loader(args.val_images, args.val_labels,
                                 args.img_size, args.batch, args.workers)
    print(f"[data] {len(ds)} ảnh val @{args.img_size} | conf={args.conf} "
          f"nms_iou={args.nms_iou} max_det={args.max_det} | seeds={args.seeds}", flush=True)

    out_path = os.path.join(HERE, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    proto = {"img_size": args.img_size, "conf": args.conf,
             "nms_iou": args.nms_iou, "max_det": args.max_det,
             "n_images": len(ds), "metric": "pycocotools bbox",
             "seeds": args.seeds, "corruption": corruption_fingerprint()}
    results = {"protocol": proto, "models": {}}
    if os.path.exists(out_path):                       # resume: giữ model đã đo
        try:
            old = json.load(open(out_path))
            old.setdefault("models", {})
            old_c = old.get("protocol", {}).get("corruption")
            if old_c is not None and old_c != proto["corruption"]:
                # BAY: _PRESETS da doi (vd sau khi hieu chuan theo phổ IMU do that).
                # Resume luc nay se tron so do bang 2 bo tham so rung khac nhau trong
                # CUNG mot bang -> dung cai loi ma script nay sinh ra de chong.
                print("\n" + "!" * 78)
                print("!! THAM SO RUNG (_PRESETS trong vib_consistency.py) DA DOI so voi lan do truoc.")
                print(f"!!   cu:  {old_c}")
                print(f"!!   moi: {proto['corruption']}")
                print("!! Moi so retention trong file cu deu VO HIEU. KHONG duoc resume.")
                print(f"!! Hay doi ten {out_path} thanh <ten>_old.json roi chay lai TU DAU.")
                print("!" * 78 + "\n")
                raise SystemExit(2)
            results = old
            results["protocol"] = proto
            print(f"[resume] đã có {len(results['models'])} model trong {out_path}", flush=True)
        except SystemExit:
            raise
        except Exception:
            pass

    jobs = []
    if args.only in ("pfdet", "all"):
        jobs += [("pfdet", n, p, v) for n, p, v in PFDET_MODELS]
    if args.only in ("yolo", "all"):
        jobs += [("yolo", n, p, None) for n, p in YOLO_MODELS]

    for kind, name, rel, ver in jobs:
        path = os.path.join(HERE, rel)
        if name in results["models"]:
            print(f"\n=== {name}: đã có, bỏ qua ===", flush=True)
            continue
        if not os.path.exists(path):
            print(f"\n=== {name}: THIẾU {rel} -> bỏ qua ===", flush=True)
            continue
        print(f"\n=== {name} ({kind}) {rel} ===", flush=True)
        try:
            if kind == "pfdet":
                pred = PFDetPredictor(path, ver, args.profile, device, args.img_size,
                                      args.conf, args.nms_iou, args.max_det)
            else:
                pred = UltralyticsPredictor(path, device, args.img_size,
                                            args.conf, args.nms_iou, args.max_det)
            results["models"][name] = eval_model(name, pred, loader, device,
                                                 args.img_size, args.seeds)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"!!! {name} lỗi: {e}", flush=True)
            continue
        del pred
        torch.cuda.empty_cache()
        json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)
        print(f"[save] {out_path}", flush=True)

    json.dump(results, open(out_path, "w"), indent=2, ensure_ascii=False)

    # ---- bảng gọn để dán vào paper ----
    print("\n" + "=" * 78)
    print(f"{'model':<16}{'params':>9}{'AP':>8}{'AP50':>8}{'AP75':>8}{'AP_s':>8}"
          f"{'mCP':>9}{'retain%':>14}")
    print("-" * 78)
    for n, r in results["models"].items():
        c, a = r["clean"], r["agg"]
        print(f"{n:<16}{r['params_M']:>8.2f}M{c['AP']:>8.4f}{c['AP50']:>8.4f}"
              f"{c['AP75']:>8.4f}{c['AP_s']:>8.4f}{a['mCP']['mean']:>9.4f}"
              f"{a['retain_pct']['mean']:>9.1f}±{a['retain_pct']['std']:<4.1f}")
    print("=" * 78)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
