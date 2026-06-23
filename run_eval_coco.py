"""
run_eval_coco.py — chạy đánh giá COCO trên 1 checkpoint, KHÔNG cần vào train.
=============================================================================

Tái dùng:
  - evaluate() trong train_v3.py  (bản bạn đã thêm cờ coco=True)
  - model PFDetNanoV15 trong models/
  - VisDronePersonDataset + collate_fn trong datasets/

Đặt file này ở thư mục gốc repo (cùng cấp train_v3.py). Ví dụ chạy:

  python run_eval_coco.py \
      --weights runs/train_v15_light_musgd_v3/best.pt \
      --val-images data/visdrone/val/images \
      --val-labels data/visdrone/val/labels \
      --img-size 640 --profile light --device cuda:0

Nếu dataset của bạn nhận tham số khác -> script in ra chữ ký __init__ để bạn (hoặc tôi) chỉnh 1 dòng.
"""

import argparse
import inspect
import torch
from torch.utils.data import DataLoader


def load_weights(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("ema", ckpt.get("model", ckpt.get("state_dict", ckpt)))
        # đôi khi 'ema'/'model' là cả nn.Module đã pickle
        if hasattr(state, "state_dict"):
            state = state.state_dict()
    else:
        state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)} "
          f"(missing nhỏ là ok; nếu missing ~ toàn bộ -> sai key checkpoint)")
    return model.to(device).eval()


def build_val_dataset(VisDronePersonDataset, val_images, val_labels, img_size):
    """Tự dò chữ ký __init__ để map tham số, tắt mọi augmentation."""
    sig = inspect.signature(VisDronePersonDataset.__init__)
    params = set(sig.parameters.keys())
    print(f"[dataset] VisDronePersonDataset.__init__{sig}")

    pool = {
        "img_dir": val_images, "image_dir": val_images, "images_dir": val_images,
        "images": val_images, "img_root": val_images,
        "label_dir": val_labels, "labels_dir": val_labels, "ann_dir": val_labels,
        "labels": val_labels, "label_root": val_labels,
        "img_size": img_size, "imgsz": img_size, "size": img_size,
        "augment": False, "train": False, "is_train": False,
        "mosaic_prob": 0.0, "mixup_prob": 0.0, "copy_paste_prob": 0.0,
        "drone_aug_prob": 0.0, "hflip_prob": 0.0, "affine_prob": 0.0, "color_jitter_prob": 0.0,
    }
    kwargs = {k: v for k, v in pool.items() if k in params}
    print(f"[dataset] dùng kwargs: { {k: kwargs[k] for k in kwargs if 'prob' not in k} }")
    return VisDronePersonDataset(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val-images", required=True)
    ap.add_argument("--val-labels", required=True)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--profile", default="light")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")

    # ---- model ----
    try:
        from models.pfdet_nano_v15 import PFDetNanoV15
        model = PFDetNanoV15(profile=args.profile)
    except Exception:
        from models import build_model
        model = build_model("v15", profile=args.profile)
    model = load_weights(model, args.weights, device)
    print(f"[model] v15 profile={args.profile} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- data ----
    from datasets import VisDronePersonDataset, collate_fn
    val_set = build_val_dataset(VisDronePersonDataset, args.val_images, args.val_labels, args.img_size)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_fn, pin_memory=True)
    print(f"[data] {len(val_set)} ảnh val")

    # ---- eval (dùng evaluate() đã thêm coco=True trong train_v3.py) ----
    from train_v3 import evaluate
    ap_, rec, prec = evaluate(model, val_loader, device,
                              img_size=args.img_size, strides=model.strides,
                              conf_thr=args.conf, progress_desc="val", coco=True)
    print(f"\n[AP@0.5 cũ] AP={ap_:.4f}  recall={rec:.4f}  precision={prec:.4f}")


if __name__ == "__main__":
    main()
