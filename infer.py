"""
Inference script for PFDet-Nano.
Supports: single image, directory of images, video, webcam.

Usage:
  python infer.py --weights runs/train/best.pt --source image.jpg
  python infer.py --weights runs/train/best.pt --source video.mp4
  python infer.py --weights runs/train/best.pt --source 0  (webcam)
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano
from utils.box_ops import decode_predictions_np, nms_numpy


def load_model(weights_path, device='cpu'):
    """Load trained model from checkpoint."""
    ckpt = torch.load(weights_path, map_location=device)
    cfg = ckpt['cfg']

    model = PFDetNano(
        base_c=cfg['model']['base_c'],
        num_bifpn=cfg['model'].get('num_bifpn', 2)
    ).to(device)

    # Load EMA weights if available, else regular weights
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
        print("[INFO] Loaded EMA weights")
    else:
        model.load_state_dict(ckpt['model'])
        print("[INFO] Loaded model weights")

    model.eval()
    img_size = cfg['model']['img_size']
    return model, img_size


def preprocess(img, img_size):
    """Preprocess image: resize, normalize, to tensor."""
    h0, w0 = img.shape[:2]

    # Letterbox resize (preserve aspect ratio)
    ratio = min(img_size / h0, img_size / w0)
    nh, nw = int(h0 * ratio), int(w0 * ratio)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    pad_h = img_size - nh
    pad_w = img_size - nw
    top = pad_h // 2
    left = pad_w // 2

    img_padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    img_padded[top:top+nh, left:left+nw] = img_resized

    # To float tensor
    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)

    return img_tensor, ratio, (top, left), (h0, w0)


def postprocess(preds, strides, img_size, conf_thr=0.3, nms_iou=0.45,
                ratio=1.0, pad=(0, 0), orig_size=(0, 0)):
    """
    Post-process model output: decode + NMS + scale to original image.
    """
    all_dets = []

    for si, pred in enumerate(preds):
        raw = pred[0].cpu().numpy()  # (7, H, W)
        dets = decode_predictions_np(raw, strides[si], img_size)
        all_dets.extend(dets)

    # Filter by confidence
    all_dets = [d for d in all_dets if d['score'] >= conf_thr]

    # NMS
    all_dets = nms_numpy(all_dets, iou_threshold=nms_iou)

    # Scale back to original image coordinates
    top, left = pad
    h0, w0 = orig_size

    for d in all_dets:
        x1, y1, x2, y2 = d['box']
        # From normalized [0,1] to padded image pixels
        x1 = x1 * img_size
        y1 = y1 * img_size
        x2 = x2 * img_size
        y2 = y2 * img_size

        # Remove padding offset
        x1 = (x1 - left) / ratio
        y1 = (y1 - top) / ratio
        x2 = (x2 - left) / ratio
        y2 = (y2 - top) / ratio

        # Clip to original image
        x1 = max(0, min(w0, x1))
        y1 = max(0, min(h0, y1))
        x2 = max(0, min(w0, x2))
        y2 = max(0, min(h0, y2))

        d['box'] = [x1, y1, x2, y2]

        # Scale foot point too
        fx, fy = d['foot']
        fx = (fx * img_size - left) / ratio
        fy = (fy * img_size - top) / ratio
        d['foot'] = [max(0, min(w0, fx)), max(0, min(h0, fy))]

    return all_dets


def draw_detections(img, detections, show_foot=True):
    """Draw bounding boxes and foot points on image."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        score = det['score']
        fx, fy = det['foot']

        # Box
        color = (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"person {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Foot point
        if show_foot:
            cv2.circle(img, (int(fx), int(fy)), 5, (0, 0, 255), -1)

    return img


def run_image(model, img_path, img_size, device, conf_thr=0.3, nms_iou=0.45,
              save_path=None, show=False):
    """Run inference on a single image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inp, ratio, pad, orig_size = preprocess(img_rgb, img_size)
    inp = inp.to(device)

    t0 = time.time()
    with torch.no_grad():
        preds = model(inp)
    dt = time.time() - t0

    dets = postprocess(preds, model.strides, img_size, conf_thr, nms_iou,
                       ratio, pad, orig_size)

    print(f"  {img_path}: {len(dets)} persons detected ({dt*1000:.1f}ms)")

    vis = draw_detections(img.copy(), dets)

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"  Saved to {save_path}")

    if show:
        cv2.imshow("PFDet-Nano", vis)
        cv2.waitKey(0)

    return dets


def run_video(model, source, img_size, device, conf_thr=0.3, nms_iou=0.45,
              save_path=None, show=True):
    """Run inference on video or webcam."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        return

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(save_path, fourcc, fps_vid, (w, h))

    fps_smooth = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp, ratio, pad, orig_size = preprocess(img_rgb, img_size)
        inp = inp.to(device)

        t0 = time.time()
        with torch.no_grad():
            preds = model(inp)
        dt = time.time() - t0

        dets = postprocess(preds, model.strides, img_size, conf_thr, nms_iou,
                           ratio, pad, orig_size)

        fps_now = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now

        vis = draw_detections(frame.copy(), dets)

        # FPS overlay
        info = f"FPS: {fps_smooth:.1f} | dets: {len(dets)} | conf: {conf_thr}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow("PFDet-Nano", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                conf_thr = min(0.9, conf_thr + 0.05)
            elif key == ord('-'):
                conf_thr = max(0.05, conf_thr - 0.05)

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames, avg FPS: {fps_smooth:.1f}")


def main():
    parser = argparse.ArgumentParser(description="PFDet-Nano Inference")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--source", required=True, help="Image/video path or webcam index")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--save", default=None, help="Save output to path")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--device", default="cuda:0", help="Device")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, img_size = load_model(args.weights, device)
    print(f"Model loaded, img_size={img_size}, device={device}")

    source = args.source
    if source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_video(model, source, img_size, device,
                  conf_thr=args.conf, nms_iou=args.nms_iou,
                  save_path=args.save, show=args.show)
    elif os.path.isdir(source):
        os.makedirs(args.save or "runs/detect", exist_ok=True)
        for fname in sorted(os.listdir(source)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(source, fname)
                save_path = os.path.join(args.save or "runs/detect", fname) if args.save else None
                run_image(model, img_path, img_size, device,
                         conf_thr=args.conf, nms_iou=args.nms_iou,
                         save_path=save_path, show=args.show)
    else:
        save_path = args.save
        run_image(model, source, img_size, device,
                 conf_thr=args.conf, nms_iou=args.nms_iou,
                 save_path=save_path, show=args.show)


if __name__ == "__main__":
    main()
