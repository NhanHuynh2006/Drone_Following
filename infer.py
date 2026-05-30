"""
Inference script for PFDet-Nano v14.

Supports single-image, directory, video, and webcam inference with the
4-scale PFDet decoder contract used by training/export/runtime paths.
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model_from_checkpoint
from utils import decode_predictions_np, nms_numpy


def soft_nms(detections, sigma=0.5, score_threshold=0.05):
    """Soft-NMS: Gaussian decay for overlapping boxes."""
    if len(detections) == 0:
        return []

    dets = list(detections)
    boxes = np.array([d['box'] for d in dets], dtype=np.float32)
    scores = np.array([d['score'] for d in dets], dtype=np.float32)
    n = len(dets)

    for i in range(n):
        max_idx = np.argmax(scores[i:]) + i
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        dets[i], dets[max_idx] = dets[max_idx], dets[i]

        if i < n - 1:
            xx1 = np.maximum(boxes[i, 0], boxes[i+1:, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[i+1:, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[i+1:, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[i+1:, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = max(0, (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))
            area_rest = (boxes[i+1:, 2] - boxes[i+1:, 0]) * (boxes[i+1:, 3] - boxes[i+1:, 1])
            union = area_i + area_rest - inter
            iou = inter / (union + 1e-7)
            scores[i+1:] *= np.exp(-(iou ** 2) / sigma)

    keep = [i for i in range(n) if scores[i] >= score_threshold]
    return [dets[i] for i in keep]


def load_model(weights_path, device='cpu', use_ema=True):
    model, ckpt, model_version, model_kwargs = build_model_from_checkpoint(
        weights_path,
        device=device,
        use_ema=use_ema,
    )
    cfg = ckpt['cfg']

    state_key = 'ema' if use_ema and 'ema' in ckpt else 'model'
    print(f"[INFO] Loaded {state_key.upper()} weights")
    if model_kwargs:
        print(f"[INFO] Model kwargs: {model_kwargs}")
    return model, cfg['model']['img_size'], model_version


def preprocess(img, img_size):
    h0, w0 = img.shape[:2]
    ratio = min(img_size / h0, img_size / w0)
    nh, nw = int(h0 * ratio), int(w0 * ratio)
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = img_size - nh
    pad_w = img_size - nw
    top = pad_h // 2
    left = pad_w // 2

    img_padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    img_padded[top:top+nh, left:left+nw] = img_resized

    img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0
    return img_tensor.unsqueeze(0), ratio, (top, left), (h0, w0)


def filter_detections(detections, min_box_area=0, min_aspect=0.0,
                      max_aspect=0.0, max_det=300):
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det['box']
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        area = bw * bh
        aspect = bh / max(bw, 1e-6)

        if area < min_box_area:
            continue
        if min_aspect > 0 and aspect < min_aspect:
            continue
        if max_aspect > 0 and aspect > max_aspect:
            continue

        filtered.append(det)
        if len(filtered) >= max_det:
            break

    return filtered


def postprocess(preds, strides, img_size, conf_thr=0.3, nms_iou=0.45,
                ratio=1.0, pad=(0, 0), orig_size=(0, 0), use_soft_nms=False,
                min_box_area=0, min_aspect=0.0, max_aspect=0.0, max_det=300):
    all_dets = []
    for si, pred in enumerate(preds):
        raw = pred[0].cpu().numpy()
        dets = decode_predictions_np(raw, strides[si], img_size)
        all_dets.extend(dets)

    all_dets = [d for d in all_dets if d['score'] >= conf_thr]

    if use_soft_nms:
        all_dets = soft_nms(all_dets, sigma=0.5, score_threshold=conf_thr)
    else:
        all_dets = nms_numpy(all_dets, iou_threshold=nms_iou)

    top, left = pad
    h0, w0 = orig_size

    for d in all_dets:
        x1, y1, x2, y2 = d['box']
        x1 = (x1 * img_size - left) / ratio
        y1 = (y1 * img_size - top) / ratio
        x2 = (x2 * img_size - left) / ratio
        y2 = (y2 * img_size - top) / ratio
        d['box'] = [max(0, min(w0, x1)), max(0, min(h0, y1)),
                     max(0, min(w0, x2)), max(0, min(h0, y2))]

        # Foot point = bottom center of bounding box (standard in literature)
        d['foot'] = [(d['box'][0] + d['box'][2]) / 2, d['box'][3]]

    return filter_detections(
        all_dets,
        min_box_area=min_box_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        max_det=max_det,
    )


def draw_detections(img, detections, show_foot=True):
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['box']]
        score = det['score']

        thickness = 2 if score > 0.5 else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        label = f"person {score:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw foot point (bottom center of box)
        if show_foot and 'foot' in det:
            fx, fy = int(det['foot'][0]), int(det['foot'][1])
            cv2.circle(img, (fx, fy), 4, (0, 0, 255), -1)

    return img


def run_image(model, img_path, img_size, device, conf_thr=0.3, nms_iou=0.45,
              save_path=None, show=False, use_soft_nms=False,
              min_box_area=0, min_aspect=0.0, max_aspect=0.0, max_det=300):
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

    dets = postprocess(
        preds, model.strides, img_size, conf_thr, nms_iou,
        ratio, pad, orig_size,
        use_soft_nms=use_soft_nms,
        min_box_area=min_box_area,
        min_aspect=min_aspect,
        max_aspect=max_aspect,
        max_det=max_det,
    )

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
              save_path=None, show=True, use_soft_nms=False,
              min_box_area=0, min_aspect=0.0, max_aspect=0.0, max_det=300):
    if source.isdigit():
        idx = int(source)
        cap = None
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(5):
                    cap.read()
                ret, test_frame = cap.read()
                if ret and test_frame.mean() > 1:
                    print(f"[INFO] Camera opened with backend {backend}")
                    break
                cap.release()
                cap = None

        if cap is None:
            print("[ERROR] Cannot open camera or camera returns black frames.")
            print("[TIP] Try: cheese  (to test camera)")
            print("[TIP] Or use a video file: --source video.mp4")
            return
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        return

    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {w}x{h} @ {fps_vid:.0f}fps")

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

        dets = postprocess(
            preds, model.strides, img_size, conf_thr, nms_iou,
            ratio, pad, orig_size,
            use_soft_nms=use_soft_nms,
            min_box_area=min_box_area,
            min_aspect=min_aspect,
            max_aspect=max_aspect,
            max_det=max_det,
        )

        fps_now = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now

        vis = draw_detections(frame.copy(), dets)

        info = f"FPS: {fps_smooth:.1f} | dets: {len(dets)} | conf>{conf_thr:.2f}"
        cv2.putText(vis, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2, cv2.LINE_AA)

        if writer:
            writer.write(vis)

        if show:
            cv2.imshow("PFDet-Nano", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                conf_thr = min(0.9, conf_thr + 0.05)
                print(f"  conf_thr -> {conf_thr:.2f}")
            elif key == ord('-'):
                conf_thr = max(0.05, conf_thr - 0.05)
                print(f"  conf_thr -> {conf_thr:.2f}")

        frame_count += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames, avg FPS: {fps_smooth:.1f}")


def main():
    parser = argparse.ArgumentParser(description="PFDet-Nano Inference")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--source", required=True, help="Image/video/webcam")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--soft-nms", action="store_true", help="Use soft-NMS instead of standard NMS")
    parser.add_argument("--min-box-area", type=float, default=64.0,
                        help="Minimum box area in original-image pixels")
    parser.add_argument("--min-aspect", type=float, default=1.0,
                        help="Minimum box aspect ratio h/w to keep")
    parser.add_argument("--max-aspect", type=float, default=5.5,
                        help="Maximum box aspect ratio h/w to keep")
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per image/frame")
    parser.add_argument("--save", default=None, help="Save output path")
    parser.add_argument("--show", action="store_true", help="Show results")
    parser.add_argument("--device", default="cuda:0", help="Device (cpu or cuda:0)")
    parser.add_argument("--no-ema", action="store_true", help="Load raw model instead of EMA")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    model, img_size, model_version = load_model(args.weights, device, use_ema=not args.no_ema)
    print(f"Model loaded: {model_version}, img_size={img_size}, device={device}")

    source = args.source
    if source.isdigit() or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        run_video(model, source, img_size, device,
                  conf_thr=args.conf, nms_iou=args.nms_iou,
                  save_path=args.save, show=args.show,
                  use_soft_nms=args.soft_nms,
                  min_box_area=args.min_box_area,
                  min_aspect=args.min_aspect,
                  max_aspect=args.max_aspect,
                  max_det=args.max_det)
    elif os.path.isdir(source):
        out_dir = args.save or "runs/detect"
        os.makedirs(out_dir, exist_ok=True)
        for fname in sorted(os.listdir(source)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img_path = os.path.join(source, fname)
                sp = os.path.join(out_dir, fname) if args.save else None
                run_image(model, img_path, img_size, device,
                         conf_thr=args.conf, nms_iou=args.nms_iou,
                         save_path=sp, show=args.show,
                         use_soft_nms=args.soft_nms,
                         min_box_area=args.min_box_area,
                         min_aspect=args.min_aspect,
                         max_aspect=args.max_aspect,
                         max_det=args.max_det)
    else:
        run_image(model, source, img_size, device,
                 conf_thr=args.conf, nms_iou=args.nms_iou,
                 save_path=args.save, show=args.show,
                 use_soft_nms=args.soft_nms,
                 min_box_area=args.min_box_area,
                 min_aspect=args.min_aspect,
                 max_aspect=args.max_aspect,
                 max_det=args.max_det)


if __name__ == "__main__":
    main()
