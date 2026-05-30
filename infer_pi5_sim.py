"""
Pi 5 Simulation Inference
=========================

Simulates Raspberry Pi 5 inference conditions on desktop:
  - CPU only (no CUDA), 4 threads (Pi 5 = 4x Cortex-A76)
  - 320px input (Pi 5 deployment target)
  - PyTorch dynamic INT8 quantization
  - Optional ONNX Runtime backend (closer to NCNN on Pi)

Usage:
  # PyTorch CPU + INT8 quantization (default)
  python infer_pi5_sim.py --weights runs/train_v15_light_adamw_v3/best.pt --source 0

  # ONNX Runtime backend (install: pip install onnxruntime)
  python infer_pi5_sim.py --weights runs/train_v15_light_adamw_v3/best.pt --source 0 --onnx

  # USB camera (usually device 1 or 2)
  python infer_pi5_sim.py --weights runs/train_v15_light_adamw_v3/best.pt --source 1

Note: x86 CPU != ARM Cortex-A76. This gives a rough estimate.
  Actual Pi 5 will be ~1.5-3x slower (no AVX/SSE, weaker per-core IPC).
  Multiply the latency you see here by ~2x for a realistic Pi 5 estimate.
"""

import argparse
import copy
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model_from_checkpoint, count_params
from utils import decode_predictions_np, nms_numpy
from infer import draw_detections, filter_detections


PI5_IMG_SIZE = 320
PI5_NUM_THREADS = 4


def prepare_model_pytorch(weights_path, quantize=True):
    """Load model for CPU inference, optionally with INT8 dynamic quantization."""
    model, ckpt, version, model_kwargs = build_model_from_checkpoint(
        weights_path, device='cpu', use_ema=True,
    )

    # Fuse BN if model supports reparameterize
    if hasattr(model, 'reparameterize'):
        model = model.reparameterize()

    if quantize:
        model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8,
        )
        print("[Pi5-sim] INT8 dynamic quantization applied")

    model.eval()
    total, _ = count_params(model)
    print(f"[Pi5-sim] Model: {version}, params: {total/1e6:.3f}M")
    return model, version


def prepare_model_onnx(weights_path, img_size):
    """Export to ONNX and load with ONNX Runtime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Install: pip install onnxruntime")
        sys.exit(1)

    # Export ONNX if not exists
    onnx_dir = os.path.dirname(weights_path)
    stem = os.path.splitext(os.path.basename(weights_path))[0]
    onnx_path = os.path.join(onnx_dir, f"{stem}_{img_size}.onnx")

    model, ckpt, version, _ = build_model_from_checkpoint(
        weights_path, device='cpu', use_ema=True,
    )
    strides = list(model.strides)

    if hasattr(model, 'reparameterize'):
        model = model.reparameterize()

    if not os.path.exists(onnx_path):
        print(f"[Pi5-sim] Exporting ONNX to {onnx_path} ...")
        dummy = torch.randn(1, 3, img_size, img_size)
        output_names = [f"output_p{i}" for i in range(2, 2 + len(strides))]
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=11,
            input_names=['input'],
            output_names=output_names,
            do_constant_folding=True,
        )
        # Simplify if possible
        try:
            import onnxsim, onnx
            m = onnx.load(onnx_path)
            m, ok = onnxsim.simplify(m)
            if ok:
                onnx.save(m, onnx_path)
                print("[Pi5-sim] ONNX simplified")
        except ImportError:
            pass

    print(f"[Pi5-sim] Loading ONNX: {onnx_path}")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = PI5_NUM_THREADS
    sess_opts.inter_op_num_threads = 1
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=['CPUExecutionProvider'])

    onnx_size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"[Pi5-sim] ONNX size: {onnx_size_mb:.2f}MB, runtime: ORT CPU, threads: {PI5_NUM_THREADS}")
    return sess, strides, version


def preprocess(img_rgb, img_size):
    """Letterbox preprocess matching infer.py."""
    h0, w0 = img_rgb.shape[:2]
    ratio = min(img_size / h0, img_size / w0)
    nh, nw = int(h0 * ratio), int(w0 * ratio)
    img_resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_h = img_size - nh
    pad_w = img_size - nw
    top = pad_h // 2
    left = pad_w // 2

    img_padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    img_padded[top:top+nh, left:left+nw] = img_resized

    blob = img_padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)
    return blob, ratio, (top, left), (h0, w0)


def postprocess(raw_outputs, strides, img_size, conf_thr, nms_iou,
                ratio, pad, orig_size):
    """Decode + NMS, rescale to original image coords."""
    all_dets = []
    for raw, stride in zip(raw_outputs, strides):
        if isinstance(raw, torch.Tensor):
            raw = raw[0].cpu().numpy()
        elif raw.ndim == 4:
            raw = raw[0]
        dets = decode_predictions_np(raw, stride, img_size)
        all_dets.extend(dets)

    all_dets = [d for d in all_dets if d['score'] >= conf_thr]
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
        d['foot'] = [(d['box'][0] + d['box'][2]) / 2, d['box'][3]]

    return filter_detections(all_dets, min_box_area=64.0, max_det=100)


def run_camera(model, strides, img_size, source, conf_thr, nms_iou, use_onnx=False):
    """Run camera inference under Pi 5 simulated conditions."""
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        print("[TIP] Try --source 0 for built-in webcam, --source 1 for USB camera")
        return

    # Set camera to 640x480 (common Pi Camera / USB cam resolution)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    backend = "ONNX Runtime" if use_onnx else "PyTorch CPU INT8"
    print(f"[Pi5-sim] Camera: {w}x{h}, inference: {img_size}px, backend: {backend}")
    print(f"[Pi5-sim] Threads: {PI5_NUM_THREADS} (Pi 5 = 4 cores)")
    print(f"[Pi5-sim] NOTE: actual Pi 5 will be ~2x slower than this (ARM vs x86)")
    print(f"[Pi5-sim] Press 'q' to quit, '+'/'-' to adjust confidence")

    fps_smooth = 0
    frame_count = 0

    # Warmup
    print("[Pi5-sim] Warming up (5 frames)...")
    for _ in range(5):
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        blob, ratio, pad, orig_size = preprocess(img_rgb, img_size)
        if use_onnx:
            model.run(None, {'input': blob})
        else:
            with torch.no_grad():
                model(torch.from_numpy(blob))
    print("[Pi5-sim] Warmup done. Starting inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        t0 = time.perf_counter()

        # Preprocess
        blob, ratio, pad, orig_size = preprocess(img_rgb, img_size)

        # Inference
        if use_onnx:
            raw_outputs = model.run(None, {'input': blob})
        else:
            with torch.no_grad():
                raw_outputs = model(torch.from_numpy(blob))

        # Postprocess
        dets = postprocess(raw_outputs, strides, img_size, conf_thr, nms_iou,
                           ratio, pad, orig_size)

        dt = time.perf_counter() - t0
        fps_now = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now

        # Draw
        vis = draw_detections(frame.copy(), dets)

        # Info overlay
        latency_ms = dt * 1000
        pi5_est_ms = latency_ms * 2.0  # rough ARM penalty
        pi5_est_fps = 1000.0 / pi5_est_ms if pi5_est_ms > 0 else 0

        info1 = f"Desktop CPU: {fps_smooth:.1f} FPS ({latency_ms:.0f}ms) | dets: {len(dets)}"
        info2 = f"Pi5 estimate: ~{pi5_est_fps:.0f} FPS (~{pi5_est_ms:.0f}ms) | {img_size}px"
        cv2.putText(vis, info1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, info2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, f"conf > {conf_thr:.2f}  [+/-] to adjust",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("PFDet-Nano Pi5 Simulation", vis)
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
    cv2.destroyAllWindows()
    print(f"\n[Pi5-sim] {frame_count} frames, desktop CPU avg: {fps_smooth:.1f} FPS")
    print(f"[Pi5-sim] Estimated Pi 5: ~{fps_smooth/2:.0f} FPS at {img_size}px")


def main():
    parser = argparse.ArgumentParser(description="PFDet-Nano Pi 5 Simulation")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--source", default="0", help="Camera index or video path")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--nms-iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=PI5_IMG_SIZE,
                        help=f"Input size (default: {PI5_IMG_SIZE}px for Pi 5)")
    parser.add_argument("--onnx", action="store_true",
                        help="Use ONNX Runtime backend (closer to NCNN on Pi)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Skip INT8 quantization (PyTorch mode only)")
    args = parser.parse_args()

    # Simulate Pi 5: CPU only, 4 threads
    torch.set_num_threads(PI5_NUM_THREADS)
    torch.set_num_interop_threads(1)
    print(f"[Pi5-sim] CPU threads: {PI5_NUM_THREADS}, input: {args.img_size}px")
    print(f"[Pi5-sim] CUDA disabled (Pi 5 has no GPU)")

    if args.onnx:
        sess, strides, version = prepare_model_onnx(args.weights, args.img_size)
        run_camera(sess, strides, args.img_size, args.source,
                   args.conf, args.nms_iou, use_onnx=True)
    else:
        model, version = prepare_model_pytorch(
            args.weights, quantize=not args.no_quantize)
        strides = list(model.strides)
        run_camera(model, strides, args.img_size, args.source,
                   args.conf, args.nms_iou, use_onnx=False)


if __name__ == "__main__":
    main()
