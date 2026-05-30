"""
Benchmark PFDet-Nano v14 on desktop and edge-oriented profiles.

Reports:
  - params
  - optional FLOPs (when `thop` is installed)
  - model-only FPS / latency
  - end-to-end FPS / latency (preprocess + model + decode + NMS)
  - AP@0.5, focus AP, recall, precision when local val data is available
"""

import argparse
import copy
import csv
import os
import sys
import time
from statistics import mean

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import VisDronePersonDataset, collate_fn
from infer import preprocess
from models import build_model_from_checkpoint, count_params
from train_v3 import evaluate
from utils import decode_predictions_np, nms_numpy


BENCHMARK_PROFILES = {
    'desktop_4060': {
        'preferred_device': 'cuda:0',
        'frame_shape': (480, 640),
        'num_threads': None,
    },
    'rpi5_cpu': {
        'preferred_device': 'cpu',
        'frame_shape': (480, 640),
        'num_threads': 4,
    },
}


def resolve_device(profile_name):
    profile = BENCHMARK_PROFILES[profile_name]
    preferred = profile['preferred_device']
    if preferred.startswith('cuda') and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device('cpu')


def maybe_configure_threads(profile_name):
    num_threads = BENCHMARK_PROFILES[profile_name]['num_threads']
    if num_threads:
        torch.set_num_threads(num_threads)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_timings(label, timings_ms):
    avg_ms = mean(timings_ms) if timings_ms else 0.0
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        'label': label,
        'avg_ms': avg_ms,
        'p50_ms': percentile(timings_ms, 50),
        'p95_ms': percentile(timings_ms, 95),
        'fps': fps,
    }


def decode_predictions(preds, strides, img_size, conf_thr=0.25, nms_iou=0.45):
    detections = []
    for stride, pred in zip(strides, preds):
        raw = pred[0].detach().cpu().numpy()
        detections.extend(decode_predictions_np(raw, stride, img_size))

    detections = [det for det in detections if det['score'] >= conf_thr]
    return nms_numpy(detections, iou_threshold=nms_iou)


def benchmark_model_only(model, device, img_size, warmup, iters):
    sample = torch.randn(1, 3, img_size, img_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(sample)
        synchronize(device)

        timings_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = model(sample)
            synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    return summarize_timings('model_only', timings_ms)


def benchmark_end_to_end(model, device, img_size, warmup, iters, frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    detections_per_iter = []

    with torch.no_grad():
        for _ in range(warmup):
            inp, _, _, _ = preprocess(frame_rgb, img_size)
            inp = inp.to(device, non_blocking=True)
            preds = model(inp)
            _ = decode_predictions(preds, model.strides, img_size)
        synchronize(device)

        timings_ms = []
        for _ in range(iters):
            t0 = time.perf_counter()
            inp, _, _, _ = preprocess(frame_rgb, img_size)
            inp = inp.to(device, non_blocking=True)
            preds = model(inp)
            dets = decode_predictions(preds, model.strides, img_size)
            synchronize(device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)
            detections_per_iter.append(len(dets))

    summary = summarize_timings('end_to_end', timings_ms)
    summary['avg_detections'] = mean(detections_per_iter) if detections_per_iter else 0.0
    return summary


def maybe_profile_flops(model, img_size):
    try:
        from thop import profile
    except ImportError:
        return None

    original_device = next(model.parameters()).device
    model = model.to('cpu')
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy,), verbose=False)
    model.to(original_device)
    model.eval()
    return flops


def maybe_fuse_model(model, enable=True):
    deploy_model = copy.deepcopy(model).eval()
    fused = False
    if enable and hasattr(deploy_model, 'reparameterize'):
        deploy_model = deploy_model.reparameterize()
        fused = True
    return deploy_model, fused


def build_val_loader(cfg, img_size, profile_name):
    data_cfg = cfg.get('data') or {}
    root = data_cfg.get('root', '')
    val_images = os.path.join(root, data_cfg.get('val_images', ''))
    val_labels = os.path.join(root, data_cfg.get('val_labels', ''))
    if not os.path.isdir(val_images) or not os.path.isdir(val_labels):
        return None, None

    val_set = VisDronePersonDataset(
        val_images,
        val_labels,
        img_size=img_size,
        augment=False,
        cache_ram=False,
    )
    if len(val_set) == 0:
        return val_set, None

    num_workers = 0 if profile_name == 'rpi5_cpu' else min(2, os.cpu_count() or 1)
    batch_size = max(1, int((cfg.get('train', {}).get('batch_size', 8)) // 2))
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(num_workers > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate_fn,
    )
    return val_set, val_loader


def load_reference_frame(cfg, img_size, fallback_shape):
    data_cfg = cfg.get('data') or {}
    root = data_cfg.get('root', '')
    val_images = os.path.join(root, data_cfg.get('val_images', ''))
    val_labels = os.path.join(root, data_cfg.get('val_labels', ''))
    if os.path.isdir(val_images) and os.path.isdir(val_labels):
        val_set = VisDronePersonDataset(
            val_images,
            val_labels,
            img_size=img_size,
            augment=False,
            cache_ram=False,
        )
        if len(val_set) > 0:
            img_path = val_set.samples[0][0]
            frame = cv2.imread(img_path)
            if frame is not None:
                return frame

    h, w = fallback_shape
    return np.full((h, w, 3), 114, dtype=np.uint8)


def evaluate_metrics(model, cfg, device, img_size, profile_name):
    _, val_loader = build_val_loader(cfg, img_size, profile_name)
    if val_loader is None:
        return None

    eval_cfg = cfg.get('eval', {})
    ap, recall, precision = evaluate(
        model,
        val_loader,
        device,
        img_size,
        model.strides,
        conf_thr=eval_cfg.get('conf_thr', 0.01),
        iou_thr=eval_cfg.get('iou_thr', 0.5),
        decode_fn=decode_predictions_np,
        min_box_area=eval_cfg.get('min_box_area', 0.0),
        min_aspect=eval_cfg.get('min_aspect', 0.0),
        max_aspect=eval_cfg.get('max_aspect', 0.0),
        max_det=int(eval_cfg.get('max_det', 300)),
        progress_desc="Benchmark val",
    )

    metrics = {
        'ap50': ap,
        'recall': recall,
        'precision': precision,
    }

    focus_min_gt_area = float(eval_cfg.get('focus_min_gt_area', 0.0))
    if focus_min_gt_area > 0:
        focus_ap, focus_recall, focus_precision = evaluate(
            model,
            val_loader,
            device,
            img_size,
            model.strides,
            conf_thr=eval_cfg.get('focus_conf_thr', eval_cfg.get('conf_thr', 0.01)),
            iou_thr=eval_cfg.get('iou_thr', 0.5),
            decode_fn=decode_predictions_np,
            min_box_area=eval_cfg.get('min_box_area', 0.0),
            min_aspect=eval_cfg.get('min_aspect', 0.0),
            max_aspect=eval_cfg.get('max_aspect', 0.0),
            max_det=int(eval_cfg.get('max_det', 300)),
            min_gt_area=focus_min_gt_area,
            progress_desc="Benchmark focus val",
        )
        metrics.update({
            'focus_ap50': focus_ap,
            'focus_recall': focus_recall,
            'focus_precision': focus_precision,
        })

    return metrics


def format_summary(summary):
    return (
        f"{summary['label']}: "
        f"{summary['fps']:.1f} FPS | "
        f"avg={summary['avg_ms']:.2f}ms | "
        f"p50={summary['p50_ms']:.2f}ms | "
        f"p95={summary['p95_ms']:.2f}ms"
    )


SCOREBOARD_FIELDS = [
    'label',
    'family',
    'benchmark_mode',
    'profile',
    'weights',
    'img_size',
    'runtime_device',
    'train_params_m',
    'deploy_params_m',
    'train_flops_g',
    'deploy_flops_g',
    'model_only_fps',
    'model_only_avg_ms',
    'model_only_p50_ms',
    'model_only_p95_ms',
    'end_to_end_fps',
    'end_to_end_avg_ms',
    'end_to_end_p50_ms',
    'end_to_end_p95_ms',
    'avg_detections',
    'ap50',
    'recall',
    'precision',
    'focus_ap50',
    'focus_recall',
    'focus_precision',
]


def append_scoreboard_row(csv_path, row):
    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SCOREBOARD_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, '') for key in SCOREBOARD_FIELDS})


def main():
    parser = argparse.ArgumentParser(description="Benchmark PFDet-Nano v14")
    parser.add_argument("--weights", required=True, help="Path to checkpoint (.pt)")
    parser.add_argument("--profile", default="desktop_4060", choices=sorted(BENCHMARK_PROFILES))
    parser.add_argument("--img-size", type=int, default=None, help="Override input size")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Measured iterations")
    parser.add_argument("--skip-metrics", action="store_true", help="Skip validation metrics")
    parser.add_argument("--no-fuse", action="store_true", help="Benchmark the training graph instead of the fused deploy graph")
    parser.add_argument("--label", default=None, help="Optional label for scoreboard rows")
    parser.add_argument("--benchmark-mode", default="full_frame", help="Logical benchmark mode label for scoreboard output")
    parser.add_argument("--scoreboard", default=None, help="Append results to a CSV scoreboard")
    args = parser.parse_args()

    maybe_configure_threads(args.profile)
    device = resolve_device(args.profile)
    model, ckpt, model_version, model_kwargs = build_model_from_checkpoint(
        args.weights,
        device=device,
        use_ema=True,
    )
    deploy_model, fused = maybe_fuse_model(model, enable=not args.no_fuse)
    cfg = ckpt.get('cfg', {})
    img_size = int(args.img_size or cfg.get('model', {}).get('img_size', 384))

    total_params, trainable_params = count_params(model)
    deploy_total_params, deploy_trainable_params = count_params(deploy_model)
    train_flops = maybe_profile_flops(model, img_size)
    deploy_flops = maybe_profile_flops(deploy_model, img_size)
    frame_bgr = load_reference_frame(
        cfg,
        img_size,
        BENCHMARK_PROFILES[args.profile]['frame_shape'],
    )

    print(f"Profile: {args.profile}")
    print(f"Device: {device}")
    print(f"Model: PFDetNano {model_version}")
    print(f"Model kwargs: {model_kwargs}")
    print(f"Input size: {img_size}")
    print(
        f"Params (train): total={total_params/1e6:.3f}M "
        f"trainable={trainable_params/1e6:.3f}M"
    )
    print(
        f"Params (deploy{' fused' if fused else ''}): "
        f"total={deploy_total_params/1e6:.3f}M "
        f"trainable={deploy_trainable_params/1e6:.3f}M"
    )
    if train_flops is not None:
        print(f"FLOPs (train): {train_flops/1e9:.3f} GFLOPs")
    else:
        print("FLOPs (train): unavailable (install `thop` to enable)")
    if deploy_flops is not None:
        print(f"FLOPs (deploy): {deploy_flops/1e9:.3f} GFLOPs")
    else:
        print("FLOPs (deploy): unavailable (install `thop` to enable)")

    model_only = benchmark_model_only(deploy_model, device, img_size, args.warmup, args.iters)
    end_to_end = benchmark_end_to_end(deploy_model, device, img_size, args.warmup, args.iters, frame_bgr)

    print(format_summary(model_only))
    print(
        f"{format_summary(end_to_end)} | "
        f"avg_dets={end_to_end['avg_detections']:.1f}"
    )

    metrics = None
    if not args.skip_metrics:
        metrics = evaluate_metrics(deploy_model, cfg, device, img_size, args.profile)
        if metrics is None:
            print("Metrics: skipped (validation dataset not found locally)")
        else:
            print(
                "Metrics: "
                f"AP@0.5={metrics['ap50']:.4f} | "
                f"Recall={metrics['recall']:.4f} | "
                f"Precision={metrics['precision']:.4f}"
            )
            if 'focus_ap50' in metrics:
                print(
                    "Focus metrics: "
                    f"AP@0.5={metrics['focus_ap50']:.4f} | "
                    f"Recall={metrics['focus_recall']:.4f} | "
                    f"Precision={metrics['focus_precision']:.4f}"
                )

    if args.scoreboard:
        row = {
            'label': args.label or f"pfdet_{model_version}",
            'family': 'pfdet',
            'benchmark_mode': args.benchmark_mode,
            'profile': args.profile,
            'weights': args.weights,
            'img_size': img_size,
            'runtime_device': str(device),
            'train_params_m': f"{total_params / 1e6:.6f}",
            'deploy_params_m': f"{deploy_total_params / 1e6:.6f}",
            'train_flops_g': f"{(train_flops / 1e9):.6f}" if train_flops is not None else '',
            'deploy_flops_g': f"{(deploy_flops / 1e9):.6f}" if deploy_flops is not None else '',
            'model_only_fps': f"{model_only['fps']:.6f}",
            'model_only_avg_ms': f"{model_only['avg_ms']:.6f}",
            'model_only_p50_ms': f"{model_only['p50_ms']:.6f}",
            'model_only_p95_ms': f"{model_only['p95_ms']:.6f}",
            'end_to_end_fps': f"{end_to_end['fps']:.6f}",
            'end_to_end_avg_ms': f"{end_to_end['avg_ms']:.6f}",
            'end_to_end_p50_ms': f"{end_to_end['p50_ms']:.6f}",
            'end_to_end_p95_ms': f"{end_to_end['p95_ms']:.6f}",
            'avg_detections': f"{end_to_end['avg_detections']:.6f}",
            'ap50': f"{metrics['ap50']:.6f}" if metrics else '',
            'recall': f"{metrics['recall']:.6f}" if metrics else '',
            'precision': f"{metrics['precision']:.6f}" if metrics else '',
            'focus_ap50': f"{metrics['focus_ap50']:.6f}" if metrics and 'focus_ap50' in metrics else '',
            'focus_recall': f"{metrics['focus_recall']:.6f}" if metrics and 'focus_recall' in metrics else '',
            'focus_precision': f"{metrics['focus_precision']:.6f}" if metrics and 'focus_precision' in metrics else '',
        }
        append_scoreboard_row(args.scoreboard, row)
        print(f"Scoreboard: appended row to {args.scoreboard}")


if __name__ == "__main__":
    main()
