"""
Run inference on image sequence and create demo video with detections.
Usage: python make_demo_video.py
"""

import os
import sys
import time
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano
from infer import load_model, preprocess, postprocess, draw_detections

# ── Config ──
WEIGHTS = "runs/train_v5/best.pt"
IMG_DIR = "/home/nolan/Documents/pfdet_drone_follow/VisDrone2019-SOT-val/sequences/uav0000024_00000_s"
OUTPUT_VIDEO = "runs/train_v5/demo_detection.mp4"
CONF_THR = 0.3
NMS_IOU = 0.45
FPS = 30
DEVICE = "cuda:0"


def main():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    model, img_size = load_model(WEIGHTS, device)
    print(f"Model loaded, img_size={img_size}, device={device}")

    # Get sorted image list
    imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
    print(f"Found {len(imgs)} images")

    # Read first image to get dimensions
    first = cv2.imread(os.path.join(IMG_DIR, imgs[0]))
    h0, w0 = first.shape[:2]
    print(f"Frame size: {w0}x{h0}")

    # Video writer
    os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (w0, h0))

    total_dets = 0
    fps_smooth = 0
    t_start = time.time()

    for i, fname in enumerate(imgs):
        img = cv2.imread(os.path.join(IMG_DIR, fname))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp, ratio, pad, orig_size = preprocess(img_rgb, img_size)
        inp = inp.to(device)

        # Only measure model forward pass
        torch.cuda.synchronize()
        t0_model = time.time()
        with torch.no_grad():
            preds = model(inp)
        torch.cuda.synchronize()
        dt_model = time.time() - t0_model

        dets = postprocess(preds, model.strides, img_size, CONF_THR, NMS_IOU,
                           ratio, pad, orig_size)

        vis = draw_detections(img.copy(), dets, show_foot=True)

        # Show model-only FPS
        fps_now = 1.0 / max(dt_model, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now
        cv2.putText(vis, f"Model FPS: {fps_smooth:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(vis)
        total_dets += len(dets)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            fps = (i + 1) / elapsed
            print(f"  [{i+1}/{len(imgs)}] {fps:.1f} img/s, avg {total_dets/(i+1):.1f} dets/frame")

    writer.release()
    elapsed = time.time() - t_start
    print(f"\nDone! {len(imgs)} frames in {elapsed:.1f}s ({len(imgs)/elapsed:.1f} img/s)")
    print(f"Avg detections/frame: {total_dets/len(imgs):.1f}")
    print(f"Video saved: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
