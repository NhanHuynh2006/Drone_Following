"""PFDet-Nano v15 detector wrapper.

Reuses the trained model and the existing infer.py preprocessing/postprocessing.
"""

import os
import sys
import time
import numpy as np
import torch

# Add parent project to path so we can import models / utils
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models import build_model_from_checkpoint
from utils import decode_predictions_np, nms_numpy
from infer import preprocess


class PersonDetector:
    def __init__(self, weights, conf_threshold=0.4, nms_iou=0.45,
                 img_size=640, use_ema=True, device='cpu'):
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.img_size = img_size
        self.device = torch.device(device)

        if not os.path.isfile(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")

        self.model, ckpt, version, _ = build_model_from_checkpoint(
            weights, device=self.device, use_ema=use_ema,
        )
        if hasattr(self.model, 'reparameterize'):
            self.model = self.model.reparameterize()
        self.model.eval()

        # On CPU, INT8 dynamic quantization helps Pi 5
        if self.device.type == 'cpu':
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model, {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8,
                )
                print("[Detector] INT8 dynamic quantization applied (CPU)")
            except Exception as e:
                print(f"[Detector] Quantization failed (continuing FP32): {e}")

        self.strides = list(self.model.strides)
        print(f"[Detector] Loaded {version}, img_size={img_size}, device={device}")

    @torch.no_grad()
    def detect(self, frame_bgr):
        """
        Run detection on a BGR frame.

        Returns list of dicts: [{'box': [x1,y1,x2,y2], 'score': float}, ...]
        Coordinates are in original image space.
        """
        if frame_bgr is None:
            return []

        h0, w0 = frame_bgr.shape[:2]
        # preprocess() expects RGB
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inp, ratio, pad, orig_size = preprocess(frame_rgb, self.img_size)
        inp = inp.to(self.device)

        preds = self.model(inp)

        # Decode all 4 scales
        all_dets = []
        for stride, pred in zip(self.strides, preds):
            raw = pred[0].cpu().numpy()
            dets = decode_predictions_np(raw, stride, self.img_size)
            all_dets.extend(dets)

        # Filter + NMS
        all_dets = [d for d in all_dets if d['score'] >= self.conf_threshold]
        all_dets = nms_numpy(all_dets, iou_threshold=self.nms_iou)

        # Rescale to original image space
        top, left = pad
        for d in all_dets:
            x1, y1, x2, y2 = d['box']
            x1 = (x1 * self.img_size - left) / ratio
            y1 = (y1 * self.img_size - top) / ratio
            x2 = (x2 * self.img_size - left) / ratio
            y2 = (y2 * self.img_size - top) / ratio
            d['box'] = [
                max(0, min(w0, x1)), max(0, min(h0, y1)),
                max(0, min(w0, x2)), max(0, min(h0, y2)),
            ]

        return all_dets
