"""
detector_hailo.py — SteadyDet-N chạy trên Hailo-8L NPU.

Cùng giao diện với `detector.py` (PyTorch), nhưng chạy HEF trên NPU thay vì CPU:

    det = PersonDetectorHailo(hef="models/pfdet_v17_640_hailo8l.hef", img_size=640)
    boxes = det.detect(frame_bgr)     # [{'box': [x1,y1,x2,y2], 'score': float}, ...]

Vì sao cần: `detector.py` chạy PyTorch trên CPU Pi 5, chậm hơn NPU hàng chục lần.
Toàn bộ ngân sách độ trễ của vòng điều khiển (30 Hz, xem runs/control_rate_2026-08-27.md)
giả định inference 17,7 ms — con số đó chỉ có khi dùng HEF.

TỰ CHỨA: không import torch, không import `models/`, `utils/`, `infer.py` của repo gốc.
Trên Pi chỉ cần `hailo_platform`, `numpy`, `cv2` (đều là lib hệ thống).

⚠️ Công thức decode PHẢI khớp đúng lúc train:
    tâm  = (ô lưới + sigmoid(dx) − 0.5) × stride / img_size
    cạnh = exp(clip(dw, −4, +4)) × stride / img_size
Dùng `tanh` thay `sigmoid−0.5` là sai nửa ô lưới — lỗi đã từng có thật trong
`hailo_eval_val.py`, xem runs/pi_hailo_measured_2026-08-24.md.
"""

import os
import time

import numpy as np

STRIDES = (4, 8, 16, 32)          # P2, P3, P4, P5


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def letterbox(img, size):
    """Thu ảnh giữ tỉ lệ rồi đệm xám 114. Trả (canvas, tỉ lệ, lề trên, lề trái)."""
    import cv2
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh),
                                                      interpolation=cv2.INTER_LINEAR)
    return canvas, r, top, left


def decode_scale(raw, stride, img_size, conf_thr):
    """raw: (5,H,W) logit -> list {'score','box'} với box chuẩn hoá 0..1 trên canvas."""
    _, H, W = raw.shape
    scores = _sigmoid(raw[0])
    mask = scores >= conf_thr
    if not mask.any():
        return []
    ys, xs = np.nonzero(mask)
    sc = scores[ys, xs]
    s = stride / img_size
    dx = _sigmoid(raw[1, ys, xs]) - 0.5
    dy = _sigmoid(raw[2, ys, xs]) - 0.5
    w = np.exp(np.clip(raw[3, ys, xs], -4.0, 4.0)) * s
    h = np.exp(np.clip(raw[4, ys, xs], -4.0, 4.0)) * s
    cx = (xs + dx) * s
    cy = (ys + dy) * s
    return [{"score": float(sc[i]),
             "box": [float(cx[i] - w[i] / 2), float(cy[i] - h[i] / 2),
                     float(cx[i] + w[i] / 2), float(cy[i] + h[i] / 2)]}
            for i in range(len(sc))]


def nms(dets, iou_thr=0.45, max_det=100, pre_nms_topk=500):
    """NMS theo IoU. `pre_nms_topk` chặn trường hợp conf thấp sinh hàng nghìn hộp."""
    if not dets:
        return []
    dets = sorted(dets, key=lambda d: -d["score"])[:pre_nms_topk]
    b = np.array([d["box"] for d in dets], np.float32)
    s = np.array([d["score"] for d in dets], np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou < iou_thr]
    return [dets[k] for k in keep]


class PersonDetectorHailo:
    """Giao diện y hệt `detector.PersonDetector`, nhưng chạy trên NPU."""

    def __init__(self, hef, conf_threshold=0.30, nms_iou=0.45, img_size=640,
                 min_box_px=0.0, max_aspect=0.0, warmup=3):
        import cv2  # noqa: F401  (kiểm sớm, đỡ chết giữa chuyến bay)
        from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams,
                                    HailoStreamInterface, InputVStreamParams,
                                    OutputVStreamParams, FormatType)

        if not os.path.isfile(hef):
            raise FileNotFoundError(f"Không thấy HEF: {hef}")

        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.img_size = img_size
        self.min_box_px = min_box_px
        self.max_aspect = max_aspect
        self.last_ms = {}

        self._hef = HEF(hef)
        self._dev = VDevice()
        cfg = ConfigureParams.create_from_hef(self._hef,
                                              interface=HailoStreamInterface.PCIe)
        self._ng = self._dev.configure(self._hef, cfg)[0]
        self._in_name = self._hef.get_input_vstream_infos()[0].name

        in_shape = self._hef.get_input_vstream_infos()[0].shape
        if int(in_shape[0]) != img_size:
            raise ValueError(
                f"HEF nhận đầu vào {in_shape[0]} nhưng config đặt img_size={img_size}. "
                f"HEF là tĩnh, không co giãn được — sửa img_size cho khớp.")

        self._pipe = InferVStreams(
            self._ng,
            InputVStreamParams.make(self._ng, format_type=FormatType.UINT8),
            OutputVStreamParams.make(self._ng, format_type=FormatType.FLOAT32))
        self._pipe.__enter__()
        self._act = self._ng.activate()
        self._act.__enter__()

        # Lần suy luận đầu tốn thêm vài chục ms (nạp trọng số). Nuốt nó ở đây,
        # đừng để nó rơi vào khung hình đầu tiên lúc đang bay.
        blank = np.zeros((1, img_size, img_size, 3), np.uint8)
        for _ in range(max(1, warmup)):
            self._pipe.infer({self._in_name: blank})

        print(f"[DetectorHailo] {os.path.basename(hef)} @ {img_size}, "
              f"conf={conf_threshold}, NPU sẵn sàng")

    # ------------------------------------------------------------------
    def detect(self, frame_bgr):
        """BGR -> [{'box':[x1,y1,x2,y2] pixel ảnh gốc, 'score':float}]"""
        if frame_bgr is None:
            return []
        import cv2
        h0, w0 = frame_bgr.shape[:2]
        sz = self.img_size

        t0 = time.perf_counter()
        canvas, r, top, left = letterbox(frame_bgr, sz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]   # NHWC uint8 RGB
        t1 = time.perf_counter()

        out = self._pipe.infer({self._in_name: rgb})
        t2 = time.perf_counter()

        # Sắp theo diện tích giảm dần -> khớp thứ tự STRIDES (P2 lớn nhất trước).
        outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2], reverse=True)
        dets = []
        for si, o in enumerate(outs):
            raw = np.transpose(o[0], (2, 0, 1))               # (H,W,5) -> (5,H,W)
            dets += decode_scale(raw, STRIDES[si], sz, self.conf_threshold)
        dets = nms(dets, self.nms_iou)
        t3 = time.perf_counter()

        result = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1 = (x1 * sz - left) / r; y1 = (y1 * sz - top) / r
            x2 = (x2 * sz - left) / r; y2 = (y2 * sz - top) / r
            bw, bh = x2 - x1, y2 - y1
            if self.min_box_px and bw * bh < self.min_box_px:
                continue
            if self.max_aspect and max(bw / (bh + 1e-9), bh / (bw + 1e-9)) > self.max_aspect:
                continue
            result.append({"score": d["score"],
                           "box": [max(0.0, min(w0, x1)), max(0.0, min(h0, y1)),
                                   max(0.0, min(w0, x2)), max(0.0, min(h0, y2))]})

        self.last_ms = {"preprocess": (t1 - t0) * 1e3,
                        "inference": (t2 - t1) * 1e3,
                        "decode_nms": (t3 - t2) * 1e3}
        return result

    # ------------------------------------------------------------------
    def close(self):
        for obj, name in ((getattr(self, "_act", None), "_act"),
                          (getattr(self, "_pipe", None), "_pipe"),
                          (getattr(self, "_dev", None), "_dev")):
            if obj is None:
                continue
            try:
                obj.__exit__(None, None, None) if name != "_dev" else obj.release()
            except Exception:
                pass
            setattr(self, name, None)

    def __del__(self):
        self.close()
