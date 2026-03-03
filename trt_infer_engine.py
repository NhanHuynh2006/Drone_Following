"""
TensorRT inference engine for PFDet-Nano on Jetson Nano.

Uses TensorRT for ~2-3x speedup over PyTorch on Jetson Nano.
Typical performance: 20-30 FPS @ 416x416 with FP16 on Jetson Nano B01.

Usage:
  1. First export: python export.py --weights best.pt --format all --fp16
  2. Build engine on Jetson: python3 best_build_trt.py
  3. Run: python3 trt_infer_engine.py --engine best_fp16.engine --camera 0
"""

import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.box_ops import decode_predictions_np, nms_numpy


class TRTEngine:
    """
    TensorRT engine wrapper for PFDet-Nano.
    Handles memory allocation, inference, and output parsing.
    """
    def __init__(self, engine_path, img_size=416):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa - initializes CUDA context

        self.img_size = img_size
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        print(f"[TRT] Loading engine: {engine_path}")
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            size = int(np.prod(shape))

            # Allocate host and device memory
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

        print(f"[TRT] Engine loaded. Input: {self.inputs[0]['shape']}, "
              f"Outputs: {[o['shape'] for o in self.outputs]}")

    def infer(self, img_np):
        """
        Run inference on preprocessed image.
        img_np: (1, 3, H, W) float32 numpy array, normalized [0,1]
        Returns: list of raw outputs [(7, H1, W1), (7, H2, W2), (7, H3, W3)]
        """
        import pycuda.driver as cuda

        # Copy input
        np.copyto(self.inputs[0]['host'], img_np.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy outputs
        results = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        for out in self.outputs:
            result = out['host'].reshape(out['shape'])
            results.append(result[0])  # Remove batch dim

        return results


def run_trt_camera(engine_path, camera_source=0, img_size=416,
                   strides=[8, 16, 32], conf_thr=0.35, show=True):
    """Run TensorRT inference on camera feed."""
    engine = TRTEngine(engine_path, img_size)

    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera: {camera_source}")
        return

    fps_smooth = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # Preprocess
        h0, w0 = frame.shape[:2]
        ratio = min(img_size / h0, img_size / w0)
        nh, nw = int(h0 * ratio), int(w0 * ratio)
        resized = cv2.resize(frame, (nw, nh))

        pad_h, pad_w = img_size - nh, img_size - nw
        top, left = pad_h // 2, pad_w // 2

        padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
        padded[top:top+nh, left:left+nw] = resized

        # To float, CHW, batch
        inp = padded.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        # Inference
        outputs = engine.infer(inp)

        # Decode
        all_dets = []
        for si, raw in enumerate(outputs):
            dets = decode_predictions_np(raw, strides[si], img_size)
            all_dets.extend(dets)

        all_dets = [d for d in all_dets if d['score'] >= conf_thr]
        all_dets = nms_numpy(all_dets, iou_threshold=0.45)

        dt = time.time() - t0
        fps_now = 1.0 / max(dt, 1e-6)
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now

        # Draw
        for det in all_dets:
            x1, y1, x2, y2 = det['box']
            x1 = int((x1 * img_size - left) / ratio)
            y1 = int((y1 * img_size - top) / ratio)
            x2 = int((x2 * img_size - left) / ratio)
            y2 = int((y2 * img_size - top) / ratio)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['score']:.2f}", (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.putText(frame, f"FPS: {fps_smooth:.1f} | TRT FP16", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        if show:
            cv2.imshow("PFDet-Nano TRT", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="TensorRT engine path")
    parser.add_argument("--camera", default="0", help="Camera source")
    parser.add_argument("--img-size", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()

    cam = int(args.camera) if args.camera.isdigit() else args.camera
    run_trt_camera(args.engine, cam, args.img_size, conf_thr=args.conf)
