"""
demo_cam.py — Chạy model E (ONNX) trên webcam, vẽ box người real-time
=====================================================================

Dùng đúng tiền xử lý của training: letterbox(pad xám 114) -> RGB -> /255 -> CHW.

Chạy:
  python demo_cam.py --onnx deploy/pfdet_E_512.onnx --img-size 512 --cam 0 --conf 0.3
Phím:  q = thoát  |  + / - = tăng/giảm ngưỡng conf
"""

import os, sys, time, argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onnxruntime as ort
from utils import decode_predictions_np, nms_numpy

STRIDES = [4, 8, 16, 32]


def letterbox(img, size):
    """BGR frame -> canvas (size,size,3) pad xám 114, giữ tỉ lệ. Trả ratio/top/left để map box về."""
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, top, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="deploy/pfdet_E_512.onnx")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--nms-iou", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--save", default=None, help="lưu video out (vd out.mp4) thay vì hiện cửa sổ")
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    # FP16 cần tắt graph-opt
    if "fp16" in args.onnx:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    sz = args.img_size
    print(f"[onnx] {args.onnx} @ {sz}px | conf={args.conf}")

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"KHÔNG mở được cam {args.cam}"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # chống tích frame cũ -> giảm độ trễ (lag)
    writer = None
    conf = args.conf
    fps_ema = None

    while True:
        ok, frame = cap.read()
        if not ok:
            print("hết frame"); break

        canvas, r, top, left = letterbox(frame, sz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        x = (rgb.transpose(2, 0, 1)[None].astype(np.float32)) / 255.0

        t0 = time.perf_counter()
        outs = sess.run(out_names, {in_name: x})
        dt = time.perf_counter() - t0
        fps_ema = 1/dt if fps_ema is None else 0.9*fps_ema + 0.1*(1/dt)

        dets = []
        for si, raw in enumerate(outs):
            dets.extend(decode_predictions_np(raw[0], STRIDES[si], sz, conf_thr=conf))
        dets = nms_numpy(dets, iou_threshold=args.nms_iou)

        n = 0
        for d in dets:
            x1, y1, x2, y2 = d["box"]                      # normalized [0,1] trong letterbox
            # về pixel letterbox -> trừ pad -> chia ratio -> toạ độ frame gốc
            x1 = (x1*sz - left)/r; x2 = (x2*sz - left)/r
            y1 = (y1*sz - top)/r;  y2 = (y2*sz - top)/r
            x1, y1, x2, y2 = [int(v) for v in (x1, y1, x2, y2)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['score']:.2f}", (x1, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            n += 1

        cv2.putText(frame, f"{fps_ema:.1f} FPS | {n} nguoi | conf={conf:.2f} | {sz}px",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(frame)
        else:
            cv2.imshow("PFDet-E webcam (q=thoat, +/- conf)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"): break
            elif k == ord("+"): conf = min(0.9, conf + 0.05)
            elif k == ord("-"): conf = max(0.05, conf - 0.05)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
