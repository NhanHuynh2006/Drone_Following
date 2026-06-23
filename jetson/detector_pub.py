"""
detector_pub.py — Chạy model (ONNX) trên Jetson, ĐẨY target qua UDP cho bộ điều khiển C++ (follow_px4).
======================================================================================================

Tách detection (Python/ONNX, nặng) khỏi control (C++, mượt 50Hz) — kiến trúc companion-computer chuẩn.
Gói UDP khớp struct TargetMsg trong follow_px4.cpp: 8 x float32 [cx,cy,w,h,distance_m,conf,t_capture,valid].

Chạy:
  python3 detector_pub.py --onnx pfdet_E_512.onnx --img-size 512 --cam 0 \
      --ctrl-ip 127.0.0.1 --ctrl-port 5600 --person-h 1.7

Lưu ý: chọn target = NGƯỜI TO NHẤT + khoá (lock) để bám ổn định. Đổi --strategy center nếu muốn giữ giữa.
"""
import os, sys, time, socket, struct, argparse
import numpy as np
import cv2

import threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import onnxruntime as ort
from utils import decode_predictions_np, nms_numpy

STRIDES = [4, 8, 16, 32]


class CameraGrabber:
    """Luồng RIÊNG đọc camera liên tục, chỉ giữ FRAME MỚI NHẤT (bỏ frame cũ).
    -> inference luôn xử lý frame tươi nhất, không nghẽn, latency thấp nhất."""

    def __init__(self, cam, width=None, height=None, fps=None):
        self.cap = cv2.VideoCapture(cam)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:  self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:    self.cap.set(cv2.CAP_PROP_FPS, fps)
        self._lock = threading.Lock()
        self._frame = None
        self._t_cap = 0.0
        self._run = True
        self._n = 0
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def opened(self):
        return self.cap.isOpened()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005); continue
            with self._lock:
                self._frame = f
                self._t_cap = time.time()
                self._n += 1

    def latest(self):
        """Trả (frame, t_capture, frame_id) mới nhất; None nếu chưa có."""
        with self._lock:
            if self._frame is None:
                return None, 0.0, -1
            return self._frame, self._t_cap, self._n

    def release(self):
        self._run = False
        time.sleep(0.05)
        self.cap.release()


def letterbox(img, size):
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(h0 * r), int(w0 * r)
    rs = cv2.resize(img, (nw, nh))
    cv = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    cv[top:top+nh, left:left+nw] = rs
    return cv, r, top, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="pfdet_E_512.onnx")
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--cam", default="0")
    ap.add_argument("--ctrl-ip", default="127.0.0.1")
    ap.add_argument("--ctrl-port", type=int, default=5600)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--person-h", type=float, default=1.7, help="chiều cao người (m) cho pinhole")
    ap.add_argument("--fy", type=float, default=700.0, help="tiêu cự dọc px (calibrate!)")
    ap.add_argument("--strategy", default="largest", choices=["largest", "center"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    so = ort.SessionOptions(); so.intra_op_num_threads = args.threads
    if "fp16" in args.onnx:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(args.onnx, sess_options=so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    sz = args.img_size

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = (args.ctrl_ip, args.ctrl_port)

    cam = int(args.cam) if args.cam.isdigit() else args.cam
    grab = CameraGrabber(cam)                      # LUỒNG camera riêng (giữ frame mới nhất)
    if not grab.opened():
        print("KHÔNG mở được camera", cam); return

    print(f"[pub] model={args.onnx}@{sz} -> UDP {dst} | strategy={args.strategy} | camera ở luồng riêng")
    locked_cx = None
    last_fid = -1
    n_infer, t0 = 0, time.time()
    while True:
        frame, t_cap, fid = grab.latest()
        if frame is None or fid == last_fid:       # chưa có / frame trùng -> bỏ qua (không infer lại)
            time.sleep(0.001); continue
        last_fid = fid
        H, W = frame.shape[:2]
        canvas, r, top, left = letterbox(frame, sz)
        x = (cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32)) / 255.0
        outs = sess.run(out_names, {in_name: x})

        dets = []
        for si, raw in enumerate(outs):
            dets.extend(decode_predictions_np(raw[0], STRIDES[si], sz, conf_thr=args.conf))
        dets = nms_numpy(dets, iou_threshold=0.5)

        # map về pixel ảnh gốc
        boxes = []
        for d in dets:
            x1, y1, x2, y2 = d["box"]
            x1 = (x1*sz - left)/r; x2 = (x2*sz - left)/r
            y1 = (y1*sz - top)/r;  y2 = (y2*sz - top)/r
            boxes.append((x1, y1, x2, y2, d["score"]))

        # chọn target
        tgt = None
        if boxes:
            if args.strategy == "center":
                cxc = W / 2
                tgt = min(boxes, key=lambda b: abs((b[0]+b[2])/2 - cxc))
            else:  # largest, ưu tiên gần locked_cx để ổn định
                def keyf(b):
                    area = (b[2]-b[0])*(b[3]-b[1])
                    pen = 0 if locked_cx is None else -abs((b[0]+b[2])/2 - locked_cx)
                    return area + pen * 50
                tgt = max(boxes, key=keyf)

        if tgt is not None:
            x1, y1, x2, y2, conf = tgt
            cx, cy = (x1+x2)/2, (y1+y2)/2
            bw, bh = max(1.0, x2-x1), max(1.0, y2-y1)
            locked_cx = cx
            dist = args.person_h * args.fy / bh    # pinhole: D = H_real * f / h_px
            msg = struct.pack("<8f", cx, cy, bw, bh, dist, conf, t_cap, 1.0)
            if args.show:
                cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
                cv2.putText(frame, f"d={dist:.1f}m c={conf:.2f}", (int(x1),int(y1)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        else:
            locked_cx = None
            msg = struct.pack("<8f", 0,0,0,0,0,0, t_cap, 0.0)   # valid=0
        sock.sendto(msg, dst)

        n_infer += 1
        if n_infer % 60 == 0:
            fps = 60.0 / (time.time() - t0); t0 = time.time()
            print(f"[pub] inference {fps:.1f} FPS")
        if args.show:
            cv2.imshow("detector_pub (q=thoat)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    grab.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
