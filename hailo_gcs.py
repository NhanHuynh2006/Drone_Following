"""
hailo_gcs.py — TRAM MAT DAT tren Pi 5 + Hailo-8L.  ArduPilot Copter.
====================================================================

Mo tren laptop/dien thoai cung WiFi:   http://<ip-cua-pi>:8080

Gop 3 thu vao mot man hinh de ngoai bai chi can nhin mot cho:
  1. Hinh camera + box detect (MJPEG, khong can man hinh/X11 tren Pi)
  2. Telemetry ArduPilot: ARM/DISARM, che do bay, do cao, pin, GPS, tu the
  3. Suc khoe he thong: FPS that cua camera, latency NPU tung chang, nhiet, throttle

CHI DOC telemetry — trang nay KHONG arm, KHONG doi che do, KHONG gui setpoint.
Dieu khien bay nam o follow_drone/ (co safety ladder rieng).

CHAY:
    ./run.sh gcs                                   # khong co MAVLink
    ./run.sh gcs --mavlink /dev/serial0 --baud 57600
    ./run.sh gcs --mavlink udp:0.0.0.0:14550       # qua mavproxy/SITL

ArduPilot: che do dieu khien tu companion la **GUIDED** (khong phai OFFBOARD nhu PX4).
Ten che do lay tu chinh xe qua mode_mapping() nen khong bao gio lech phien ban.
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import cv2
import numpy as np
from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams,
                            HailoStreamInterface, InputVStreamParams,
                            OutputVStreamParams, FormatType)

from flight_record import STRIDES, decode_scale, nms, letterbox, soc_temp_c, throttled
from cam_ctl import SoftAE, apply_locked_fps, max_exposure_for_fps
from mav_telem import MavTelem

HERE = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────────────────────────────────────── trang thai chia se
class Shared:
    def __init__(self):
        self.jpeg = None
        self.seq = 0
        self.cv = threading.Condition()
        self.conf = 0.30                 # 0.30 = diem F1 tot nhat do tren flight_val
        self.stats = {}
        self.models = []
        self.model_idx = 0
        self.want_model = None           # index model muon chuyen sang
        self.draw = True
        self.run = True

    def publish(self, jpeg, stats):
        with self.cv:
            self.jpeg, self.stats = jpeg, stats
            self.seq += 1
            self.cv.notify_all()

    def wait_next(self, last_seq, timeout=5.0):
        with self.cv:
            if self.seq == last_seq:
                self.cv.wait(timeout)
            return self.jpeg, self.seq, dict(self.stats)


SH = Shared()
MAV = MavTelem()


# ────────────────────────────────────────────────────────────────────── camera
class Cam:
    def __init__(self, cam, w, h, fps):
        self.cap = cv2.VideoCapture(int(cam) if str(cam).isdigit() else cam, cv2.CAP_V4L2)
        # YUYV 720p nghen bang thong USB2 -> 5 fps. Phai ep MJPG.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # BUFFERSIZE=1 rot DUNG MOT NUA khung (30->15 fps). Xem
        # runs/pi_camera_measured_2026-08-27.md. Dung 2.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if w: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if fps: self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.f, self.t, self.n, self.run = None, 0.0, 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            ok, f = self.cap.read()
            if ok:
                self.f, self.t, self.n = f, time.time(), self.n + 1
            else:
                time.sleep(0.005)

    def read(self):
        return self.f, self.t, self.n

    def opened(self):
        return self.cap.isOpened()

    def props(self):
        return (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                float(self.cap.get(cv2.CAP_PROP_FPS)))

    def release(self):
        self.run = False
        time.sleep(0.05)
        self.cap.release()


# ─────────────────────────────────────────────────────────────── vong xu ly anh
def detect_loop(args):
    cam = Cam(args.cam, args.width, args.height, args.fps)
    time.sleep(1.0)
    if not cam.opened():
        print(f"[gcs] KHONG mo duoc camera {args.cam}")
        SH.run = False
        return
    w, h, drv_fps = cam.props()
    print(f"[gcs] camera {w}x{h} (driver bao {drv_fps:.0f} fps)")

    max_exp = args.max_exposure or max_exposure_for_fps(args.fps)
    ae = None
    if args.ae:
        ae = SoftAE(dev=args.dev, max_exposure=max_exp, exposure=args.exposure,
                    gain=args.gain, prefer_gain=args.prefer_gain)
        print(f"[gcs] SoftAE bat, tran phoi sang {max_exp} (={max_exp/10:.1f} ms)")
    else:
        apply_locked_fps(args.dev, min(args.exposure, max_exp), args.gain)
    ae_e, ae_g, ae_l = args.exposure, args.gain, 0.0

    enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
    fps_ema = cam_fps = 0.0
    n_prev, t_prev_cap = -1, 0.0
    last_dets, ms = [], {}

    while SH.run:
        # ---- (nap lai) model
        idx = SH.want_model if SH.want_model is not None else SH.model_idx
        SH.want_model = None
        SH.model_idx = idx
        m = SH.models[idx]
        print(f"[gcs] nap model: {m['name']} @ {m['img_size']}")
        hef = HEF(m["path"])
        sz = m["img_size"]

        with VDevice() as dev:
            cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            ng = dev.configure(hef, cfg)[0]
            in_name = hef.get_input_vstream_infos()[0].name
            with InferVStreams(ng,
                               InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                               OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)) as pipe, \
                 ng.activate():
                while SH.run and SH.want_model is None:
                    frame, t_cap, n = cam.read()
                    if frame is None or n == n_prev:
                        time.sleep(0.002)
                        continue
                    if t_prev_cap:
                        inst = 1.0 / max(1e-6, t_cap - t_prev_cap)
                        cam_fps = inst if cam_fps == 0 else 0.9 * cam_fps + 0.1 * inst
                    n_prev, t_prev_cap = n, t_cap
                    frame = frame.copy()

                    if ae is not None:
                        r = ae.step(frame)
                        if r:
                            ae_e, ae_g, ae_l = r

                    t0 = time.perf_counter()
                    canvas, r_, top, left = letterbox(frame, sz)
                    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                    t1 = time.perf_counter()
                    out = pipe.infer({in_name: rgb})
                    t2 = time.perf_counter()
                    outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2],
                                  reverse=True)
                    dets = []
                    for si, o in enumerate(outs):
                        dets += decode_scale(np.transpose(o[0], (2, 0, 1)),
                                             STRIDES[si], sz, SH.conf)
                    dets = nms(dets, 0.5)
                    t3 = time.perf_counter()

                    last_dets = []
                    for d in dets:
                        x1, y1, x2, y2 = d["box"]
                        last_dets.append((int((x1 * sz - left) / r_), int((y1 * sz - top) / r_),
                                          int((x2 * sz - left) / r_), int((y2 * sz - top) / r_),
                                          d["score"]))
                    ms = {"preprocess": round((t1 - t0) * 1e3, 2),
                          "inference": round((t2 - t1) * 1e3, 2),
                          "decode_nms": round((t3 - t2) * 1e3, 2)}
                    ms["total"] = round(sum(ms.values()), 2)

                    if SH.draw:
                        H, W = frame.shape[:2]
                        for x1, y1, x2, y2, sc in last_dets:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 120), 2)
                            lbl = f"{sc:.2f}"
                            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)),
                                          (x1 + tw + 6, y1), (80, 220, 120), -1)
                            cv2.putText(frame, lbl, (x1 + 3, max(10, y1 - 4)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 18, 14), 1,
                                        cv2.LINE_AA)
                        # thap tam de ngam huong bay
                        cx, cy = W // 2, H // 2
                        cv2.line(frame, (cx - 18, cy), (cx - 6, cy), (120, 200, 255), 1)
                        cv2.line(frame, (cx + 6, cy), (cx + 18, cy), (120, 200, 255), 1)
                        cv2.line(frame, (cx, cy - 18), (cx, cy - 6), (120, 200, 255), 1)
                        cv2.line(frame, (cx, cy + 6), (cx, cy + 18), (120, 200, 255), 1)

                    now = time.perf_counter()
                    fps_ema = cam_fps

                    temp = soc_temp_c()
                    ok, buf = cv2.imencode(".jpg", frame, enc)
                    if ok:
                        best = max((d[4] for d in last_dets), default=0.0)
                        SH.publish(buf.tobytes(), {
                            "cam_fps": round(cam_fps, 1),
                            "dets": len(last_dets),
                            "best_conf": round(best, 2),
                            "boxes": [[a, b, c, d, round(e, 3)] for a, b, c, d, e in last_dets],
                            "frame_wh": [frame.shape[1], frame.shape[0]],
                            "conf": SH.conf,
                            "temp_C": round(temp, 1),
                            "throttled": throttled(),
                            "frames": n,
                            "ms": ms,
                            "exposure": ae_e, "gain": ae_g, "luma": round(ae_l, 1),
                            "model": m["name"], "img_size": sz,
                            "models": [x["name"] for x in SH.models],
                            "model_idx": idx,
                        })
        if not SH.run:
            break
    cam.release()


# ────────────────────────────────────────────────────────────────────────── web
def load_page():
    p = os.path.join(HERE, "gcs.html")
    with open(p, encoding="utf-8") as f:
        return f.read().encode("utf-8")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json"):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)

        if u.path == "/":
            try:
                self._send(load_page(), "text/html; charset=utf-8")
            except FileNotFoundError:
                self._send(b"thieu gcs.html", "text/plain; charset=utf-8")

        elif u.path == "/api/state":
            _, _, st = SH.wait_next(-1, 0.01)
            st["mav"] = MAV.get()
            st["ts"] = time.time()
            self._send(json.dumps(st).encode())

        elif u.path == "/api/conf":
            try:
                SH.conf = max(0.01, min(0.95, float(q.get("v", ["0.3"])[0])))
            except ValueError:
                pass
            self._send(json.dumps({"conf": SH.conf}).encode())

        elif u.path == "/api/model":
            try:
                i = int(q.get("i", ["0"])[0])
                if 0 <= i < len(SH.models) and i != SH.model_idx:
                    SH.want_model = i
            except ValueError:
                pass
            self._send(json.dumps({"model_idx": SH.want_model
                                   if SH.want_model is not None else SH.model_idx}).encode())

        elif u.path == "/api/draw":
            SH.draw = q.get("v", ["1"])[0] not in ("0", "false")
            self._send(json.dumps({"draw": SH.draw}).encode())

        elif u.path == "/snapshot":
            jpg, _, _ = SH.wait_next(-1, 5.0)
            if jpg is None:
                self.send_error(503)
                return
            self._send(jpg, "image/jpeg")

        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            seq = -1
            try:
                while SH.run:
                    jpg, seq, _ = SH.wait_next(seq)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() +
                                     b"\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/pfdet_v17_640_hailo8l.hef:640,"
                                        "models/pfdet_v17_640_fair_hailo8l.hef:640,"
                                        "models/pfdet_v17_320_hailo8l.hef:320",
                    help="danh sach hef:imgsize, phan cach bang dau phay")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--cam", default="0")
    ap.add_argument("--dev", default="/dev/video0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80)
    ap.add_argument("--ae", dest="ae", action="store_true", default=True)
    ap.add_argument("--no-ae", dest="ae", action="store_false")
    ap.add_argument("--max-exposure", type=int, default=0)
    ap.add_argument("--exposure", type=int, default=120)
    ap.add_argument("--gain", type=int, default=32)
    ap.add_argument("--prefer-gain", action="store_true")
    ap.add_argument("--mavlink", default=None,
                    help="/dev/serial0 | udp:0.0.0.0:14550 | tcp:127.0.0.1:5760")
    ap.add_argument("--baud", type=int, default=57600)
    a = ap.parse_args()

    SH.conf = a.conf
    SH.models = []
    for spec in a.models.split(","):
        spec = spec.strip()
        if not spec:
            continue
        path, _, size = spec.partition(":")
        if not os.path.exists(path):
            print(f"[gcs] bo qua (khong co file): {path}")
            continue
        SH.models.append(dict(path=path, img_size=int(size or 640),
                              name=os.path.basename(path).replace("_hailo8l.hef", "")))
    if not SH.models:
        print("[gcs] khong co model nao dung duoc."); return
    print(f"[gcs] {len(SH.models)} model: " + ", ".join(m["name"] for m in SH.models))

    global MAV
    MAV = MavTelem(a.mavlink, a.baud)
    if a.mavlink:
        print(f"[gcs] MAVLink (ArduPilot, CHI DOC): {a.mavlink} @ {a.baud}")
        MAV.start()
    else:
        print("[gcs] khong co --mavlink -> panel bay se hien 'NO LINK'")

    threading.Thread(target=detect_loop, args=(a,), daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    srv.daemon_threads = True
    print(f"[gcs] http://0.0.0.0:{a.port}   (Ctrl-C de dung)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[gcs] dung.")
    finally:
        SH.run = False
        MAV.stop()
        with SH.cv:
            SH.cv.notify_all()
        srv.shutdown()


if __name__ == "__main__":
    main()
