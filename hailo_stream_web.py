"""
hailo_stream_web.py — CHẠY TRÊN PI 5. Phát hình camera + box detect lên TRÌNH DUYỆT.
=====================================================================================

Giống `hailo_demo_cam.py` nhưng KHÔNG cần màn hình / X11: nó dựng một web server nhỏ,
đẩy MJPEG. Mở trên laptop (hoặc điện thoại cùng WiFi):

    http://<ip-cua-pi>:8080

Chạy:
    ./run.sh stream                      # mặc định HEF 640, cam 0, cổng 8080
    ./run.sh stream --img-size 320 --hef models/pfdet_v17_320_hailo8l.hef
    ./run.sh stream --conf 0.3 --quality 70 --detect-every 2

Vì sao MJPEG chứ không X11-forward: X11 đẩy pixel THÔ (640x480 BGR ~ 13 MB/s ở 15fps),
qua WiFi là giật. MJPEG nén JPEG từng khung nên chỉ tốn ~0.5-1.5 MB/s, và mở được ở
bất kỳ trình duyệt nào, không cần cài gì trên máy xem.

Camera chạy LUỒNG RIÊNG (giữ khung mới nhất) — giống flight_record.py — để việc
detect không làm trễ việc thu, và ngược lại.
"""
import argparse
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


# --------------------------------------------------------------- trạng thái chia sẻ
class Shared:
    """Khung JPEG mới nhất + số liệu, chia sẻ giữa luồng detect và các luồng HTTP."""

    def __init__(self):
        self.jpeg = None
        self.seq = 0
        self.cv = threading.Condition()
        self.conf = 0.4
        self.stats = {}
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


# --------------------------------------------------------------- camera (luồng riêng)
class Cam:
    def __init__(self, cam, w, h, fps):
        self.cap = cv2.VideoCapture(int(cam) if str(cam).isdigit() else cam,
                                    cv2.CAP_V4L2)
        # MJPG cho phép C270 chạy 720p@30; YUYV 720p bị nghẽn băng thông USB (5 fps).
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # BUFFERSIZE=1 làm RỚT ĐÚNG MỘT NỬA khung (30->15 fps): một buffer thì driver
        # không có chỗ ghi khung kế trong lúc app đang giữ buffer. Đo trên C270/Pi5
        # 2026-08-27. Dùng 2 — vẫn tươi vì luồng _loop luôn ghi đè khung mới nhất
        # (đo được: khung cũ p50 15.7 ms, p95 31.5 ms = nửa chu kỳ khung, đúng kỳ vọng).
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


# --------------------------------------------------------------- vòng detect
def detect_loop(args):
    cam = Cam(args.cam, args.width, args.height, args.fps)
    if not cam.opened():
        print(f"KHONG mo duoc camera {args.cam}")
        SH.run = False
        return
    w, h, fps = cam.props()
    print(f"[stream] camera {w}x{h} @{fps:.0f}fps (driver)")

    # Auto-exposure PHAN MEM: giu sang ma KHONG cho camera tu ha fps.
    # (C270 o auto-exposure keo phoi sang > 1/30s khi thieu sang -> tut con 15 fps.)
    max_exp = args.max_exposure or max_exposure_for_fps(args.fps)
    if args.ae:
        ae = SoftAE(dev=args.dev, max_exposure=max_exp, exposure=args.exposure,
                    gain=args.gain, prefer_gain=args.prefer_gain)
        print(f"[stream] SoftAE bat | tran phoi sang {max_exp} "
              f"(={max_exp/10:.1f} ms) -> fps khong tut duoi {args.fps}")
    else:
        ae = None
        apply_locked_fps(args.dev, min(args.exposure, max_exp), args.gain)
    ae_e, ae_g, ae_l = args.exposure, args.gain, 0.0

    sz = args.img_size
    hef = HEF(args.hef)
    with VDevice() as dev:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = dev.configure(hef, cfg)[0]
        in_name = hef.get_input_vstream_infos()[0].name
        in_vp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_vp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)

        enc = [int(cv2.IMWRITE_JPEG_QUALITY), args.quality]
        fps_ema, last_dets, i = None, [], 0
        t_prev = time.perf_counter()
        n_prev, t_cap_prev, cam_fps = -1, 0.0, 0.0

        with InferVStreams(ng, in_vp, out_vp) as pipe, ng.activate():
            print(f"[stream] http://0.0.0.0:{args.port}  (Ctrl-C de dung)")
            while SH.run:
                frame, t_cap, n = cam.read()
                if frame is None:
                    time.sleep(0.01)
                    continue
                if n == n_prev:                     # camera chua co khung moi
                    time.sleep(0.002)
                    continue
                cam_fps = 0.9 * cam_fps + 0.1 * (1.0 / max(1e-6, t_cap - t_cap_prev)) \
                          if t_cap_prev else 0.0
                n_prev, t_cap_prev = n, t_cap
                frame = frame.copy()
                i += 1

                if ae is not None:
                    r = ae.step(frame)
                    if r:
                        ae_e, ae_g, ae_l = r

                ms = {}
                if i % max(1, args.detect_every) == 0:
                    t0 = time.perf_counter()
                    canvas, r, top, left = letterbox(frame, sz)
                    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                    t1 = time.perf_counter()
                    out = pipe.infer({in_name: rgb})
                    t2 = time.perf_counter()

                    outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2], reverse=True)
                    dets = []
                    for si, o in enumerate(outs):
                        raw = np.transpose(o[0], (2, 0, 1))       # (H,W,5) -> (5,H,W)
                        dets += decode_scale(raw, STRIDES[si], sz, SH.conf)
                    dets = nms(dets, 0.5)
                    t3 = time.perf_counter()

                    # đưa box từ canvas letterbox về toạ độ pixel của khung gốc
                    last_dets = []
                    for d in dets:
                        x1, y1, x2, y2 = d["box"]
                        last_dets.append((int((x1 * sz - left) / r), int((y1 * sz - top) / r),
                                          int((x2 * sz - left) / r), int((y2 * sz - top) / r),
                                          d["score"]))
                    ms = {"preprocess": round((t1 - t0) * 1e3, 2),
                          "inference": round((t2 - t1) * 1e3, 2),
                          "decode_nms": round((t3 - t2) * 1e3, 2)}

                for x1, y1, x2, y2, sc in last_dets:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{sc:.2f}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                now = time.perf_counter()
                inst = 1.0 / max(1e-6, now - t_prev)
                t_prev = now
                fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

                temp = soc_temp_c()
                cv2.putText(frame, f"{cam_fps:4.1f} FPS cam | {len(last_dets)} nguoi | "
                                   f"conf={SH.conf:.2f} | {temp:.0f}C",
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.putText(frame, f"exp={ae_e} gain={ae_g} luma={ae_l:.0f} | "
                                   f"infer {ms.get('inference', 0):.1f}ms",
                            (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

                ok, buf = cv2.imencode(".jpg", frame, enc)
                if ok:
                    SH.publish(buf.tobytes(),
                               {"fps": round(fps_ema, 1), "dets": len(last_dets),
                                "conf": SH.conf, "temp_C": round(temp, 1),
                                "throttled": throttled(), "frames": n, "ms": ms,
                                "exposure": ae_e, "gain": ae_g, "luma": round(ae_l, 1),
                                "cam_fps": round(cam_fps, 1)})
    cam.release()


# --------------------------------------------------------------- web
PAGE = """<!doctype html><meta charset=utf-8><title>SteadyDet-N · Pi 5 + Hailo</title>
<style>
 body{margin:0;background:#0d1117;color:#e6edf3;font:14px/1.5 system-ui,sans-serif;
      display:flex;flex-direction:column;align-items:center;gap:12px;padding:16px}
 h1{font-size:16px;font-weight:600;margin:0;color:#7ee787}
 img{max-width:100%;border:1px solid #30363d;border-radius:8px;background:#000}
 .bar{display:flex;gap:16px;align-items:center;flex-wrap:wrap;justify-content:center}
 .k{color:#8b949e}
 input[type=range]{width:180px}
</style>
<h1>SteadyDet-N · Raspberry Pi 5 + Hailo-8L</h1>
<img id=v src="/stream">
<div class=bar>
  <span><span class=k>conf</span> <input type=range id=c min=5 max=90 value=40>
        <span id=cv>0.40</span></span>
  <span class=k>FPS</span><b id=f>-</b>
  <span class=k>nguoi</span><b id=d>-</b>
  <span class=k>nhiet</span><b id=t>-</b>
  <span class=k>infer</span><b id=i>-</b>
  <span class=k>exp/gain</span><b id=e>-</b>
</div>
<script>
const c=document.getElementById('c');
c.oninput=()=>{const v=(c.value/100).toFixed(2);document.getElementById('cv').textContent=v;
                fetch('/conf?v='+v)};
setInterval(async()=>{try{const s=await (await fetch('/stats')).json();
  document.getElementById('f').textContent=s.fps;
  document.getElementById('d').textContent=s.dets;
  document.getElementById('t').textContent=s.temp_C+'C';
  document.getElementById('i').textContent=(s.ms&&s.ms.inference?s.ms.inference+'ms':'-');
  document.getElementById('f').textContent=s.cam_fps;
  document.getElementById('e').textContent=s.exposure+'/'+s.gain;
}catch(e){}},1000);
</script>
"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                   # đừng spam terminal ngoài bãi

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif u.path == "/conf":
            v = parse_qs(u.query).get("v", ["0.4"])[0]
            try:
                SH.conf = max(0.01, min(0.95, float(v)))
            except ValueError:
                pass
            self._json(b'{"ok":true}')

        elif u.path == "/stats":
            import json
            _, _, st = SH.wait_next(-1, 0.01)
            self._json(json.dumps(st).encode())

        elif u.path == "/snapshot":
            jpg, _, _ = SH.wait_next(-1, 5.0)
            if jpg is None:
                self.send_error(503); return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpg)))
            self.end_headers()
            self.wfile.write(jpg)

        elif u.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            seq = -1
            try:
                while SH.run:
                    jpg, seq, _ = SH.wait_next(seq)
                    if jpg is None:
                        continue
                    self.wfile.write(b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n")
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass                            # trình duyệt đóng tab — bình thường
        else:
            self.send_error(404)

    def _json(self, body):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="models/pfdet_v17_640_hailo8l.hef")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--cam", default="0")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--quality", type=int, default=80, help="chat luong JPEG 1-100")
    ap.add_argument("--detect-every", type=int, default=1, help="detect 1 khung moi N khung")
    ap.add_argument("--dev", default="/dev/video0", help="node v4l2 de chinh exposure/gain")
    ap.add_argument("--ae", dest="ae", action="store_true", default=True,
                    help="auto-exposure PHAN MEM (mac dinh bat) — sang ma khong tut fps")
    ap.add_argument("--no-ae", dest="ae", action="store_false")
    ap.add_argument("--max-exposure", type=int, default=0,
                    help="tran phoi sang, don vi 100us. 0 = tinh tu --fps (30fps -> 333)")
    ap.add_argument("--exposure", type=int, default=120, help="phoi sang khoi diem")
    ap.add_argument("--gain", type=int, default=32, help="gain khoi diem")
    ap.add_argument("--prefer-gain", action="store_true",
                    help="LUC BAY: giu shutter ngan chong nhoe, de gain ganh phan sang")
    args = ap.parse_args()

    SH.conf = args.conf
    th = threading.Thread(target=detect_loop, args=(args,), daemon=True)
    th.start()

    srv = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[stream] dung.")
    finally:
        SH.run = False
        with SH.cv:
            SH.cv.notify_all()
        srv.shutdown()


if __name__ == "__main__":
    main()
