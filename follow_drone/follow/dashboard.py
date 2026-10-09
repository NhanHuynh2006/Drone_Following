"""
dashboard.py — BẢNG ĐIỀU KHIỂN GIÁM SÁT nhúng thẳng vào vòng bay.

Vì sao nhúng chứ không chạy riêng: **NPU Hailo chỉ một tiến trình dùng được**
(`HAILO_OUT_OF_PHYSICAL_DEVICES` nếu mở lần hai), và kể cả có dùng chế độ chia sẻ
nhiều tiến trình của HailoRT thì hai tiến trình sẽ mỗi bên chạy inference riêng
trên cùng khung hình — 60 lượt mỗi giây trên con chip chỉ làm được 56, cả hai
cùng chậm đi. Gộp lại thì detect MỘT lần, dùng kết quả cho cả điều khiển lẫn hiển thị.

Dùng:
    dash = Dashboard(port=8080).start()
    ...
    dash.publish(frame_bgr, state_dict)     # gọi mỗi vòng lặp

CHỈ ĐỌC: trang này không có nút nào gửi lệnh xuống máy bay. Nó chỉ hiển thị.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_HTML_CANDIDATES = [
    os.path.join(_HERE, "gcs.html"),
    os.path.join(_HERE, "..", "gcs.html"),
    os.path.join(_HERE, "..", "..", "gcs.html"),
]


class Dashboard:
    def __init__(self, port=8080, quality=75, html_path=None, allow_retarget=True):
        self.port = port
        self.quality = quality
        self.html_path = html_path
        # NGOẠI LỆ DUY NHẤT của "chỉ đọc": cho phép bỏ khoá mục tiêu để chọn lại người.
        # Nó KHÔNG arm, KHÔNG đổi chế độ bay, KHÔNG gửi vận tốc — chỉ đổi xem bám ai,
        # trong đúng cùng một khung an toàn. Cần có vì ngoài bãi không có màn hình cắm
        # vào Pi, mà phím 'u' của cửa sổ OpenCV chỉ dùng được khi có màn hình.
        self.allow_retarget = allow_retarget
        self._retarget_req = False
        self._jpeg = None
        self._state = {}
        self._seq = 0
        self._cv = threading.Condition()
        self._srv = None
        self.running = False

    # ------------------------------------------------------------------
    def _html(self):
        cands = ([self.html_path] if self.html_path else []) + _HTML_CANDIDATES
        for p in cands:
            if p and os.path.isfile(p):
                with open(p, encoding="utf-8") as f:
                    return f.read().encode("utf-8")
        return (b"<h1>Thieu gcs.html</h1><p>Dat gcs.html canh main.py hoac truyen "
                b"html_path.</p><img src='/stream' style='max-width:100%'>")

    def take_retarget_request(self):
        """Vòng bay gọi mỗi khung: có yêu cầu đổi mục tiêu đang chờ không (và xoá cờ)."""
        with self._cv:
            req, self._retarget_req = self._retarget_req, False
        return req

    def publish(self, frame_bgr, state):
        """Đăng khung hình đã vẽ + trạng thái. Gọi mỗi vòng lặp, rất rẻ."""
        if frame_bgr is None:
            return
        ok, buf = cv2.imencode(".jpg", frame_bgr,
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return
        with self._cv:
            self._jpeg = buf.tobytes()
            self._state = state
            self._seq += 1
            self._cv.notify_all()

    def _wait(self, last_seq, timeout=5.0):
        with self._cv:
            if self._seq == last_seq:
                self._cv.wait(timeout)
            return self._jpeg, self._seq, dict(self._state)

    # ------------------------------------------------------------------
    def start(self):
        dash = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass                      # đừng spam terminal ngoài bãi bay

            def _send(self, body, ctype):
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self._send(dash._html(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    _, _, st = dash._wait(-1, 0.01)
                    st = dict(st)
                    st["ts"] = time.time()
                    st["allow_retarget"] = dash.allow_retarget
                    self._send(json.dumps(st).encode(), "application/json")
                elif path == "/api/retarget":
                    if not dash.allow_retarget:
                        self._send(b'{"ok":false,"reason":"tat trong config"}',
                                   "application/json")
                        return
                    with dash._cv:
                        dash._retarget_req = True
                    print("[Dashboard] người vận hành yêu cầu ĐỔI MỤC TIÊU")
                    self._send(b'{"ok":true}', "application/json")
                elif path == "/snapshot":
                    jpg, _, _ = dash._wait(-1, 5.0)
                    if jpg is None:
                        self.send_error(503); return
                    self._send(jpg, "image/jpeg")
                elif path == "/stream":
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Content-Type",
                                     "multipart/x-mixed-replace; boundary=FRAME")
                    self.end_headers()
                    seq = -1
                    try:
                        while dash.running:
                            jpg, seq, _ = dash._wait(seq)
                            if jpg is None:
                                continue
                            self.wfile.write(
                                b"--FRAME\r\nContent-Type: image/jpeg\r\n"
                                b"Content-Length: " + str(len(jpg)).encode() +
                                b"\r\n\r\n" + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        pass              # đóng tab — bình thường
                else:
                    # Mọi đường dẫn điều khiển khác đều bị chặn ở đây. Trang này chỉ
                    # đọc, trừ /api/retarget ở trên.
                    self.send_error(404)

        self._srv = ThreadingHTTPServer(("0.0.0.0", self.port), H)
        self._srv.daemon_threads = True
        self.running = True
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        print(f"[Dashboard] http://0.0.0.0:{self.port}  "
              f"(chỉ đọc{'; cho phép đổi mục tiêu' if self.allow_retarget else ''})")
        return self

    def stop(self):
        self.running = False
        with self._cv:
            self._cv.notify_all()
        if self._srv:
            try:
                self._srv.shutdown()
            except Exception:
                pass
