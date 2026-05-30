"""Camera capture in background thread.

Drops old frames if main loop is slow → always read latest frame for low-latency control.
"""

import threading
import time
import cv2


class CameraThread:
    def __init__(self, source=0, width=640, height=480, fps=30):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.thread = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.frame_count = 0

    def start(self):
        if isinstance(self.source, str) and self.source.isdigit():
            self.source = int(self.source)

        self.cap = cv2.VideoCapture(self.source, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source: {self.source}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always latest frame

        # Warmup
        for _ in range(5):
            ret, _ = self.cap.read()
            if not ret:
                time.sleep(0.05)

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        print(f"[Camera] Started: {self.width}x{self.height} @ {self.fps} FPS")

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = time.time()
                self.frame_count += 1

    def get_frame(self):
        """Return (frame, timestamp) or (None, 0) if not ready."""
        with self.lock:
            if self.latest_frame is None:
                return None, 0
            return self.latest_frame.copy(), self.latest_timestamp

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("[Camera] Stopped")
