"""
flight_record.py — CHẠY TRÊN PI 5. Thu data chuyến bay, đúng theo FLIGHT_DATA_SPEC.md.
=======================================================================================

Gộp 3 việc mà spec đòi, vào 1 lệnh duy nhất, để ngoài bãi chỉ cần gõ 1 dòng:

  1. Thu VIDEO gốc từ camera (KHÔNG phải quay màn hình GCS) + tách frame theo nhịp.
  2. Chạy detector Hailo NGAY LÚC BAY, ghi jsonl: timestamp mỗi frame, box dự đoán,
     latency từng chặng (capture / preprocess / inference / decode+NMS).
  3. Ghi sức khoẻ Pi suốt chuyến: nhiệt SoC, vcgencmd get_throttled, tần số CPU.

Và ghi luôn METADATA lượt bay (độ cao, người, cảnh, chế độ bay, shutter/ISO) vào
manifest.json — thay cho "ghi giấy" trong spec, thứ mà bay xong hay quên nhất.

CHẠY (mỗi LƯỢT BAY một lệnh, đổi --alt/--scene cho đúng):

  # Lượt bay có detector (cần Hailo):
  python3 flight_record.py --alt 5 --scene grass --persons 3 --mode hover \
      --shutter 1/1000 --iso 200 --detect --hef models/pfdet_v17_640_hailo8l.hef --img-size 640

  # Lượt bay CHỈ thu data, không cần Hailo (nhẹ CPU hơn, video mượt hơn):
  python3 flight_record.py --alt 12 --scene concrete --persons 3 --mode forward --no-detect

  # Camera CSI (Camera Module) thay vì USB:
  python3 flight_record.py --source csi --alt 3 ...

BÀN PHÍM (gõ vào terminal rồi Enter — chạy được cả khi ssh, không cần màn hình):
  m  = đánh MỐC ĐỒNG BỘ (dùng lúc lắc drone sau khi arm -> căn video với .ulg)
  q  = dừng lượt bay

Ra thư mục  flight_data/<session>/ :
  video.mp4        video gốc toàn lượt
  images/*.jpg     frame đã tách (tên = timestamp epoch, khớp với .ulg)
  detections.jsonl 1 dòng / frame đã detect: ts, boxes, latency từng chặng
  health.csv       ts, nhiet_C, throttled, cpu_MHz
  markers.csv      ts của mỗi mốc đồng bộ
  manifest.json    metadata lượt bay + thông số camera thật + thông tin model
"""
import argparse
import csv
import json
import os
import select
import subprocess
import sys
import threading
import time

import numpy as np

STRIDES = (4, 8, 16, 32)

try:
    from cam_ctl import SoftAE, max_exposure_for_fps
except ImportError:                     # chay le, khong co cam_ctl
    SoftAE = None


# ------------------------------------------------------------------ decode / nms
# GIỮ ĐÚNG công thức của utils/box_ops.py:decode_predictions_np (sigmoid-0.5, clip +-4).
# Đừng đổi sang tanh: sai encoding -> tâm box lệch nửa ô lưới.
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def decode_scale(raw, stride, img_size, conf_thr):
    """raw: (5,H,W) logits -> list {'score', 'box':[x1,y1,x2,y2] chuẩn hoá 0-1}."""
    _, H, W = raw.shape
    scores = _sig(raw[0])
    mask = scores >= conf_thr
    if not mask.any():
        return []
    ys, xs = np.nonzero(mask)
    sc = scores[ys, xs]
    dx = _sig(raw[1, ys, xs]) - 0.5
    dy = _sig(raw[2, ys, xs]) - 0.5
    w = np.exp(np.clip(raw[3, ys, xs], -4.0, 4.0)) * (stride / img_size)
    h = np.exp(np.clip(raw[4, ys, xs], -4.0, 4.0)) * (stride / img_size)
    cx = (xs + dx) * (stride / img_size)
    cy = (ys + dy) * (stride / img_size)
    return [{"score": float(sc[i]),
             "box": [float(cx[i] - w[i] / 2), float(cy[i] - h[i] / 2),
                     float(cx[i] + w[i] / 2), float(cy[i] + h[i] / 2)]}
            for i in range(len(sc))]


def nms(dets, iou_thr=0.5, max_det=100):
    if not dets:
        return []
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


def letterbox(img, size):
    import cv2
    h0, w0 = img.shape[:2]
    r = min(size / h0, size / w0)
    nh, nw = int(round(h0 * r)), int(round(w0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, top, left


# ------------------------------------------------------------------ camera
class UsbCam:
    """Luồng riêng giữ frame MỚI NHẤT (không để detector làm trễ việc thu)."""

    def __init__(self, cam, w, h, fps):
        import cv2
        self.cap = cv2.VideoCapture(int(cam) if str(cam).isdigit() else cam, cv2.CAP_V4L2)
        # YUYV 720p nghen bang thong USB2 -> chi 5 fps. Phai ep MJPG.
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # BUFFERSIZE=1 lam ROT DUNG MOT NUA khung (30->15 fps): mot buffer thi driver khong
        # co cho ghi khung ke trong luc app dang giu buffer. Do tren C270/Pi5 2026-08-27.
        # Dung 2 — van tuoi vi luong _loop luon ghi de khung moi nhat.
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
        import cv2
        return dict(width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=float(self.cap.get(cv2.CAP_PROP_FPS)))

    def release(self):
        self.run = False
        time.sleep(0.05)
        self.cap.release()


class CsiCam:
    """Camera Module qua picamera2 (Pi 5 không dùng được V4L2 legacy cho CSI)."""

    def __init__(self, w, h, fps):
        from picamera2 import Picamera2
        self.pc = Picamera2()
        cfg = self.pc.create_video_configuration(main={"size": (w, h), "format": "RGB888"})
        self.pc.configure(cfg)
        self.pc.start()
        time.sleep(1.0)
        self._w, self._h, self._fps = w, h, fps
        self.f, self.t, self.n, self.run = None, 0.0, 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:
            arr = self.pc.capture_array()          # picamera2 "RGB888" trả BGR cho OpenCV
            self.f, self.t, self.n = arr, time.time(), self.n + 1

    def read(self):
        return self.f, self.t, self.n

    def opened(self):
        return True

    def props(self):
        return dict(width=self._w, height=self._h, fps=float(self._fps))

    def release(self):
        self.run = False
        time.sleep(0.05)
        self.pc.stop()



class SynthCam:
    """Nguon anh GIA — de smoke-test toan bo pipeline khi CHUA cam camera.
    KHONG dung de thu data that: anh la nhieu ngau nhien, khong co nguoi."""

    def __init__(self, w, h, fps):
        self._w, self._h, self._fps = w, h, fps
        base = np.random.randint(0, 255, (h, w, 3), np.uint8)
        self.f, self.t, self.n, self.run = base, time.time(), 0, True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.run:                     # gia lap dung nhip fps cua camera that
            self.f = np.roll(self.f, 3, axis=1)
            self.t, self.n = time.time(), self.n + 1
            time.sleep(1.0 / max(1, self._fps))

    def read(self):
        return self.f, self.t, self.n

    def opened(self):
        return True

    def props(self):
        return dict(width=self._w, height=self._h, fps=float(self._fps), synthetic=True)

    def release(self):
        self.run = False
        time.sleep(0.05)


# ------------------------------------------------------------------ sức khoẻ Pi
def soc_temp_c():
    try:
        return round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0, 1)
    except Exception:
        return float("nan")


def throttled():
    """0x0 = chưa từng throttle. bit0/1/2/3 = under-volt / freq-cap / throttle / soft-limit."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        return out.split("=")[-1]
    except Exception:
        return "n/a"


def cpu_mhz():
    try:
        khz = int(open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").read())
        return round(khz / 1000.0)
    except Exception:
        return -1


class Health(threading.Thread):
    def __init__(self, path, period=2.0):
        super().__init__(daemon=True)
        self.path, self.period, self.run_ = path, period, True
        self.last = ""

    def run(self):
        with open(self.path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ts", "temp_C", "throttled", "cpu_MHz"])
            while self.run_:
                t, th, mhz = soc_temp_c(), throttled(), cpu_mhz()
                self.last = f"{t}C {th} {mhz}MHz"
                w.writerow([f"{time.time():.3f}", t, th, mhz])
                fh.flush()
                time.sleep(self.period)


# ------------------------------------------------------------------ bàn phím qua stdin
def poll_key():
    """Đọc 1 dòng từ stdin nếu có, KHÔNG chặn. Chạy được qua ssh, không cần cửa sổ."""
    if not sys.stdin.isatty():
        return None
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.readline().strip().lower()
    return None


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    # --- metadata lượt bay (spec mục 3: phải biết clip nào ở độ cao nào) ---
    ap.add_argument("--alt", type=float, required=True, help="độ cao mét (3/5/8/12)")
    ap.add_argument("--scene", default="unknown", help="concrete / grass / shadow ...")
    ap.add_argument("--persons", type=int, default=1, help="số người trong khung")
    ap.add_argument("--mode", default="hover", help="hover / forward / yaw")
    ap.add_argument("--vib", default="normal", choices=["normal", "high"],
                    help="lượt rung mạnh hay không (spec mục 4)")
    ap.add_argument("--shutter", default="", help="vd 1/1000 — spec mục 1, BẮT BUỘC ghi")
    ap.add_argument("--iso", default="", help="vd 200")
    ap.add_argument("--note", default="", help="ghi chú tự do")
    # --- thu ---
    ap.add_argument("--out", default="flight_data")
    ap.add_argument("--source", choices=["usb", "csi", "synth"], default="usb",
                    help="synth = anh gia, chi de smoke-test khi chua cam camera")
    ap.add_argument("--cam", default="0")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--every", type=int, default=15, help="lưu 1 ảnh mỗi N frame (~1-2 ảnh/s)")
    ap.add_argument("--no-video", action="store_true", help="chỉ lưu ảnh, không lưu video")
    ap.add_argument("--minutes", type=float, default=0, help=">0 = tự dừng sau N phút")
    # --- detector ---
    ap.add_argument("--detect", dest="detect", action="store_true", default=True)
    ap.add_argument("--no-detect", dest="detect", action="store_false")
    ap.add_argument("--hef", default="models/pfdet_v17_640_hailo8l.hef")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.4, help="conf DEPLOY (eval mới dùng 0.01)")
    ap.add_argument("--detect-every", type=int, default=1, help="detect 1 frame mỗi N frame")
    ap.add_argument("--dev", default="/dev/video0", help="node v4l2 để chỉnh exposure/gain")
    ap.add_argument("--ae", dest="ae", action="store_true", default=True,
                    help="auto-exposure PHẦN MỀM (mặc định bật): sáng mà KHÔNG tụt fps")
    ap.add_argument("--no-ae", dest="ae", action="store_false")
    ap.add_argument("--max-exposure", type=int, default=0,
                    help="trần phơi sáng, đơn vị 100us. 0 = tính từ --fps. "
                         "LÚC BAY nên 20 (1/500s) để khỏi nhoè vì rung")
    ap.add_argument("--prefer-gain", action="store_true",
                    help="LÚC BAY: giữ shutter ngắn chống nhoè, để gain gánh phần sáng")
    args = ap.parse_args()

    import cv2

    sess = time.strftime("%Y%m%d_%H%M%S") + f"_alt{args.alt:g}m_{args.scene}_{args.mode}"
    root = os.path.join(args.out, sess)
    img_dir = os.path.join(root, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"[REC] session -> {root}")

    # ---- camera
    if args.source == "synth":
        print("[REC] !! NGUON ANH GIA (--source synth): chi de smoke-test, KHONG phai data bay")
        cam = SynthCam(args.width, args.height, args.fps)
    elif args.source == "csi":
        cam = CsiCam(args.width, args.height, args.fps)
    else:
        cam = UsbCam(args.cam, args.width, args.height, args.fps)
        time.sleep(1.0)
        if not cam.opened():
            print(f"[LỖI] không mở được camera {args.cam}. "
                  f"Kiểm: ls /dev/video*  |  hoặc dùng --source csi")
            return
    time.sleep(0.8)
    frame, _, _ = cam.read()
    t_wait = time.time()
    while frame is None and time.time() - t_wait < 5:
        time.sleep(0.05)
        frame, _, _ = cam.read()
    if frame is None:
        print("[LỖI] camera mở được nhưng không ra frame."); cam.release(); return
    H, W = frame.shape[:2]
    print(f"[REC] camera {W}x{H} | {cam.props()}")

    # ---- auto-exposure PHẦN MỀM (chỉ cho camera USB).
    # Ở auto-exposure của firmware, thiếu sáng là nó kéo dài phơi sáng rồi TỰ HẠ fps
    # (đo được 30 -> 19.7 fps trong nhà). SoftAE giữ sáng nhưng chặn trần phơi sáng
    # nên fps không bao giờ tụt. Lúc BAY nên --max-exposure 20 (1/500s) + --prefer-gain
    # để shutter ngắn, khỏi nhoè vì rung.
    ae = None
    if args.ae and args.source == "usb" and SoftAE is not None:
        mx = args.max_exposure or max_exposure_for_fps(args.fps)
        ae = SoftAE(dev=args.dev, max_exposure=mx, prefer_gain=args.prefer_gain)
        print(f"[REC] SoftAE bật | trần phơi sáng {mx} (={mx/10:.1f} ms = 1/{int(1e4/mx)} s)")
    ae_state = {}

    # ---- model (tuỳ chọn)
    pipe = ng_ctx = vdev = None
    in_name = None
    model_info = None
    if args.detect:
        if not os.path.exists(args.hef):
            print(f"[LỖI] không thấy HEF: {args.hef}  (chạy lại với --no-detect nếu chỉ thu data)")
            cam.release(); return
        from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, InferVStreams,
                                    FormatType)
        hef = HEF(args.hef)
        vdev = VDevice()
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = vdev.configure(hef, cfg)[0]
        in_name = hef.get_input_vstream_infos()[0].name
        pipe = InferVStreams(ng, InputVStreamParams.make(ng, format_type=FormatType.UINT8),
                             OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32))
        pipe.__enter__()
        ng_ctx = ng.activate(); ng_ctx.__enter__()
        model_info = dict(hef=os.path.abspath(args.hef), img_size=args.img_size,
                          conf=args.conf, input_vstream=in_name)
        print(f"[REC] detector BẬT: {args.hef} @ {args.img_size}px conf={args.conf}")
    else:
        print("[REC] detector TẮT (--no-detect)")

    # ---- writer + log
    writer = None
    if not args.no_video:
        # Không tin VideoWriter im lặng: nếu codec hỏng nó vẫn trả object nhưng
        # isOpened()=False -> bay xong mới biết video rỗng. Thử mp4v, rớt thì MJPG/.avi.
        for fourcc, ext in (("mp4v", "mp4"), ("avc1", "mp4"), ("MJPG", "avi")):
            vpath = os.path.join(root, f"video.{ext}")
            writer = cv2.VideoWriter(vpath, cv2.VideoWriter.fourcc(*fourcc), args.fps, (W, H))
            if writer.isOpened():
                print(f"[REC] video -> {vpath}  (codec {fourcc})")
                break
            writer.release()
            writer = None
        if writer is None:
            print("[LỖI] KHÔNG mở được VideoWriter với codec nào. Vẫn thu ẢNH bình thường.")

    health = Health(os.path.join(root, "health.csv")); health.start()
    det_fh = open(os.path.join(root, "detections.jsonl"), "w") if args.detect else None
    mk_fh = open(os.path.join(root, "markers.csv"), "w", newline="")
    mk = csv.writer(mk_fh); mk.writerow(["ts", "label"]); mk_fh.flush()

    manifest = dict(
        session=sess, started_ts=time.time(),
        started_local=time.strftime("%Y-%m-%d %H:%M:%S"),
        flight=dict(alt_m=args.alt, scene=args.scene, persons=args.persons,
                    mode=args.mode, vibration=args.vib, shutter=args.shutter,
                    iso=args.iso, note=args.note),
        SYNTHETIC=(args.source == "synth"),   # True = KHONG phai data bay that
        camera=dict(source=args.source, cam=args.cam, requested=[args.width, args.height, args.fps],
                    actual=[W, H], props=cam.props(), save_every_n_frames=args.every,
                    soft_ae=bool(ae),
                    max_exposure=(args.max_exposure or max_exposure_for_fps(args.fps))
                                 if ae else None,
                    prefer_gain=args.prefer_gain),
        model=model_info,
        host=os.uname().nodename,
    )
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 68)
    print("  ĐANG THU.  gõ  m<Enter> = MỐC ĐỒNG BỘ (lắc drone sau khi arm)")
    print("             gõ  q<Enter> = dừng lượt bay      (Ctrl+C cũng được)")
    print("=" * 68 + "\n")

    saved = ndet = 0
    last_id = -1
    t_start = time.time()
    lat_hist = []
    try:
        while True:
            k = poll_key()
            if k == "q":
                break
            if k == "m":
                ts = time.time()
                mk.writerow([f"{ts:.3f}", "sync"]); mk_fh.flush()
                print(f"  >> MỐC ĐỒNG BỘ tại ts={ts:.3f}")
            if args.minutes > 0 and time.time() - t_start > args.minutes * 60:
                print("  >> hết --minutes, dừng.")
                break

            t_cap0 = time.perf_counter()
            frame, fts, fid = cam.read()
            if frame is None or fid == last_id:
                time.sleep(0.002)
                continue
            last_id = fid
            t_cap = time.perf_counter() - t_cap0

            if ae is not None:
                r = ae.step(frame)
                if r:
                    ae_state = dict(exposure=r[0], gain=r[1], luma=round(r[2], 1))

            # Ghi dia do RIENG, KHONG tinh vao latency detection: day la chi phi cua
            # viec THU DATA, luc bay that (deploy) khong co. Gop vao la thoi phong e2e.
            t_rec0 = time.perf_counter()
            if writer is not None:
                writer.write(frame)
            if fid % args.every == 0:
                cv2.imwrite(os.path.join(img_dir, f"{fts:.3f}.jpg".replace(".", "_", 1)), frame)
                saved += 1
            t_rec = time.perf_counter() - t_rec0

            if args.detect and fid % args.detect_every == 0:
                t0 = time.perf_counter()
                canvas, r, top, left = letterbox(frame, args.img_size)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                t1 = time.perf_counter()
                out = pipe.infer({in_name: rgb})
                t2 = time.perf_counter()
                outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2], reverse=True)
                dets = []
                for si, o in enumerate(outs):
                    dets += decode_scale(np.transpose(o[0], (2, 0, 1)), STRIDES[si],
                                         args.img_size, args.conf)
                dets = nms(dets, 0.5)
                t3 = time.perf_counter()

                boxes = []
                for d in dets:                       # đưa về pixel của ẢNH GỐC
                    x1, y1, x2, y2 = d["box"]
                    boxes.append([round((x1 * args.img_size - left) / r, 1),
                                  round((y1 * args.img_size - top) / r, 1),
                                  round((x2 * args.img_size - left) / r, 1),
                                  round((y2 * args.img_size - top) / r, 1),
                                  round(d["score"], 3)])
                # total = DUONG DETECT thuc su: capture + preprocess + inference + decode.
                # Bang dung tong 4 chang -> reviewer cong lai la khop.
                total_ms = (t_cap + (t3 - t0)) * 1e3
                lat_hist.append(total_ms)
                det_fh.write(json.dumps(dict(
                    ts=round(fts, 3), frame=fid, n=len(boxes), boxes=boxes,
                    ms=dict(capture=round(t_cap * 1e3, 2), preprocess=round((t1 - t0) * 1e3, 2),
                            inference=round((t2 - t1) * 1e3, 2),
                            decode_nms=round((t3 - t2) * 1e3, 2), total=round(total_ms, 2),
                            record_overhead=round(t_rec * 1e3, 2)),
                    temp_C=soc_temp_c())) + "\n")
                ndet += 1
                if ndet % 30 == 0:
                    det_fh.flush()
                    a = np.array(lat_hist[-300:])
                    print(f"  [{time.time()-t_start:6.0f}s] anh:{saved:5d} det:{ndet:6d} "
                          f"nguoi:{len(boxes):2d} | e2e p50={np.percentile(a,50):5.1f}ms "
                          f"p99={np.percentile(a,99):5.1f}ms | {health.last}")
            elif saved and saved % 50 == 0 and fid % args.every == 0:
                print(f"  [{time.time()-t_start:6.0f}s] anh:{saved:5d} | {health.last}")
    except KeyboardInterrupt:
        print("\n  >> Ctrl+C, đang đóng file...")

    # ---- đóng
    health.run_ = False
    cam.release()
    if writer is not None:
        writer.release()
    if det_fh:
        det_fh.close()
    mk_fh.close()
    if ng_ctx is not None:
        ng_ctx.__exit__(None, None, None)
    if pipe is not None:
        pipe.__exit__(None, None, None)
    if vdev is not None:
        try: vdev.release()
        except Exception: pass

    dur = time.time() - t_start
    manifest["ended_ts"] = time.time()
    manifest["duration_s"] = round(dur, 1)
    manifest["saved_images"] = saved
    manifest["detected_frames"] = ndet
    manifest["throttled_end"] = throttled()
    manifest["temp_end_C"] = soc_temp_c()
    if lat_hist:
        a = np.array(lat_hist)
        manifest["e2e_ms_note"] = ("capture+preprocess+inference+decode_nms; "
                                   "KHONG gom ghi video/anh (xem ms.record_overhead)")
        manifest["e2e_ms"] = dict(mean=round(float(a.mean()), 2),
                                  p50=round(float(np.percentile(a, 50)), 2),
                                  p95=round(float(np.percentile(a, 95)), 2),
                                  p99=round(float(np.percentile(a, 99)), 2))
    with open(os.path.join(root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[XONG] {dur:.0f}s | {saved} ảnh | {ndet} frame đã detect -> {root}")
    if lat_hist:
        e = manifest["e2e_ms"]
        print(f"       e2e p50={e['p50']}ms p95={e['p95']}ms p99={e['p99']}ms")
    print(f"       nhiệt cuối {manifest['temp_end_C']}C | throttled={manifest['throttled_end']} "
          f"(0x0 = chưa từng throttle)")
    print(f"       NHỚ: copy .ulg của lượt này về, và đối chiếu mốc trong markers.csv")


if __name__ == "__main__":
    main()
