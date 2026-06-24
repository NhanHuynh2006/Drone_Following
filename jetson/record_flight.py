"""
record_flight.py — CÔNG CỤ RIÊNG: thu ảnh + video lúc bay (để làm val set 3-10m thật).
=====================================================================================

ĐỘC LẬP hoàn toàn với code deploy (detector_pub/follow_px4). Chỉ chạy khi BẠN muốn thu data.
Không cần model, không cần torch — chỉ camera + opencv.

Chạy (trên Jetson hoặc máy nào có camera):
  # Thu ẢNH (mặc định): lưu 1 frame mỗi 15 frame (~1-2 ảnh/giây) vào flight_data/images/
  python3 record_flight.py --out flight_data --every 15

  # Thu cả VIDEO đầy đủ + ảnh:
  python3 record_flight.py --out flight_data --every 15 --video

  # Xem trực tiếp lúc thu (bấm q để dừng, space để chụp ngay 1 ảnh):
  python3 record_flight.py --out flight_data --show

Sau khi thu: label bằng tay (hoặc dùng model pre-label) -> thành val set 3-10m cho riêng bài toán bạn.
"""
import os
import time
import argparse
import threading
import cv2


class Grab:
    """Luồng camera riêng giữ frame mới nhất (không lag)."""
    def __init__(self, cam, w=None, h=None):
        self.cap = cv2.VideoCapture(int(cam) if str(cam).isdigit() else cam)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if w: self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        if h: self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        self.f = None; self.n = 0; self.run = True
        self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()
    def _loop(self):
        while self.run:
            ok, f = self.cap.read()
            if ok: self.f = f; self.n += 1     # đếm frame camera THẬT
            else: time.sleep(0.005)
    def read(self):
        return self.f, self.n                  # trả frame + id (để biết frame mới)
    def opened(self):
        return self.cap.isOpened()
    def release(self):
        self.run = False; time.sleep(0.05); self.cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="flight_data", help="thư mục lưu")
    ap.add_argument("--cam", default="0")
    ap.add_argument("--every", type=int, default=15, help="lưu 1 ảnh mỗi N frame (15 ~ 1-2/giây)")
    ap.add_argument("--video", action="store_true", help="lưu thêm video đầy đủ .mp4")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--show", action="store_true", help="hiện cửa sổ (q=dừng, space=chụp ngay)")
    args = ap.parse_args()

    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"[REC] lưu ảnh -> {img_dir}/  (mỗi {args.every} frame)")

    grab = Grab(args.cam, args.width, args.height)
    time.sleep(1.0)
    if not grab.opened():
        print("KHÔNG mở được camera", args.cam); return

    writer = None
    if args.video:
        f0, _ = grab.read()
        while f0 is None:
            time.sleep(0.05); f0, _ = grab.read()
        h, w = f0.shape[:2]
        vpath = os.path.join(args.out, f"video_{int(time.time())}.mp4")
        writer = cv2.VideoWriter(vpath, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        print(f"[REC] lưu video -> {vpath}")

    saved, last_id = 0, -1
    print("[REC] đang thu... (Ctrl+C để dừng)")
    try:
        while True:
            frame, fid = grab.read()
            if frame is None or fid == last_id:   # chỉ xử lý FRAME CAMERA MỚI (không spin CPU)
                time.sleep(0.002); continue
            last_id = fid
            if writer is not None:
                writer.write(frame)

            if fid % args.every == 0:              # lưu mỗi N frame camera THẬT
                ts = f"{time.time():.3f}".replace(".", "_")
                cv2.imwrite(os.path.join(img_dir, f"{ts}.jpg"), frame)
                saved += 1
                if saved % 20 == 0:
                    print(f"[REC] đã lưu {saved} ảnh")

            if args.show:
                disp = frame.copy()
                cv2.putText(disp, f"REC  anh:{saved}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("record_flight (q=dung, space=chup)", disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'):
                    break
                if k == ord(' '):   # chụp ngay 1 ảnh
                    ts = f"{time.time():.3f}".replace(".", "_")
                    cv2.imwrite(os.path.join(img_dir, f"{ts}_manual.jpg"), frame)
                    saved += 1
    except KeyboardInterrupt:
        pass

    grab.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"\n[REC] XONG. Tổng {saved} ảnh ở {img_dir}/" + (f" + video" if args.video else ""))
    print("      -> label tay (hoặc pre-label bằng model) -> val set 3-10m của bạn.")


if __name__ == "__main__":
    main()
