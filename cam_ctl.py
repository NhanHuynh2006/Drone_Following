"""
cam_ctl.py — GIỮ CAMERA USB Ở ĐÚNG FPS, và tự chỉnh sáng mà KHÔNG tụt fps.
==========================================================================

VẤN ĐỀ (đã đo trên chính con Pi này, C270, 2026-08-27):
  Ở chế độ auto-exposure, khi thiếu sáng firmware camera **kéo dài thời gian phơi**
  vượt 1/30 s → nó tự hạ khung hình xuống **15 fps**, ở MỌI độ phân giải và MỌI
  fourcc. Đây là lý do e2e của cả pipeline bị kẹt 15 FPS trong khi NPU mới dùng
  17.8/33.3 ms. Không phải lỗi Hailo, không phải lỗi băng thông USB.

  `exposure_dynamic_framerate=0` (cờ UVC sinh ra đúng để cấm hành vi này)
  **KHÔNG ăn thua** trên C270 — firmware Logitech phớt lờ. Đã thử, vẫn 15.0 fps.

CÁCH SỬA:
  Khoá `auto_exposure=1` (Manual) → camera giữ đúng 30 fps. Nhưng khoá cứng thì
  đổi cảnh là ảnh tối thui hoặc cháy trắng. Nên lớp `SoftAE` dưới đây làm
  auto-exposure BẰNG PHẦN MỀM: đo độ sáng trung bình rồi chỉnh
  `exposure_time_absolute` + `gain`, với **trần phơi sáng cứng** =
  `max_exposure` → không bao giờ được phép rơi xuống dưới fps mục tiêu.

  Đơn vị `exposure_time_absolute` của UVC là **100 µs**:
      333 = 33.3 ms = 1/30 s   -> trần để giữ 30 fps
      166 = 16.6 ms = 1/60 s
       10 =  1.0 ms = 1/1000 s -> shutter FLIGHT_DATA_SPEC yêu cầu khi bay

  Lúc BAY: đặt `--max-exposure 20` (1/500 s) trở xuống để khỏi nhoè do rung/
  chuyển động, và để `gain` gánh phần sáng. Trong nhà thì cho tới 333 vì
  gain cao là ảnh đầy nhiễu, detector ghét nhiễu hơn ghét mờ nhẹ.

DÙNG:
    python3 cam_ctl.py --show                       # xem control hiện tại
    python3 cam_ctl.py --apply --max-exposure 333   # khoa 30fps, trong nha
    python3 cam_ctl.py --apply --max-exposure 20    # khoa 30fps, cau hinh BAY
    python3 cam_ctl.py --restore                    # tra ve auto cua camera
"""
import argparse
import subprocess
import time

# 100 µs/đơn vị -> trần phơi sáng theo fps mục tiêu
EXP_UNIT_US = 100


def max_exposure_for_fps(fps):
    """Trần phơi sáng (đơn vị UVC) để camera còn giữ được `fps`."""
    return max(1, int(1e6 / fps / EXP_UNIT_US))          # 30 fps -> 333


def _v4l2(dev, args):
    return subprocess.run(["v4l2-ctl", "-d", dev] + args,
                          capture_output=True, text=True, timeout=5)


def get_ctrls(dev="/dev/video0"):
    """Đọc mọi control -> dict {ten: gia_tri_int}."""
    out = _v4l2(dev, ["--list-ctrls"]).stdout
    d = {}
    for line in out.splitlines():
        if "value=" not in line:
            continue
        name = line.split()[0]
        for tok in line.split():
            if tok.startswith("value="):
                try:
                    d[name] = int(tok.split("=")[1])
                except ValueError:
                    pass
    return d


def set_ctrl(dev, name, value):
    r = _v4l2(dev, ["-c", f"{name}={value}"])
    return r.returncode == 0


def apply_locked_fps(dev="/dev/video0", exposure=120, gain=32,
                     power_line=1, verbose=True):
    """Khoá camera ở chế độ thủ công -> fps ổn định. Trả về dict đã đặt."""
    applied = {}
    for name, val in (("power_line_frequency", power_line),   # 1 = 50 Hz (dien VN)
                      ("auto_exposure", 1),                   # 1 = Manual Mode
                      ("exposure_dynamic_framerate", 0),      # vo hai; C270 phot lo
                      ("exposure_time_absolute", exposure),
                      ("gain", gain)):
        if set_ctrl(dev, name, val):
            applied[name] = val
    if verbose:
        print(f"[cam_ctl] khoa {dev}: exposure={exposure} (={exposure*EXP_UNIT_US/1000:.1f} ms "
              f"= 1/{int(1e6/(exposure*EXP_UNIT_US)):d} s), gain={gain}, 50 Hz")
    return applied


def restore_auto(dev="/dev/video0", verbose=True):
    set_ctrl(dev, "auto_exposure", 3)                          # Aperture Priority
    set_ctrl(dev, "exposure_dynamic_framerate", 1)
    if verbose:
        print(f"[cam_ctl] {dev} tra ve auto-exposure (fps se lai tut khi thieu sang)")


class SoftAE:
    """
    Auto-exposure bằng phần mềm, có TRẦN phơi sáng nên fps không bao giờ tụt.

        ae = SoftAE(max_exposure=333)      # 1/30 s
        ae.step(frame)                     # goi moi khung; no tu gian cach

    Ưu tiên chỉnh `exposure` trước (ít nhiễu hơn), chạm trần mới tăng `gain`.
    Đặt `prefer_gain=True` cho lúc BAY: giữ shutter ngắn chống nhoè, để gain
    gánh phần sáng.
    """

    def __init__(self, dev="/dev/video0", target=110, tol=12,
                 max_exposure=333, min_exposure=3, max_gain=255,
                 exposure=120, gain=32, interval=0.4, prefer_gain=False):
        self.dev, self.target, self.tol = dev, target, tol
        self.max_e, self.min_e, self.max_g = max_exposure, min_exposure, max_gain
        self.e, self.g = min(exposure, max_exposure), gain
        self.interval, self.prefer_gain = interval, prefer_gain
        self._last = 0.0
        apply_locked_fps(dev, self.e, self.g, verbose=False)

    def step(self, frame):
        """frame: BGR uint8. Trả (exposure, gain, do_sang) hoặc None nếu chưa tới nhịp."""
        now = time.time()
        if now - self._last < self.interval:
            return None
        self._last = now

        small = frame[::8, ::8]                      # lấy mẫu thưa: đủ để đo sáng, gần như free
        mean = float(small.mean())
        err = self.target - mean
        if abs(err) <= self.tol:
            return self.e, self.g, mean

        # bước tỉ lệ với sai số, chặn để khỏi dao động
        step = max(1, int(abs(err) / self.target * self.e * 0.6)) if self.e else 1
        step = min(step, max(1, self.max_e // 4))

        if err > 0:                                   # quá TỐI -> sáng lên
            if self.prefer_gain and self.g < self.max_g:
                self.g = min(self.max_g, self.g + max(4, int(abs(err) / 3)))
            elif self.e < self.max_e:
                self.e = min(self.max_e, self.e + step)
            elif self.g < self.max_g:
                self.g = min(self.max_g, self.g + max(4, int(abs(err) / 3)))
        else:                                         # quá SÁNG -> tối đi
            if self.g > 0 and (self.prefer_gain or self.e <= self.min_e):
                self.g = max(0, self.g - max(4, int(abs(err) / 3)))
            elif self.e > self.min_e:
                self.e = max(self.min_e, self.e - step)
            elif self.g > 0:
                self.g = max(0, self.g - max(4, int(abs(err) / 3)))

        set_ctrl(self.dev, "exposure_time_absolute", self.e)
        set_ctrl(self.dev, "gain", self.g)
        return self.e, self.g, mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/video0")
    ap.add_argument("--show", action="store_true", help="in control hien tai roi thoat")
    ap.add_argument("--apply", action="store_true", help="khoa manual -> fps on dinh")
    ap.add_argument("--restore", action="store_true", help="tra ve auto cua camera")
    ap.add_argument("--fps", type=float, default=30.0, help="fps muc tieu (dat tran phoi sang)")
    ap.add_argument("--max-exposure", type=int, default=0,
                    help="tran phoi sang (don vi 100us). 0 = tinh tu --fps")
    ap.add_argument("--exposure", type=int, default=120)
    ap.add_argument("--gain", type=int, default=32)
    a = ap.parse_args()

    if a.show or not (a.apply or a.restore):
        c = get_ctrls(a.dev)
        print(f"[cam_ctl] {a.dev}")
        for k in ("auto_exposure", "exposure_time_absolute", "exposure_dynamic_framerate",
                  "gain", "power_line_frequency", "white_balance_automatic",
                  "backlight_compensation", "brightness", "contrast"):
            if k in c:
                print(f"  {k:30s} = {c[k]}")
        mx = max_exposure_for_fps(a.fps)
        print(f"  -> tran phoi sang de giu {a.fps:.0f} fps: {mx} "
              f"(={mx*EXP_UNIT_US/1000:.1f} ms)")
        return

    if a.restore:
        restore_auto(a.dev)
        return

    mx = a.max_exposure or max_exposure_for_fps(a.fps)
    apply_locked_fps(a.dev, min(a.exposure, mx), a.gain)
    print(f"[cam_ctl] tran phoi sang = {mx} -> fps khong tut duoi {a.fps:.0f}")


if __name__ == "__main__":
    main()
