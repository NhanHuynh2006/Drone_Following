"""Hiệu chuẩn camera bằng bảng checkerboard.

VÌ SAO PHẢI LÀM: distance.py tính cự ly bằng  d = fy * 1.7 / h_px.  fy trong
config.yaml đang để 700.0 — số tròn ĐOÁN, chưa đo. fy sai bao nhiêu phần trăm thì
cự ly sai bấy nhiêu, mà cự ly chính là thứ điều khiển tiến/lùi VÀ là thứ chốt an
toàn d_min = 3 m dựa vào. Chưa hiệu chuẩn thì đừng tin cự ly.

CHUẨN BỊ: in bảng checkerboard 9x6 GÓC TRONG (= 10x7 ô), mỗi ô 25 mm. Dán lên bìa
cứng cho phẳng — bảng cong là hỏng số. Ô khác 25 mm thì truyền --square.

CHẠY (mặc định KHÔNG CẦN MÀN HÌNH — chạy được qua ssh):

    # trên Pi, qua ssh:
    cd ~/NhanHuynh/follow_drone
    ../venv/bin/python3 scripts/calibrate_camera.py --need 20

  Rồi cầm bảng đưa qua lại trước ống kính: gần/xa, nghiêng trái/phải/trên/dưới,
  bốn góc khung hình. Script TỰ bắt ảnh khi thấy bảng ở tư thế MỚI (không cần bấm
  gì), in "bắt được i/N" mỗi lần. Đủ N là nó tự tính rồi in kết quả.

    --write    ghi thẳng fx/fy/cx/cy vào ../config.yaml (có sao lưu .bak)
    --gui      bản cũ: mở cửa sổ, bấm SPACE từng ảnh (cần màn hình)
    --from-dir <thư mục>   hiệu chuẩn lại từ ảnh đã lưu, không cần camera

MẸO: đứng đủ xa để bảng chiếm ~1/3 khung, và NHỚ nghiêng bảng. Chụp 20 ảnh bảng
song song với ống kính thì fx/fy ra sai — cần đa dạng góc nghiêng mới tách được
tiêu cự khỏi khoảng cách.
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

# Default checkerboard
PATTERN_SIZE = (9, 6)        # inner corners (số góc trong, không phải số ô)
SQUARE_SIZE_MM = 25.0


def _solve(objpoints, imgpoints, shape_wh, args):
    """Chạy calibrateCamera rồi in kết quả. Trả về dict hoặc None."""
    if len(objpoints) < 10:
        print(f"\nChỉ có {len(objpoints)} ảnh — cần ít nhất 10. Không tính.")
        return None
    print(f"\nĐang tính từ {len(objpoints)} ảnh...")
    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, shape_wh, None, None)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    print("\n=== KẾT QUẢ ===")
    print(f"  fx = {fx:.2f}")
    print(f"  fy = {fy:.2f}")
    print(f"  cx = {cx:.2f}   (tâm ảnh lý thuyết {shape_wh[0]/2:.0f})")
    print(f"  cy = {cy:.2f}   (tâm ảnh lý thuyết {shape_wh[1]/2:.0f})")
    print(f"  sai số tái chiếu (RMS) = {rms:.4f} px")
    print(f"  méo = {np.round(dist.flatten(), 4)}")

    # Cổng kiểm — số xấu thì nói thẳng, đừng để đem đi bay.
    bad = []
    if rms > 1.0:
        bad.append(f"RMS {rms:.2f} px > 1.0 — ảnh mờ/bảng cong/thiếu góc nghiêng")
    if abs(fx - fy) / max(fx, fy) > 0.15:
        bad.append(f"fx và fy lệch {abs(fx-fy)/max(fx,fy)*100:.0f}% — thường là thiếu đa dạng tư thế")
    if not (0.25 < cx / shape_wh[0] < 0.75 and 0.25 < cy / shape_wh[1] < 0.75):
        bad.append("tâm quang lệch xa tâm ảnh — nghi ngờ số")
    if bad:
        print("\n  ⚠️ SỐ NÀY ĐÁNG NGỜ, ĐỪNG ĐEM BAY:")
        for b in bad:
            print(f"     - {b}")
        print("     Chụp lại, nhớ nghiêng bảng nhiều hướng và giữ bảng thật phẳng.")
    else:
        print("\n  ✅ Số trông hợp lý.")

    d_old = 700.0 * 1.7 / 100.0
    d_new = fy * 1.7 / 100.0
    print(f"\n  Ảnh hưởng thực tế — người cao 1,7 m cho hộp cao 100 px:")
    print(f"     fy=700 (đang dùng) -> {d_old:.2f} m")
    print(f"     fy={fy:.0f} (đo được) -> {d_new:.2f} m")
    print(f"     lệch {abs(d_new-d_old)/d_old*100:.1f}%")

    if args.write:
        _write_config(args.write_path, fx, fy, cx, cy)
    else:
        print(f"\nChép vào {args.write_path} mục 'camera:', hoặc chạy lại kèm --write")
    return dict(fx=fx, fy=fy, cx=cx, cy=cy, rms=rms)


def _write_config(path, fx, fy, cx, cy):
    """Ghi fx/fy/cx/cy vào config.yaml, giữ nguyên chú thích, có sao lưu."""
    import re, shutil
    if not os.path.exists(path):
        print(f"\nKhông thấy {path} — bỏ qua --write.")
        return
    shutil.copy2(path, path + ".bak")
    txt = open(path, encoding="utf-8").read()
    for key, val in (("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)):
        pat = re.compile(r"^(\s*%s:\s*)[0-9.]+" % key, re.M)
        txt, n = pat.subn(lambda m: "%s%.1f" % (m.group(1), val), txt, count=1)
        if n == 0:
            print(f"  !! không tìm thấy dòng '{key}:' trong config — chưa ghi")
    open(path, "w", encoding="utf-8").write(txt)
    print(f"\n✅ Đã ghi vào {path} (bản cũ ở {path}.bak)")
    print("   Nhớ đẩy sang Pi: bash scripts/sync_to_pi.sh --go")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=20, help="số ảnh cần bắt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--square", type=float, default=SQUARE_SIZE_MM,
                    help="cạnh ô checkerboard, mm")
    ap.add_argument("--cols", type=int, default=PATTERN_SIZE[0], help="số GÓC TRONG theo chiều ngang")
    ap.add_argument("--rows", type=int, default=PATTERN_SIZE[1], help="số GÓC TRONG theo chiều dọc")
    ap.add_argument("--min-move", type=float, default=40.0,
                    help="px — tư thế phải khác ảnh cũ ít nhất bấy nhiêu mới nhận")
    ap.add_argument("--timeout", type=float, default=300.0, help="giây, tự dừng")
    ap.add_argument("--save-dir", default="", help="lưu ảnh đã nhận vào đây")
    ap.add_argument("--from-dir", default="", help="hiệu chuẩn từ ảnh có sẵn, không dùng camera")
    ap.add_argument("--write", action="store_true", help="ghi kết quả vào config.yaml")
    ap.add_argument("--write-path", default=os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    ap.add_argument("--gui", action="store_true", help="bản cũ: cửa sổ + bấm SPACE")
    args = ap.parse_args()
    args.write_path = os.path.normpath(args.write_path)

    pattern = (args.cols, args.rows)
    objp = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)
    objp *= args.square

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints, imgpoints = [], []

    # ── Chế độ ảnh có sẵn ────────────────────────────────────────
    if args.from_dir:
        import glob
        files = sorted(sum([glob.glob(os.path.join(args.from_dir, e))
                            for e in ("*.jpg", "*.png", "*.jpeg")], []))
        if not files:
            print(f"Không có ảnh trong {args.from_dir}"); return
        shape_wh = None
        for f in files:
            img = cv2.imread(f)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            shape_wh = gray.shape[::-1]
            ok, corners = cv2.findChessboardCorners(gray, pattern, None)
            if ok:
                objpoints.append(objp)
                imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
                print(f"  ok  {os.path.basename(f)}")
            else:
                print(f"  bỏ  {os.path.basename(f)} (không thấy bảng)")
        _solve(objpoints, imgpoints, shape_wh, args)
        return

    # ── Mở camera ────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"Không mở được camera {args.cam}"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.gui:
        _loop_gui(cap, pattern, objp, objpoints, imgpoints, criteria, args)
    else:
        _loop_headless(cap, pattern, objp, objpoints, imgpoints, criteria, args)

    gray_shape = (args.width, args.height)
    cap.release()
    if args.gui:
        cv2.destroyAllWindows()
    _solve(objpoints, imgpoints, gray_shape, args)


def _pose_is_new(corners, imgpoints, min_move):
    """Tư thế mới = cách MỌI ảnh đã nhận ít nhất min_move px (trung bình)."""
    c = corners.reshape(-1, 2)
    for prev in imgpoints:
        if np.linalg.norm(c - prev.reshape(-1, 2), axis=1).mean() < min_move:
            return False
    return True


def _loop_headless(cap, pattern, objp, objpoints, imgpoints, criteria, args):
    print("\n=== HIỆU CHUẨN CAMERA (không cần màn hình) ===")
    print(f"Cầm bảng {pattern[0]}x{pattern[1]} góc trong, ô {args.square:g} mm, đưa qua lại trước ống kính.")
    print("Gần/xa, NGHIÊNG trái phải trên dưới, và bốn góc khung hình.")
    print(f"Script tự bắt khi thấy tư thế mới. Cần {args.need} ảnh. Ctrl+C để dừng sớm.\n")
    t0 = time.time()
    seen = 0
    last_report = 0.0
    try:
        while len(imgpoints) < args.need and time.time() - t0 < args.timeout:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
            now = time.time()
            if found:
                seen += 1
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                if _pose_is_new(corners2, imgpoints, args.min_move):
                    objpoints.append(objp)
                    imgpoints.append(corners2)
                    n = len(imgpoints)
                    print(f"  bắt được {n}/{args.need}   (đổi tư thế đi)")
                    if args.save_dir:
                        cv2.imwrite(os.path.join(args.save_dir, f"calib_{n:02d}.jpg"), frame)
                    time.sleep(0.4)
            elif now - last_report > 3.0:
                last_report = now
                if seen == 0:
                    print("  ...chưa thấy bảng. Kiểm tra: đủ sáng? bảng phẳng? đúng 9x6 GÓC TRONG?")
                else:
                    print(f"  ...mất bảng (đã nhận {len(imgpoints)}/{args.need})")
    except KeyboardInterrupt:
        print("\n  (Ctrl+C — dừng sớm, tính với những gì đã có)")


def _loop_gui(cap, pattern, objp, objpoints, imgpoints, criteria, args):
    print("\n=== CALIBRATION (GUI) ===  SPACE = chụp, q = xong\n")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, None)
        disp = frame.copy()
        if found:
            cv2.drawChessboardCorners(disp, pattern, corners, found)
            cv2.putText(disp, "DETECTED - press SPACE", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(disp, f"Captured: {len(imgpoints)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Calibration', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and found:
            objpoints.append(objp)
            imgpoints.append(cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria))
            print(f"Captured {len(imgpoints)}")
        elif key == ord('q'):
            break


if __name__ == '__main__':
    main()
