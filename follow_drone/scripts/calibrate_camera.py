"""Camera calibration with checkerboard.

In trước khi bay outdoor:
  1. In checkerboard 9x6 ô (mỗi ô 25mm) — search "OpenCV checkerboard 9x6"
  2. Chạy script này
  3. Cầm checkerboard ở các góc khác nhau, chụp 20-30 ảnh
  4. Script tính fx, fy, cx, cy → update vào config.yaml

Usage:
    python scripts/calibrate_camera.py
"""

import os
import sys
import numpy as np
import cv2

# Default checkerboard
PATTERN_SIZE = (9, 6)        # inner corners (số góc trong, không phải số ô)
SQUARE_SIZE_MM = 25.0


def main():
    # Prepare 3D points
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM

    objpoints = []
    imgpoints = []

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n=== CAMERA CALIBRATION ===")
    print("Hold checkerboard at different angles/positions.")
    print("Press SPACE to capture (need 20+ samples).")
    print("Press 'q' when done.\n")

    captured = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)
        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, PATTERN_SIZE, corners, ret_corners)
            cv2.putText(display, "DETECTED - press SPACE to capture",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)
        cv2.putText(display, f"Captured: {captured}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Calibration', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' ') and ret_corners:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured += 1
            print(f"Captured {captured}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if captured < 10:
        print(f"Only {captured} samples — need at least 10.")
        return

    print("\nCalibrating...")
    ret, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    print("\n=== RESULTS ===")
    print(f"  fx = {fx:.2f}")
    print(f"  fy = {fy:.2f}")
    print(f"  cx = {cx:.2f}")
    print(f"  cy = {cy:.2f}")
    print(f"  reprojection error = {ret:.4f}")
    print(f"  distortion = {dist.flatten()}")
    print("\nCopy these values into config.yaml under 'camera:'")


if __name__ == '__main__':
    main()
