"""Chạy lại NGOẠI TUYẾN vòng bám người của follow_drone/ trên video chuyến bay thử 21/07/2026.

Nguồn: '2026-07-21 09-43-12.mkv' (cùng chuyến bay với demo_flight/flight_deploy_v17.mp4),
cắt bỏ giao diện trạm mặt đất (trên 66 px, trái 105 px) như tập flight_val.

Mỗi khung: SteadyDet-N (runs/train_v17_deploy/best.pt, @640, ngưỡng 0,25)
 -> OC-SORT (follow_drone/follow/ocsort.py)          -- đúng tham số config.yaml
 -> TargetSelector (follow_drone/follow/target_selector.py)
 -> YawPID (follow_drone/follow/pid.py)  + thang FOLLOW/COAST/HOVER như safety.py
Kết quả vẽ lên khung và ghi ra _replay.mp4, kèm replay_log.csv.
Đây là chạy lại trên video đã quay: lệnh được TÍNH, không gửi tới máy bay.

Chạy: /usr/bin/python3 replay_follow.py
"""
import csv
import math
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "follow_drone"))

from infer import load_model, preprocess, postprocess          # noqa: E402
from follow.ocsort import OCSORT                               # noqa: E402
from follow.target_selector import TargetSelector              # noqa: E402
from follow.pid import YawPID                                  # noqa: E402

SRC = os.path.join(ROOT, "2026-07-21 09-43-12.mkv")
WEIGHTS = os.path.join(ROOT, "runs", "train_v17_deploy", "best.pt")
OUT = os.path.join(HERE, "_replay.mp4")
LOG = os.path.join(HERE, "replay_log.csv")
CROP_T, CROP_L, W, H = 66, 105, 1174, 654
T_START, T_END = 18.0, 150.0          # đoạn bay ổn định (như flight_val)
CONF = float(os.environ.get("CONF", 0.25))
NMS_IOU = 0.45
IMG = int(os.environ.get("IMG", 640))
# config.yaml: min_box_px 200 cho khung 640x480 -> quy đổi theo diện tích khung 1174x654
MIN_AREA = float(os.environ.get("MIN_AREA", 200.0 * (W * H) / (640 * 480)))
MAX_ASPECT = 6.0
# Khoá mục tiêu: mặc định config.yaml 3,0 s / 5,0 s (cảnh nhiều người). Phạm vi thuyết minh là
# MỘT người trong khung, nên cửa sổ tìm lại có thể ngắn hơn -> đặt qua biến môi trường.
GRACE = float(os.environ.get("GRACE", 3.0))
REACQ = float(os.environ.get("REACQ", 5.0))
NOWRITE = bool(os.environ.get("NOWRITE"))
UI_ZONES = [(370, 590, 470, 654), (1100, 0, 1174, 100)]   # nút giao diện dưới đáy và góc phải trên
CENTER = (535, 294)   # tâm quang học = dấu thập FRAME CENTER của trạm mặt đất trên khung đã cắt
FX = 700.0 * W / 640.0               # fx của config.yaml quy đổi theo bề rộng khung
COAST_S = 3.0
F_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
F_R = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"


def main():
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, _, _ = load_model(WEIGHTS, device=dev)
    model.eval()
    strides = [4, 8, 16, 32]

    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_MSEC, T_START * 1000)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracker = OCSORT(max_age=45, min_hits=3, iou_threshold=0.3, delta_t=3)
    selector = TargetSelector(strategy="largest", lock_on_first=True, image_width=W,
                              image_height=H, lost_grace_s=GRACE, reacquire_s=REACQ,
                              auto_relock=True)
    yaw_pid = YawPID(image_width=W, fx=FX, Kp=2.0, Ki=0.0, Kd=0.3, rate_max=1.0)

    f_big, f_mid, f_small = (ImageFont.truetype(F_B, 22), ImageFont.truetype(F_R, 19),
                             ImageFont.truetype(F_R, 16))
    trail, lost_since, n = [], None, 0
    stats = {"frames": 0, "follow": 0, "coast": 0, "hover": 0}
    logf = open(LOG, "w", newline="")
    log = csv.writer(logf)
    log.writerow(["t", "n_det", "target_id", "state", "cx", "cy", "err_px", "yaw_err_deg", "yaw_rate_deg_s"])

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = T_START + n / fps
        if t > T_END:
            break
        n += 1
        img = frame[CROP_T:CROP_T + H, CROP_L:CROP_L + W].copy()

        with torch.no_grad():
            inp, ratio, pad, orig = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG)
            preds = model(inp.to(dev))
        dets = postprocess(preds, strides, IMG, conf_thr=CONF, nms_iou=NMS_IOU,
                           ratio=ratio, pad=pad, orig_size=orig)
        def _ok(b):
            bw, bh = b[2] - b[0], b[3] - b[1]
            cxb, cyb = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            # bỏ các biểu tượng giao diện của trạm mặt đất vẽ đè lên video (không phải ảnh camera)
            for zx1, zy1, zx2, zy2 in UI_ZONES:
                if zx1 <= cxb <= zx2 and zy1 <= cyb <= zy2:
                    return False
            return bw * bh >= MIN_AREA and max(bw, bh) / max(1.0, min(bw, bh)) <= MAX_ASPECT
        dets = [d for d in dets if _ok(d["box"])]
        tracks = tracker.update([{"box": d["box"], "score": d["score"]} for d in dets])
        target = selector.select(tracks, current_time=t)

        yaw_err = yaw_rate = err_px = None
        if target is not None:
            lost_since = None
            state = "FOLLOW"
            x1, y1, x2, y2 = target["box"]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            err_px = cx - CENTER[0]
            yaw_err = math.degrees(err_px / FX)
            yaw_rate = math.degrees(yaw_pid.compute(cx - CENTER[0] + W / 2, t))
            trail.append((int(cx), int(cy)))
        else:
            yaw_pid.reset()
            lost_since = t if lost_since is None else lost_since
            state = "COAST" if t - lost_since < COAST_S else "HOVER"
            cx = cy = None
        # Khi mất phát hiện trong thời gian giữ khoá, bộ chọn vẫn giữ ID; OC-SORT vẫn có hộp dự đoán.
        pred_box = None
        if target is None and selector.locked_id is not None:
            trk = tracker.get_track_by_id(selector.locked_id)
            if trk is not None:
                pred_box = [float(v) for v in trk.get_box()]
        trail = trail[-45:] if state != "HOVER" else []
        stats["frames"] += 1
        stats[state.lower()] += 1
        log.writerow([f"{t:.3f}", len(dets), target["id"] if target else "", state,
                      "" if cx is None else f"{cx:.1f}", "" if cy is None else f"{cy:.1f}",
                      "" if err_px is None else f"{err_px:.1f}",
                      "" if yaw_err is None else f"{yaw_err:.2f}",
                      "" if yaw_rate is None else f"{yaw_rate:.2f}"])
        if NOWRITE:
            continue

        # ---------------- vẽ ----------------
        for tr in tracks:
            if target is None or tr["id"] != target["id"]:
                bx = [int(v) for v in tr["box"]]
                cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (200, 200, 200), 2)
        for i in range(1, len(trail)):
            cv2.line(img, trail[i - 1], trail[i], (0, 200, 255), 2)
        c0 = CENTER
        color = {"FOLLOW": (80, 200, 60), "COAST": (0, 170, 255), "HOVER": (60, 60, 230)}[state]
        if target is not None:
            bx = [int(v) for v in target["box"]]
            cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (80, 220, 60), 3)
            cv2.arrowedLine(img, c0, (int(cx), int(cy)), (0, 200, 255), 2, tipLength=0.04)
        elif pred_box is not None:
            bx = [int(v) for v in pred_box]
            cv2.rectangle(img, (bx[0], bx[1]), (bx[2], bx[3]), (0, 170, 255), 2)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil, "RGBA")
        d.rounded_rectangle((12, 12, 372, 170), 8, fill=(0, 0, 0, 150))
        d.text((24, 20), f"TRẠNG THÁI: {state}", font=f_big, fill=tuple(int(c) for c in color[::-1]))
        lock_txt = (f"Mục tiêu khoá: ID {target['id']}" if target else
                    (f"Mục tiêu khoá: ID {selector.locked_id} (dự đoán)" if selector.locked_id is not None else "Mục tiêu khoá: —"))
        d.text((24, 52), lock_txt, font=f_mid, fill="white")
        d.text((24, 78), f"Số người phát hiện: {len(dets)}", font=f_mid, fill="white")
        d.text((24, 104), "Sai số hướng: " + ("—" if yaw_err is None else f"{yaw_err:+.1f}°"), font=f_mid, fill="white")
        d.text((24, 130), "Lệnh quay hướng: " + ("0 (giữ vị trí)" if yaw_rate is None else f"{yaw_rate:+.1f}°/s"), font=f_mid, fill="white")
        if target is not None:
            x1, y1 = int(target["box"][0]), int(target["box"][1])
            d.rectangle((x1, max(0, y1 - 24), x1 + 150, max(0, y1 - 24) + 22), fill=(40, 160, 40, 220))
            d.text((x1 + 5, max(0, y1 - 23)), f"ID {target['id']}  {target['score']:.2f}", font=f_small, fill="white")
        img = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        writer.write(img)

    writer.release()
    logf.close()
    print("frames", stats)


if __name__ == "__main__":
    main()
