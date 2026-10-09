"""Follow Drone — Main entry point.

Flow:
  1. Load config + connect MAVLink
  2. Pre-flight checks
  3. Wait for arm command (from RC or this script)
  4. Takeoff to follow altitude
  5. Switch to GUIDED mode
  6. Main loop: detection → tracking → control → MAVLink @ 10 Hz
  7. On exit: land or RTL

Run:
    python main.py --config config.yaml [--sitl] [--no-fly]

Flags:
    --sitl    Use SITL connection (UDP localhost:14550) instead of UART
    --no-fly  Run detection/tracking pipeline but don't send velocity commands
              (useful for testing on ground with motors off)
"""

import argparse
import csv
import os
import signal
import sys
import time

import cv2
import numpy as np
import yaml

# Local imports
from follow.camera import CameraThread
from follow.ocsort import OCSORT
from follow.target_selector import TargetSelector
from follow.distance import PinholeDistanceEstimator, DistanceSmoother
from follow.visual_servo import VisualServo
from follow.mavlink_client import MAVLinkClient
from follow.safety import SafetyManager, SafetyState
from follow.dashboard import Dashboard


# ─── Globals for clean shutdown ───────────────────────────────────
shutdown_flag = False


def signal_handler(signum, frame):
    global shutdown_flag
    print(f"\n[Main] Signal {signum} received, shutting down...")
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── Logging ──────────────────────────────────────────────────────

class FlightLogger:
    def __init__(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(log_dir, f"flight_{timestamp}.csv")
        self.f = open(self.csv_path, 'w', newline='')
        self.writer = csv.writer(self.f)
        self.writer.writerow([
            'time', 'state', 'mode', 'armed',
            'drone_x', 'drone_y', 'drone_z',
            'drone_vx', 'drone_vy', 'drone_vz',
            'drone_yaw', 'altitude_agl',
            'battery_v', 'battery_pct', 'gps_fix',
            'target_id', 'bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h',
            'distance_raw', 'distance_smooth',
            'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_yaw_rate',
            # rung thật + chất lượng phát hiện, thêm 2026-08-27
            'vibe_x', 'vibe_y', 'vibe_z', 'clip',
            'n_detections', 'best_conf',
        ])
        print(f"[Logger] Logging to {self.csv_path}")

    def log(self, **kwargs):
        row = [kwargs.get(k, '') for k in [
            'time', 'state', 'mode', 'armed',
            'drone_x', 'drone_y', 'drone_z',
            'drone_vx', 'drone_vy', 'drone_vz',
            'drone_yaw', 'altitude_agl',
            'battery_v', 'battery_pct', 'gps_fix',
            'target_id', 'bbox_cx', 'bbox_cy', 'bbox_w', 'bbox_h',
            'distance_raw', 'distance_smooth',
            'cmd_vx', 'cmd_vy', 'cmd_vz', 'cmd_yaw_rate',
            'vibe_x', 'vibe_y', 'vibe_z', 'clip',
            'n_detections', 'best_conf',
        ]]
        self.writer.writerow(row)
        self.f.flush()

    def close(self):
        self.f.close()


# ─── Visualization ────────────────────────────────────────────────

def draw_overlay(frame, tracks, target_id, distance, state, fps,
                  command, drone_state):
    """Draw debug overlay on frame for monitoring."""
    h, w = frame.shape[:2]
    img = frame.copy()

    # Draw all tracks
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t['box']]
        is_target = (t['id'] == target_id)
        color = (0, 255, 0) if is_target else (200, 200, 200)
        thickness = 3 if is_target else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img, f"ID{t['id']}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Center crosshair
    cv2.drawMarker(img, (w // 2, h // 2), (0, 255, 255),
                   cv2.MARKER_CROSS, 30, 1)

    # Status bar (top)
    bar_h = 80
    cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
    text_lines = [
        f"State: {state} | FPS: {fps:.1f} | Target ID: {target_id}",
        f"Distance: {distance:.2f}m" if distance else "Distance: ?",
    ]
    if command:
        vx, vy, vz, yr = command
        text_lines.append(
            f"CMD: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} yaw_rate={yr:+.2f}"
        )
    if drone_state:
        alt = drone_state.get('altitude_agl', 0)
        bat = drone_state.get('battery_voltage')
        bat_s = f"{bat:.1f}V" if bat else "?"
        mode = drone_state.get('mode', '?')
        text_lines.append(f"Mode: {mode} | Alt: {alt:.1f}m | Bat: {bat_s}")
    for i, line in enumerate(text_lines[:4]):
        cv2.putText(img, line, (10, 18 + i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return img


# ─── Main ─────────────────────────────────────────────────────────

# Tên chế độ ArduPilot -> nhóm, để bảng giám sát tô màu và giải thích.
_MODE_CLASS = {
    "STABILIZE": ("manual", "Tay lái hoàn toàn, không giữ độ cao"),
    "ACRO": ("manual", "Tay lái, không tự cân bằng"),
    "ALT_HOLD": ("assist", "Giữ độ cao, ngang vẫn tay lái"),
    "LOITER": ("assist", "Giữ vị trí + độ cao (cần GPS)"),
    "POSHOLD": ("assist", "Giữ vị trí, cho lái tay mượt"),
    "GUIDED": ("auto", "NHẬN LỆNH TỪ MÁY TÍNH ĐỒNG HÀNH — vòng bám đang chạy ở đây"),
    "GUIDED_NOGPS": ("auto", "GUIDED không GPS"),
    "AUTO": ("auto", "Chạy nhiệm vụ"),
    "CIRCLE": ("auto", "Bay vòng quanh điểm"),
    "RTL": ("safety", "Tự bay về nhà"),
    "SMART_RTL": ("safety", "Về nhà theo đường đã bay"),
    "LAND": ("safety", "Tự hạ cánh"),
    "BRAKE": ("safety", "Phanh gấp, dừng tại chỗ"),
}
_GPS_FIX = {0: "NO GPS", 1: "NO FIX", 2: "2D", 3: "3D", 4: "DGPS",
            5: "RTK float", 6: "RTK fixed"}


def build_web_state(cfg, drone, detections, target, tracks, distance,
                    safety_state, command, fps, det_ms, frame_shape,
                    selector=None):
    """Gộp trạng thái bay + phát hiện thành đúng dạng mà gcs.html đọc."""
    mode = drone.get("mode")
    cls, desc = _MODE_CLASS.get(mode, ("other", ""))
    att = drone.get("attitude") or (None, None, None)
    deg = lambda v: None if v is None else round(v * 57.2958, 1)
    hb = drone.get("last_heartbeat") or 0
    age = round(time.time() - hb, 1) if hb else None
    vibe = drone.get("vibe")

    best = max((d["score"] for d in detections), default=0.0) if detections else 0.0
    ms = {k: round(v, 2) for k, v in (det_ms or {}).items()}
    if ms:
        ms["total"] = round(sum(ms.values()), 2)

    return {
        "cam_fps": round(fps, 1),
        "dets": len(detections),
        "best_conf": round(best, 2),
        "conf": cfg["model"]["conf_threshold"],
        "temp_C": _soc_temp(),
        "throttled": _throttled(),
        "ms": ms,
        "model": os.path.basename(str(cfg["model"].get("hef", "torch"))),
        "img_size": cfg["model"]["img_size"],
        "models": [], "model_idx": 0,
        "frame_wh": [frame_shape[1], frame_shape[0]],
        # --- riêng của vòng bám ---
        "follow": {
            "state": str(getattr(safety_state, "name", safety_state)).split(".")[-1].upper(),
            "awaiting_operator": bool(getattr(selector, "awaiting_operator", False)),
            "target_id": target["id"] if target else None,
            "tracks": len(tracks),
            "distance_m": round(distance, 2) if distance else None,
            "hold_distance_m": cfg["control"]["follow_distance_m"],
            "cmd": ([round(c, 2) for c in command] if command else None),
        },
        "mav": {
            "enabled": True, "connected": age is not None and age < 3.0,
            "err": None, "heartbeat_age": age,
            "armed": drone.get("armed"), "mode": mode,
            "mode_class": cls, "mode_desc": desc,
            "alt_agl": round(drone.get("altitude_agl") or 0.0, 2),
            "climb": drone.get("climb"), "groundspeed": drone.get("groundspeed"),
            "heading": drone.get("heading"),
            "roll": deg(att[0]), "pitch": deg(att[1]), "yaw": deg(att[2]),
            "volt": drone.get("battery_voltage"),
            "batt_pct": drone.get("battery_remaining"),
            "batt_current": None, "batt_used_mah": None,
            "gps_fix": drone.get("gps_fix"),
            "gps_fix_txt": _GPS_FIX.get(drone.get("gps_fix"), "?"),
            "gps_sats": drone.get("gps_sats"),
            "gps_hdop": (round(drone.get("gps_eph", 9999) / 100.0, 2)
                         if drone.get("gps_eph", 9999) != 9999 else None),
            "vibe": list(vibe) if vibe else None,
            "clipping": list(drone.get("clipping")) if drone.get("clipping") else None,
            "ekf_var": drone.get("ekf_var"),
            "prearm_ok": drone.get("prearm_ok"),
            "prearm_msg": drone.get("prearm_msg"),
            "cpu_load": drone.get("cpu_load"), "vcc": drone.get("vcc"),
            "rc_channels": None, "msg_rate": None,
            "sensors_bad": [],
            "statustext": drone.get("statustext"),
            "statustext_ts": drone.get("statustext_ts"),
            "autopilot": "ArduPilot", "vehicle": "QUADROTOR",
        },
    }


def _soc_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000.0, 1)
    except Exception:
        return None


def _throttled():
    try:
        import subprocess
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=1).stdout.strip()
        return out.split("=")[-1] if "=" in out else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--sitl', action='store_true',
                        help="Use SITL connection (UDP)")
    parser.add_argument('--no-fly', action='store_true',
                        help="Run pipeline but don't send velocity commands")
    parser.add_argument('--no-takeoff', action='store_true',
                        help="Skip takeoff (assumes drone already in air)")
    parser.add_argument('--show', action='store_true',
                        help="Show live video window")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.sitl:
        cfg['mavlink']['connection'] = 'udpin:127.0.0.1:14550'
        print("[Main] SITL mode: connecting to UDP 14550")

    print("=" * 60)
    print("FOLLOW DRONE — autonomous person follower")
    print("=" * 60)
    print(f"  Mode:        {'SITL' if args.sitl else 'REAL HARDWARE'}")
    print(f"  Fly:         {'NO (--no-fly)' if args.no_fly else 'YES'}")
    print(f"  Takeoff:     {'NO (--no-takeoff)' if args.no_takeoff else 'YES'}")
    print(f"  Show video:  {'YES' if args.show else 'NO'}")
    print("=" * 60)

    # ─── Initialize components ───────────────────────────────────

    # ─── Detector: Hailo NPU hay PyTorch CPU ─────────────────────
    #
    # Mặc định `hailo`. PyTorch trên CPU Pi 5 chậm hơn NPU hàng chục lần, và toàn
    # bộ ngân sách độ trễ của vòng điều khiển (30 Hz) giả định inference 17,7 ms —
    # con số đó chỉ có khi chạy HEF. Giữ nhánh `torch` để chạy thử trên máy bàn.
    backend = str(cfg['model'].get('backend', 'hailo')).lower()
    print(f"\n[Main] Loading detector (backend = {backend})...")
    if backend == 'hailo':
        from follow.detector_hailo import PersonDetectorHailo
        detector = PersonDetectorHailo(
            hef=cfg['model']['hef'],
            conf_threshold=cfg['model']['conf_threshold'],
            nms_iou=cfg['model']['nms_iou'],
            img_size=cfg['model']['img_size'],
            min_box_px=cfg['model'].get('min_box_px', 0.0),
            max_aspect=cfg['model'].get('max_aspect', 0.0),
        )
    elif backend == 'torch':
                detector = PersonDetector(
            weights=cfg['model']['weights'],
            conf_threshold=cfg['model']['conf_threshold'],
            nms_iou=cfg['model']['nms_iou'],
            img_size=cfg['model']['img_size'],
            use_ema=cfg['model']['use_ema'],
            device=cfg['model']['device'],
        )
    else:
        raise ValueError(f"model.backend không hợp lệ: {backend!r} (dùng 'hailo' hoặc 'torch')")

    print("[Main] Starting camera...")
    camera = CameraThread(
        source=cfg['camera']['source'],
        width=cfg['camera']['width'],
        height=cfg['camera']['height'],
        fps=cfg['camera']['fps'],
    )
    camera.start()

    # Bảng giám sát nhúng: dùng LẠI khung hình và kết quả phát hiện của chính vòng
    # bay này, nên không tốn thêm một lượt inference nào. Xem follow/dashboard.py.
    web_cfg = cfg.get('web', {}) or {}
    dashboard = None
    if web_cfg.get('enabled', True):
        dashboard = Dashboard(port=int(web_cfg.get('port', 8080)),
                              quality=int(web_cfg.get('quality', 75)),
                              allow_retarget=bool(web_cfg.get('allow_retarget', True))
                              ).start()

    tracker = OCSORT(**cfg['tracking'])
    selector = TargetSelector(
        strategy=cfg['target_selection']['strategy'],
        lock_on_first=cfg['target_selection']['lock_on_first'],
        image_width=cfg['camera']['width'],
        image_height=cfg['camera']['height'],
        lost_grace_s=float(cfg['target_selection'].get('lost_grace_s', 3.0)),
        reacquire_s=float(cfg['target_selection'].get('reacquire_s', 5.0)),
        auto_relock=bool(cfg['target_selection'].get('auto_relock', True)),
    )
    distance_estimator = PinholeDistanceEstimator(
        fy=cfg['camera']['fy'],
        person_height_m=cfg['distance']['person_height_m'],
        min_bbox_height_px=cfg['distance']['min_bbox_height_px'],
        max_distance_m=cfg['distance']['max_distance_m'],
    )
    distance_smoother = DistanceSmoother(initial_distance=cfg['control']['follow_distance_m'])

    visual_servo = VisualServo(
        cfg=cfg['control'],
        image_width=cfg['camera']['width'],
        image_height=cfg['camera']['height'],
        fx=cfg['camera']['fx'],
        fy=cfg['camera']['fy'],
    )

    print("[Main] Connecting MAVLink...")
    mav = MAVLinkClient(
        connection_string=cfg['mavlink']['connection'],
        baud=cfg['mavlink']['baud'],
        source_system=cfg['mavlink']['source_system'],
        source_component=cfg['mavlink']['source_component'],
    )
    mav.connect(timeout=15)
    safety = SafetyManager(mav, cfg['safety'])

    logger = None
    if cfg.get('logging', {}).get('csv_log', True):
        logger = FlightLogger(cfg['logging']['log_dir'])

    # ─── Pre-flight ──────────────────────────────────────────────

    print("\n[Main] Pre-flight check...")
    time.sleep(2)  # Let state messages arrive

    if not args.no_fly:
        ok, warnings = safety.preflight_check()
        if warnings:
            print("[Main] Warnings:")
            for w in warnings:
                print(f"  - {w}")
        if not ok:
            print("[Main] Pre-flight check FAILED. Aborting.")
            print("       To override, use --no-fly to test without flight.")
            shutdown(camera, mav, logger, dashboard)
            return

    # ─── Takeoff ─────────────────────────────────────────────────

    if not args.no_fly and not args.no_takeoff:
        print("\n[Main] === TAKEOFF SEQUENCE ===")
        print("[Main] Switching to GUIDED mode...")
        if not mav.set_mode('GUIDED'):
            print("[Main] Failed to set GUIDED. Aborting.")
            shutdown(camera, mav, logger, dashboard)
            return

        print("[Main] Arming motors (hands clear of propellers!)...")
        time.sleep(2)
        if not mav.arm(timeout=cfg['mavlink']['arm_check_timeout_s']):
            print("[Main] Failed to arm. Aborting.")
            shutdown(camera, mav, logger, dashboard)
            return

        print("[Main] Taking off...")
        if not mav.takeoff(cfg['mavlink']['takeoff_altitude_m'], timeout=20):
            print("[Main] Takeoff did not reach altitude in time.")
        time.sleep(2)

    # ─── Main loop ───────────────────────────────────────────────

    print("\n[Main] Entering FOLLOW loop. Press Ctrl+C to stop and RTL.")
    target_dt = 1.0 / cfg['control']['command_rate_hz']
    fps_smooth = 0.0
    loop_count = 0
    last_health_warn = None
    last_loop_time = None
    guided_warned = False

    try:
        while not shutdown_flag:
            t_loop_start = time.time()

            # 1. Get latest frame
            frame, frame_ts = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # 2. Detect
            detections = detector.detect(frame)

            # 3. Track
            tracks = tracker.update(detections)

            # 4. Select target
            # Người vận hành bấm "đổi mục tiêu" trên bảng giám sát -> bỏ khoá hiện tại
            # và chọn lại ngay khung này. Đây là thứ DUY NHẤT trang web tác động được.
            if dashboard is not None and dashboard.take_retarget_request():
                selector.retarget()
            target = selector.select(tracks, current_time=t_loop_start)

            # 5. Get drone state
            drone_state = mav.get_state()
            altitude = drone_state.get('altitude_agl', 0)

            # 6. Compute control
            distance_raw = None
            distance_smooth = None
            command = None
            target_present = False

            if target is not None:
                target_present = True
                distance_raw = distance_estimator.estimate(target['box'])
                d_var = distance_estimator.variance(distance_raw)
                distance_smooth = distance_smoother.update(
                    distance_raw, t_loop_start, measurement_var=d_var
                )

                vx_body, vy_body, vz_ned, yaw_rate = visual_servo.compute(
                    target['box'], distance_smooth, altitude, t_loop_start
                )

                # Safety clamps
                vx_body = safety.clamp_forward_by_distance(vx_body, distance_smooth)
                vz_ned = safety.clamp_altitude(vz_ned, altitude)
                vx_body, vy_body, vz_ned, yaw_rate = safety.clamp_velocity(
                    vx_body, vy_body, vz_ned, yaw_rate
                )
                command = (vx_body, vy_body, vz_ned, yaw_rate)
            else:
                visual_servo.reset()

            # 7. Update safety state machine
            safety_state = safety.update_target_status(target_present)

            # 8. Runtime health
            ok, reason = safety.runtime_health_check()
            if not ok:
                if args.no_fly:
                    # Chạy thử trên bàn: in cảnh báo nhưng KHÔNG dừng vòng lặp, để
                    # còn kiểm tra được detect/track/điều khiển trong nhà (nơi không
                    # có GPS). Không lệnh nào được gửi ở chế độ này.
                    if reason != last_health_warn:
                        print(f"[Main] (--no-fly) bỏ qua lỗi sức khoẻ: {reason}")
                        last_health_warn = reason
                else:
                    # KHÔNG đổi chế độ bay. Người lái sở hữu mode (sự cố 2026-09-04).
                    # Dừng gửi lệnh và thoát vòng lặp; máy bay giữ nguyên chế độ
                    # hiện tại, ArduPilot tự treo khi hết GUID_TIMEOUT.
                    print(f"[Main] LỖI SỨC KHOẺ: {reason}")
                    print("[Main] NGỪNG GỬI LỆNH. Chế độ bay GIỮ NGUYÊN — "
                          "gạt công tắc sang LOITER/STABILIZE để lái tay.")
                    break
            else:
                last_health_warn = None

            # 9. Send command
            #
            # CHỐT QUYỀN ĐIỀU KHIỂN: chỉ gửi lệnh khi máy bay ĐANG Ở GUIDED.
            # Gạt công tắc sang LOITER/STABILIZE/bất cứ đâu là người lái đã giành
            # quyền — từ giây đó phần mềm phải câm. ArduPilot vốn bỏ qua setpoint
            # khi không ở GUIDED, nhưng chặn ngay từ đây thì log mới trung thực và
            # không có đường nào để phần mềm tranh lái. Xem sự cố 2026-09-04.
            in_guided = (drone_state.get('mode') == 'GUIDED')
            if not args.no_fly and in_guided:
                if safety_state == SafetyState.FOLLOW and command:
                    mav.send_velocity_body(command[0], command[1], command[2], command[3])
                elif safety_state == SafetyState.COAST and command:
                    # Bám theo dự đoán, chậm lại một nửa
                    mav.send_velocity_body(command[0]*0.5, command[1]*0.5,
                                           command[2]*0.5, command[3]*0.5)
                else:
                    # HOVER, COAST không có lệnh, và MỌI trạng thái khác: đứng yên.
                    # Đứng yên MÃI cho tới khi người lái tự gạt công tắc. Không RTL.
                    mav.send_zero_velocity()
            elif not args.no_fly and not in_guided and not guided_warned:
                print(f"[Main] Máy bay đang ở {drone_state.get('mode')}, không phải "
                      f"GUIDED — phần mềm KHÔNG gửi lệnh nào. Bạn đang lái tay.")
                guided_warned = True
            if in_guided:
                guided_warned = False

            # 10. Log
            attitude = drone_state.get('attitude') or (0, 0, 0)
            local_pos = drone_state.get('local_position') or (0, 0, 0)
            velocity = drone_state.get('velocity') or (0, 0, 0)
            bbox_cx = bbox_cy = bbox_w = bbox_h = 0
            if target:
                x1, y1, x2, y2 = target['box']
                bbox_cx = (x1 + x2) / 2
                bbox_cy = (y1 + y2) / 2
                bbox_w = x2 - x1
                bbox_h = y2 - y1
            if logger:
                logger.log(
                    time=t_loop_start, state=safety_state,
                    mode=drone_state.get('mode'),
                    armed=drone_state.get('armed'),
                    drone_x=local_pos[0], drone_y=local_pos[1], drone_z=local_pos[2],
                    drone_vx=velocity[0], drone_vy=velocity[1], drone_vz=velocity[2],
                    drone_yaw=attitude[2], altitude_agl=altitude,
                    battery_v=drone_state.get('battery_voltage'),
                    battery_pct=drone_state.get('battery_remaining'),
                    gps_fix=drone_state.get('gps_fix'),
                    target_id=target['id'] if target else '',
                    bbox_cx=bbox_cx, bbox_cy=bbox_cy,
                    bbox_w=bbox_w, bbox_h=bbox_h,
                    distance_raw=distance_raw if distance_raw else '',
                    distance_smooth=distance_smooth if distance_smooth else '',
                    cmd_vx=command[0] if command else '',
                    cmd_vy=command[1] if command else '',
                    cmd_vz=command[2] if command else '',
                    cmd_yaw_rate=command[3] if command else '',
                    # RUNG THẬT của khung máy bay, do ArduPilot đo. Đây là dữ liệu
                    # để đối chiếu mô hình rung mô phỏng lúc train với rung thật lúc
                    # bay — đúng thứ bài báo cần mà trước giờ chưa có.
                    vibe_x=(drone_state.get('vibe') or (None,))[0] or '',
                    vibe_y=(drone_state.get('vibe') or (None, None))[1]
                            if drone_state.get('vibe') else '',
                    vibe_z=(drone_state.get('vibe') or (None, None, None))[2]
                            if drone_state.get('vibe') else '',
                    clip=sum(drone_state.get('clipping') or ()) or 0,
                    n_detections=len(detections),
                    best_conf=round(max((d['score'] for d in detections), default=0.0), 3),
                )

            # 11. Visualization — vẽ MỘT lần, dùng cho cả cửa sổ tại chỗ lẫn web
            # Nhịp vòng lặp THẬT = khoảng cách giữa hai lần BẮT ĐẦU vòng. Bản cũ đo từ
            # cuối vòng trước nên bỏ sót thời gian ngủ và báo cao hơn thực tế.
            if last_loop_time:
                fps_now = 1.0 / max(1e-6, t_loop_start - last_loop_time)
                fps_smooth = (0.9 * fps_smooth + 0.1 * fps_now) if fps_smooth > 0 else fps_now
            last_loop_time = t_loop_start
            vis = None
            if args.show or dashboard is not None:
                vis = draw_overlay(
                    frame, tracks, target['id'] if target else None,
                    distance_smooth, safety_state, fps_smooth, command, drone_state,
                )
            if dashboard is not None:
                dashboard.publish(vis, build_web_state(
                    cfg, drone_state, detections, target, tracks,
                    distance_smooth, safety_state, command, fps_smooth,
                    getattr(detector, 'last_ms', {}), frame.shape, selector))
            if args.show:
                cv2.imshow('Follow Drone', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('u'):
                    selector.unlock()
                    print("[Main] Target unlocked manually")

            loop_count += 1

            # 12. Rate limit to command_rate_hz
            elapsed = time.time() - t_loop_start
            sleep_t = max(0, target_dt - elapsed)
            time.sleep(sleep_t)

    except Exception as e:
        print(f"[Main] EXCEPTION in main loop: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # ─── Shutdown sequence ───────────────────────────────────
        print("\n[Main] Shutting down...")
        if not args.no_fly:
            # KHÔNG RTL. Bản cũ gọi set_mode('RTL') ở đây; ArduPilot RTL leo lên
            # RTL_ALT (mặc định 15 m) trước khi bay về, nên chỉ cần Ctrl+C là máy
            # bay vọt lên trời — đúng cú xảy ra ngày 2026-09-04, ngay sau khi log
            # dừng. Giờ chỉ gửi vận tốc 0 rồi buông: máy bay đứng yên tại chỗ,
            # giữ nguyên chế độ, chờ người lái.
            print("[Main] Gửi vận tốc 0 và buông quyền. KHÔNG đổi chế độ bay.")
            print("[Main] ⚠️ MÁY BAY VẪN ĐANG TRÊN KHÔNG — gạt công tắc sang "
                  "LOITER/STABILIZE và hạ cánh bằng tay.")
            try:
                for _ in range(6):          # ~1 s ở 5 Hz, đủ để lệnh tới nơi
                    mav.send_zero_velocity()
                    time.sleep(0.2)
            except Exception as e:
                print(f"[Main] Lỗi khi gửi vận tốc 0: {e}")
        shutdown(camera, mav, logger, dashboard)


def shutdown(camera, mav, logger, dashboard=None):
    print("[Main] Cleanup...")
    if camera:
        camera.stop()
    if mav:
        mav.close()
    if logger:
        logger.close()
    cv2.destroyAllWindows()
    print("[Main] Done")
    if dashboard is not None:
        dashboard.stop()


if __name__ == '__main__':
    main()
