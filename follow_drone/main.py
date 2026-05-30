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
from follow.detector import PersonDetector
from follow.ocsort import OCSORT
from follow.target_selector import TargetSelector
from follow.distance import PinholeDistanceEstimator, DistanceSmoother
from follow.visual_servo import VisualServo
from follow.mavlink_client import MAVLinkClient
from follow.safety import SafetyManager, SafetyState


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

    print("\n[Main] Loading detector...")
    detector = PersonDetector(
        weights=cfg['model']['weights'],
        conf_threshold=cfg['model']['conf_threshold'],
        nms_iou=cfg['model']['nms_iou'],
        img_size=cfg['model']['img_size'],
        use_ema=cfg['model']['use_ema'],
        device=cfg['model']['device'],
    )

    print("[Main] Starting camera...")
    camera = CameraThread(
        source=cfg['camera']['source'],
        width=cfg['camera']['width'],
        height=cfg['camera']['height'],
        fps=cfg['camera']['fps'],
    )
    camera.start()

    tracker = OCSORT(**cfg['tracking'])
    selector = TargetSelector(
        strategy=cfg['target_selection']['strategy'],
        lock_on_first=cfg['target_selection']['lock_on_first'],
        image_width=cfg['camera']['width'],
        image_height=cfg['camera']['height'],
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
            shutdown(camera, mav, logger)
            return

    # ─── Takeoff ─────────────────────────────────────────────────

    if not args.no_fly and not args.no_takeoff:
        print("\n[Main] === TAKEOFF SEQUENCE ===")
        print("[Main] Switching to GUIDED mode...")
        if not mav.set_mode('GUIDED'):
            print("[Main] Failed to set GUIDED. Aborting.")
            shutdown(camera, mav, logger)
            return

        print("[Main] Arming motors (hands clear of propellers!)...")
        time.sleep(2)
        if not mav.arm(timeout=cfg['mavlink']['arm_check_timeout_s']):
            print("[Main] Failed to arm. Aborting.")
            shutdown(camera, mav, logger)
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
    last_loop_time = time.time()

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
                print(f"[Main] Health check failed: {reason}. Switching to LOITER.")
                if not args.no_fly:
                    mav.set_mode('LOITER')
                break

            # 9. Send command
            if not args.no_fly:
                if safety_state == SafetyState.FOLLOW and command:
                    mav.send_velocity_body(command[0], command[1], command[2], command[3])
                elif safety_state == SafetyState.COAST:
                    # Continue with last command but slow it down
                    if command:
                        mav.send_velocity_body(command[0]*0.5, command[1]*0.5,
                                               command[2]*0.5, command[3]*0.5)
                    else:
                        mav.send_zero_velocity()
                elif safety_state == SafetyState.HOVER:
                    mav.send_zero_velocity()
                # LOITER and RTL are handled by FCU after mode change

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
                )

            # 11. Visualization
            if args.show:
                fps_now = 1.0 / max(0.001, time.time() - last_loop_time)
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now
                vis = draw_overlay(
                    frame, tracks, target['id'] if target else None,
                    distance_smooth, safety_state, fps_smooth, command, drone_state,
                )
                cv2.imshow('Follow Drone', vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('u'):
                    selector.unlock()
                    print("[Main] Target unlocked manually")

            last_loop_time = time.time()
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
            print("[Main] Sending zero velocity & switching to RTL...")
            try:
                mav.send_zero_velocity()
                time.sleep(0.5)
                mav.set_mode('RTL')
                print("[Main] RTL initiated. Drone should return to launch.")
                # Wait a bit so RTL command is acknowledged
                time.sleep(2)
            except Exception as e:
                print(f"[Main] RTL command error: {e}")
        shutdown(camera, mav, logger)


def shutdown(camera, mav, logger):
    print("[Main] Cleanup...")
    if camera:
        camera.stop()
    if mav:
        mav.close()
    if logger:
        logger.close()
    cv2.destroyAllWindows()
    print("[Main] Done")


if __name__ == '__main__':
    main()
