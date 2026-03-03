"""
Drone Follow-Me Controller for Jetson Nano + Pixhawk 6C
=========================================================

This is the main flight controller that:
  1. Captures frames from camera
  2. Runs PFDet-Nano person detection
  3. Selects target person (largest/closest or specific ID)
  4. Computes PID-based velocity commands
  5. Sends MAVLink commands to Pixhawk 6C via serial/UDP

Requirements:
  pip install pymavlink opencv-python numpy torch

Hardware setup:
  Jetson Nano <--serial/USB--> Pixhawk 6C
  Camera (CSI or USB) connected to Jetson Nano

IMPORTANT: Test in GUIDED mode first, then switch to actual flight!

Usage:
  python drone_follow.py --weights runs/train/best.pt --camera 0
  python drone_follow.py --weights runs/train/best.pt --camera /dev/video0 --pixhawk /dev/ttyTHS1
"""

import os
import sys
import time
import math
import argparse
import threading
import numpy as np
import cv2
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano
from utils.box_ops import decode_predictions_np, nms_numpy


# ============================================================
#  PID Controller
# ============================================================

class PIDController:
    """
    PID controller with anti-windup and derivative smoothing.

    Tuning guide:
      - Start with P only, increase until oscillation
      - Add D to dampen oscillation (usually D = P * 0.3)
      - Add small I for steady-state error (I = P * 0.05)
      - Reduce all gains by 50% for first real flights
    """
    def __init__(self, kp=1.0, ki=0.0, kd=0.0,
                 output_min=-1.0, output_max=1.0,
                 integral_max=0.5, deadband=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max
        self.deadband = deadband

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._derivative_smooth = 0.0  # EMA-smoothed derivative

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None
        self._derivative_smooth = 0.0

    def compute(self, error, current_time=None):
        """
        Compute PID output.
        error: setpoint - measured_value
        Returns: control output (clamped)
        """
        if current_time is None:
            current_time = time.time()

        # Deadband
        if abs(error) < self.deadband:
            error = 0.0

        # Time delta
        if self._prev_time is None:
            dt = 0.02  # default 50Hz
        else:
            dt = current_time - self._prev_time
            dt = max(0.001, min(dt, 0.5))  # clamp dt

        # Proportional
        p_term = self.kp * error

        # Integral with anti-windup
        self._integral += error * dt
        self._integral = max(-self.integral_max, min(self.integral_max, self._integral))
        i_term = self.ki * self._integral

        # Derivative (smoothed to reduce noise)
        if dt > 0:
            raw_derivative = (error - self._prev_error) / dt
            alpha = 0.3  # smoothing factor
            self._derivative_smooth = alpha * raw_derivative + (1 - alpha) * self._derivative_smooth
        d_term = self.kd * self._derivative_smooth

        # Total output
        output = p_term + i_term + d_term
        output = max(self.output_min, min(self.output_max, output))

        self._prev_error = error
        self._prev_time = current_time

        return output


# ============================================================
#  Target Tracker (simple centroid-based)
# ============================================================

class TargetTracker:
    """
    Simple target tracker using centroid distance + size similarity.
    Maintains a target person across frames even with brief occlusions.
    """
    def __init__(self, max_lost_frames=15, distance_threshold=0.15):
        self.target = None           # Current target dict
        self.lost_count = 0          # Frames since target was last seen
        self.max_lost = max_lost_frames
        self.dist_thr = distance_threshold
        self.target_locked = False

    def update(self, detections):
        """
        Update tracker with new detections.
        Returns: selected target detection or None
        """
        if len(detections) == 0:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.target = None
                self.target_locked = False
            return self.target if self.lost_count <= self.max_lost else None

        if self.target is None or not self.target_locked:
            # Select new target: prefer largest person (closest to drone)
            best = max(detections, key=lambda d: _box_area(d['box']))
            self.target = best
            self.lost_count = 0
            self.target_locked = True
            return best

        # Match to existing target by distance + size
        prev_cx = (self.target['box'][0] + self.target['box'][2]) / 2
        prev_cy = (self.target['box'][1] + self.target['box'][3]) / 2
        prev_area = _box_area(self.target['box'])

        best_det = None
        best_cost = float('inf')

        for det in detections:
            cx = (det['box'][0] + det['box'][2]) / 2
            cy = (det['box'][1] + det['box'][3]) / 2
            area = _box_area(det['box'])

            # Cost = distance + size difference
            dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
            size_diff = abs(area - prev_area) / max(prev_area, 1e-6)

            cost = dist + 0.5 * size_diff

            if cost < best_cost and dist < self.dist_thr:
                best_cost = cost
                best_det = det

        if best_det is not None:
            self.target = best_det
            self.lost_count = 0
            return best_det
        else:
            self.lost_count += 1
            if self.lost_count > self.max_lost:
                self.target = None
                self.target_locked = False
            return self.target if self.lost_count <= self.max_lost else None

    def force_reselect(self):
        """Force tracker to select a new target."""
        self.target = None
        self.target_locked = False
        self.lost_count = 0


def _box_area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ============================================================
#  MAVLink Communication
# ============================================================

class PixhawkController:
    """
    Controls Pixhawk 6C via MAVLink.
    Sends velocity commands for follow-me behavior.

    Connection options:
      Serial: /dev/ttyTHS1 (Jetson UART) or /dev/ttyACM0 (USB)
      UDP: udp:127.0.0.1:14550 (via mavproxy)

    SAFETY:
      - Always test with propellers removed first
      - Use low gains initially
      - Keep a manual override (RC transmitter) ready
      - Set GCS failsafe in Mission Planner
    """
    def __init__(self, connection_string='/dev/ttyTHS1', baudrate=921600):
        self.connection_string = connection_string
        self.baudrate = baudrate
        self.vehicle = None
        self.connected = False

        # Safety limits (m/s)
        self.max_horizontal_speed = 2.0   # Start conservative!
        self.max_vertical_speed = 1.0
        self.max_yaw_rate = 30.0          # deg/s

    def connect(self):
        """Connect to Pixhawk via MAVLink."""
        try:
            from pymavlink import mavutil
            print(f"[MAV] Connecting to {self.connection_string} @ {self.baudrate}...")

            self.vehicle = mavutil.mavlink_connection(
                self.connection_string,
                baud=self.baudrate,
                source_system=1,
                source_component=191
            )

            print("[MAV] Waiting for heartbeat...")
            self.vehicle.wait_heartbeat(timeout=10)
            print(f"[MAV] Connected! System {self.vehicle.target_system}, "
                  f"Component {self.vehicle.target_component}")
            self.connected = True

        except ImportError:
            print("[MAV] pymavlink not installed. Running in SIMULATION mode.")
            print("  Install: pip install pymavlink")
            self.connected = False
        except Exception as e:
            print(f"[MAV] Connection failed: {e}")
            print("[MAV] Running in SIMULATION mode.")
            self.connected = False

    def set_guided_mode(self):
        """Switch to GUIDED mode (required for velocity commands)."""
        if not self.connected:
            return
        from pymavlink import mavutil

        # Request GUIDED mode (mode number 4 for ArduCopter)
        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            4  # GUIDED mode
        )
        print("[MAV] Requested GUIDED mode")

    def send_velocity(self, vx, vy, vz, yaw_rate=0):
        """
        Send velocity command to Pixhawk.

        Coordinate frame: NED (North-East-Down) body frame
          vx: forward (+) / backward (-)  m/s
          vy: right (+) / left (-)        m/s
          vz: down (+) / up (-)           m/s
          yaw_rate: clockwise (+)         deg/s

        IMPORTANT: Commands must be sent at >2Hz or Pixhawk will stop.
        """
        # Clamp velocities
        vx = max(-self.max_horizontal_speed, min(self.max_horizontal_speed, vx))
        vy = max(-self.max_horizontal_speed, min(self.max_horizontal_speed, vy))
        vz = max(-self.max_vertical_speed, min(self.max_vertical_speed, vz))
        yaw_rate = max(-self.max_yaw_rate, min(self.max_yaw_rate, yaw_rate))

        if self.connected:
            from pymavlink import mavutil

            yaw_rate_rad = math.radians(yaw_rate)

            self.vehicle.mav.set_position_target_local_ned_send(
                0,                                              # timestamp (0 = auto)
                self.vehicle.target_system,
                self.vehicle.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,             # body frame
                0b0000_0111_11_000_111,                         # type_mask: velocity + yaw_rate only
                0, 0, 0,                                        # position (ignored)
                vx, vy, vz,                                     # velocity
                0, 0, 0,                                        # acceleration (ignored)
                0,                                              # yaw (ignored)
                yaw_rate_rad,                                   # yaw_rate
            )
        else:
            # Simulation mode: just print
            print(f"  [SIM] vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} yaw={yaw_rate:+.1f}°/s",
                  end='\r')

    def send_stop(self):
        """Emergency stop: zero velocity."""
        self.send_velocity(0, 0, 0, 0)


# ============================================================
#  Main Follow Controller
# ============================================================

class DroneFollowController:
    """
    Main controller that orchestrates:
      Camera -> Detection -> Tracking -> PID -> MAVLink

    Control strategy:
      - YAW: Keep target person centered horizontally
      - FORWARD/BACK: Maintain desired target size (distance)
      - UP/DOWN: Keep target at desired vertical position
      - LEFT/RIGHT: Optional lateral adjustment

    Target behavior:
      - target_cx = 0.5 (center of frame)
      - target_cy = 0.55 (slightly below center - better for follow)
      - target_size = 0.15-0.25 (desired bbox height ratio)
    """
    def __init__(self, model, img_size, device,
                 pixhawk=None, target_size=0.20):
        self.model = model
        self.img_size = img_size
        self.device = device
        self.pixhawk = pixhawk

        # Target setpoints (normalized)
        self.target_cx = 0.5     # Horizontal center
        self.target_cy = 0.55    # Slightly below center
        self.target_size = target_size  # Desired bbox height as fraction of frame

        # PID controllers
        # Yaw: keep person centered (horizontal error -> yaw rate)
        self.pid_yaw = PIDController(
            kp=60.0, ki=2.0, kd=15.0,
            output_min=-30, output_max=30,    # deg/s
            deadband=0.03
        )

        # Forward/back: maintain distance (size error -> forward velocity)
        self.pid_forward = PIDController(
            kp=3.0, ki=0.1, kd=1.0,
            output_min=-1.5, output_max=1.5,  # m/s
            deadband=0.02
        )

        # Altitude: keep person at target_cy (vertical error -> vertical velocity)
        self.pid_altitude = PIDController(
            kp=2.0, ki=0.05, kd=0.8,
            output_min=-0.8, output_max=0.8,  # m/s
            deadband=0.03
        )

        # Target tracker
        self.tracker = TargetTracker(max_lost_frames=15, distance_threshold=0.15)

        # Detection params
        self.conf_thr = 0.35
        self.nms_iou = 0.45

        # Safety
        self.armed = False
        self.frames_no_target = 0
        self.max_no_target = 30  # frames before stopping

    def detect(self, frame):
        """Run detection on a single frame."""
        h0, w0 = frame.shape[:2]

        # Letterbox resize
        ratio = min(self.img_size / h0, self.img_size / w0)
        nh, nw = int(h0 * ratio), int(w0 * ratio)
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        pad_h = self.img_size - nh
        pad_w = self.img_size - nw
        top = pad_h // 2
        left = pad_w // 2

        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[top:top+nh, left:left+nw] = resized

        # Inference
        inp = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        inp = inp.to(self.device)

        with torch.no_grad():
            preds = self.model(inp)

        # Decode all scales
        all_dets = []
        for si, pred in enumerate(preds):
            raw = pred[0].cpu().numpy()
            dets = decode_predictions_np(raw, self.model.strides[si], self.img_size)
            all_dets.extend(dets)

        all_dets = [d for d in all_dets if d['score'] >= self.conf_thr]
        all_dets = nms_numpy(all_dets, iou_threshold=self.nms_iou)

        # Scale back to normalized [0,1] relative to original frame
        for d in all_dets:
            x1, y1, x2, y2 = d['box']
            x1 = (x1 * self.img_size - left) / (ratio * w0)
            y1 = (y1 * self.img_size - top) / (ratio * h0)
            x2 = (x2 * self.img_size - left) / (ratio * w0)
            y2 = (y2 * self.img_size - top) / (ratio * h0)
            d['box'] = [
                max(0, min(1, x1)), max(0, min(1, y1)),
                max(0, min(1, x2)), max(0, min(1, y2))
            ]
            fx, fy = d['foot']
            d['foot'] = [
                max(0, min(1, (fx * self.img_size - left) / (ratio * w0))),
                max(0, min(1, (fy * self.img_size - top) / (ratio * h0)))
            ]

        return all_dets

    def compute_control(self, target):
        """
        Compute velocity commands from target detection.
        Returns: (vx, vy, vz, yaw_rate) or None if no target.
        """
        if target is None:
            self.frames_no_target += 1
            if self.frames_no_target > self.max_no_target:
                return (0, 0, 0, 0)  # Stop if target lost too long
            return None  # Hold previous command briefly

        self.frames_no_target = 0
        x1, y1, x2, y2 = target['box']

        # Target properties
        cx = (x1 + x2) / 2      # center x (normalized)
        cy = (y1 + y2) / 2      # center y (normalized)
        bh = y2 - y1            # bbox height (proxy for distance)

        # Errors
        err_yaw = self.target_cx - cx           # positive = target is left -> yaw left
        err_forward = self.target_size - bh     # positive = target too small -> move forward
        err_altitude = cy - self.target_cy      # positive = target below center -> move down

        t = time.time()

        # PID outputs
        yaw_rate = self.pid_yaw.compute(err_yaw, t)       # deg/s
        vx = self.pid_forward.compute(err_forward, t)       # m/s forward
        vz = self.pid_altitude.compute(err_altitude, t)     # m/s down (NED)

        return (vx, 0.0, vz, yaw_rate)

    def run(self, camera_source=0, show=True, save_path=None):
        """
        Main loop: capture -> detect -> track -> control -> send.
        """
        cap = cv2.VideoCapture(camera_source if isinstance(camera_source, int)
                                else camera_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open camera: {camera_source}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[CAM] Opened {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")

        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_path, fourcc, 30, (w, h))

        fps_smooth = 0
        self.armed = True

        print("\n[FOLLOW] Starting follow-me mode!")
        print("  Press 'q' to quit, 'r' to reselect target, 's' to emergency stop")
        print("  Press '+'/'-' to adjust confidence threshold")
        print()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 1. Detect
                detections = self.detect(frame_rgb)

                # 2. Track
                target = self.tracker.update(detections)

                # 3. Control
                cmd = self.compute_control(target)

                # 4. Send to Pixhawk
                if self.pixhawk and self.armed and cmd:
                    vx, vy, vz, yaw = cmd
                    self.pixhawk.send_velocity(vx, vy, vz, yaw)
                elif self.pixhawk and (cmd is None or not self.armed):
                    self.pixhawk.send_velocity(0, 0, 0, 0)

                dt = time.time() - t0
                fps_now = 1.0 / max(dt, 1e-6)
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_now if fps_smooth > 0 else fps_now

                # 5. Visualize
                vis = self._draw_hud(frame, detections, target, cmd, fps_smooth)

                if writer:
                    writer.write(vis)

                if show:
                    cv2.imshow("Drone Follow-Me", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        self.tracker.force_reselect()
                        print("[FOLLOW] Target reselect!")
                    elif key == ord('s'):
                        self.armed = False
                        if self.pixhawk:
                            self.pixhawk.send_stop()
                        print("[FOLLOW] EMERGENCY STOP!")
                    elif key == ord('a'):
                        self.armed = True
                        print("[FOLLOW] Armed!")
                    elif key == ord('+'):
                        self.conf_thr = min(0.9, self.conf_thr + 0.05)
                    elif key == ord('-'):
                        self.conf_thr = max(0.1, self.conf_thr - 0.05)

        finally:
            # Always stop on exit
            if self.pixhawk:
                self.pixhawk.send_stop()
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            print("[FOLLOW] Stopped.")

    def _draw_hud(self, frame, detections, target, cmd, fps):
        """Draw HUD overlay."""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw all detections
        for det in detections:
            x1, y1, x2, y2 = [int(v * s) for v, s in
                               zip(det['box'], [w, h, w, h])]
            color = (100, 100, 100)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)

        # Draw target (highlighted)
        if target:
            x1, y1, x2, y2 = [int(v * s) for v, s in
                               zip(target['box'], [w, h, w, h])]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(vis, f"TARGET {target['score']:.2f}",
                       (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.7, (0, 255, 0), 2)

            # Foot point
            fx = int(target['foot'][0] * w)
            fy = int(target['foot'][1] * h)
            cv2.circle(vis, (fx, fy), 8, (0, 0, 255), -1)

            # Crosshair at target center
            tcx = int((target['box'][0] + target['box'][2]) / 2 * w)
            tcy = int((target['box'][1] + target['box'][3]) / 2 * h)
            cv2.drawMarker(vis, (tcx, tcy), (0, 255, 255),
                          cv2.MARKER_CROSS, 20, 2)

        # Center crosshair
        cv2.drawMarker(vis, (w // 2, int(h * self.target_cy)),
                      (255, 255, 255), cv2.MARKER_CROSS, 30, 1)

        # Status bar
        status = f"FPS: {fps:.1f} | Dets: {len(detections)} | Conf: {self.conf_thr:.2f}"
        if self.armed:
            status += " | ARMED"
        else:
            status += " | DISARMED"

        cv2.putText(vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

        # Velocity commands
        if cmd:
            vx, vy, vz, yaw = cmd
            cmd_text = f"Vx:{vx:+.1f} Vz:{vz:+.1f} Yaw:{yaw:+.1f}"
            cv2.putText(vis, cmd_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 200, 0), 2)

        return vis


# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Drone Follow-Me Controller")
    parser.add_argument("--weights", required=True, help="PFDet-Nano weights (.pt)")
    parser.add_argument("--camera", default="0", help="Camera source (index or path)")
    parser.add_argument("--pixhawk", default=None,
                        help="Pixhawk connection string (e.g. /dev/ttyTHS1, udp:127.0.0.1:14550)")
    parser.add_argument("--baudrate", type=int, default=921600, help="Serial baudrate")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence")
    parser.add_argument("--target-size", type=float, default=0.20,
                        help="Desired target bbox height (0.15=far, 0.30=close)")
    parser.add_argument("--save", default=None, help="Save video to file")
    parser.add_argument("--no-show", action="store_true", help="Disable display")
    parser.add_argument("--device", default="cuda:0", help="Compute device")
    args = parser.parse_args()

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)
    cfg = ckpt['cfg']

    model = PFDetNano(
        base_c=cfg['model']['base_c'],
        num_bifpn=cfg['model'].get('num_bifpn', 2)
    ).to(device)

    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema'])
    else:
        model.load_state_dict(ckpt['model'])
    model.eval()

    img_size = cfg['model']['img_size']
    print(f"[MODEL] PFDet-Nano loaded (base_c={cfg['model']['base_c']}, img={img_size})")

    # Connect to Pixhawk
    pixhawk = None
    if args.pixhawk:
        pixhawk = PixhawkController(args.pixhawk, args.baudrate)
        pixhawk.connect()

    # Camera source
    cam_src = int(args.camera) if args.camera.isdigit() else args.camera

    # Create controller
    controller = DroneFollowController(
        model=model,
        img_size=img_size,
        device=device,
        pixhawk=pixhawk,
        target_size=args.target_size,
    )
    controller.conf_thr = args.conf

    # Run!
    controller.run(camera_source=cam_src, show=not args.no_show, save_path=args.save)


if __name__ == "__main__":
    main()
