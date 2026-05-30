"""MAVLink client wrapper for ArduPilot Copter.

Handles:
  - Connection to Pixhawk over UART/UDP
  - Mode switching (GUIDED, LOITER, RTL, LAND)
  - Arm/disarm
  - Takeoff
  - Velocity setpoint sending (10 Hz target)
  - State polling (position, attitude, battery, GPS)
  - Heartbeat thread

Uses pymavlink (low-level, official from ArduPilot org).
"""

import math
import threading
import time
from collections import deque

from pymavlink import mavutil


class MAVLinkClient:
    def __init__(self, connection_string, baud=921600,
                 source_system=255, source_component=1):
        self.connection_string = connection_string
        self.baud = baud
        self.source_system = source_system
        self.source_component = source_component

        self.master = None
        self.target_system = None
        self.target_component = None

        # Latest state (updated by background thread)
        self.lock = threading.Lock()
        self.state = {
            'local_position': None,    # NED (x, y, z) meters
            'velocity': None,           # NED (vx, vy, vz) m/s
            'attitude': None,           # (roll, pitch, yaw) rad
            'battery_voltage': None,    # Volt
            'battery_remaining': None,  # %
            'gps_fix': 0,               # 0=no, 3=3D
            'gps_eph': 9999,            # HDOP × 100
            'gps_sats': 0,
            'mode': None,
            'armed': False,
            'system_status': 0,
            'last_heartbeat': 0.0,
            'altitude_agl': 0.0,        # meters
            'home_position': None,
        }

        self.running = False
        self.recv_thread = None
        self.heartbeat_thread = None

    # ───── Connection ─────────────────────────────────────────────

    def connect(self, timeout=15):
        print(f"[MAVLink] Connecting to {self.connection_string} (baud={self.baud})...")
        if self.connection_string.startswith(('udp', 'tcp')):
            self.master = mavutil.mavlink_connection(
                self.connection_string,
                source_system=self.source_system,
                source_component=self.source_component,
            )
        else:
            self.master = mavutil.mavlink_connection(
                self.connection_string, baud=self.baud,
                source_system=self.source_system,
                source_component=self.source_component,
            )

        print("[MAVLink] Waiting for heartbeat...")
        hb = self.master.wait_heartbeat(timeout=timeout)
        if hb is None:
            raise RuntimeError("No heartbeat received")
        self.target_system = self.master.target_system
        self.target_component = self.master.target_component
        print(f"[MAVLink] Connected: system={self.target_system}, "
              f"component={self.target_component}")

        # Request data streams
        self._request_data_streams()

        # Start background threads
        self.running = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()

        # Wait briefly for first state messages
        time.sleep(1.0)

    def _request_data_streams(self):
        """Request high-rate data streams from FCU."""
        # Set message intervals (microseconds)
        intervals = {
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: 100000,   # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 100000,             # 10 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 200000,  # 5 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS: 1000000,      # 1 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_GPS_RAW_INT: 200000,          # 5 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS: 500000,           # 2 Hz
            mavutil.mavlink.MAVLINK_MSG_ID_HOME_POSITION: 5000000,       # 0.2 Hz
        }
        for msg_id, interval_us in intervals.items():
            self.master.mav.command_long_send(
                self.target_system, self.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                msg_id, interval_us,
                0, 0, 0, 0, 0,
            )

    def _heartbeat_loop(self):
        """Send heartbeat to FCU at 1 Hz."""
        while self.running:
            try:
                self.master.mav.heartbeat_send(
                    mavutil.mavlink.MAV_TYPE_GCS,
                    mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                    0, 0, 0,
                )
            except Exception as e:
                print(f"[MAVLink] Heartbeat send error: {e}")
            time.sleep(1.0)

    def _recv_loop(self):
        """Receive messages and update state dict."""
        while self.running:
            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
                if msg is None:
                    continue
                self._handle_message(msg)
            except Exception as e:
                print(f"[MAVLink] Recv error: {e}")
                time.sleep(0.1)

    def _handle_message(self, msg):
        mtype = msg.get_type()
        with self.lock:
            if mtype == 'HEARTBEAT':
                self.state['last_heartbeat'] = time.time()
                self.state['armed'] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                self.state['system_status'] = msg.system_status
                # Mode mapping
                try:
                    mode_map = self.master.mode_mapping()
                    if mode_map:
                        rev_map = {v: k for k, v in mode_map.items()}
                        self.state['mode'] = rev_map.get(msg.custom_mode, str(msg.custom_mode))
                except Exception:
                    pass
            elif mtype == 'LOCAL_POSITION_NED':
                self.state['local_position'] = (msg.x, msg.y, msg.z)
                self.state['velocity'] = (msg.vx, msg.vy, msg.vz)
                self.state['altitude_agl'] = -msg.z  # NED Z is down
            elif mtype == 'ATTITUDE':
                self.state['attitude'] = (msg.roll, msg.pitch, msg.yaw)
            elif mtype == 'BATTERY_STATUS':
                # voltages[0] is in millivolts
                if msg.voltages and msg.voltages[0] != 65535:
                    self.state['battery_voltage'] = msg.voltages[0] / 1000.0
                self.state['battery_remaining'] = msg.battery_remaining
            elif mtype == 'GPS_RAW_INT':
                self.state['gps_fix'] = msg.fix_type
                self.state['gps_eph'] = msg.eph
                self.state['gps_sats'] = msg.satellites_visible
            elif mtype == 'GLOBAL_POSITION_INT':
                # Use relative altitude as backup AGL
                if self.state.get('altitude_agl', 0) == 0:
                    self.state['altitude_agl'] = msg.relative_alt / 1000.0
            elif mtype == 'HOME_POSITION':
                self.state['home_position'] = (msg.latitude, msg.longitude, msg.altitude)

    # ───── Getters (thread-safe) ─────────────────────────────────

    def get_state(self):
        with self.lock:
            return dict(self.state)

    def heartbeat_age(self):
        with self.lock:
            t = self.state['last_heartbeat']
        return time.time() - t if t else float('inf')

    # ───── Mode control ──────────────────────────────────────────

    def set_mode(self, mode_name):
        """Set ArduPilot mode by name. Returns True if confirmed."""
        mode_map = self.master.mode_mapping()
        if mode_name not in mode_map:
            raise ValueError(f"Unknown mode: {mode_name}. Available: {list(mode_map)}")
        mode_id = mode_map[mode_name]

        self.master.mav.set_mode_send(
            self.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )

        # Confirm
        deadline = time.time() + 5.0
        while time.time() < deadline:
            with self.lock:
                if self.state.get('mode') == mode_name:
                    print(f"[MAVLink] Mode set: {mode_name}")
                    return True
            time.sleep(0.1)
        print(f"[MAVLink] WARNING: mode {mode_name} not confirmed within 5s")
        return False

    # ───── Arm / disarm ──────────────────────────────────────────

    def arm(self, force=False, timeout=10):
        print("[MAVLink] Arming...")
        param2 = 21196 if force else 0  # magic number to force arm
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, param2, 0, 0, 0, 0, 0,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                if self.state['armed']:
                    print("[MAVLink] Armed")
                    return True
            time.sleep(0.2)
        print("[MAVLink] WARNING: arm not confirmed")
        return False

    def disarm(self, force=False):
        print("[MAVLink] Disarming...")
        param2 = 21196 if force else 0
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, param2, 0, 0, 0, 0, 0,
        )
        time.sleep(1.0)

    # ───── Takeoff ───────────────────────────────────────────────

    def takeoff(self, altitude_m, timeout=15):
        print(f"[MAVLink] Takeoff to {altitude_m}m")
        self.master.mav.command_long_send(
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude_m,
        )
        # Wait until close to target altitude
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self.lock:
                alt = self.state.get('altitude_agl', 0)
            if alt >= altitude_m * 0.95:
                print(f"[MAVLink] Reached takeoff altitude ({alt:.2f}m)")
                return True
            time.sleep(0.5)
        print(f"[MAVLink] WARNING: takeoff timeout, alt={alt:.2f}m")
        return False

    # ───── Velocity setpoint ─────────────────────────────────────

    def send_velocity_ned(self, vx, vy, vz, yaw_rate=0.0):
        """Send velocity setpoint in local NED frame.

        vx, vy, vz: NED velocity (vz negative = up)
        yaw_rate:   rad/s (positive = clockwise viewed from above)
        """
        # type_mask: ignore position, ignore acceleration, use velocity + yaw_rate
        # Bit positions (LSB):
        #  0-2: ignore x, y, z position
        #  3-5: ignore vx, vy, vz velocity
        #  6-8: ignore ax, ay, az acceleration
        #  9:   force (not used)
        #  10:  ignore yaw
        #  11:  ignore yaw_rate
        # We want: ignore position (1,1,1), use velocity (0,0,0), ignore accel (1,1,1),
        #         ignore yaw (1), use yaw_rate (0)
        type_mask = (
            0b0000_1111_1100_0111
            # 0xDC7
        )
        type_mask = 0b0000_1011_1100_0111  # ignore pos+accel+yaw, use vel+yaw_rate

        self.master.mav.set_position_target_local_ned_send(
            0,  # time_boot_ms
            self.target_system, self.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            type_mask,
            0, 0, 0,        # x, y, z (ignored)
            vx, vy, vz,     # velocity
            0, 0, 0,        # acceleration (ignored)
            0,              # yaw (ignored)
            yaw_rate,       # rad/s
        )

    def send_velocity_body(self, vx_body, vy_body, vz_body, yaw_rate=0.0):
        """Send velocity in body frame (forward-right-down).

        Converts to NED using current yaw.
        """
        with self.lock:
            attitude = self.state.get('attitude')
        yaw = attitude[2] if attitude else 0.0

        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx_ned = vx_body * cos_y - vy_body * sin_y
        vy_ned = vx_body * sin_y + vy_body * cos_y
        vz_ned = vz_body  # NED z down, body z down — same
        self.send_velocity_ned(vx_ned, vy_ned, vz_ned, yaw_rate)

    def send_zero_velocity(self):
        self.send_velocity_ned(0, 0, 0, 0)

    # ───── Cleanup ───────────────────────────────────────────────

    def close(self):
        self.running = False
        time.sleep(0.5)
        if self.recv_thread:
            self.recv_thread.join(timeout=1.0)
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=1.0)
        if self.master:
            self.master.close()
        print("[MAVLink] Disconnected")
