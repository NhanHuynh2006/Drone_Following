"""
mav_telem.py — DOC telemetry ArduPilot cho dashboard. CHI DOC, KHONG GUI LENH.

Co y khong co arm/set_mode/takeoff: trang web la de QUAN SAT. Muon dieu khien thi
dung follow_drone/ (co safety ladder rieng). Mot trang web ma arm duoc drone la
mot tai nan dang cho.

ArduPilot khac PX4:
  - Ten che do lay tu `mavutil.mode_mapping()` cua chinh xe (STABILIZE / ALT_HOLD /
    LOITER / GUIDED / RTL / LAND / POSHOLD / AUTO...). PX4 dung custom_mode dang
    main/sub khac han -> khong hardcode duoc, phai hoi xe.
  - ArduPilot khong co che do OFFBOARD; dieu khien tu companion = **GUIDED**.
"""
import threading
import time

# Phan loai che do ArduPilot Copter -> mau sac + y nghia tren dashboard
MODE_CLASS = {
    "STABILIZE": ("manual", "Tay lai hoan toan, khong giu do cao"),
    "ACRO":      ("manual", "Tay lai, khong tu can bang"),
    "ALT_HOLD":  ("assist", "Giu do cao, ngang van tay lai"),
    "LOITER":    ("assist", "Giu vi tri + do cao (can GPS)"),
    "POSHOLD":   ("assist", "Giu vi tri, cho lai tay muot"),
    "DRIFT":     ("assist", "Lai kieu xe hoi"),
    "SPORT":     ("assist", "Giu do cao, lai the thao"),
    "GUIDED":    ("auto",   "NHAN LENH TU COMPANION — che do bay bam chay o day"),
    "GUIDED_NOGPS": ("auto", "GUIDED khong GPS"),
    "AUTO":      ("auto",   "Chay mission"),
    "CIRCLE":    ("auto",   "Bay vong quanh diem"),
    "RTL":       ("safety", "Tu bay ve nha"),
    "SMART_RTL": ("safety", "Ve nha theo duong da bay"),
    "LAND":      ("safety", "Tu ha canh"),
    "BRAKE":     ("safety", "Phanh gap, dung tai cho"),
    "THROW":     ("manual", "Nem de cat canh"),
    "AVOID_ADSB":("safety", "Tranh may bay khac"),
    "FLIP":      ("manual", "Lat"),
    "AUTOTUNE":  ("auto",   "Tu tinh chinh PID"),
}

GPS_FIX = {0: "NO GPS", 1: "NO FIX", 2: "2D", 3: "3D", 4: "DGPS",
           5: "RTK float", 6: "RTK fixed"}

# Tên tiếng Việt cho các bit của MAV_SYS_STATUS_SENSOR.
#
# ⚠️ ĐỪNG gõ tay giá trị bit. Bản đầu của file này gõ tay và lệch: 0x400 bị gán
# nhãn "thu RC" (thật ra là ANGULAR_RATE_CONTROL, RC_RECEIVER là 0x10000), và bit
# prearm dùng 0x20000000 (thật ra là OBSTACLE_AVOIDANCE; PREARM_CHECK là 0x10000000)
# nên prearm không bao giờ đọc được. Nay tra thẳng enum của pymavlink.
_VN = {
    "SENSOR_3D_GYRO": "con quay hồi chuyển", "SENSOR_3D_ACCEL": "gia tốc kế",
    "SENSOR_3D_MAG": "la bàn", "SENSOR_ABSOLUTE_PRESSURE": "cảm biến áp suất",
    "SENSOR_GPS": "GPS", "SENSOR_OPTICAL_FLOW": "cảm biến quang lưu",
    "SENSOR_LASER_POSITION": "cảm biến khoảng cách",
    "SENSOR_ANGULAR_RATE_CONTROL": "điều khiển tốc độ góc",
    "SENSOR_ATTITUDE_STABILIZATION": "ổn định tư thế",
    "SENSOR_YAW_POSITION": "giữ hướng", "SENSOR_Z_ALTITUDE_CONTROL": "giữ độ cao",
    "SENSOR_XY_POSITION_CONTROL": "giữ vị trí ngang",
    "SENSOR_MOTOR_OUTPUTS": "đầu ra động cơ", "SENSOR_RC_RECEIVER": "thu RC",
    "SENSOR_3D_GYRO2": "con quay hồi chuyển 2", "SENSOR_3D_ACCEL2": "gia tốc kế 2",
    "SENSOR_3D_MAG2": "la bàn 2", "GEOFENCE": "hàng rào địa lý", "AHRS": "AHRS",
    "TERRAIN": "dữ liệu địa hình", "LOGGING": "ghi log", "SENSOR_BATTERY": "pin",
    "SENSOR_PROXIMITY": "cảm biến cận", "PREARM_CHECK": "kiểm tra trước khi arm",
    "OBSTACLE_AVOIDANCE": "tránh vật cản", "SENSOR_PROPULSION": "hệ đẩy",
}


def _sensor_bits():
    """[(bit, tên tiếng Việt)] lấy từ enum thật của pymavlink."""
    try:
        from pymavlink import mavutil
        out = []
        for v, e in mavutil.mavlink.enums["MAV_SYS_STATUS_SENSOR"].items():
            if not v or e.name.endswith("ENUM_END") or v & (v - 1):   # bỏ 0 và giá trị không phải 1 bit
                continue
            key = e.name.replace("MAV_SYS_STATUS_", "")
            out.append((v, _VN.get(key, key.lower().replace("_", " "))))
        return sorted(out)
    except Exception:
        return []


SENSOR_BITS = _sensor_bits()
PREARM_BIT = 0x10000000          # MAV_SYS_STATUS_PREARM_CHECK


class MavTelem:
    """Doc telemetry ArduPilot o luong nen. state luon doc duoc, ke ca khi chua noi."""

    def __init__(self, conn=None, baud=57600, source_system=254):
        self.conn, self.baud, self.source_system = conn, baud, source_system
        self.lock = threading.Lock()
        self.state = self._blank()
        self.running = False
        self.master = None
        self._mode_map_rev = None

    @staticmethod
    def _blank():
        return dict(
            enabled=False, connected=False, err=None,
            heartbeat_age=None, armed=None, mode=None,
            mode_class=None, mode_desc=None,
            alt_agl=None, climb=None,
            groundspeed=None, airspeed=None, heading=None,
            roll=None, pitch=None, yaw=None,
            volt=None, current=None, batt_pct=None,
            gps_fix=None, gps_fix_txt=None, gps_sats=None, gps_hdop=None,
            lat=None, lon=None,
            local_pos=None, vel_ned=None,
            rc_rssi=None, rc_channels=None, sys_status=None,
            statustext=None, statustext_ts=None, statustext_sev=None,
            # rung — trục chính của đề tài, ArduPilot gửi sẵn qua VIBRATION
            vibe=None, clipping=None,
            # sức khoẻ
            ekf_flags=None, ekf_var=None, prearm_ok=None, prearm_msg=None,
            cpu_load=None, sensors_bad=None, batt_current=None, batt_used_mah=None,
            vcc=None, msg_rate=None,
            autopilot="ArduPilot", vehicle=None,
        )

    # ------------------------------------------------------------------
    def start(self):
        if not self.conn:
            return self
        self.running = True
        with self.lock:
            self.state["enabled"] = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self):
        self.running = False

    def get(self):
        with self.lock:
            s = dict(self.state)
        if s.get("_hb"):
            s["heartbeat_age"] = round(time.time() - s["_hb"], 1)
            s["connected"] = s["heartbeat_age"] < 3.0
            s.pop("_hb", None)
        return s

    # ------------------------------------------------------------------
    def _connect(self):
        from pymavlink import mavutil
        if self.conn.startswith(("udp:", "udpin:", "udpout:", "tcp:")):
            m = mavutil.mavlink_connection(self.conn, source_system=self.source_system)
        else:
            m = mavutil.mavlink_connection(self.conn, baud=self.baud,
                                           source_system=self.source_system)
        m.wait_heartbeat(timeout=15)
        return m

    # Bản tin dashboard cần. Chỉ XIN NẾU THIẾU.
    NEED = ("ATTITUDE", "VFR_HUD", "GLOBAL_POSITION_INT", "LOCAL_POSITION_NED",
            "SYS_STATUS", "GPS_RAW_INT", "RC_CHANNELS", "VIBRATION",
            "EKF_STATUS_REPORT", "BATTERY_STATUS")
    WANT_US = {"ATTITUDE": 100000, "VFR_HUD": 100000, "GLOBAL_POSITION_INT": 100000,
               "LOCAL_POSITION_NED": 200000, "SYS_STATUS": 200000,
               "GPS_RAW_INT": 200000, "RC_CHANNELS": 500000,
               "VIBRATION": 500000, "EKF_STATUS_REPORT": 500000,
               "BATTERY_STATUS": 500000}

    def _request_missing(self, seen):
        """CHỈ xin những bản tin chưa thấy.

        Đừng xin bừa tất cả. ArduPilot qua USB đã tự phát ATTITUDE / VFR_HUD /
        GLOBAL_POSITION_INT ở 10 Hz và SYS_STATUS / GPS_RAW_INT ở 5 Hz. Gửi
        SET_MESSAGE_INTERVAL với chu kỳ DÀI hơn sẽ **bóp** nhịp đang tốt xuống —
        đúng lỗi của bản đầu file này, phát hiện khi đo thật trên Pixhawk 6C
        ngày 2026-08-27 (VFR_HUD 10→5 Hz, GPS_RAW_INT 5→2 Hz).
        """
        from pymavlink import mavutil
        comp = self.master.target_component or 1     # 0 = broadcast; nhắm autopilot cho gọn
        for name in self.NEED:
            if seen.get(name):
                continue
            mid = getattr(mavutil.mavlink, f"MAVLINK_MSG_ID_{name}", None)
            if mid is None:
                continue
            self.master.mav.command_long_send(
                self.master.target_system, comp,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                mid, self.WANT_US.get(name, 500000), 0, 0, 0, 0, 0)

    def _loop(self):
        while self.running:
            try:
                self.master = self._connect()
                with self.lock:
                    self.state["err"] = None
                try:
                    self._mode_map_rev = {v: k for k, v in
                                          (self.master.mode_mapping() or {}).items()}
                except Exception:
                    self._mode_map_rev = None
                # Nghe 3 giây trước để biết xe TỰ phát sẵn những gì, rồi mới xin phần thiếu.
                seen, t_probe = {}, time.time()
                while self.running and time.time() - t_probe < 3.0:
                    msg = self.master.recv_match(blocking=True, timeout=1.0)
                    if msg is not None:
                        seen[msg.get_type()] = True
                        self._handle(msg)
                self._request_missing(seen)

                n_msg, t_rate = 0, time.time()
                while self.running:
                    msg = self.master.recv_match(blocking=True, timeout=2.0)
                    if msg is None:
                        continue
                    self._handle(msg)
                    n_msg += 1
                    dt = time.time() - t_rate
                    if dt > 2.0:
                        with self.lock:
                            self.state["msg_rate"] = round(n_msg / dt, 1)
                        n_msg, t_rate = 0, time.time()
            except Exception as e:                     # rot ket noi -> thu lai
                with self.lock:
                    self.state["err"] = f"{type(e).__name__}: {e}"
                    self.state["connected"] = False
                time.sleep(2.0)

    def _handle(self, msg):
        t = msg.get_type()
        with self.lock:
            s = self.state
            if t == "HEARTBEAT":
                from pymavlink import mavutil
                s["_hb"] = time.time()
                s["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                s["sys_status"] = msg.system_status
                name = None
                if self._mode_map_rev:
                    name = self._mode_map_rev.get(msg.custom_mode)
                if name is None:
                    try:
                        self._mode_map_rev = {v: k for k, v in
                                              (self.master.mode_mapping() or {}).items()}
                        name = self._mode_map_rev.get(msg.custom_mode)
                    except Exception:
                        pass
                s["mode"] = name or f"mode#{msg.custom_mode}"
                if s.get("vehicle") is None:
                    try:
                        from pymavlink import mavutil as _mu
                        s["vehicle"] = _mu.mavlink.enums["MAV_TYPE"][msg.type].name \
                                          .replace("MAV_TYPE_", "")
                    except Exception:
                        pass
                cls, desc = MODE_CLASS.get(s["mode"], ("other", ""))
                s["mode_class"], s["mode_desc"] = cls, desc
            elif t == "ATTITUDE":
                s["roll"] = round(msg.roll * 57.2958, 1)
                s["pitch"] = round(msg.pitch * 57.2958, 1)
                s["yaw"] = round((msg.yaw * 57.2958) % 360, 1)
            elif t == "VFR_HUD":
                s["alt_agl"] = round(msg.alt, 2)
                s["climb"] = round(msg.climb, 2)
                s["groundspeed"] = round(msg.groundspeed, 2)
                s["airspeed"] = round(msg.airspeed, 2)
                s["heading"] = msg.heading
            elif t == "GLOBAL_POSITION_INT":
                s["lat"] = msg.lat / 1e7
                s["lon"] = msg.lon / 1e7
                if s.get("alt_agl") is None:
                    s["alt_agl"] = round(msg.relative_alt / 1000.0, 2)
            elif t == "LOCAL_POSITION_NED":
                s["local_pos"] = [round(msg.x, 2), round(msg.y, 2), round(msg.z, 2)]
                s["vel_ned"] = [round(msg.vx, 2), round(msg.vy, 2), round(msg.vz, 2)]
            elif t == "SYS_STATUS":
                if s.get("volt") is None and msg.voltage_battery not in (0, 65535):
                    s["volt"] = round(msg.voltage_battery / 1000.0, 2)
                if msg.battery_remaining != -1:
                    s["batt_pct"] = msg.battery_remaining
                s["cpu_load"] = round(msg.load / 10.0, 1)
                # cảm biến nào present+enabled mà KHÔNG healthy
                bad = [nm for bit, nm in SENSOR_BITS
                       if (msg.onboard_control_sensors_enabled & bit)
                       and not (msg.onboard_control_sensors_health & bit)]
                s["sensors_bad"] = bad
                if msg.onboard_control_sensors_present & PREARM_BIT:
                    ok = bool(msg.onboard_control_sensors_health & PREARM_BIT)
                    s["prearm_ok"] = ok
                    if ok:
                        s["prearm_msg"] = None
            elif t == "GPS_RAW_INT":
                s["gps_fix"] = msg.fix_type
                s["gps_fix_txt"] = GPS_FIX.get(msg.fix_type, str(msg.fix_type))
                s["gps_sats"] = msg.satellites_visible
                s["gps_hdop"] = round(msg.eph / 100.0, 2) if msg.eph != 65535 else None
            elif t == "RC_CHANNELS":
                s["rc_rssi"] = msg.rssi
                s["rc_channels"] = msg.chancount
            elif t == "VIBRATION":
                # Rung là trục nghiên cứu của đề tài. ArduPilot ngưỡng: <30 tốt,
                # 30-60 cảnh báo, >60 xấu (m/s^2). clipping phải bằng 0.
                s["vibe"] = [round(msg.vibration_x, 2), round(msg.vibration_y, 2),
                             round(msg.vibration_z, 2)]
                s["clipping"] = [msg.clipping_0, msg.clipping_1, msg.clipping_2]
            elif t == "EKF_STATUS_REPORT":
                s["ekf_flags"] = msg.flags
                s["ekf_var"] = {
                    "vel": round(msg.velocity_variance, 3),
                    "pos_h": round(msg.pos_horiz_variance, 3),
                    "pos_v": round(msg.pos_vert_variance, 3),
                    "mag": round(msg.compass_variance, 3),
                }
            elif t == "BATTERY_STATUS":
                if msg.voltages and msg.voltages[0] != 65535:
                    s["volt"] = round(msg.voltages[0] / 1000.0, 2)
                if msg.current_battery != -1:
                    s["batt_current"] = round(msg.current_battery / 100.0, 2)
                s["batt_used_mah"] = msg.current_consumed
                if msg.battery_remaining != -1:
                    s["batt_pct"] = msg.battery_remaining
            elif t == "POWER_STATUS":
                s["vcc"] = round(msg.Vcc / 1000.0, 2)
            elif t == "STATUSTEXT":
                txt = msg.text.decode() if isinstance(msg.text, bytes) else str(msg.text)
                s["statustext"] = txt
                s["statustext_ts"] = time.time()
                s["statustext_sev"] = msg.severity
                # ArduPilot báo lý do không arm được qua đúng kênh này.
                if txt.startswith("PreArm:") or txt.startswith("Arm:"):
                    s["prearm_ok"] = False
                    s["prearm_msg"] = txt
