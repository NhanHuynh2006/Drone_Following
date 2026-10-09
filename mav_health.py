"""Kiem tra suc khoe may bay: prearm, EKF, rung, cam bien. KHONG gui lenh dieu khien."""
import sys, time
from pymavlink import mavutil

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
m = mavutil.mavlink_connection(dev, baud=115200, source_system=254)
m.wait_heartbeat(timeout=12)
print(f"[health] noi duoc system={m.target_system}\n")

# ⚠️ KHONG go tay bit — ban dau go tay bi lech nhan (0x400 khong phai RC receiver,
# 0x20000000 khong phai prearm). Lay thang tu enum cua pymavlink.
from mav_telem import SENSOR_BITS as SENSORS, PREARM_BIT
EKF_FLAGS = [
 (1,"attitude"),(2,"velocity horiz"),(4,"velocity vert"),(8,"pos horiz rel"),
 (16,"pos horiz abs"),(32,"pos vert abs"),(64,"pos vert agl"),(128,"const pos mode"),
 (256,"pred pos horiz rel"),(512,"pred pos horiz abs"),(1024,"uninitialized"),
]

t0, seen, texts = time.time(), {}, []
while time.time() - t0 < 14:
    msg = m.recv_match(blocking=True, timeout=1.0)
    if msg is None: continue
    t = msg.get_type()
    if t in ("SYS_STATUS","EKF_STATUS_REPORT","VIBRATION","GPS_RAW_INT",
             "BATTERY_STATUS","POWER_STATUS","HEARTBEAT","RC_CHANNELS"):
        seen[t] = msg
    elif t == "STATUSTEXT":
        txt = msg.text.decode() if isinstance(msg.text, bytes) else msg.text
        texts.append((msg.severity, txt))

s = seen.get("SYS_STATUS")
if s:
    print("=== CAM BIEN (present / enabled / healthy) ===")
    bad = []
    for bit, name in SENSORS:
        p = bool(s.onboard_control_sensors_present & bit)
        e = bool(s.onboard_control_sensors_enabled & bit)
        h = bool(s.onboard_control_sensors_health & bit)
        if not p: continue
        flag = "OK " if h else "!! "
        if not h: bad.append(name)
        print(f"  {flag}{name:26s} present={int(p)} enabled={int(e)} healthy={int(h)}")
    print(f"\n  -> {'TAT CA LANH MANH' if not bad else 'CO VAN DE: ' + ', '.join(bad)}")
    print(f"  tai CPU: {s.load/10:.1f}%   loi truyen: {s.errors_comm}")

e = seen.get("EKF_STATUS_REPORT")
if e:
    print("\n=== EKF3 ===")
    ok = [n for b, n in EKF_FLAGS if e.flags & b]
    print("  co san  : " + ", ".join(ok))
    for k in ("velocity_variance","pos_horiz_variance","pos_vert_variance",
              "compass_variance","terrain_alt_variance"):
        v = getattr(e, k, None)
        if v is not None:
            mark = "OK " if v < 0.5 else "!! "
            print(f"  {mark}{k:22s} {v:.3f}   (an toan khi < 0.5)")

v = seen.get("VIBRATION")
if v:
    print("\n=== RUNG (day la truc chinh cua de tai) ===")
    for ax, val in (("X", v.vibration_x), ("Y", v.vibration_y), ("Z", v.vibration_z)):
        mark = "OK " if val < 30 else ("!! " if val < 60 else "XX ")
        print(f"  {mark}vibe {ax}: {val:6.2f} m/s^2   (<30 tot, 30-60 canh bao, >60 xau)")
    print(f"  clipping: {v.clipping_0} / {v.clipping_1} / {v.clipping_2}  (phai la 0)")

g = seen.get("GPS_RAW_INT")
if g:
    FIX = {0:"NO GPS",1:"NO FIX",2:"2D",3:"3D",4:"DGPS",5:"RTK float",6:"RTK fixed"}
    mark = "OK " if g.fix_type >= 3 else "!! "
    print(f"\n=== GPS ===\n  {mark}fix={FIX.get(g.fix_type,g.fix_type)}  sats={g.satellites_visible}"
          f"  HDOP={g.eph/100 if g.eph!=65535 else '-'}")

b = seen.get("BATTERY_STATUS"); p = seen.get("POWER_STATUS")
if b:
    mv = b.voltages[0] if b.voltages and b.voltages[0] != 65535 else None
    print(f"\n=== PIN ===\n  {mv/1000.0 if mv else '-'} V | {b.current_battery/100.0 if b.current_battery!=-1 else '-'} A"
          f" | da dung {b.current_consumed} mAh | con {b.battery_remaining}%")
if p:
    print(f"  nguon board: Vcc={p.Vcc/1000.0:.2f}V  servo={p.Vservo/1000.0:.2f}V  flags=0x{p.flags:x}")

r = seen.get("RC_CHANNELS")
if r:
    mark = "OK " if r.chancount > 0 else "!! "
    print(f"\n=== DIEU KHIEN TAY ===\n  {mark}{r.chancount} kenh, rssi={r.rssi}")
    if r.chancount:
        print("  " + "  ".join(f"ch{i}={getattr(r,f'chan{i}_raw')}" for i in range(1,9)))

if texts:
    print("\n=== MAY BAY BAO (STATUSTEXT) ===")
    SEV = {0:"EMERG",1:"ALERT",2:"CRIT",3:"ERROR",4:"WARN",5:"NOTICE",6:"INFO",7:"DEBUG"}
    for sev, t in texts[-15:]:
        print(f"  [{SEV.get(sev,sev)}] {t}")
else:
    print("\n=== MAY BAY BAO ===\n  (khong co canh bao nao trong 14s)")
