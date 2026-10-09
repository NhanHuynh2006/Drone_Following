"""Do thu ket noi MAVLink toi may bay: cong nao, he ban tin gi, tan so bao nhieu."""
import sys, time, collections
from pymavlink import mavutil

dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
secs = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

print(f"[probe] mo {dev} ...")
m = mavutil.mavlink_connection(dev, baud=115200, source_system=254)
t0 = time.time()
hb = m.wait_heartbeat(timeout=12)
if hb is None:
    print("  !! KHONG nhan duoc heartbeat trong 12s"); sys.exit(1)
print(f"  OK heartbeat sau {time.time()-t0:.1f}s")
print(f"  he thong  : system={m.target_system} component={m.target_component}")
print(f"  autopilot : {mavutil.mavlink.enums['MAV_AUTOPILOT'][hb.autopilot].name}")
print(f"  loai xe   : {mavutil.mavlink.enums['MAV_TYPE'][hb.type].name}")
print(f"  mavlink   : v{m.WIRE_PROTOCOL_VERSION}")

mm = m.mode_mapping() or {}
print(f"  so che do : {len(mm)}")
if mm:
    print("  che do    : " + ", ".join(sorted(mm, key=lambda k: mm[k])))

print(f"\n[probe] nghe {secs:.0f}s, dem ban tin ...")
cnt = collections.Counter()
t0 = time.time()
snap = {}
while time.time() - t0 < secs:
    msg = m.recv_match(blocking=True, timeout=1.0)
    if msg is None: continue
    t = msg.get_type()
    if t == "BAD_DATA": continue
    cnt[t] += 1
    if t == "HEARTBEAT":
        snap["armed"] = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
        rev = {v: k for k, v in mm.items()}
        snap["mode"] = rev.get(msg.custom_mode, f"#{msg.custom_mode}")
        snap["sys_status"] = mavutil.mavlink.enums['MAV_STATE'][msg.system_status].name
    elif t == "SYS_STATUS":
        snap["volt"] = msg.voltage_battery/1000.0 if msg.voltage_battery not in (0,65535) else None
        snap["batt%"] = msg.battery_remaining
    elif t == "GPS_RAW_INT":
        snap["gps_fix"] = msg.fix_type; snap["sats"] = msg.satellites_visible
    elif t == "ATTITUDE":
        snap["roll/pitch/yaw"] = tuple(round(v*57.2958,1) for v in (msg.roll,msg.pitch,msg.yaw))
    elif t == "VFR_HUD":
        snap["alt"] = msg.alt; snap["groundspeed"] = msg.groundspeed
    elif t == "STATUSTEXT":
        txt = msg.text.decode() if isinstance(msg.text,bytes) else msg.text
        snap.setdefault("statustext", []).append(txt)

dt = time.time() - t0
print(f"\n{'ban tin':32s} {'so luong':>9} {'Hz':>7}")
for k, v in cnt.most_common():
    print(f"{k:32s} {v:>9} {v/dt:>7.1f}")

print("\n[probe] trang thai doc duoc:")
for k, v in snap.items():
    print(f"  {k:16s} = {v}")

need = ["HEARTBEAT","ATTITUDE","VFR_HUD","SYS_STATUS","GPS_RAW_INT","GLOBAL_POSITION_INT"]
print("\n[probe] ban tin DASHBOARD can:")
for n in need:
    print(f"  {'OK ' if cnt[n] else '!! '} {n:24s} {cnt[n]:>5} goi  ({cnt[n]/dt:.1f} Hz)")
