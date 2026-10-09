"""Chốt chặn sự cố bay 2026-09-04: phần mềm KHÔNG ĐƯỢC đổi chế độ bay.

Chuyến 2026-09-04: camera không thấy người lần nào suốt 249 s. Thang mất-mục-tiêu
bản cũ leo tới LOITER rồi RTL và gọi thẳng mav.set_mode(). main.py gọi thang này
vô điều kiện (không nằm trong `if not args.no_fly`), nên phần mềm cướp chế độ bay
khỏi tay người lái. RTL của ArduPilot leo lên RTL_ALT (mặc định 15 m) trước khi
bay về -> máy bay vọt thẳng lên trời.

Test này thất bại nếu ai đó đưa set_mode trở lại thang mất-mục-tiêu.

Chạy:  python3 tests/test_safety_no_mode_change.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from follow.safety import SafetyManager, SafetyState

CFG = {
    'd_min': 3.0, 'd_max': 15.0,
    'v_max_horizontal': 5.0, 'yaw_rate_max': 1.0,
    'altitude_min_agl': 2.0, 'altitude_max_agl': 25.0,
    'target_lost': {'coast_duration_s': 3.0, 'hover_duration_s': 5.0,
                    'loiter_duration_s': 7.0},
    'command_timeout_s': 1.0,
    'require_gps_3d_fix': True, 'max_hdop': 200,
    'battery_warn_v': 14.8, 'battery_critical_v': 14.0,
}


class SpyMav:
    """Ghi lại MỌI lệnh gửi đi. set_mode bị gọi là test hỏng."""
    def __init__(self):
        self.mode_calls = []
        self.velocity_calls = []

    def set_mode(self, mode):
        self.mode_calls.append(mode)
        return True

    def send_velocity_body(self, *a):
        self.velocity_calls.append(a)

    def send_zero_velocity(self):
        self.velocity_calls.append((0, 0, 0, 0))

    def get_state(self):
        return {'gps_fix': 3, 'gps_eph': 80, 'gps_sats': 12,
                'battery_voltage': 15.2, 'home_position': (0, 0, 0),
                'altitude_agl': 5.0, 'mode': 'LOITER'}

    def heartbeat_age(self):
        return 0.2


fails = []


def check(cond, msg):
    print(("  OK   " if cond else "  HONG ") + msg)
    if not cond:
        fails.append(msg)


print("=== Mất mục tiêu 60 s: thang phải dừng ở HOVER, KHÔNG đổi mode ===")
mav = SpyMav()
s = SafetyManager(mav, CFG)
s.update_target_status(True)                      # thấy người 1 lần
seen = []
t_start = time.time()
s.target_lost_since = t_start                     # giả lập mất từ đây
for dt in (0.5, 2.9, 3.1, 7.9, 8.1, 15.0, 30.0, 60.0, 120.0):
    s.target_lost_since = time.time() - dt
    seen.append((dt, s.update_target_status(False)))
for dt, st in seen:
    print(f"     mất {dt:6.1f}s -> {st}")

check(all(st in (SafetyState.COAST, SafetyState.HOVER) for _, st in seen),
      "thang chỉ ra COAST hoặc HOVER")
check(not any(st == SafetyState.RTL for _, st in seen),
      "KHÔNG BAO GIỜ ra trạng thái RTL")
check(not any(st == SafetyState.LOITER for _, st in seen),
      "KHÔNG BAO GIỜ ra trạng thái LOITER")
check(seen[-1][1] == SafetyState.HOVER,
      "mất 120 s vẫn đứng ở HOVER (đứng yên chờ người lái)")
check(mav.mode_calls == [],
      f"KHÔNG gọi set_mode lần nào (thực tế: {mav.mode_calls})")

print("\n=== Health check trượt: cũng không được đổi mode ===")
mav2 = SpyMav()
mav2.get_state = lambda: {'gps_fix': 0, 'gps_eph': 999, 'gps_sats': 0,
                          'battery_voltage': 13.0, 'home_position': None,
                          'altitude_agl': 5.0, 'mode': 'LOITER'}
s2 = SafetyManager(mav2, CFG)
ok, reason = s2.runtime_health_check()
print(f"     health -> ok={ok} reason={reason}")
check(not ok, "phát hiện được lỗi sức khoẻ")
check(mav2.mode_calls == [],
      f"KHÔNG gọi set_mode khi health trượt (thực tế: {mav2.mode_calls})")

print("\n=== Tìm lại được người thì quay lại FOLLOW ===")
mav3 = SpyMav()
s3 = SafetyManager(mav3, CFG)
s3.target_lost_since = time.time() - 60.0
check(s3.update_target_status(False) == SafetyState.HOVER, "mất 60 s -> HOVER")
check(s3.update_target_status(True) == SafetyState.FOLLOW, "thấy lại -> FOLLOW")
check(mav3.mode_calls == [], "vẫn không đổi mode lần nào")

print("\n=== Mã nguồn: safety.py không được chứa lời gọi set_mode ===")
src = open(os.path.join(os.path.dirname(__file__), '..', 'follow', 'safety.py'),
           encoding='utf-8').read()
code = "\n".join(l for l in src.splitlines() if not l.strip().startswith('#'))
check('set_mode(' not in code,
      "không có lời gọi set_mode( nào ngoài phần chú thích")

print()
if fails:
    print("=== HỎNG %d MỤC ===" % len(fails))
    for f in fails:
        print("   - " + f)
    sys.exit(1)
print("TAT CA DEU DAT")
