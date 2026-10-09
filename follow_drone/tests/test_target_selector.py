"""
Test hồi quy cho TargetSelector — chạy: python follow_drone/tests/test_target_selector.py

Câu hỏi gốc: người rời khung rồi quay lại thì OC-SORT cấp ID MỚI (nó đếm tăng dần,
không tái dùng ID, không có re-identification theo ngoại hình). Vậy drone có bám lại
đúng người đó không, hay vồ sang người lạ đứng gần?
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from follow.target_selector import TargetSelector

W, H = 640, 480
def T(i, box): return {"id": i, "box": box, "score": 0.9}
def new(**kw):
    d = dict(strategy="largest", lock_on_first=True, image_width=W, image_height=H,
             lost_grace_s=3.0, reacquire_s=5.0)
    d.update(kw)
    return TargetSelector(**d)

fails = []
def check(name, got, want):
    ok = got == want
    print(f"  {'OK ' if ok else '!! '}{name}: got={got} want={want}")
    if not ok: fails.append(name)

A_BOX = [280, 180, 380, 420]
B_BOX = [80, 240, 120, 360]

print("1) chon nguoi TO NHAT roi khoa vao ID do")
s = new(); t = 100.0
check("khoa ID", s.select([T(1, A_BOX), T(2, B_BOX)], current_time=t)["id"], 1)

print("2) khoa roi thi KHONG doi du nguoi khac to hon")
BIG = [200, 100, 440, 470]
check("giu ID cu", s.select([T(1, A_BOX), T(3, BIG)], current_time=t + 0.1)["id"], 1)

print("3) muc tieu mat -> giu khoa suot pha coast, KHONG vo nguoi la")
s = new(); s.select([T(1, A_BOX), T(2, B_BOX)], current_time=200.0)
for dt in (1.0, 2.0, 2.9):
    check(f"t=+{dt}s van None", s.select([T(2, B_BOX)], current_time=200.0 + dt), None)
check("t=+4s van None (dang tim lai nguoi cu)",
      s.select([T(2, B_BOX)], current_time=204.0), None)

print("4) nguoi cu quay lai voi ID MOI -> phai bam lai dung nguoi do")
r = s.select([T(77, [290, 185, 385, 425]), T(2, B_BOX)], current_time=204.5)
check("bam lai ID moi cua nguoi cu", r["id"], 77)

print("5) nguoi cu KHONG quay lai -> het cua so tim lai moi chap nhan nguoi khac")
s = new(); s.select([T(1, A_BOX)], current_time=300.0)
# t=305: qua grace (3s) -> nha khoa, cua so tim lai bat dau TU DAY (khong phai tu 300)
check("trong cua so: chua chon", s.select([T(9, B_BOX)], current_time=305.0), None)
check("van trong cua so o t=309", s.select([T(9, B_BOX)], current_time=309.0), None)
r = s.select([T(9, B_BOX)], current_time=310.5)   # 305 + 5.0 = 310 -> da het
check("het cua so: chon nguoi con lai", r["id"] if r else None, 9)

print("6) auto_relock=False -> dung cho nguoi van hanh, khong tu chon ai")
s = new(auto_relock=False); s.select([T(1, A_BOX)], current_time=400.0)
s.select([T(2, B_BOX)], current_time=404.0)
check("khong tu chon", s.select([T(2, B_BOX)], current_time=420.0), None)
check("co cho nguoi van hanh", s.awaiting_operator, True)
s.retarget()
check("sau khi bam doi muc tieu", s.select([T(2, B_BOX)], current_time=421.0)["id"], 2)

print("7) bam DOI MUC TIEU phai bo ca tri nho vi tri")
s = new(); s.select([T(1, A_BOX), T(2, B_BOX)], current_time=500.0)
s.retarget()
check("chon nguoi con lai", s.select([T(2, B_BOX)], current_time=500.1)["id"], 2)

print()
if fails:
    print(f"THAT BAI {len(fails)}: {fails}"); sys.exit(1)
print("TAT CA DEU DAT")
