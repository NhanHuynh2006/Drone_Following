"""Dựng video bay thử của hồ sơ:
  thẻ mở đầu -> đoạn video bay thử (demo_flight) -> đoạn chạy vòng bám người trên chính chuyến bay đó
  (_replay.mp4 do replay_follow.py sinh ra) -> thẻ kết quả.
Chạy: /usr/bin/python3 replay_follow.py (với GRACE=1.5 REACQ=1.0 CONF=0.30 IMG=512) rồi /usr/bin/python3 make_video.py
"""
import csv
import os
import statistics as st
import subprocess

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DEMO = os.path.join(ROOT, "demo_flight", "flight_deploy_v17.mp4")
REPLAY = os.path.join(HERE, "_replay.mp4")
LOGO = os.path.join(ROOT, "nckh_SV2026-474", "hinh", "logo_hcmute.png")
W, H = 1280, 720
F_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
F_R = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
NAVY = (11, 60, 120)


def font(p, s):
    return ImageFont.truetype(p, s)


def center(d, y, text, f, fill="white"):
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, y), text, font=f, fill=fill)


def card(path, lines):
    im = Image.new("RGB", (W, H), NAVY)
    d = ImageDraw.Draw(im)
    logo = Image.open(LOGO).convert("RGBA")
    logo = logo.resize((int(logo.width * 70 / logo.height), 70))
    box = Image.new("RGB", (logo.width + 24, 94), "white")
    box.paste(logo, (12, 12), logo)
    im.paste(box, ((W - box.width) // 2, 40))
    y = 170
    for text, f, gap, color in lines:
        center(d, y, text, f, color)
        y += gap
    im.save(path)


def bar(path, line1, line2):
    b = Image.new("RGBA", (W, 66), (11, 60, 120, 215))
    d = ImageDraw.Draw(b)
    d.text((24, 8), line1, font=font(F_B, 24), fill="white")
    d.text((24, 38), line2, font=font(F_R, 20), fill=(225, 235, 250))
    b.save(path)


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---- số liệu từ nhật ký chạy vòng bám
rows = list(csv.DictReader(open(os.path.join(HERE, "replay_log.csv"))))
n = len(rows)
follow = [r for r in rows if r["state"] == "FOLLOW"]
held = sum(r["state"] != "HOVER" for r in rows)
med_yaw = st.median(abs(float(r["yaw_err_deg"])) for r in follow)

card(os.path.join(HERE, "_intro.png"), [
    ("TRƯỜNG ĐẠI HỌC CÔNG NGHỆ KỸ THUẬT TP.HCM", font(F_R, 26), 40, "white"),
    ("ĐỀ TÀI NGHIÊN CỨU KHOA HỌC SINH VIÊN NĂM 2026 — MÃ SỐ SV2026-474", font(F_R, 24), 80, (220, 230, 245)),
    ("MÁY BAY KHÔNG NGƯỜI LÁI TỰ ĐỘNG THEO DÕI", font(F_B, 46), 60, "white"),
    ("CON NGƯỜI BẰNG TRÍ TUỆ NHÂN TẠO", font(F_B, 46), 100, "white"),
    ("VIDEO BAY THỬ VÀ VÒNG BÁM NGƯỜI", font(F_B, 34), 70, (255, 200, 120)),
    ("SV thực hiện: Huỳnh Lê Thành Nhân, Trịnh Trần Quốc Bảo", font(F_R, 26), 40, "white"),
    ("Giảng viên hướng dẫn: PGS.TS. Lê Mỹ Hà", font(F_R, 26), 40, "white"),
])
card(os.path.join(HERE, "_mid.png"), [
    ("PHẦN 2", font(F_R, 30), 60, (220, 230, 245)),
    ("VÒNG BÁM NGƯỜI TRÊN VIDEO BAY THỬ", font(F_B, 44), 90, "white"),
    ("Phát hiện người (SteadyDet-N) → theo vết OC-SORT → khoá mục tiêu", font(F_R, 30), 50, "white"),
    ("→ sai số hướng → lệnh điều khiển → trạng thái FOLLOW / COAST / HOVER", font(F_R, 30), 90, "white"),
    ("Chương trình follow_drone xử lý từng khung của chuyến bay 21/07/2026 (≈ 6 m)", font(F_R, 26), 40, (255, 200, 120)),
])
card(os.path.join(HERE, "_outro.png"), [
    ("KẾT QUẢ CHÍNH", font(F_B, 38), 70, (255, 200, 120)),
    ("AP@0,5 trên dữ liệu bay thật (INT8, trên máy bay): 0,917  •  F1 = 0,880", font(F_R, 28), 48, "white"),
    ("30 khung hình/giây đầu-cuối trên Raspberry Pi 5 + Hailo-8L", font(F_R, 28), 48, "white"),
    ("Huấn luyện nhận biết rung: tỷ lệ giữ độ chính xác 16,8% → 36,2%", font(F_R, 28), 48, "white"),
    (f"Vòng bám trên video bay thử: giữ khoá mục tiêu {100 * held / n:.1f}% thời gian,"
     .replace(".", ","), font(F_R, 28), 40, "white"),
    (f"sai số hướng trung vị {med_yaw:.1f}° khi đang bám".replace(".", ","), font(F_R, 28), 70, "white"),
    ("Trường Đại học Công nghệ Kỹ thuật TP.HCM — 2026", font(F_R, 24), 40, (220, 230, 245)),
])
bar(os.path.join(HERE, "_bar1.png"),
    "Phần 1 — Video bay thử, độ cao ≈ 6 m: SteadyDet-N phát hiện người trên từng khung hình",
    "Hộp xanh: người được phát hiện  •  Dấu thập: tâm khung hình")
bar(os.path.join(HERE, "_bar2.png"),
    "Phần 2 — Vòng bám người: phát hiện → OC-SORT → khoá mục tiêu → lệnh quay hướng",
    "Xanh lá: mục tiêu đang khoá (FOLLOW)  •  Cam: vị trí dự đoán khi mất phát hiện (COAST)  •  Vệt vàng: quỹ đạo")

enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-preset", "medium", "-crf", "20"]
sil = ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]


def still(img, sec, out):
    run(["ffmpeg", "-y", "-loop", "1", "-t", str(sec), "-i", img, *sil,
         "-vf", f"fade=t=in:st=0:d=0.6,fade=t=out:st={sec - 0.6}:d=0.6", *enc, "-c:a", "aac", "-shortest", out])


def clip(src, barimg, out, speed=1.0):
    vf = (f"[0:v]setpts=PTS/{speed},scale={W}:-2:flags=lanczos,pad={W}:{H}:(ow-iw)/2:0:black[v];"
          f"[v][1:v]overlay=0:{H}-66[o]")
    run(["ffmpeg", "-y", "-i", src, "-i", barimg, *sil, "-filter_complex", vf,
         "-map", "[o]", "-map", "2:a", *enc, "-c:a", "aac", "-shortest", out])


parts = [os.path.join(HERE, p) for p in ("_p1.mp4", "_p2.mp4", "_p3.mp4", "_p4.mp4", "_p5.mp4")]
still(os.path.join(HERE, "_intro.png"), 6, parts[0])
clip(DEMO, os.path.join(HERE, "_bar1.png"), parts[1])
still(os.path.join(HERE, "_mid.png"), 5, parts[2])
clip(REPLAY, os.path.join(HERE, "_bar2.png"), parts[3])
still(os.path.join(HERE, "_outro.png"), 8, parts[4])
lst = os.path.join(HERE, "_list.txt")
with open(lst, "w") as f:
    for p in parts:
        f.write(f"file '{p}'\n")
run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", "-movflags", "+faststart",
     os.path.join(HERE, "video_bay_thu.mp4")])
for p in parts + [lst, os.path.join(HERE, "_bar1.png"), os.path.join(HERE, "_bar2.png")]:
    os.remove(p)
print("xong")
