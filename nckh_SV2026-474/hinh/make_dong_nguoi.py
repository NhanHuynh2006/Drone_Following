"""Vẽ ba ảnh đông người có kết quả phát hiện tốt nhất thành một hình cho slide."""
import os, sys, cv2, torch, numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT = "/home/nhanhuynh/Documents/pfdet_drone_follow"
sys.path.insert(0, ROOT)
from infer import load_model, preprocess, postprocess

CHON = [("0000291_03401_d_0000885", "Ngã tư giờ cao điểm"),
        ("0000072_02834_d_0000003", "Quảng trường"),
        ("0000086_00592_d_0000002", "Sân thể thao trường học")]
CONF, IMG = 0.30, 640
OUT = os.path.join(ROOT, "nckh_SV2026-474/hinh/dong_nguoi.png")
F_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model, _, _ = load_model(os.path.join(ROOT, "runs/train_v17_deploy/best.pt"), device=dev)
model.eval(); strides = [4, 8, 16, 32]

tam = []
for ten, nhan in CHON:
    img = cv2.imread(os.path.join(ROOT, "data/visdrone/val/images", ten + ".jpg"))
    with torch.no_grad():
        inp, ratio, pad, orig = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG)
        preds = model(inp.to(dev))
    dets = postprocess(preds, strides, IMG, conf_thr=CONF, nms_iou=0.45,
                       ratio=ratio, pad=pad, orig_size=orig)
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 60), 2)
    n_that = sum(1 for l in open(os.path.join(ROOT, "data/visdrone/val/labels", ten + ".txt")) if l.strip())
    tam.append((Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
                f"{nhan} — {n_that} người trong ảnh"))
    print(f"{ten}: {n_that} người, {len(dets)} hộp")

W = 980
th = []
for im, cap in tam:
    im = im.resize((W, int(im.height * W / im.width)), Image.LANCZOS)
    th.append((im, cap))
h = min(i.height for i, _ in th)
th = [(i.crop((0, 0, W, h)), c) for i, c in th]
BAR, KHE = 48, 10
sheet = Image.new("RGB", (W * 3 + KHE * 2, h + BAR), "white")
d = ImageDraw.Draw(sheet)
f_b = ImageFont.truetype(F_B, 27)
for k, (im, cap) in enumerate(th):
    x = k * (W + KHE)
    sheet.paste(im, (x, 0))
    d.rectangle((x, h, x + W, h + BAR), fill=(11, 60, 120))
    d.text((x + 16, h + 11), cap, font=f_b, fill="white")
sheet.save(OUT)
print("đã lưu", OUT, sheet.size)
