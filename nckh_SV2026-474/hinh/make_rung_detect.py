"""Dựng hình: cùng một ảnh ở bốn mức rung, so phát hiện của mô hình gốc và mô hình
huấn luyện nhận biết rung. Chạy: ~/.venvs/pfdet/bin/python make_rung_detect.py"""
import math, os, sys
import cv2, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, ROOT)
from infer import load_model, preprocess, postprocess

ANH = "0000086_00592_d_0000002"          # sân thể thao, 55 người, hộp đủ to để nhìn rõ
MUC = {1: (5, 0.02, 0.10), 2: (9, 0.04, 0.20), 3: (15, 0.06, 0.30)}   # (dài nhoè, nhiễu, sáng)
GOC_ANGLE, CONF, IMG, IOU = 35.0, 0.30, 640, 0.3
MO_HINH = [("runs/train_v17_light/best.pt", "SteadyDet-N gốc"),
           ("runs/train_v17_vib/best.pt",   "SteadyDet-N + huấn luyện nhận biết rung")]
OUT = os.path.join(ROOT, "nckh_SV2026-474/hinh/rung_detect.png")
F_B = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
F_R = "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
RNG = np.random.default_rng(0)


def nhan_nhoe(length, angle_deg):
    length = max(3, int(length) | 1)
    k = np.zeros((length, length), np.float32)
    c = (length - 1) / 2.0
    dx, dy = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    for t in range(length):
        x, y = int(round(c + (t - c) * dx)), int(round(c + (t - c) * dy))
        if 0 <= x < length and 0 <= y < length:
            k[y, x] = 1.0
    return k / k.sum()


def lam_rung(img, muc):
    """Đúng toán tử của giao thức VisDrone-Shake: nhoè + nhiễu + dao động sáng."""
    if muc == 0:
        return img.copy()
    L, ns, br = MUC[muc]
    f = img.astype(np.float32) / 255.0
    f = cv2.filter2D(f, -1, nhan_nhoe(L, GOC_ANGLE), borderType=cv2.BORDER_REFLECT)
    f = f + RNG.normal(0, ns, f.shape)
    return np.clip(f * (1.0 + br * 0.6), 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)


def iou_ma_tran(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    sa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); sb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (sa[:, None] + sb[None, :] - inter + 1e-9)


goc = cv2.imread(os.path.join(ROOT, "data/visdrone/val/images", ANH + ".jpg"))
h, w = goc.shape[:2]
gt = []
for l in open(os.path.join(ROOT, "data/visdrone/val/labels", ANH + ".txt")):
    t = l.split()
    if len(t) >= 5:
        cx, cy, bw, bh = [float(v) for v in t[1:5]]
        gt.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
gt = np.array(gt, dtype=np.float32)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
strides = [4, 8, 16, 32]
anh_rung = {m: lam_rung(goc, m) for m in (0, 1, 2, 3)}
hang = []
for w_path, ten in MO_HINH:
    model, _, _ = load_model(os.path.join(ROOT, w_path), device=dev); model.eval()
    o = []
    for m in (0, 1, 2, 3):
        img = anh_rung[m].copy()
        with torch.no_grad():
            inp, ratio, pad, orig = preprocess(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG)
            preds = model(inp.to(dev))
        dets = postprocess(preds, strides, IMG, conf_thr=CONF, nms_iou=0.45,
                           ratio=ratio, pad=pad, orig_size=orig)
        db = np.array([d["box"] for d in dets], np.float32) if dets else np.zeros((0, 4), np.float32)
        M = iou_ma_tran(db, gt); tp, da = 0, set()
        if M.size:
            for i in np.argsort(-M.max(axis=1)):
                j = int(np.argmax(M[i]))
                if M[i, j] >= IOU and j not in da:
                    da.add(j); tp += 1
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            cv2.rectangle(img, (x1, y1), (x2, y2), (80, 220, 60), 2)
        o.append((Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), tp))
        print(f"{ten:42} mức {m}: bắt đúng {tp}/{len(gt)}")
    hang.append((ten, o))

W = 760
TEN_COT = ["Ảnh sạch", "Rung mức 1", "Rung mức 2", "Rung mức 3"]
mau = hang[0][1][0][0]
H = int(mau.height * W / mau.width)
CAO_TEN, CAO_NHAN, KHE = 40, 42, 8
sheet = Image.new("RGB", (W * 4 + KHE * 3, CAO_TEN + (H + CAO_NHAN + KHE) * 2), "white")
d = ImageDraw.Draw(sheet)
f_b, f_r = ImageFont.truetype(F_B, 26), ImageFont.truetype(F_R, 24)
for c, ten in enumerate(TEN_COT):
    d.text((c * (W + KHE) + 14, 8), ten, font=f_b, fill=(11, 60, 120))
for r, (ten_mh, o) in enumerate(hang):
    y = CAO_TEN + r * (H + CAO_NHAN + KHE)
    for c, (im, tp) in enumerate(o):
        x = c * (W + KHE)
        sheet.paste(im.resize((W, H), Image.LANCZOS), (x, y))
        mau_nen = (11, 60, 120) if r else (90, 100, 115)
        d.rectangle((x, y + H, x + W, y + H + CAO_NHAN), fill=mau_nen)
        chu = f"bắt đúng {tp}/{len(gt)} người"
        d.text((x + 14, y + H + 9), chu, font=f_r, fill="white")
    d.text((KHE + 6, y + 6), ten_mh, font=f_b, fill="white",
           stroke_width=3, stroke_fill=(0, 0, 0))
sheet.save(OUT)
print("đã lưu", OUT, sheet.size)
