"""Sinh toàn bộ hình số liệu (tiếng Việt) cho báo cáo, tóm tắt, poster, slide.

Chạy:  cd nckh_SV2026-474/hinh && /usr/bin/python3 make_figures.py
Mọi con số lấy từ các file kết quả trong repo:
  runs/paper_queue/UNIFIED.json        (VisDrone-val, 1 giao thức, 3 seed)
  runs/paper_queue/FLIGHT.json         (92 khung bay thật)
  runs/pi_hailo_measured_2026-08-24.md (FPS NPU, INT8, YOLO26n)
  runs/pi_camera_measured_2026-08-27.md(e2e, camera)
  runs/control_rate_2026-08-27.md      (nhịp lệnh)
  runs/retrain_coco_2026-08-27.md      (HEF triển khai, INT8 theo miền)
"""
import json
import os
import statistics as st

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.dirname(os.path.abspath(__file__))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#8a8986",
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e4e3df",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "legend.frameon": False,
})
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY, DGRAY = "#b5b4ae", "#6b6a66"
INK = "#0b0b0b"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("->", name)


# ------------------------------------------------------------------ số liệu
U = json.load(open(os.path.join(ROOT, "runs/paper_queue/UNIFIED.json")))["models"]


def agg(key):
    v = U[key]
    c = v["clean"]["AP"]
    ps = v["per_seed"]
    sev = {s: [ps[x][s]["AP"] for x in ps] for s in ("sev1", "sev2", "sev3")}
    mcp = [sum(ps[x][s]["AP"] for s in sev) / 3 for x in ps]
    ret = [100 * m / c for m in mcp]
    return {
        "clean": c,
        "sev_mean": [st.mean(sev[s]) for s in ("sev1", "sev2", "sev3")],
        "sev_std": [st.stdev(sev[s]) for s in ("sev1", "sev2", "sev3")],
        "mcp": st.mean(mcp),
        "ret": st.mean(ret),
        "ret_sd": st.stdev(ret),
        "params": v["params_M"],
    }


# 1) Đường cong AP theo mức rung ---------------------------------------------
def fig_curve():
    series = [
        ("v17-vib-augonly", "SteadyDet-N + huấn luyện rung", BLUE, "o", "-"),
        ("v17-baseline", "SteadyDet-N (gốc)", BLUE, "o", "--"),
        ("yolo11n", "YOLO11-n", ORANGE, "s", "-"),
        ("rtdetr-l", "RT-DETR-l", AQUA, "^", "-"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    x = [0, 1, 2, 3]
    for key, lab, col, mk, ls in series:
        a = agg(key)
        y = [a["clean"]] + a["sev_mean"]
        e = [0] + a["sev_std"]
        ax.errorbar(x, y, yerr=e, color=col, marker=mk, ms=5, lw=2, ls=ls,
                    capsize=2, label=lab, mfc="white" if ls == "--" else col)
    ax.set_xticks(x, ["sạch", "mức 1", "mức 2", "mức 3"])
    ax.set_ylabel("AP@[.50:.95]")
    ax.set_xlabel("Mức rung VisDrone-Shake")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 0.30)
    save(fig, "duong_cong_rung.pdf")


# 2) Tỷ lệ giữ độ chính xác (retain %) -----------------------------------------
def fig_retain():
    rows = [
        ("v15-augonly", "SteadyDet-W + huấn luyện rung", BLUE),
        ("v17-vib-augonly", "SteadyDet-N + huấn luyện rung", BLUE),
        ("v17-vibcons", "SteadyDet-N + ràng buộc nhất quán", "#8fb8ea"),
        ("rtdetr-l", "RT-DETR-l", GRAY),
        ("v15-baseline", "SteadyDet-W (gốc)", GRAY),
        ("yolo11n", "YOLO11-n", GRAY),
        ("yolov10n", "YOLOv10-n", GRAY),
        ("yolov8n", "YOLOv8-n", GRAY),
        ("yolo26n", "YOLO26-n", GRAY),
        ("v17-ft-control", "SteadyDet-N + 120 epoch không rung", GRAY),
        ("v17-baseline", "SteadyDet-N (gốc)", GRAY),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ys = np.arange(len(rows))[::-1]
    for y, (k, lab, col) in zip(ys, rows):
        a = agg(k)
        ax.barh(y, a["ret"], color=col, height=0.62)
        ax.errorbar(a["ret"], y, xerr=a["ret_sd"], color=DGRAY, capsize=2, lw=1)
        ax.text(a["ret"] + a["ret_sd"] + 0.6, y, f"{a['ret']:.1f}".replace(".", ","),
                va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [r[1] for r in rows], fontsize=8)
    ax.set_xlabel("Tỷ lệ giữ độ chính xác khi rung, retain (%) — trung bình ± độ lệch chuẩn, 3 seed")
    ax.set_xlim(0, 45)
    ax.grid(axis="y", visible=False)
    save(fig, "retain.pdf")


# 3) Đánh đổi: AP sạch vs mCP --------------------------------------------------
def fig_tradeoff():
    pts = [
        ("v17-baseline", "SteadyDet-N gốc", GRAY, (6, -10)),
        ("v17-vib-augonly", "SteadyDet-N + rung", BLUE, (6, 4)),
        ("v15-baseline", "SteadyDet-W gốc", GRAY, (-20, -14)),
        ("v15-augonly", "SteadyDet-W + rung", BLUE, (6, 2)),
        ("yolo11n", "YOLO11-n", GRAY, (6, 4)),
        ("yolov10n", "YOLOv10-n", GRAY, (-24, 8)),
        ("yolov8n", "YOLOv8-n", GRAY, (-48, 2)),
        ("yolo26n", "YOLO26-n", GRAY, (2, -14)),
        ("rtdetr-l", "RT-DETR-l (32,8 M)", GRAY, (6, 4)),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    for k, lab, col, off in pts:
        a = agg(k)
        ax.scatter(a["clean"], a["mcp"], s=46, color=col, edgecolor="white", lw=1.5, zorder=3)
        ax.annotate(lab, (a["clean"], a["mcp"]), xytext=off, textcoords="offset points",
                    fontsize=7.5, color=INK)
    for a0, a1 in (("v17-baseline", "v17-vib-augonly"), ("v15-baseline", "v15-augonly")):
        p, q = agg(a0), agg(a1)
        ax.annotate("", xy=(q["clean"], q["mcp"]), xytext=(p["clean"], p["mcp"]),
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax.set_xlabel("AP@[.50:.95] trên ảnh sạch")
    ax.set_ylabel("mCP (AP trung bình khi rung)")
    ax.set_xlim(0.16, 0.30)
    ax.set_ylim(0.025, 0.095)
    save(fig, "danh_doi.pdf")


# 4) Bay thật (92 khung) --------------------------------------------------------
def fig_flight():
    F = json.load(open(os.path.join(ROOT, "runs/paper_queue/FLIGHT.json")))["models"]
    groups = [("deploy", "Huấn luyện VisDrone + COCO-person"), ("light", "Huấn luyện chỉ VisDrone")]
    treat = [("baseline", "không xử lý", GRAY), ("augonly", "huấn luyện rung", BLUE),
             ("vibcons", "ràng buộc nhất quán", AQUA)]
    mets = [("AP50", "AP@0,5"), ("AP", "AP@[.5:.95]"), ("AP75", "AP@0,75")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    w = 0.26
    for ax, (g, title) in zip(axes, groups):
        for i, (t, lab, col) in enumerate(treat):
            vals = [F[f"{g}-{t}"][m] for m, _ in mets]
            xs = np.arange(3) + (i - 1) * w
            ax.bar(xs, vals, w - 0.02, color=col, label=lab)
        ax.set_xticks(np.arange(3), [m[1] for m in mets])
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Giá trị")
    axes[0].legend(fontsize=7.5, loc="upper right")
    save(fig, "bay_that.pdf")


# 5) INT8 theo miền -----------------------------------------------------------
def fig_int8():
    vis = {"AP@[.5:.95]": (0.1775, 0.1584), "AP@0,5": (0.4946, 0.4625), "AP@0,75": (0.0742, 0.0555)}
    fli = {"AP@[.5:.95]": (0.4356, 0.4349), "AP@0,5": (0.8822, 0.9171), "AP@0,75": (0.2561, 0.2552)}
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, (d, title) in zip(axes, ((vis, "VisDrone-val (548 ảnh, người rất nhỏ)"),
                                     (fli, "Bay thật ~6 m (92 khung, người lớn)"))):
        xs = np.arange(3)
        f = [v[0] for v in d.values()]
        q = [v[1] for v in d.values()]
        ax.bar(xs - 0.18, f, 0.34, color=GRAY, label="FP32 (GPU)")
        ax.bar(xs + 0.18, q, 0.34, color=BLUE, label="INT8 (Hailo-8L)")
        for x, a, b in zip(xs, f, q):
            ax.text(x + 0.18, b + 0.015, f"{b:.3f}".replace(".", ","), ha="center", fontsize=7)
        ax.set_xticks(xs, list(d))
        ax.set_title(title, fontsize=8.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(fontsize=7.5, loc="upper left")
    save(fig, "int8.pdf")


# 6) Độ trễ từng chặng ----------------------------------------------------------
def fig_latency():
    stages = [("Chờ camera", 12.47, GRAY), ("Tiền xử lý", 0.99, YELLOW),
              ("Suy luận NPU", 17.72, BLUE), ("Giải mã + NMS", 0.84, ORANGE)]
    fig, ax = plt.subplots(figsize=(6.2, 1.3))
    left = 0
    for lab, v, col in stages:
        ax.barh(0, v, left=left, color=col, height=0.5, edgecolor="white", lw=2, label=lab)
        if v > 3:
            ax.text(left + v / 2, 0, f"{lab}\n{v:.2f} ms".replace(".", ","), ha="center",
                    va="center", fontsize=7.5, color="white" if col == BLUE else INK)
        left += v
    ax.axvline(33.3, color=INK, lw=1, ls=":")
    ax.text(33.5, 0.33, "ngân sách 33,3 ms (30 Hz)", fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlim(0, 42)
    ax.set_xlabel("Trung vị (p50) theo chặng, @640, camera USB thật, 3.594 khung (ms)")
    ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.55))
    ax.grid(axis="y", visible=False)
    save(fig, "do_tre.pdf")


# 7) Sửa lỗi camera ------------------------------------------------------------
def fig_camera():
    rows = [("BUFFERSIZE=1\n+ tự phơi sáng (cũ)", 11.1), ("BUFFERSIZE=1\n+ phơi sáng tay", 15.0),
            ("BUFFERSIZE=2\n+ tự phơi sáng", 19.7), ("BUFFERSIZE=2\n+ SoftAE (mới)", 30.0)]
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    xs = np.arange(4)
    ax.bar(xs, [r[1] for r in rows], 0.55, color=[GRAY, GRAY, GRAY, BLUE])
    for x, (_, v) in zip(xs, rows):
        ax.text(x, v + 0.6, f"{v:.1f}".replace(".", ","), ha="center", fontsize=8)
    ax.set_xticks(xs, [r[0] for r in rows], fontsize=7.5)
    ax.set_ylabel("Tốc độ khung thực (FPS)")
    ax.set_ylim(0, 34)
    ax.grid(axis="x", visible=False)
    save(fig, "camera_fps.pdf")


# 8) Hailo: SteadyDet-N vs YOLO26n + batch -------------------------------------
def fig_hailo():
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    ax = axes[0]
    ax.bar([0, 1], [63.66, 37.46], 0.5, color=[BLUE, GRAY])
    for x, v, c in ((0, 63.66, "4 context"), (1, 37.46, "6 context")):
        ax.text(x, v + 1.5, f"{v:.2f} FPS\n{c}".replace(".", ","), ha="center", fontsize=7.5)
    ax.set_xticks([0, 1], ["SteadyDet-N\n1,17 M · 4,6 GFLOPs", "YOLO26-n\n2,38 M · 5,2 GFLOPs"], fontsize=7.5)
    ax.set_ylabel("FPS @640, batch = 1")
    ax.set_ylim(0, 80)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    b = [1, 4, 8, 16]
    lat = [8.40, 17.05, 28.35, 50.48]
    ax.plot(b, lat, color=BLUE, marker="o", lw=2)
    for x, y, f in zip(b, lat, [112.55, 188.78, 214.63, 231.82]):
        ax.annotate(f"{f:.0f} FPS", (x, y), xytext=(4, -10), textcoords="offset points", fontsize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(b, [str(v) for v in b])
    ax.set_xlabel("Kích thước lô (@320)")
    ax.set_ylabel("Độ trễ phần cứng (ms)")
    save(fig, "hailo.pdf")


# 9) Nhịp gửi lệnh ------------------------------------------------------------
def fig_rate():
    rates = ["10 Hz", "20 Hz", "30 Hz", "50 Hz"]
    mean = [84.4, 59.4, 51.1, 44.4]
    worst = [138.4, 88.4, 71.7, 58.4]
    xs = np.arange(4)
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.bar(xs - 0.18, mean, 0.34, color=BLUE, label="trung bình")
    ax.bar(xs + 0.18, worst, 0.34, color=GRAY, label="xấu nhất")
    for x, m, w_ in zip(xs, mean, worst):
        ax.text(x - 0.18, m + 2, f"{m:.1f}".replace(".", ","), ha="center", fontsize=7)
        ax.text(x + 0.18, w_ + 2, f"{w_:.1f}".replace(".", ","), ha="center", fontsize=7)
    ax.set_xticks(xs, rates)
    ax.set_ylabel("Độ trễ ảnh → lệnh tới FCU (ms)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 155)
    ax.grid(axis="x", visible=False)
    save(fig, "nhip_lenh.pdf")


# 10) Quét ngưỡng tin cậy HEF triển khai --------------------------------------
def fig_conf():
    c = [0.20, 0.25, 0.30, 0.40]
    f1 = [0.848, 0.880, 0.823, 0.551]
    fig, ax = plt.subplots(figsize=(3.1, 2.3))
    ax.plot(c, f1, color=BLUE, marker="o", lw=2)
    for x, y in zip(c, f1):
        ax.annotate(f"{y:.3f}".replace(".", ","), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set_xlabel("Ngưỡng tin cậy")
    ax.set_ylabel("F1 (92 khung bay thật)")
    ax.set_ylim(0.5, 0.95)
    ax.set_xticks(c, [f"{v:.2f}".replace(".", ",") for v in c])
    save(fig, "nguong_conf.pdf")


# 11) Đường cong huấn luyện -----------------------------------------------------
def fig_training():
    import csv
    def load(p):
        r = list(csv.DictReader(open(os.path.join(ROOT, p))))
        ep = [int(x["epoch"]) for x in r]
        loss = [float(x["train_loss"]) for x in r]
        v = [(int(x["epoch"]), float(x["ap"])) for x in r if x["ap"]]
        return ep, loss, v
    e0, l0, v0 = load("runs/train_v17_light/training_log.csv")
    e1, l1, v1 = load("runs/train_v17_vib_augonly/training_log.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    axes[0].plot(e0, l0, color=GRAY, lw=1.5, label="huấn luyện gốc (300 epoch)")
    axes[0].plot([300 + e for e in e1], l1, color=BLUE, lw=1.5, label="tinh chỉnh rung (120 epoch)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Hàm mất mát huấn luyện")
    axes[0].legend(fontsize=7)
    axes[1].plot([a for a, _ in v0], [b for _, b in v0], color=GRAY, lw=1.5, marker="o", ms=2)
    axes[1].plot([300 + a for a, _ in v1], [b for _, b in v1], color=BLUE, lw=1.5, marker="o", ms=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AP@0,5 kiểm định (ảnh sạch)")
    save(fig, "huan_luyen.pdf")


# ------------------------------------------------------------------ ảnh
_PRESETS = {1: (5, 0.02, 0.10), 2: (9, 0.04, 0.20), 3: (15, 0.06, 0.30)}


def corrupt(img01, sev, angle=35.0, bright_sign=-1.0, seed=0):
    """Bản numpy đúng công thức vib_consistency.corrupt_batch (hướng/độ sáng cố định để minh hoạ)."""
    L, ns, br = _PRESETS[sev]
    L = max(3, int(L) | 1)
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2
    a = np.radians(angle)
    for t in range(L):
        x = int(round(c + (t - c) * np.cos(a)))
        y = int(round(c + (t - c) * np.sin(a)))
        k[y, x] = 1.0
    k /= k.sum()
    out = cv2.filter2D(img01, -1, k, borderType=cv2.BORDER_CONSTANT)
    rng = np.random.default_rng(seed)
    out = out + rng.normal(0, ns, out.shape).astype(np.float32)
    out = out * (1.0 + bright_sign * br * 0.5)
    return np.clip(out, 0, 1)


def draw_boxes(img, lbl, color=(42, 120, 214), th=2):
    h, w = img.shape[:2]
    if not os.path.exists(lbl):
        return img
    for line in open(lbl):
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
    return img


def fig_vibration_images():
    vd = cv2.imread(os.path.join(ROOT, "data/visdrone/val/images/0000001_02999_d_0000005.jpg"))
    fl = cv2.imread(os.path.join(ROOT, "data/flight_val/images/flight_0056.jpg"))
    rows = []
    for img in (vd, fl):
        img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
        f = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        rows.append([f] + [corrupt(f, s) for s in (1, 2, 3)])
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 2.1))
    titles = ["Ảnh sạch", "Mức 1\n(5 px; σ=0,02; ±10%)", "Mức 2\n(9 px; σ=0,04; ±20%)", "Mức 3\n(15 px; σ=0,06; ±30%)"]
    for r in range(2):
        for cidx in range(4):
            ax = axes[r, cidx]
            ax.imshow(rows[r][cidx])
            ax.set_xticks([]); ax.set_yticks([])
            ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(titles[cidx], fontsize=7)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    fig.savefig(os.path.join(OUT, "rung_minh_hoa.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("-> rung_minh_hoa.png")


def fig_dataset_samples():
    items = [
        ("data/visdrone/val/images/0000001_02999_d_0000005.jpg", "data/visdrone/val/labels/0000001_02999_d_0000005.txt",
         "VisDrone-person (bay cao)"),
        (None, None, "COCO-person (cự ly gần)"),
        ("data/flight_val/images/flight_0056.jpg", "data/flight_val/labels/flight_0056.txt", "Bay thật ~6 m (flight_val)"),
    ]
    coco_dir = os.path.join(ROOT, "data/coco_person/images")
    names = sorted(os.listdir(coco_dir))
    for n in names:
        lbl = os.path.join(ROOT, "data/coco_person/labels", os.path.splitext(n)[0] + ".txt")
        if os.path.exists(lbl) and sum(1 for _ in open(lbl)) == 1 and float(open(lbl).read().split()[4]) > 0.6:
            items[1] = (os.path.join("data/coco_person/images", n), os.path.relpath(lbl, ROOT), items[1][2])
            break
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.9))
    for ax, (im, lb, t) in zip(axes, items):
        img = cv2.imread(os.path.join(ROOT, im))
        img = draw_boxes(img, os.path.join(ROOT, lb), color=(0, 200, 255), th=max(2, img.shape[1] // 300))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(t, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(os.path.join(OUT, "mau_du_lieu.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("-> mau_du_lieu.png")


def fig_demo_frames():
    cap = cv2.VideoCapture(os.path.join(ROOT, "demo_flight/flight_deploy_v17.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    for t in (15, 45, 80):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ok, f = cap.read()
        if ok:
            frames.append((t, cv2.cvtColor(f, cv2.COLOR_BGR2RGB)))
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 1.3))
    for ax, (t, f) in zip(axes, frames):
        ax.imshow(f)
        ax.set_title(f"giây thứ {t}", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(os.path.join(OUT, "demo_khung.png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    for t, f in frames:
        cv2.imwrite(os.path.join(OUT, f"demo_t{t}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    print("-> demo_khung.png")


if __name__ == "__main__":
    fig_curve(); fig_retain(); fig_tradeoff(); fig_flight(); fig_int8()
    fig_latency(); fig_camera(); fig_hailo(); fig_rate(); fig_conf(); fig_training()
    fig_vibration_images(); fig_dataset_samples(); fig_demo_frames()
