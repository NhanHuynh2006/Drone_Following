"""
summarize_baselines.py — Tính lại params/GFLOPs/mAP cho MỌI baseline đã train, xuất bảng sạch.

Vì sao cần: train_yolo_baselines*.py cũ dùng model.info(verbose=False) -> ultralytics 8.4.x trả None
-> SUMMARY ghi params=0.00M GFLOPs=0.0 cho tất cả. Script này đọc best.pt và tính trực tiếp.

Chạy: python summarize_baselines.py     (không cần train lại gì)
Xuất: runs/paper_queue/BASELINES_TABLE.md
"""
import os, glob
from ultralytics import YOLO, RTDETR
from ultralytics.utils.torch_utils import get_flops

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "runs/paper_queue/BASELINES_TABLE.md")

# (thư mục runs, data yaml, nhãn nhánh)
GROUPS = [
    (os.path.join(ROOT, "runs/yolo"),        "configs/visdrone_person_ultralytics.yaml",
     "PHASE-1 (no-COCO): train VisDrone-only, val VisDrone-val 548"),
    (os.path.join(ROOT, "runs/yolo_deploy"), "configs/visdrone_coco_ultralytics.yaml",
     "PHASE-2 (deploy): train VisDrone+COCO 9125, val VisDrone-val 548"),
]


def epochs_of(d):
    csv = os.path.join(d, "results.csv")
    if not os.path.exists(csv):
        return 0
    with open(csv) as f:
        return max(0, sum(1 for _ in f) - 1)


def main():
    lines = ["# BASELINES — params / GFLOPs / mAP (tính lại chuẩn)", ""]
    for proj, data, label in GROUPS:
        lines += [f"## {label}", "",
                  "| model | params (M) | GFLOPs@640 | mAP50 | mAP50-95 | epoch đã train |",
                  "|---|---|---|---|---|---|"]
        if not os.path.isdir(proj):
            lines += ["| *(chưa có run nào)* | | | | | |", ""]
            continue
        for d in sorted(glob.glob(os.path.join(proj, "*"))):
            name = os.path.basename(d)
            best = os.path.join(d, "weights", "best.pt")
            if not os.path.exists(best):
                continue
            Cls = RTDETR if "rtdetr" in name else YOLO
            try:
                m = Cls(best)
                params = sum(p.numel() for p in m.model.parameters()) / 1e6
                try:
                    flops = get_flops(m.model, imgsz=640)
                except Exception:
                    flops = 0.0
                r = m.val(data=data, imgsz=640, split="val", verbose=False)
                lines.append(f"| {name} | {params:.2f} | {flops:.1f} | "
                             f"{r.box.map50:.4f} | {r.box.map:.4f} | {epochs_of(d)} |")
                print(f"[ok] {name}: {params:.2f}M {flops:.1f}G mAP50={r.box.map50:.4f}")
            except Exception as e:
                lines.append(f"| {name} | LỖI: {str(e)[:40]} | | | | {epochs_of(d)} |")
                print(f"[loi] {name}: {str(e)[:80]}")
        lines.append("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[done] -> {OUT}")


if __name__ == "__main__":
    main()
