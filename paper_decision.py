"""
paper_decision.py — TỰ ĐỘNG so no-COCO vs COCO(deploy) rồi ra khuyến nghị model-only vs system paper.
====================================================================================================
Chạy sau khi queue train xong. Eval cả 2 model trên VisDrone-val @640, tính gap, ghi DECISION.md.
Quy tắc user chốt: gap NHỎ -> system paper (model COCO + deploy) ; gap LỚN -> model-only (no-COCO).
"""
import os, re, subprocess, datetime

PY = "/home/nolan/Documents/Reinforcement_Learning/train/bin/python"
VAL_I = "data/visdrone/val/images"
VAL_L = "data/visdrone/val/labels"
OUT = "runs/paper_queue/DECISION.md"

MODELS = {
    "no-COCO (paper baseline)": "runs/train_v17_light/best.pt",
    "COCO (deploy)":            "runs/train_v17_deploy/best.pt",
    "no-COCO + vibration":      "runs/train_v17_vib/best.pt",
}


def eval_ap(weights):
    """Trả dict AP từ run_eval_coco.py (overall). None nếu thiếu file/lỗi."""
    if not os.path.exists(weights):
        return None
    try:
        out = subprocess.run(
            [PY, "run_eval_coco.py", "--weights", weights, "--val-images", VAL_I,
             "--val-labels", VAL_L, "--img-size", "640", "--profile", "light", "--device", "cuda:0"],
            capture_output=True, text=True, timeout=1800).stdout
        m = re.search(r"\[COCO\] overall:\s*(\{.*\})", out)
        if not m:
            return None
        d = eval(m.group(1))
        return {"AP": d.get("AP", 0), "AP50": d.get("AP50", 0), "AP75": d.get("AP75", 0)}
    except Exception:
        return None


def main():
    res = {name: eval_ap(w) for name, w in MODELS.items()}
    nc = res.get("no-COCO (paper baseline)")
    co = res.get("COCO (deploy)")

    lines = [f"# QUYẾT ĐỊNH HƯỚNG PAPER (tự động {datetime.datetime.now():%Y-%m-%d %H:%M})", ""]
    lines.append("## Số liệu (VisDrone-val @640, AP@.5:.95 / AP50 / AP75)")
    for name, r in res.items():
        if r:
            lines.append(f"- **{name}**: AP={r['AP']:.4f}  AP50={r['AP50']:.4f}  AP75={r['AP75']:.4f}")
        else:
            lines.append(f"- {name}: (chưa có / lỗi eval)")
    lines.append("")

    rec = "CHƯA ĐỦ DỮ LIỆU"
    if nc and co:
        gap_ap = nc["AP"] - co["AP"]
        gap_ap50 = nc["AP50"] - co["AP50"]
        lines.append(f"## Gap (no-COCO trừ COCO): ΔAP={gap_ap:+.4f}  ΔAP50={gap_ap50:+.4f}")
        lines.append("")
        # quy tắc: gap AP50 < 0.03 -> nhỏ -> SYSTEM ; ngược lại -> model-only
        if abs(gap_ap50) < 0.03 and abs(gap_ap) < 0.02:
            rec = ("SYSTEM PAPER (model COCO + deploy). Gap NHỎ trên val, mà COCO tốt hơn cho 3-10m "
                   "-> dùng model COCO cho cả paper lẫn bay, thêm phần Hailo + PX4 control + bay thật.")
        else:
            rec = ("MODEL-ONLY (no-COCO). Gap LỚN trên benchmark -> COCO hy sinh số quá nhiều. "
                   "Viết model-only (vibration + efficiency), deploy để future work.")
        lines.append(f"## 🎯 KHUYẾN NGHỊ: **{rec}**")
        lines.append("")
        lines.append("> Lưu ý: VisDrone-val toàn người tí xíu (lợi cho no-COCO). Deploy thật 3-10m người to "
                     "-> COCO tốt hơn. Nếu gap val nhỏ, COCO là lựa chọn rõ ràng tốt hơn cho hệ thống.")
    else:
        lines.append(f"## KHUYẾN NGHỊ: {rec} (thiếu eval 1 trong 2 model)")

    os.makedirs("runs/paper_queue", exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    # append vào SUMMARY luôn
    with open("runs/paper_queue/SUMMARY.txt", "a") as f:
        f.write("\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
