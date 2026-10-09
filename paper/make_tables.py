"""
make_tables.py — sinh dòng LaTeX cho các bảng paper TRỰC TIẾP từ runs/paper_queue/UNIFIED.json.

Mục đích: không chép số bằng tay. Chép tay là chỗ sinh lỗi mà reviewer bắt được (số trong bảng
khác số trong body). Chạy lại sau mỗi lần đo thêm model:

    python paper/make_tables.py

rồi dán khối in ra vào sections/09_exp_main.tex, 10_exp_generality.tex, 11_exp_baselines.tex.
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "..", "runs", "paper_queue", "UNIFIED.json")

# params / GFLOPs @640 — nguồn: runs/paper_queue/BASELINES_TABLE.md (summarize_baselines.py)
FLOPS = {"yolov8n": 8.2, "yolo11n": 6.4, "yolo26n": 5.8, "yolov10n": 8.4,
         "rtdetr-l": 108.0, "v17-baseline": 4.6, "v17-vibcons": 4.6,
         "v17-nmsfree": 4.6, "v15-baseline": 12.9, "v15-vibcons": 12.9}
STRIDE = {"yolov8n": 8, "yolo11n": 8, "yolo26n": 8, "yolov10n": 8, "rtdetr-l": 8}

PRETTY = {
    "v17-baseline": "SteadyDet-N", "v17-vibcons": "\\quad + vib.-cons.",
    "v17-nmsfree": "\\quad NMS-free", "v15-baseline": "SteadyDet-W",
    "v15-vibcons": "\\quad + vib.-cons.", "v15-vibcons-last": "v15 + vib.-cons. (final ckpt)",
    "yolov8n": "YOLOv8-n", "yolo11n": "YOLO11-n", "yolo26n": "YOLO26-n",
    "yolov10n": "YOLOv10-n", "rtdetr-l": "RT-DETR-l",
    "v17-deploy": "v17 deploy (+COCO)", "v17-vib-deploy": "\\quad + vib.-cons.",
}


def ms(mean, std, dec=4):
    """\\ms{0.1098}{.0010} — bo so 0 dau cua std cho gon cot."""
    s = f"{std:.{dec}f}".lstrip("0")
    return f"\\ms{{{mean:.{dec}f}}}{{{s}}}"


def row_robust(name, r, bold=False):
    c, a = r["clean"], r["agg"]
    b = (lambda x: f"\\best{{{x}}}") if bold else (lambda x: x)
    return (f"{PRETTY.get(name, name):<26} & {c['AP']:.4f} & {c['AP50']:.4f} & {c['AP75']:.4f}\n"
            f"    & {b(ms(a['sev1']['mean'], a['sev1']['std']))}"
            f" & {b(ms(a['sev2']['mean'], a['sev2']['std']))}"
            f" & {b(ms(a['sev3']['mean'], a['sev3']['std']))}"
            f" & {b(ms(a['retain_pct']['mean'], a['retain_pct']['std'], 1))} \\\\")


def row_compare(name, r, bold_ap=False, bold_ret=False):
    c, a = r["clean"], r["agg"]
    g = FLOPS.get(name)
    st = STRIDE.get(name, 4)
    ap = f"\\best{{{c['AP50']:.4f}}}" if bold_ap else f"{c['AP50']:.4f}"
    rt = ms(a["retain_pct"]["mean"], a["retain_pct"]["std"], 1)
    if bold_ret:
        rt = f"\\best{{{rt}}}"
    return (f"{PRETTY.get(name, name):<26} & {r['params_M']:.2f} & "
            f"{'--' if g is None else f'{g:.1f}'} & {st} & "
            f"{ap} & {c['AP']:.4f} & {c['AP_s']:.4f} & {rt} \\\\")


def main():
    d = json.load(open(JSON))
    M = d["models"]
    p = d["protocol"]
    print(f"% protocol: {p['n_images']} images @{p['img_size']}, conf={p['conf']}, "
          f"nms_iou={p['nms_iou']}, max_det={p['max_det']}, seeds={p['seeds']}, {p['metric']}")

    print("\n%%%%%%%%%%%%%% Table: MAIN (v17) — sections/09_exp_main.tex %%%%%%%%%%%%%%")
    for n, bold in [("v17-baseline", False), ("v17-vibcons", True), ("v17-nmsfree", False)]:
        if n in M:
            print(row_robust(n, M[n], bold))

    print("\n%%%%%%%%%%%%%% Table: GENERALITY (v15) — sections/10_exp_generality.tex %%%%%%%%%%%%%%")
    for n, bold in [("v15-baseline", False), ("v15-vibcons", True), ("v15-vibcons-last", False)]:
        if n in M:
            print(row_robust(n, M[n], bold))

    print("\n%%%%%%%%%%%%%% Table: COMPARISON — sections/11_exp_baselines.tex %%%%%%%%%%%%%%")
    order = ["rtdetr-l", "yolov10n", "yolo11n", "yolov8n", "yolo26n",
             "v15-baseline", "v17-baseline", "v17-nmsfree", "v17-vibcons"]
    best_ap = max((M[n]["clean"]["AP50"], n) for n in order if n in M)[1]
    best_rt = max((M[n]["agg"]["retain_pct"]["mean"], n) for n in order if n in M)[1]
    for n in order:
        if n in M:
            print(row_compare(n, M[n], bold_ap=(n == best_ap), bold_ret=(n == best_rt)))

    print("\n%%%%%%%%%%%%%% VisDrone-val cua nhanh deploy (cho Section flight) %%%%%%%%%%%%%%")
    for n in ["v17-deploy", "v17-vib-deploy"]:
        if n in M:
            c = M[n]["clean"]
            print(f"% {n}: AP50={c['AP50']:.4f} AP={c['AP']:.4f} AP75={c['AP75']:.4f}")

    print("\n%%%%%%%%%%%%%% So dan xuat dung trong body %%%%%%%%%%%%%%")
    if "v17-baseline" in M and "v17-vibcons" in M:
        b, v = M["v17-baseline"], M["v17-vibcons"]
        rb, rv = b["agg"]["retain_pct"]["mean"], v["agg"]["retain_pct"]["mean"]
        print(f"% v17 retain: {rb:.1f} -> {rv:.1f}  (+{rv - rb:.1f} diem, "
              f"+{100 * (rv - rb) / rb:.0f}% tuong doi)")
        print(f"% v17 clean AP: {b['clean']['AP']:.4f} -> {v['clean']['AP']:.4f} "
              f"({100 * (v['clean']['AP'] - b['clean']['AP']) / b['clean']['AP']:+.1f}%)")
        print(f"% v17 clean AP50: {100 * (v['clean']['AP50'] - b['clean']['AP50']) / b['clean']['AP50']:+.1f}%  "
              f"AP75: {100 * (v['clean']['AP75'] - b['clean']['AP75']) / b['clean']['AP75']:+.1f}%")
        for s in ["sev1", "sev2", "sev3"]:
            print(f"% v17 {s}: {b['agg'][s]['mean']:.4f} -> {v['agg'][s]['mean']:.4f} "
                  f"= {v['agg'][s]['mean'] / b['agg'][s]['mean']:.2f}x")
        print(f"% v17 retain tai sev2: baseline {100 * b['agg']['sev2']['mean'] / b['clean']['AP']:.1f}% "
              f"-> vib {100 * v['agg']['sev2']['mean'] / v['clean']['AP']:.1f}%")
        print(f"% gain/std ratio: {(rv - rb) / max(b['agg']['retain_pct']['std'], 1e-9):.0f}x")
    if "v15-baseline" in M and "v15-vibcons" in M:
        b, v = M["v15-baseline"], M["v15-vibcons"]
        rb, rv = b["agg"]["retain_pct"]["mean"], v["agg"]["retain_pct"]["mean"]
        print(f"% v15 retain: {rb:.1f} -> {rv:.1f} (+{100 * (rv - rb) / rb:.0f}% tuong doi); "
              f"clean AP {b['clean']['AP']:.4f} -> {v['clean']['AP']:.4f} "
              f"({100 * (v['clean']['AP'] - b['clean']['AP']) / b['clean']['AP']:+.1f}%)")
        print(f"% v15 sev2: {b['agg']['sev2']['mean']:.4f} -> {v['agg']['sev2']['mean']:.4f} "
              f"= {v['agg']['sev2']['mean'] / b['agg']['sev2']['mean']:.2f}x")
    rets = [(r["agg"]["retain_pct"]["mean"], n) for n, r in M.items()
            if n in ("yolov8n", "yolo11n", "yolo26n", "yolov10n", "rtdetr-l")]
    if rets:
        lo, hi = min(rets), max(rets)
        print(f"% baseline retain range: {lo[0]:.1f}% ({lo[1]}) .. {hi[0]:.1f}% ({hi[1]}) "
              f"-> mat {100 - hi[0]:.0f}-{100 - lo[0]:.0f}% AP")


if __name__ == "__main__":
    main()
