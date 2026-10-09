"""
hailo_bench_latency.py — CHẠY TRÊN PI 5. Đo latency end-to-end + throughput bền + nhiệt.
=========================================================================================

Vì sao cần: "186 FPS" trong paper hiện là FPS *inference*, đo bằng EMA trong
hailo_demo_cam.py. Reviewer của một bài deployment sẽ hỏi ba thứ mà con số đó không trả lời:

  1. End-to-end là bao nhiêu? (capture + preprocess + inference + decode/NMS)
     -> cái quyết định vòng điều khiển là tổng, không phải riêng inference.
  2. p99 chứ không phải mean. Bộ điều khiển bay chết vì frame CHẬM NHẤT, không phải trung bình.
     Đặc biệt decode/NMS phụ thuộc số box -> cảnh đông người là lúc latency phình.
  3. Giữ được bao lâu? Pi 5 tản nhiệt bị động -> throttle sau vài phút là bình thường.
     FPS đo trong 10 giây đầu không phải FPS lúc bay.

Chạy:
    python hailo_bench_latency.py --hef pfdet_v17_640_hailo8l.hef --img-size 640 \
        --minutes 10 --source cam        # hoặc --source synth (không cần camera)

In ra bảng dán thẳng vào paper: từng chặng p50/p95/p99, FPS bền theo từng phút,
nhiệt độ SoC, và có throttle hay không.
"""

import argparse
import subprocess
import time

import numpy as np

from hailo_eval_val import STRIDES, decode_scale, letterbox, nms


def soc_temp_c():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0
    except Exception:
        return float("nan")


def throttled():
    """0x0 = chua bao gio throttle. Bit 0/1/2/3 = under-volt / freq-cap / throttle / soft-limit."""
    try:
        out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                             text=True, timeout=3).stdout.strip()
        return out.split("=")[-1]
    except Exception:
        return "n/a"


def pct(a):
    a = np.asarray(a)
    return dict(mean=float(a.mean()), p50=float(np.percentile(a, 50)),
                p95=float(np.percentile(a, 95)), p99=float(np.percentile(a, 99)),
                max=float(a.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="pfdet_v17_640_hailo8l.hef")
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--source", choices=["cam", "synth"], default="cam")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4, help="conf DEPLOY (khong phai 0.01 nhu eval)")
    args = ap.parse_args()

    import cv2
    from hailo_platform import (HEF, VDevice, ConfigureParams, HailoStreamInterface,
                                InputVStreamParams, OutputVStreamParams, InferVStreams,
                                FormatType)
    sz = args.img_size
    cap = None
    if args.source == "cam":
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # YUYV 720p nghen USB -> 5 fps
        # BUFFERSIZE=1 lam ROT DUNG MOT NUA khung (30->15 fps): mot buffer thi driver
        # khong co cho ghi khung ke trong luc app dang giu buffer. Do tren C270/Pi5
        # 2026-08-27. Dung 2 — van tuoi vi vong doc luon lay khung moi nhat.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            print("KHONG mo duoc camera -> chuyen sang --source synth")
            cap, args.source = None, "synth"
    synth = np.random.randint(0, 255, (720, 1280, 3), np.uint8)

    T = {k: [] for k in ("capture", "preprocess", "inference", "decode_nms", "total")}
    ndets = []
    per_min = []
    hef = HEF(args.hef)
    print(f"[bench] {args.hef} @ {sz} | source={args.source} | {args.minutes} phut")
    print(f"[bench] nhiet luc bat dau: {soc_temp_c():.1f}C | throttled={throttled()}")

    with VDevice() as dev:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = dev.configure(hef, cfg)[0]
        in_name = hef.get_input_vstream_infos()[0].name
        in_vp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_vp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
        with InferVStreams(ng, in_vp, out_vp) as pipe, ng.activate():
            t_end = time.time() + args.minutes * 60
            mark, nmark = time.time() + 60, 0
            while time.time() < t_end:
                t0 = time.perf_counter()
                if cap is not None:
                    ok, frame = cap.read()
                    if not ok:
                        break
                else:
                    frame = synth
                t1 = time.perf_counter()
                canvas = letterbox(frame, sz)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]
                t2 = time.perf_counter()
                out = pipe.infer({in_name: rgb})
                t3 = time.perf_counter()
                outs = sorted(out.values(), key=lambda a: a.shape[1] * a.shape[2],
                              reverse=True)
                dets = []
                for si, o in enumerate(outs):
                    dets += decode_scale(np.transpose(o[0], (2, 0, 1)), STRIDES[si],
                                         sz, args.conf)
                dets = nms(dets, 0.5)
                t4 = time.perf_counter()

                T["capture"].append((t1 - t0) * 1e3)
                T["preprocess"].append((t2 - t1) * 1e3)
                T["inference"].append((t3 - t2) * 1e3)
                T["decode_nms"].append((t4 - t3) * 1e3)
                T["total"].append((t4 - t0) * 1e3)
                ndets.append(len(dets))
                nmark += 1
                if time.time() >= mark:
                    per_min.append((len(per_min) + 1, nmark / 60.0, soc_temp_c()))
                    print(f"  phut {len(per_min)}: {nmark/60.0:.1f} FPS  "
                          f"{soc_temp_c():.1f}C  throttled={throttled()}", flush=True)
                    mark, nmark = mark + 60, 0

    if cap is not None:
        cap.release()
    print(f"\n[bench] nhiet cuoi: {soc_temp_c():.1f}C | throttled={throttled()}")
    print(f"[bench] {len(T['total'])} frame | so box/frame: p50={np.percentile(ndets,50):.0f} "
          f"p99={np.percentile(ndets,99):.0f}")
    print("\n%-12s %8s %8s %8s %8s %8s" % ("chang", "mean", "p50", "p95", "p99", "max"))
    for k in ("capture", "preprocess", "inference", "decode_nms", "total"):
        s = pct(T[k])
        print("%-12s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % (k, s["mean"], s["p50"], s["p95"], s["p99"], s["max"]))
    tot = pct(T["total"])
    print(f"\nFPS end-to-end: p50 {1000/tot['p50']:.0f} | p99 (worst) {1000/tot['p99']:.0f}")
    if per_min:
        f = [p[1] for p in per_min]
        print(f"FPS ben: phut dau {f[0]:.1f} -> phut cuoi {f[-1]:.1f} "
              f"({100*(f[-1]-f[0])/f[0]:+.1f}%)")
    print("\nCAU CHO BAI BAO: bao p50 VA p99 end-to-end, kem nhiet + throttle. "
          "Dung bao FPS inference-only.")


if __name__ == "__main__":
    main()
