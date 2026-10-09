"""
hailo_demo_cam.py — TEST detect trên Raspberry Pi 5 + Hailo-8L (chỉ model + camera, KHÔNG control).
====================================================================================================

Chạy TRÊN PI: cắm camera + Hailo AI Kit, xem model detect người đúng không (vẽ box + FPS).
Độc lập: chỉ cần `hailort` (HailoRT), `opencv-python`, `numpy` — KHÔNG cần torch/repo.
Decode + NMS nhúng sẵn trong file này.

  python3 hailo_demo_cam.py --hef pfdet_v17_hailo8l.hef --img-size 320 --cam 0 --conf 0.4

Phím: q = thoát | + / - = chỉnh ngưỡng conf
"""
import argparse, time
import numpy as np
import cv2

from hailo_platform import (HEF, VDevice, InferVStreams, ConfigureParams,
                            HailoStreamInterface, InputVStreamParams, OutputVStreamParams, FormatType)

STRIDES = [4, 8, 16, 32]   # P2, P3, P4, P5


# ---------- decode + NMS (numpy thuần, nhúng từ utils/box_ops.py) ----------
def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


def decode_scale(raw, stride, img_size, conf_thr):
    """raw: (5,H,W) logits -> list {'score','box':[x1,y1,x2,y2] normalized 0-1}."""
    C, H, W = raw.shape
    scores = _sig(raw[0])
    mask = scores >= conf_thr
    if not mask.any():
        return []
    ys, xs = np.where(mask)
    sc = scores[ys, xs]
    dx = _sig(raw[1, ys, xs]) - 0.5
    dy = _sig(raw[2, ys, xs]) - 0.5
    w = np.exp(np.clip(raw[3, ys, xs], -4, 4)) * (stride / img_size)
    h = np.exp(np.clip(raw[4, ys, xs], -4, 4)) * (stride / img_size)
    cx = (xs + dx) * (stride / img_size)
    cy = (ys + dy) * (stride / img_size)
    out = []
    for i in range(len(sc)):
        out.append({'score': float(sc[i]),
                    'box': [float(cx[i]-w[i]/2), float(cy[i]-h[i]/2),
                            float(cx[i]+w[i]/2), float(cy[i]+h[i]/2)]})
    return out


def nms(dets, iou_thr=0.5):
    if not dets:
        return []
    b = np.array([d['box'] for d in dets], np.float32)
    s = np.array([d['score'] for d in dets], np.float32)
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = (x2-x1)*(y2-y1)
    order = s.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        order = order[1:][iou < iou_thr]
    return [dets[k] for k in keep]


def letterbox(img, size):
    h0, w0 = img.shape[:2]
    r = min(size/h0, size/w0)
    nh, nw = int(h0*r), int(w0*r)
    rs = cv2.resize(img, (nw, nh))
    cv = np.full((size, size, 3), 114, np.uint8)
    top, left = (size-nh)//2, (size-nw)//2
    cv[top:top+nh, left:left+nw] = rs
    return cv, r, top, left


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", default="pfdet_v17_hailo8l.hef")
    ap.add_argument("--img-size", type=int, default=320)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4)
    args = ap.parse_args()
    sz = args.img_size

    hef = HEF(args.hef)
    with VDevice() as dev:
        cfg = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        ng = dev.configure(hef, cfg)[0]
        in_info = hef.get_input_vstream_infos()[0]
        in_name = in_info.name
        in_vp = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
        out_vp = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)  # dequantize -> logits float

        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # YUYV 720p nghen USB -> 5 fps
        # BUFFERSIZE=1 lam ROT DUNG MOT NUA khung (30->15 fps): mot buffer thi driver
        # khong co cho ghi khung ke trong luc app dang giu buffer. Do tren C270/Pi5
        # 2026-08-27. Dung 2 — van tuoi vi vong doc luon lay khung moi nhat.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        if not cap.isOpened():
            print("KHÔNG mở được camera", args.cam); return
        print(f"[hailo] {args.hef} @ {sz}px | q=thoat, +/- conf")
        conf = args.conf
        fps = None

        with InferVStreams(ng, in_vp, out_vp) as pipe, ng.activate():
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                canvas, r, top, left = letterbox(frame, sz)
                rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)[None]   # (1,sz,sz,3) uint8 RGB

                t0 = time.perf_counter()
                out = pipe.infer({in_name: rgb})
                dt = time.perf_counter() - t0
                fps = 1/dt if fps is None else 0.9*fps + 0.1*(1/dt)

                # gom 4 output, sắp theo kích thước (lớn nhất = P2/stride4)
                outs = list(out.values())
                outs.sort(key=lambda a: a.shape[1]*a.shape[2], reverse=True)  # NHWC: (1,H,W,5)
                dets = []
                for si, o in enumerate(outs):
                    raw = np.transpose(o[0], (2, 0, 1))   # (H,W,5) -> (5,H,W)
                    dets += decode_scale(raw, STRIDES[si], sz, conf)
                dets = nms(dets, 0.5)

                for d in dets:
                    x1, y1, x2, y2 = d["box"]
                    x1 = int((x1*sz-left)/r); x2 = int((x2*sz-left)/r)
                    y1 = int((y1*sz-top)/r);  y2 = int((y2*sz-top)/r)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{d['score']:.2f}", (x1, y1-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                cv2.putText(frame, f"{fps:.1f} FPS | {len(dets)} nguoi | conf={conf:.2f}", (8, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

                cv2.imshow("Hailo detect (q=thoat, +/- conf)", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord('q'): break
                elif k == ord('+'): conf = min(0.9, conf+0.05)
                elif k == ord('-'): conf = max(0.05, conf-0.05)

        cap.release(); cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
