"""
build_deploy.py — Đóng gói model E (vibration) cho deploy Pi 5
==============================================================

Từ checkpoint train (E = density wrapper, có density_head chỉ lúc train) -> tạo bộ deploy:
  1) deploy/<name>.pt        : base v15 light SẠCH (bỏ density_head) — load lại được bình thường.
  2) deploy/<name>_320.onnx  : ONNX FP32 @320 (đã fuse conv+bn) — chạy ONNX Runtime trên Pi.
  3) deploy/<name>_320_int8.onnx : ONNX INT8 (dynamic quant) — nhỏ ~4×, nhanh hơn trên CPU ARM.

Vibration là TRAIN-TIME ONLY -> model deploy y hệt baseline (1.003M params, 1.62 GMACs @320).

Dùng:
  python build_deploy.py --weights runs/train_v15_vib_E/last.pt --name pfdet_E --img-size 320
"""

import os
import sys
import copy
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNanoV15, count_params, DEFAULT_EXPORT_OUTPUT_NAMES
from export import export_onnx, prepare_export_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="checkpoint train (vd runs/train_v15_vib_E/last.pt)")
    ap.add_argument("--name", default="pfdet_E", help="tên file output")
    ap.add_argument("--img-size", type=int, default=320, help="resolution deploy (Pi = 320)")
    ap.add_argument("--profile", default="light")
    ap.add_argument("--out-dir", default="deploy")
    ap.add_argument("--calib-images", default=None, help="ảnh calibrate INT8 (mặc định val)")
    ap.add_argument("--calib-labels", default=None)
    ap.add_argument("--use-ema", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device("cpu")

    # ---- 1) Nạp weight train vào model DEPLOY base (bỏ density_head) ----
    ck = torch.load(args.weights, map_location=dev, weights_only=False)
    state = ck.get("ema") if (args.use_ema and "ema" in ck) else ck.get("model", ck)
    model = PFDetNanoV15(profile=args.profile).eval()
    miss, unexp = model.load_state_dict(state, strict=False)
    print(f"[load] {args.weights}")
    print(f"  missing={len(miss)} (phải 0)  unexpected={len(unexp)} (density_head train-only, bỏ OK)")
    assert len(miss) == 0, f"THIẾU weight deploy: {miss[:5]}"

    total, _ = count_params(model)
    print(f"[model] base v15 {args.profile}: {total/1e6:.3f}M params (deploy)")

    # ---- 2) Lưu checkpoint deploy SẠCH (.pt) ----
    cfg = ck.get("cfg", {})
    cfg = copy.deepcopy(cfg) if cfg else {"model": {}, "export": {}}
    cfg.setdefault("model", {})
    cfg["model"]["version"] = "v15"
    cfg["model"]["profile"] = args.profile
    cfg["model"]["img_size"] = args.img_size
    cfg.setdefault("export", {})["output_names"] = list(DEFAULT_EXPORT_OUTPUT_NAMES)
    pt_path = os.path.join(args.out_dir, f"{args.name}.pt")
    torch.save({"model": model.state_dict(), "ema": model.state_dict(),
                "cfg": cfg, "model_version": "v15",
                "ap": ck.get("ap", None), "source": args.weights}, pt_path)
    print(f"[save] {pt_path} ({os.path.getsize(pt_path)/1e6:.2f}MB) — checkpoint deploy sạch")

    # ---- 3) Export ONNX FP32 @img_size (fuse conv+bn) ----
    export_model, fused = prepare_export_model(model, fuse=True)
    print(f"[fuse] reparameterize conv+bn: {'yes' if fused else 'no'}")
    onnx_path = os.path.join(args.out_dir, f"{args.name}_{args.img_size}.onnx")
    export_onnx(export_model, args.img_size, onnx_path,
                list(DEFAULT_EXPORT_OUTPUT_NAMES), opset=13, simplify=True)  # 13: per-channel INT8 cần axis

    # ---- 4) Quantize INT8 STATIC (QLinearConv — chạy được trên ORT CPU/Pi; dynamic ConvInteger thì KHÔNG) ----
    try:
        from onnxruntime.quantization import quantize_static, QuantType, QuantFormat, CalibrationDataReader
        from datasets import VisDronePersonDataset

        class _CalibReader(CalibrationDataReader):
            """Cấp ~n_calib ảnh val thật để calibrate scale/zero-point INT8."""
            def __init__(self, images_dir, labels_dir, img_size, input_name, n_calib=120):
                ds = VisDronePersonDataset(images_dir, labels_dir, img_size=img_size, augment=False)
                self.data = ({input_name: ds[i][0].unsqueeze(0).numpy().astype("float32")}
                             for i in range(min(n_calib, len(ds))))
            def get_next(self):
                return next(self.data, None)

        calib_imgs = args.calib_images or os.path.join("data/visdrone/val/images")
        calib_lbls = args.calib_labels or os.path.join("data/visdrone/val/labels")
        int8_path = os.path.join(args.out_dir, f"{args.name}_{args.img_size}_int8.onnx")
        reader = _CalibReader(calib_imgs, calib_lbls, args.img_size, "input")
        # LOẠI head detection (cls/box) khỏi quantize -> giữ FP32 cho head, chỉ quant backbone/neck.
        # Quant head làm hỏng logits/box -> AP sụp. Đây là cách chuẩn cho detector.
        import onnx as _onnx
        _g = _onnx.load(onnx_path).graph
        exclude = [n.name for n in _g.node if n.op_type == "Conv" and "head" in n.name.lower()]
        print(f"[int8] loại {len(exclude)} conv head khỏi quantize (giữ FP32)")
        quantize_static(onnx_path, int8_path, reader,
                        quant_format=QuantFormat.QDQ,
                        per_channel=True,
                        weight_type=QuantType.QInt8,
                        activation_type=QuantType.QUInt8,
                        nodes_to_exclude=exclude)
        s_fp32 = os.path.getsize(onnx_path) / 1e6
        s_int8 = os.path.getsize(int8_path) / 1e6
        print(f"[int8] {int8_path} ({s_int8:.2f}MB) — static QDQ, nén {s_fp32/s_int8:.1f}× so với FP32 ({s_fp32:.2f}MB)")
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"[int8] bỏ qua (lỗi: {e}) — dùng FP32 ONNX cũng được")

    print("\n✅ XONG. File deploy ở thư mục:", args.out_dir)
    print("   Test độ chính xác:  python test_accuracy_onnx.py --onnx <file.onnx> --img-size", args.img_size)


if __name__ == "__main__":
    main()
