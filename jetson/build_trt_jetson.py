"""
build_trt_jetson.py — Build TensorRT FP16 engine từ ONNX, CHẠY TRÊN JETSON NANO B01.
====================================================================================

TRT engine phụ thuộc thiết bị -> phải build trên chính Jetson. JetPack có sẵn TensorRT.

  python3 build_trt_jetson.py --onnx pfdet_E_512.onnx --fp16
  -> tạo pfdet_E_512_fp16.engine

Sau đó benchmark:  python3 bench_trt.py --engine pfdet_E_512_fp16.engine --img-size 512
"""
import argparse, os
import tensorrt as trt


def build(onnx_path, engine_path, fp16=True, workspace_mb=512):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("parse ONNX lỗi")

    config = builder.create_builder_config()
    # API khác nhau giữa TRT 7 (Nano JetPack4) và TRT 8
    try:
        config.max_workspace_size = workspace_mb * (1 << 20)
    except Exception:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20))
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[trt] FP16 BẬT (Maxwell tối ưu FP16)")

    print("[trt] đang build engine... (vài phút trên Nano)")
    try:
        engine = builder.build_engine(network, config)          # TRT7
        blob = engine.serialize()
    except AttributeError:
        blob = builder.build_serialized_network(network, config)  # TRT8+
    with open(engine_path, "wb") as f:
        f.write(blob)
    print(f"[trt] xong: {engine_path} ({os.path.getsize(engine_path)/1e6:.1f}MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.onnx.replace(".onnx", "_fp16.engine")
    build(args.onnx, out, fp16=args.fp16)
