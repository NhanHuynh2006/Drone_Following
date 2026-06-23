"""
bench_trt.py — Đo FPS thật của engine TensorRT trên Jetson Nano B01.
====================================================================

  python3 bench_trt.py --engine pfdet_E_512_fp16.engine --img-size 512 --iters 200

In: ms/frame + FPS (pure inference). Đây là con số quyết định có cần đổi kiến trúc không.
"""
import argparse, time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--img-size", type=int, default=512)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as f, trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    ctx = engine.create_execution_context()

    # cấp buffer cho mọi binding
    bufs, host_in = [], None
    sz = args.img_size
    for i in range(engine.num_bindings):
        shape = engine.get_binding_shape(i)
        shape = [d if d > 0 else 1 for d in shape]
        if any(d <= 0 for d in shape) or len(shape) == 4:
            shape = [1, 3, sz, sz] if engine.binding_is_input(i) else shape
        vol = int(np.prod(shape))
        dev = cuda.mem_alloc(vol * 4)
        bufs.append(int(dev))
        if engine.binding_is_input(i):
            host_in = (np.random.rand(vol).astype(np.float32), dev)

    stream = cuda.Stream()
    cuda.memcpy_htod(host_in[1], host_in[0])

    for _ in range(args.warmup):
        ctx.execute_v2(bufs)
    cuda.Context.synchronize()

    t0 = time.perf_counter()
    for _ in range(args.iters):
        ctx.execute_v2(bufs)
    cuda.Context.synchronize()
    dt = (time.perf_counter() - t0) / args.iters

    print(f"[Jetson TRT] {args.engine} @ {sz}px")
    print(f"  {dt*1000:.2f} ms/frame  |  {1/dt:.1f} FPS  (pure inference, FP16)")
    print("  realtime (>25 FPS)?", "✅ CÓ" if 1/dt >= 25 else "⚠️ chưa — cân nhắc NMS-free/giảm res")


if __name__ == "__main__":
    main()
