"""
Export PFDet-Nano v14 to ONNX and TensorRT-compatible artifacts.

The export contract mirrors the 4-scale v14 runtime path:
  input -> output_p2, output_p3, output_p4, output_p5
"""

import os
import sys
import argparse
import copy
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DEFAULT_EXPORT_OUTPUT_NAMES, build_model_from_checkpoint, count_params


def prepare_export_model(model, fuse=True):
    export_model = copy.deepcopy(model).eval()
    fused = False
    if fuse and hasattr(export_model, 'reparameterize'):
        export_model = export_model.reparameterize()
        fused = True
    return export_model, fused


def export_onnx(model, img_size, output_path, output_names, opset=11, simplify=True):
    """
    Export model to ONNX format.
    """
    print(f"\n[ONNX] Exporting to {output_path} (opset={opset})")

    dummy_input = torch.randn(1, 3, img_size, img_size)
    model.eval()

    # Dynamic axes for potential batch size flexibility
    dynamic_axes = {'input': {0: 'batch'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch'}

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=['input'],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
    )

    print(f"[ONNX] Exported: {output_path} ({os.path.getsize(output_path)/1e6:.2f}MB)")

    # Simplify ONNX (reduces graph, improves TRT conversion)
    if simplify:
        try:
            import onnxsim
            import onnx
            model_onnx = onnx.load(output_path)
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "ONNX simplification failed"
            onnx.save(model_onnx, output_path)
            print("[ONNX] Simplified successfully")
        except ImportError:
            print("[ONNX] onnxsim not installed, skipping simplification")
            print("  Install with: pip install onnxsim")
        except Exception as e:
            print(f"[ONNX] Simplification failed: {e}")

    # Verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(output_path)
        dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        outputs = sess.run(None, {'input': dummy})
        print(f"[ONNX] Verification passed. Output shapes: {[o.shape for o in outputs]}")
    except ImportError:
        print("[ONNX] onnxruntime not installed, skipping verification")

    return output_path


def export_tensorrt(onnx_path, output_path, img_size, fp16=True, workspace_gb=1):
    """
    Export ONNX to TensorRT engine.
    Must be run on the target device (Jetson Nano).
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("[TRT] tensorrt not installed. Run this on Jetson Nano.")
        print("  On Jetson Nano, TensorRT is pre-installed with JetPack.")

        # Generate a helper script instead
        script_path = output_path.replace('.engine', '_build_trt.py')
        with open(script_path, 'w') as f:
            f.write(f'''#!/usr/bin/env python3
"""Run this script on Jetson Nano to build TensorRT engine."""
import tensorrt as trt

ONNX_PATH = "{os.path.basename(onnx_path)}"
ENGINE_PATH = "{os.path.basename(output_path)}"
IMG_SIZE = {img_size}
FP16 = {fp16}

logger = trt.Logger(trt.Logger.INFO)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

config = builder.create_builder_config()
config.max_workspace_size = {workspace_gb} * (1 << 30)

if FP16 and builder.platform_has_fast_fp16:
    config.set_flag(trt.BuilderFlag.FP16)
    print("FP16 mode enabled")

# Set input shape
profile = builder.create_optimization_profile()
profile.set_shape("input",
    (1, 3, IMG_SIZE, IMG_SIZE),   # min
    (1, 3, IMG_SIZE, IMG_SIZE),   # opt
    (1, 3, IMG_SIZE, IMG_SIZE))   # max
config.add_optimization_profile(profile)

print("Building TensorRT engine... (this takes several minutes on Jetson Nano)")
engine = builder.build_engine(network, config)

with open(ENGINE_PATH, 'wb') as f:
    f.write(engine.serialize())

print(f"Engine saved to {{ENGINE_PATH}}")
''')
        print(f"[TRT] Generated build script: {script_path}")
        print(f"  Copy {os.path.basename(onnx_path)} and {os.path.basename(script_path)} to Jetson Nano")
        print(f"  Then run: python3 {os.path.basename(script_path)}")
        return script_path

    # If TensorRT is available, build directly
    print(f"\n[TRT] Building engine from {onnx_path}")
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.max_workspace_size = workspace_gb * (1 << 30)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[TRT] FP16 mode enabled")

    profile = builder.create_optimization_profile()
    profile.set_shape("input",
        (1, 3, img_size, img_size),
        (1, 3, img_size, img_size),
        (1, 3, img_size, img_size))
    config.add_optimization_profile(profile)

    print("[TRT] Building engine... (this may take several minutes)")
    engine = builder.build_engine(network, config)

    with open(output_path, 'wb') as f:
        f.write(engine.serialize())

    print(f"[TRT] Engine saved: {output_path} ({os.path.getsize(output_path)/1e6:.2f}MB)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export PFDet-Nano")
    parser.add_argument("--weights", required=True, help="PyTorch weights (.pt)")
    parser.add_argument("--format", default="onnx", choices=["onnx", "trt", "all"])
    parser.add_argument("--fp16", action="store_true", help="FP16 for TensorRT")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version")
    parser.add_argument("--no-fuse", action="store_true", help="Export the training graph instead of the fused deploy graph")
    args = parser.parse_args()

    # Load model
    device = torch.device('cpu')
    model, ckpt, model_version, model_kwargs = build_model_from_checkpoint(
        args.weights,
        device=device,
        use_ema=True,
    )
    cfg = ckpt['cfg']

    img_size = cfg['model']['img_size']
    export_cfg = cfg.get('export', {})
    output_names = list(export_cfg.get('output_names') or DEFAULT_EXPORT_OUTPUT_NAMES)
    if len(output_names) != len(model.strides):
        raise ValueError(
            f"Expected {len(model.strides)} export outputs for PFDet v14, got {len(output_names)}"
        )
    export_model, fused = prepare_export_model(model, fuse=not args.no_fuse)
    total, _ = count_params(model)
    deploy_total, _ = count_params(export_model)
    print(
        f"Model: PFDetNano {model_version} "
        f"(base_c={cfg['model'].get('base_c', 'n/a')}, "
        f"train_params={total/1e6:.2f}M, deploy_params={deploy_total/1e6:.2f}M)"
    )
    if model_kwargs:
        print(f"Model kwargs: {model_kwargs}")
    print(f"Input size: {img_size}x{img_size}")
    print(f"Export outputs: {output_names}")
    print(f"Fused for export: {'yes' if fused else 'no'}")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.weights)
    os.makedirs(output_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(args.weights))[0]

    if args.format in ("onnx", "all"):
        onnx_path = os.path.join(output_dir, f"{stem}.onnx")
        export_onnx(export_model, img_size, onnx_path, output_names, opset=args.opset)

    if args.format in ("trt", "all"):
        onnx_path = os.path.join(output_dir, f"{stem}.onnx")
        if not os.path.exists(onnx_path):
            export_onnx(export_model, img_size, onnx_path, output_names, opset=args.opset)

        trt_path = os.path.join(output_dir, f"{stem}_fp16.engine" if args.fp16 else f"{stem}.engine")
        export_tensorrt(onnx_path, trt_path, img_size, fp16=args.fp16)


if __name__ == "__main__":
    main()
