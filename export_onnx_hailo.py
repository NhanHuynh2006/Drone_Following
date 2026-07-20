#!/usr/bin/env python3
"""
BƯỚC 1 — Export ONNX "sạch" cho Hailo từ checkpoint PFDet-Nano v15.

Khác export.py gốc ở chỗ:
  - KHÔNG dùng dynamic_axes (Hailo cần static shape, batch=1)
  - Fuse RepConv (reparameterize) trước khi export
  - opset 11 (an toàn nhất với Hailo parser)
  - Chạy onnxsim để dọn graph (constant folding giúp parser dễ ăn hơn)

Đặt file này vào THƯ MỤC GỐC của repo Drone_Following rồi chạy:
    python export_onnx_hailo.py \
        --weights runs/train_v15_light_musgd_v3/best.pt \
        --img-size 320 \
        --output pfdet_v15_hailo.onnx

Lưu ý: img-size nên đúng size bạn sẽ chạy trên Pi (320), vì HEF là static.
"""

import os
import sys
import copy
import argparse

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import DEFAULT_EXPORT_OUTPUT_NAMES, build_model_from_checkpoint, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--img-size', type=int, default=320)
    ap.add_argument('--output', default='pfdet_v17_hailo.onnx')
    ap.add_argument('--opset', type=int, default=11)
    ap.add_argument('--no-simplify', action='store_true')
    args = ap.parse_args()

    model, _ck, _ver, _mk = build_model_from_checkpoint(args.weights, device='cpu', use_ema=True)
    print(f'[export] version={_ver}')
    model = copy.deepcopy(model).eval()
    if hasattr(model, 'reparameterize'):
        model = model.reparameterize()
        print('[export] RepConv fused (reparameterized)')

    total, _ = count_params(model)
    print(f'[export] params: {total / 1e6:.2f}M | input: 1x3x{args.img_size}x{args.img_size}')

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    output_names = list(DEFAULT_EXPORT_OUTPUT_NAMES)

    # QUAN TRỌNG: không truyền dynamic_axes -> graph hoàn toàn static
    torch.onnx.export(
        model, dummy, args.output,
        opset_version=args.opset,
        input_names=['input'],
        output_names=output_names,
        do_constant_folding=True,
    )
    print(f'[export] ONNX saved: {args.output} ({os.path.getsize(args.output)/1e6:.2f}MB)')

    if not args.no_simplify:
        try:
            import onnx, onnxsim
            m = onnx.load(args.output)
            m, ok = onnxsim.simplify(m)
            assert ok, 'onnxsim check failed'
            # Hailo parser cũ đôi khi kén ir_version cao
            if m.ir_version > 8:
                m.ir_version = 8
            onnx.save(m, args.output)
            print('[export] simplified + ir_version<=8')
        except ImportError:
            print('[export] onnxsim chưa cài (pip install onnxsim onnx) — bỏ qua')

    # verify
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(args.output, providers=['CPUExecutionProvider'])
        outs = sess.run(None, {'input': np.random.randn(1, 3, args.img_size, args.img_size).astype(np.float32)})
        print('[export] verify OK, output shapes:', [o.shape for o in outs])
        print('[export] output names:', [o.name for o in sess.get_outputs()])
    except ImportError:
        pass

    print('\nTiếp theo: mở file này bằng Netron (netron.app) để ghi lại tên 4 node Conv')
    print('cuối cùng của 4 head (mỗi node ra 5 kênh) — sẽ dùng làm end-nodes nếu parser fail.')


if __name__ == '__main__':
    main()
