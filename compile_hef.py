#!/usr/bin/env python3
"""
BƯỚC 2 — Convert ONNX -> HEF bằng Hailo Dataflow Compiler (ClientRunner API).

Chạy trên máy x86_64 Linux (Legion/Ubuntu của bạn), TRONG venv đã cài
hailo_dataflow_compiler (.whl tải từ hailo.ai Developer Zone, cần đăng ký).
KHÔNG chạy được trên Raspberry Pi.

    python compile_hef.py \
        --onnx pfdet_v15_hailo.onnx \
        --hw-arch hailo8l \
        --calib-dir data/visdrone/train/images \
        --img-size 320

hw-arch:
  hailo8l  -> AI Kit / AI HAT+ 13 TOPS (phổ biến nhất)
  hailo8   -> AI HAT+ 26 TOPS
(chạy `hailortcli scan` trên Pi nếu không chắc mình có con nào)

Nếu bước parse FAIL vì unsupported layer:
  - Đọc kỹ message: parser thường in luôn "Please try to parse the model again,
    using these end node names: ..." -> copy các tên đó vào --end-nodes
  - Nhưng với v15, layer fail (AreaAttention/LSK) nằm GIỮA backbone nên không
    cắt end-node được -> phải sửa model (xem README_HAILO.md mục 4) rồi
    finetune + export lại.
"""

import os
import glob
import argparse

import numpy as np
from PIL import Image

from hailo_sdk_client import ClientRunner


def load_calib(calib_dir, img_size, n=64):
    paths = sorted(
        glob.glob(os.path.join(calib_dir, '*.jpg'))
        + glob.glob(os.path.join(calib_dir, '*.jpeg'))
        + glob.glob(os.path.join(calib_dir, '*.png'))
    )
    assert paths, f'Không tìm thấy ảnh trong {calib_dir}'
    paths = paths[:: max(1, len(paths) // n)][:n]
    print(f'[calib] dùng {len(paths)} ảnh calibration')

    data = np.zeros((len(paths), img_size, img_size, 3), dtype=np.float32)
    for i, p in enumerate(paths):
        im = Image.open(p).convert('RGB').resize((img_size, img_size))
        data[i] = np.asarray(im, dtype=np.float32)  # 0..255, normalize để on-chip lo (xem model script)
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--hw-arch', default='hailo8l', choices=['hailo8', 'hailo8l'])
    ap.add_argument('--calib-dir', required=True)
    ap.add_argument('--img-size', type=int, default=320)
    ap.add_argument('--name', default='pfdet_v17')
    ap.add_argument('--end-nodes', nargs='*', default=None,
                    help='Tên các node ONNX làm output (chỉ cần khi parser gợi ý)')
    ap.add_argument('--calib-n', type=int, default=64,
                    help='So anh calibration. DFC muon 1024; 64 la it, no se canh bao '
                         '"didn\'t have enough data for dataset_size of 1024". Nhieu anh '
                         '= luong tu hoa sat thuc te hon, doi lai compile lau hon.')
    ap.add_argument('--opt-level', type=int, default=2,
                    help='model_optimization_flavor optimization_level (0-4)')
    args = ap.parse_args()

    runner = ClientRunner(hw_arch=args.hw_arch)

    # ---------- 1) PARSE: ONNX -> HAR ----------
    print(f'\n===== PARSE ({args.hw_arch}) =====')
    # Doc TEN INPUT thang tu file ONNX thay vi gan cung 'input': moi exporter dat ten
    # khac nhau (SteadyDet='input', Ultralytics='images'). Gan cung -> parser bao
    # "start_node_names and net_input_shapes keys must contain the same names".
    import onnx as _onnx
    _m = _onnx.load(args.onnx)
    _init = {i.name for i in _m.graph.initializer}
    _inputs = [i.name for i in _m.graph.input if i.name not in _init]
    if len(_inputs) != 1:
        raise SystemExit(f'ONNX co {len(_inputs)} input {_inputs}; script nay chi ho tro 1.')
    _in_name = _inputs[0]
    print(f'[parse] input ONNX = {_in_name!r}')
    parse_kwargs = dict(
        net_input_shapes={_in_name: [1, 3, args.img_size, args.img_size]},
    )
    if args.end_nodes:
        parse_kwargs['end_node_names'] = args.end_nodes
    runner.translate_onnx_model(args.onnx, args.name, **parse_kwargs)
    har_parsed = f'{args.name}_parsed.har'
    runner.save_har(har_parsed)
    print(f'[parse] OK -> {har_parsed}')

    # ---------- 2) OPTIMIZE: quantize INT8 ----------
    print('\n===== OPTIMIZE (INT8 quantization) =====')
    # Normalize on-chip: ảnh calib đưa vào dạng 0..255, chip tự chia 255
    # (nếu training của bạn normalize khác — mean/std ImageNet chẳng hạn —
    #  sửa lại dòng normalization cho khớp utils/dataset của repo!)
    model_script = (
        f'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])\n'
        f'model_optimization_flavor(optimization_level={args.opt_level})\n'
    )
    runner.load_model_script(model_script)

    calib = load_calib(args.calib_dir, args.img_size, n=args.calib_n)
    runner.optimize(calib)
    har_q = f'{args.name}_quantized.har'
    runner.save_har(har_q)
    print(f'[optimize] OK -> {har_q}')

    # ---------- 3) COMPILE: HAR -> HEF ----------
    print('\n===== COMPILE -> HEF =====')
    hef = runner.compile()
    hef_path = f'{args.name}_{args.hw_arch}.hef'
    with open(hef_path, 'wb') as f:
        f.write(hef)
    print(f'\n✅ DONE: {hef_path} — copy file này qua Raspberry Pi 5')


if __name__ == '__main__':
    main()
