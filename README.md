# PFDet Drone Follow

Lightweight person detection for drone footage, with a re-baselining workflow against modern Ultralytics detectors.

The repo now has two parallel goals:

1. Keep `PFDet-v14` as a research detector for tiny aerial persons.
2. Re-benchmark it fairly against current strong baselines such as `YOLO26`, `YOLO11`, and `RT-DETR`.

## Current Model

`PFDet-v14` is the active in-repo detector.

- Backbone: `EdgeContextStem + UIB`
- Pyramid: `P2/P3/P4/P5` with strides `[4, 8, 16, 32]`
- Neck: `BiFPN`
- Head: `RepConv` decoupled head
- Optional: `AreaAttention`
- Export/runtime contract: `output_p2`, `output_p3`, `output_p4`, `output_p5`

## Config Roles

Use the config role before training anything:

- `configs/train_config_v14_clean.yaml`
  Role: `research_baseline`
  Purpose: clean PFDet baseline for ablation.
  Keeps only the core architecture and simple loss.

- `configs/train_config_v14.yaml`
  Role: `research_enhanced`
  Purpose: stronger PFDet config with extra assignment/loss/data tricks.
  Use this only after `v14_clean` has been benchmarked.

- `configs/train_config_v14_light.yaml`
  Role: `edge_candidate`
  Purpose: smaller PFDet profile for edge deployment.
  `AreaAttention` is disabled here on purpose.

## Training PFDet

Clean baseline:

```bash
python train_v3.py --config configs/train_config_v14_clean.yaml --model v14
```

Enhanced research config:

```bash
python train_v3.py --config configs/train_config_v14.yaml --model v14
```

Edge candidate:

```bash
python train_v3.py --config configs/train_config_v14_light.yaml --model v14
```

## PFDet Benchmarking

`benchmark.py` now benchmarks the fused deploy graph by default. This matters because `RepConv` must be fused before export/speed measurement.

Example:

```bash
python benchmark.py \
  --weights runs/train_v14_clean/best.pt \
  --profile desktop_4060 \
  --img-size 640 \
  --scoreboard runs/rebaseline/scoreboard.csv \
  --label pfdet_v14_clean
```

Reported fields include:

- train params / deploy params
- train FLOPs / deploy FLOPs
- model-only FPS
- end-to-end FPS
- AP@0.5
- focus AP / recall / precision

## Export

`export.py` now exports the fused deploy graph by default.

```bash
python export.py --weights runs/train_v14_clean/best.pt --format onnx
```

Use `--no-fuse` only if you intentionally want the training graph.

## Re-Baselining Against World Models

### 1. Prepare an Ultralytics-friendly dataset view

PFDet labels can contain 7 columns (`cls cx cy w h fx fy`), while Ultralytics detectors expect 5 columns.

Prepare a mirrored dataset:

```bash
python scripts/prepare_ultralytics_dataset.py \
  --src ./data/visdrone \
  --dst ./data/visdrone_ultralytics
```

This writes labels compatible with:

- `YOLO26n`
- `YOLO26s`
- `YOLO11n`
- `RT-DETR-L`

Dataset YAML:

- `configs/visdrone_person_ultralytics.yaml`

### 2. Fine-tune official pretrained baselines

```bash
python scripts/rebaseline_world_models.py train \
  --models yolo26n.pt yolo26s.pt yolo11n.pt rtdetr-l.pt \
  --data configs/visdrone_person_ultralytics.yaml \
  --img-size 640 \
  --epochs 100 \
  --project runs/rebaseline
```

### 3. Benchmark full-frame and SAHI

```bash
python scripts/rebaseline_world_models.py benchmark \
  --models \
    runs/rebaseline/yolo26n_640/weights/best.pt \
    runs/rebaseline/yolo26s_640/weights/best.pt \
    runs/rebaseline/yolo11n_640/weights/best.pt \
    runs/rebaseline/rtdetr-l_640/weights/best.pt \
  --data configs/visdrone_person_ultralytics.yaml \
  --profile desktop_4060 \
  --img-size 640 \
  --slice-sizes 512 640 \
  --scoreboard runs/rebaseline/scoreboard.csv
```

This produces standardized rows for:

- `full_frame`
- `sahi_512`
- `sahi_640`

## Model Selection Rule

The decision rule is encoded in `configs/rebaseline_candidates.yaml`.

Choose the winner by:

1. `focus_ap50`
2. `precision`
3. `end_to_end_fps`

That means the final production model does not have to remain PFDet if a baseline wins clearly.

## Notes On PFDet-Clean

`train_config_v14_clean.yaml` intentionally disables the confounding extras:

- `AreaAttention`
- `copy_paste`
- `multiscale`
- `NWD`
- `ASL`
- `OHEM`
- progressive multi-positive assignment
- COCO-person extra mixing

This gives a baseline that is much easier to trust during ablation.

## Requirements

Core repo:

```bash
pip install torch torchvision pyyaml tqdm opencv-python matplotlib
```

Optional for full re-baselining:

```bash
pip install ultralytics sahi thop onnxruntime onnxsim
```

## Important Caveat

Do not compare:

- PFDet at `320/384` full-frame
- YOLO26/YOLO11/RT-DETR at `640`

and then conclude one model is better.

For tiny aerial persons, the comparison must be done at the same benchmark input strategy, and ideally with both:

- full-frame inference
- tiled / SAHI inference
