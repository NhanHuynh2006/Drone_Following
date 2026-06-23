"""
Training script for PFDet-Nano v14.
===================================
Optimized for Drone Person Detection (VisDrone/UAVDT).
- Đã ÉP CỨNG in ảnh/mAP mỗi 5 Epoch.
- Đã BẬT TÍNH NĂNG TRỘN COCO (30%) cho kịch bản bay thấp.
"""

import os
# Chống phân mảnh VRAM do multi-scale (11 size đổi qua hàng nghìn iter) -> tránh OOM tích lũy.
# PHẢI set TRƯỚC khi import torch (đọc lúc khởi tạo CUDA). setdefault: không đè nếu user đã set.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import sys
import math
import copy
import argparse
import traceback
import yaml
import csv
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Subset
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import build_model, count_params, normalize_model_version, PFDetNanoV15
from datasets import VisDronePersonDataset, collate_fn
from utils import (
    PFDetLossV14,
    PFDetLossV15,
    PFDetLossV16,
    decode_predictions_np,
    nms_numpy,
    xywh2xyxy,
)
# DGS (Density-Guided Supervision) + KD + Vibration-consistency — train-time only, 0 chi phí inference
from density_aux import PFDetNanoV15Density, build_density_targets, density_loss
from distillation import load_teacher, DistillationLoss, TeacherFeatCapture
from vib_consistency import corrupt_batch, VibConsistencyLoss


def _labels_to_pixel_boxes(labels_list, img_size, person_class=0):
    """labels (N,5)[cls,cx,cy,w,h] chuẩn hoá 0-1 -> list per-image box PIXEL [cx,cy,w,h] (chỉ person).

    Dùng cho build_density_targets (cần toạ độ pixel ở kích thước img hiện tại)."""
    out = []
    for lb in labels_list:
        if len(lb) == 0:
            out.append(torch.zeros(0, 4)); continue
        person = lb[lb[:, 0] == person_class]
        if len(person) == 0:
            out.append(torch.zeros(0, 4)); continue
        out.append(person[:, 1:5].clone().float() * float(img_size))
    return out

# ============================================================
#  EMA
# ============================================================

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)
            for ema_b, model_b in zip(self.ema.buffers(), model.buffers()):
                ema_b.copy_(model_b)

    def state_dict(self):
        return self.ema.state_dict()


def build_train_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=collate_fn, drop_last=True,
    )

def _dataset_aug_kwargs(data_cfg):
    return {
        'mosaic_prob': data_cfg.get('mosaic_prob', 0.2),
        'mixup_prob': data_cfg.get('mixup_prob', 0.0),
        'drone_aug_prob': data_cfg.get('drone_aug_prob', 0.5),
        'copy_paste_prob': data_cfg.get('copy_paste_prob', 0.0),
        'hflip_prob': data_cfg.get('hflip_prob', 0.5),
        'affine_prob': data_cfg.get('affine_prob', 0.8),       
        'color_jitter_prob': data_cfg.get('color_jitter_prob', 0.8),
    }

def build_mixed_train_set(base_train_set, extra_set, extra_sample_ratio=0.3, seed=42):
    if extra_set is None:
        return base_train_set, 0
    ratio = float(extra_sample_ratio)
    if ratio <= 0.0:
        return base_train_set, 0
    if ratio >= 1.0:
        sampled_extra = extra_set
    else:
        n_extra = max(1, min(len(extra_set), int(round(len(base_train_set) * ratio))))
        rng = random.Random(seed)
        indices = list(range(len(extra_set)))
        rng.shuffle(indices)
        sampled_extra = Subset(extra_set, indices[:n_extra])
    return ConcatDataset([base_train_set, sampled_extra]), len(sampled_extra)

def _set_aug_mix(dataset, mosaic_prob, mixup_prob):
    if isinstance(dataset, ConcatDataset):
        for ds in dataset.datasets:
            _set_aug_mix(ds, mosaic_prob, mixup_prob)
        return
    if isinstance(dataset, Subset):
        _set_aug_mix(dataset.dataset, mosaic_prob, mixup_prob)
        return
    if hasattr(dataset, 'mosaic_prob'):
        dataset.mosaic_prob = mosaic_prob
    if hasattr(dataset, 'mixup_prob'):
        dataset.mixup_prob = mixup_prob

def disable_train_aug_mix(dataset):
    _set_aug_mix(dataset, mosaic_prob=0.0, mixup_prob=0.0)

def resolve_optimizer_cfg(train_cfg):
    opt_cfg = dict(train_cfg.get('optimizer') or {})
    return {
        'name': str(opt_cfg.get('name', train_cfg.get('optimizer_name', 'adamw'))).lower(),
        'lr': float(opt_cfg.get('lr', train_cfg.get('lr', 1e-3))),
        'lr_min': float(opt_cfg.get('lr_min', train_cfg.get('lr_min', 1e-5))),
        'weight_decay': float(opt_cfg.get('weight_decay', train_cfg.get('weight_decay', 5e-4))),
        'momentum': float(opt_cfg.get('momentum', train_cfg.get('momentum', 0.937))),
        'fused': opt_cfg.get('fused', 'auto'),
    }

def build_optimizer(model, optimizer_cfg, device=None):
    opt_name = optimizer_cfg['name']
    wd       = optimizer_cfg['weight_decay']
    lr       = optimizer_cfg['lr']

    def _is_nodecay(pname, p):
        return p.ndim == 1 or pname.endswith('.bias') or 'bn' in pname.lower()

    if opt_name == 'musgd':
        # Matches Ultralytics YOLO26 (commit f2d3aed):
        #   - 2D weight params → use_muon=True: blend Muon 50% + SGD 50%
        #   - 1D params (bias, BN) → use_muon=False: plain SGD
        #   - NO separate AdamW path — 50-50 blend handles head layers naturally
        #     (thin 1×48 head conv: Muon ≡ normalization + scale=1, SGD provides update)
        from utils.musgd import MuSGD
        muon_params, sgd_params = [], []
        muon_nodecay, sgd_nodecay = [], []
        for pname, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_nodecay(pname, p):
                sgd_nodecay.append(p)   # bias, BN → plain SGD, no decay
            elif p.ndim >= 2:
                muon_params.append(p)   # 2D weights → Muon+SGD blend, with decay
            else:
                sgd_params.append(p)    # fallback (shouldn't happen)
        param_groups = [
            {'params': muon_params,  'weight_decay': wd,  'use_muon': True,  'lr': lr, 'lr_ratio': 1.0},
            {'params': sgd_nodecay,  'weight_decay': 0.0, 'use_muon': False, 'lr': lr, 'lr_ratio': 1.0},
        ]
        if sgd_params:
            param_groups.append(
                {'params': sgd_params, 'weight_decay': wd, 'use_muon': False, 'lr': lr, 'lr_ratio': 1.0}
            )
        return MuSGD(param_groups, lr=lr,
                     momentum=optimizer_cfg.get('momentum', 0.9),
                     weight_decay=wd,
                     nesterov=True)

    decay_params, no_decay_params = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay_params if _is_nodecay(pname, p) else decay_params).append(p)
    param_groups = [
        {'params': decay_params,    'weight_decay': wd},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ]
    fused_cfg = optimizer_cfg.get('fused', 'auto')
    use_fused = bool(device is not None and device.type == 'cuda') if fused_cfg == 'auto' else bool(fused_cfg)

    if opt_name == 'adamw':
        try: return torch.optim.AdamW(param_groups, lr=lr, fused=use_fused)
        except TypeError: return torch.optim.AdamW(param_groups, lr=lr)
    if opt_name == 'sgd':
        return torch.optim.SGD(param_groups, lr=lr, momentum=optimizer_cfg['momentum'], nesterov=True)
    raise ValueError(f"Unsupported optimizer.name: {opt_name!r}")

def dets_sorted(detections):
    return sorted(detections, key=lambda d: d.get('score', 0.0), reverse=True)

def filter_eval_detections(detections, img_size, min_box_area=0.0, min_aspect=0.0, max_aspect=0.0, max_det=300):
    filtered = []
    for det in dets_sorted(detections):
        x1, y1, x2, y2 = det['box']
        bw = max(0.0, (x2 - x1) * img_size)
        bh = max(0.0, (y2 - y1) * img_size)
        area = bw * bh
        aspect = bh / max(bw, 1e-6)
        if area < min_box_area: continue
        if min_aspect > 0 and aspect < min_aspect: continue
        if max_aspect > 0 and aspect > max_aspect: continue
        filtered.append(det)
        if len(filtered) >= max_det: break
    return filtered

# ============================================================
#  LR Scheduler
# ============================================================

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs=5, lr_start=1e-4, lr_max=1e-3, lr_min=1e-5):
    if epoch < warmup_epochs:
        lr = lr_start + (lr_max - lr_start) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        # lr_ratio lets different groups (e.g. AdamW head vs Muon body) scale together.
        pg['lr'] = lr * pg.get('lr_ratio', 1.0)
    return lr

# ============================================================
#  Validation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, img_size, strides, conf_thr=0.01, iou_thr=0.5,
             decode_fn=decode_predictions_np, min_box_area=0.0, min_aspect=0.0,
             max_aspect=0.0, max_det=300, min_gt_area=0.0, progress_desc=None,
             coco=False):                                          # >>> COCO: thêm cờ
    model.eval()
    all_scores, all_correct = [], []
    total_gt, total_tp, total_fp = 0, 0, 0
    pred_all, gt_all, ngt_all = [], [], []                        # >>> COCO: gom theo ảnh

    iterator = tqdm(loader, desc=progress_desc, leave=False, total=len(loader)) if progress_desc else loader

    for imgs, labels_list, _ in iterator:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        for b in range(imgs.shape[0]):
            gt_labels = labels_list[b]
            gt_person = gt_labels[gt_labels[:, 0] == 0] if len(gt_labels) > 0 else gt_labels
            gt_boxes_xywh = gt_person[:, 1:5].numpy() if len(gt_person) > 0 else np.zeros((0, 4))
            if min_gt_area > 0 and len(gt_boxes_xywh) > 0:
                gt_area = gt_boxes_xywh[:, 2] * img_size * gt_boxes_xywh[:, 3] * img_size
                gt_boxes_xywh = gt_boxes_xywh[gt_area >= min_gt_area]
            n_gt = len(gt_boxes_xywh)
            total_gt += n_gt

            dets = []
            for si, pred in enumerate(preds):
                raw = pred[b].cpu().numpy()
                scale_dets = decode_fn(raw, strides[si], img_size, conf_thr=conf_thr)
                dets.extend(scale_dets)
            dets = nms_numpy(dets, iou_threshold=0.5)
            dets = filter_eval_detections(dets, img_size=img_size, min_box_area=min_box_area,
                                          min_aspect=min_aspect, max_aspect=max_aspect, max_det=max_det)

            pred_all.append(dets)                                 # >>> COCO
            gt_all.append(gt_boxes_xywh)                          # >>> COCO
            ngt_all.append(n_gt)                                  # >>> COCO

            if n_gt == 0:
                for d in dets:
                    all_scores.append(d['score']); all_correct.append(0); total_fp += 1
                continue
            if len(dets) == 0:
                continue

            det_boxes = np.array([d['box'] for d in dets])
            gt_boxes_xyxy = np.array([xywh2xyxy(g) for g in gt_boxes_xywh])
            iou_matrix = np.zeros((len(dets), n_gt))
            for di in range(len(dets)):
                for gi in range(n_gt):
                    iou_matrix[di, gi] = _iou_single(det_boxes[di], gt_boxes_xyxy[gi])

            matched_gt = set()
            for di in range(len(dets)):
                best_gi = iou_matrix[di].argmax()
                best_iou = iou_matrix[di, best_gi]
                if best_iou >= iou_thr and best_gi not in matched_gt:
                    all_scores.append(dets[di]['score']); all_correct.append(1)
                    matched_gt.add(best_gi); total_tp += 1
                else:
                    all_scores.append(dets[di]['score']); all_correct.append(0); total_fp += 1

    # ---- AP@0.5 cũ (giữ nguyên) ----
    if len(all_scores) == 0 or total_gt == 0:
        ap = recall = precision = 0.0
    else:
        indices = np.argsort(-np.array(all_scores))
        correct = np.array(all_correct)[indices]
        tp_cumsum = np.cumsum(correct)
        fp_cumsum = np.cumsum(1 - correct)
        recalls = tp_cumsum / total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        mrec = np.concatenate(([0.], recalls, [1.]))
        mpre = np.concatenate(([1.], precisions, [0.]))
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        recall = tp_cumsum[-1] / total_gt if total_gt > 0 else 0.0
        precision = total_tp / max(1, total_tp + total_fp)

    # ---- >>> COCO: AP@.5:.95 + AP_small/medium/large + theo mật độ ----
    if coco:
        try:
            from eval_coco import coco_eval
            overall = coco_eval(pred_all, gt_all, img_size, verbose=False)
            print("\n[COCO] overall:", {k: round(v, 4) for k, v in overall.items()})
            ngt_arr = np.array(ngt_all)
            for lo, hi in [(0, 10), (10, 30), (30, 1e9)]:
                idx = np.where((ngt_arr >= lo) & (ngt_arr < hi))[0]
                tag = f"{lo}-{int(hi) if hi < 1e8 else 'inf'} người (n={len(idx)})"
                if len(idx) == 0:
                    print(f"[COCO] {tag}: (rỗng)"); continue
                mm = coco_eval([pred_all[i] for i in idx], [gt_all[i] for i in idx], img_size, verbose=False)
                print(f"[COCO] {tag}: AP50={mm['AP50']:.3f} AP={mm['AP']:.3f} "
                      f"AP_s={mm['AP_s']:.3f} AR_s={mm['AR_s']:.3f}")
        except Exception as e:
            print(f"[COCO] bỏ qua (lỗi: {e})")

    return ap, recall, precision

def _iou_single(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter / (area1 + area2 - inter + 1e-7)

# ============================================================
#  Val Sample Visualization
# ============================================================

@torch.no_grad()
def save_val_samples(model, val_set, device, img_size, strides, save_dir, epoch,
                     n_samples=8, conf_thr=0.3, decode_fn=decode_predictions_np,
                     min_box_area=0.0, min_aspect=0.0, max_aspect=0.0, max_det=300):
    model.eval()
    out_dir = os.path.join(save_dir, 'val_samples')
    os.makedirs(out_dir, exist_ok=True)
    indices = random.sample(range(len(val_set)), min(n_samples, len(val_set)))
    grid_imgs = []
    
    for idx in indices:
        img_tensor, labels_tensor, _ = val_set[idx]
        inp = img_tensor.unsqueeze(0).to(device)
        preds = model(inp)
        dets = []
        for si, pred in enumerate(preds):
            raw = pred[0].cpu().numpy()
            scale_dets = decode_fn(raw, strides[si], img_size, conf_thr=conf_thr)
            dets.extend(scale_dets)
        dets = nms_numpy(dets, iou_threshold=0.5)
        dets = filter_eval_detections(dets, img_size=img_size, min_box_area=min_box_area, min_aspect=min_aspect, max_aspect=max_aspect, max_det=max_det)

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        if len(labels_tensor) > 0:
            for lb in labels_tensor:
                cx, cy, w, h = lb[1].item(), lb[2].item(), lb[3].item(), lb[4].item()
                x1, y1 = int((cx - w/2) * img_size), int((cy - h/2) * img_size)
                x2, y2 = int((cx + w/2) * img_size), int((cy + h/2) * img_size)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 100, 0), 1)

        for det in dets:
            x1, y1, x2, y2 = det['box']
            x1, y1, x2, y2 = int(x1 * img_size), int(y1 * img_size), int(x2 * img_size), int(y2 * img_size)
            score = det['score']
            color = (0, 255, 0) if score > 0.5 else (0, 200, 0)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_bgr, f'{score:.2f}', (x1, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.putText(img_bgr, f'Epoch {epoch+1}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        grid_imgs.append(img_bgr)

    n_cols = 4
    n_rows = (len(grid_imgs) + n_cols - 1) // n_cols
    cell_h, cell_w = img_size, img_size
    grid = np.full((n_rows * cell_h, n_cols * cell_w, 3), 114, dtype=np.uint8)
    for i, img in enumerate(grid_imgs):
        r, c = i // n_cols, i % n_cols
        h, w = img.shape[:2]
        grid[r*cell_h:r*cell_h+h, c*cell_w:c*cell_w+w] = img

    save_path = os.path.join(out_dir, f'epoch_{epoch+1:03d}.jpg')
    cv2.imwrite(save_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return save_path

# ============================================================
#  Training Curves Plot
# ============================================================

def plot_training_curves(csv_path, save_dir, model_name="v6"):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs, train_loss, obj_loss, box_loss = [], [], [], []
    val_epochs, ap_list, recall_list, precision_list = [], [], [], []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['epoch'])
            epochs.append(ep)
            train_loss.append(float(row['train_loss']))
            obj_loss.append(float(row['obj_loss']))
            box_loss.append(float(row['box_loss']))
            if row['ap'] != '':
                val_epochs.append(ep)
                ap_list.append(float(row['ap']))
                recall_list.append(float(row['recall']))
                precision_list.append(float(row['precision']))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'PFDet-Nano {model_name} Training', fontsize=14, fontweight='bold')

    axes[0, 0].plot(epochs, train_loss, 'b-', label='Total Loss')
    axes[0, 0].set_title('Total Loss'); axes[0, 0].legend()

    axes[0, 1].plot(epochs, obj_loss, 'r-', label='Obj Loss')
    axes[0, 1].plot(epochs, box_loss, 'g-', label='Box Loss')
    axes[0, 1].set_title('Obj & Box Loss'); axes[0, 1].legend()

    if val_epochs:
        axes[1, 0].plot(val_epochs, ap_list, 'b-o', label='AP@0.5')
        axes[1, 0].set_title(f'AP@0.5 (best: {max(ap_list):.4f})'); axes[1, 0].legend()

        axes[1, 1].plot(val_epochs, recall_list, 'g-o', label='Recall')
        axes[1, 1].plot(val_epochs, precision_list, 'r-o', label='Precision')
        axes[1, 1].set_title('Recall & Precision'); axes[1, 1].legend()

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_curves.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    return plot_path

# ============================================================
#  Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_config_v14_clean.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--model", default=None, choices=["v14", "v15", "v16"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg.setdefault('train', {})
    if args.batch_size is not None: train_cfg['batch_size'] = int(args.batch_size)
    if args.num_workers is not None: train_cfg['num_workers'] = int(args.num_workers)

    device = torch.device(train_cfg.get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    torch.set_float32_matmul_precision('high')

    model_cfg = dict(cfg.get('model') or {})
    # Model version: CLI arg overrides config; config overrides default 'v14'
    cfg_version = model_cfg.get('version', 'v14')
    model_name = normalize_model_version(args.model if args.model is not None else cfg_version)
    img_size = model_cfg.get('img_size', 384)
    model_kwargs = {k:v for k,v in model_cfg.items() if k not in ['img_size', 'version']}

    # ---- DGS + KD + Vibration-consistency config ----
    dgs_cfg = dict(cfg.get('dgs') or {})
    kd_cfg  = dict(cfg.get('kd') or {})
    vib_cfg = dict(cfg.get('vib') or {})
    use_density       = bool(dgs_cfg.get('use_density', False))
    use_kd            = bool(kd_cfg.get('use_kd', False))
    use_vib           = bool(vib_cfg.get('use_vib', False))
    density_scale_idx = int(dgs_cfg.get('density_scale_idx', 1))
    kd_feat_on        = use_kd and bool(kd_cfg.get('use_feature_kd', True))
    vib_feat_on       = use_vib and bool(vib_cfg.get('use_feature', True))

    # neck_feats (forward return_aux) cần khi: density loss, KD-feature, hoặc vib-feature.
    # -> dùng chung density wrapper; density LOSS chỉ bật khi use_density.
    need_aux = use_density or kd_feat_on or vib_feat_on
    if need_aux:
        if model_name != 'v15':
            raise ValueError("DGS / vib-feature / KD-feature hiện chỉ hỗ trợ model v15.")
        model = PFDetNanoV15Density(density_scale_idx=density_scale_idx, **model_kwargs).to(device)
        if use_density:
            print(f"[DGS] Density head BẬT (scale_idx={density_scale_idx}, stride={model.density_stride})")
        else:
            print("[AUX] Dùng density wrapper để lấy neck_feats (density LOSS tắt).")
    else:
        model = build_model(model_name, **model_kwargs).to(device)
    total_p, train_p = count_params(model)
    loss_name = {
        'v14': 'PFDetLossV14',
        'v15': 'PFDetLossV15 (STAL+ProgLoss)',
        'v16': 'PFDetLossV16 (STAL+ProgLoss)',
        'v17': 'PFDetLossV15 (FLOPs-efficient: P2 mỏng + LiteHead)',
    }.get(model_name, 'PFDetLossV14')
    print(f"[MODEL] PFDet-Nano {model_name.upper()} | {train_p/1e6:.3f}M params | "
          f"Loss: {loss_name}")
    if device.type == 'cuda':
        # Multi-scale đổi 11 size khác nhau -> cudnn.benchmark sẽ re-benchmark + cache workspace
        # mỗi size => phình VRAM và chậm lúc đổi size. Chỉ bật khi size CỐ ĐỊNH.
        _ms_on = bool((train_cfg.get('multiscale') or {}).get('enabled', False))
        torch.backends.cudnn.benchmark = not _ms_on
        torch.backends.cuda.matmul.allow_tf32 = True
        print(f"[CUDNN] benchmark={not _ms_on} (tắt khi multi-scale để khỏi phình VRAM theo size)")

    decode_fn = decode_predictions_np
    data_root = cfg['data']['root']
    data_aug_kwargs = _dataset_aug_kwargs(cfg['data'])
    
    # 1. Khởi tạo tập Drone (Base)
    base_train_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['train_images']),
        os.path.join(data_root, cfg['data']['train_labels']),
        img_size=img_size, augment=True, **data_aug_kwargs,
        cache_ram=bool(cfg['data'].get('cache_ram', False)),
    )
    
    # 2. Khởi tạo và TRỘN tập COCO (Extra) với tỷ lệ 30%
    extra_set = None
    if 'extra_images' in cfg['data'] and 'extra_labels' in cfg['data']:
        extra_set = VisDronePersonDataset(
            os.path.join(data_root, cfg['data']['extra_images']),
            os.path.join(data_root, cfg['data']['extra_labels']),
            img_size=img_size, augment=True, **data_aug_kwargs,
            cache_ram=bool(cfg['data'].get('cache_ram', False))
        )
    
    extra_enable_epoch  = int(cfg['data'].get('extra_enable_epoch', 0))
    extra_disable_epoch = int(cfg['data'].get('extra_disable_epoch', 9999))
    extra_ratio         = float(cfg['data'].get('extra_sample_ratio', 0.3))

    # Train set khởi tạo: chưa có COCO (epoch 0 chưa biết sẽ enable chưa)
    # Sẽ rebuild loader theo epoch trong vòng lặp train.
    print("\n[INFO] Đang chuẩn bị dữ liệu Train...")
    if extra_set is not None:
        print(f"  -> COCO extra: {len(extra_set)} ảnh, enable epoch {extra_enable_epoch}-{extra_disable_epoch}")
    train_set = base_train_set
    _coco_active = False  # tracker trạng thái hiện tại

    val_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['val_images']),
        os.path.join(data_root, cfg['data']['val_labels']),
        img_size=img_size, augment=False, cache_ram=bool(cfg['data'].get('cache_ram', False)),
    )

    nw = train_cfg['num_workers']
    batch_size = train_cfg['batch_size']
    eval_cfg = cfg.get('eval', {})
    train_loader = build_train_loader(train_set, batch_size, nw)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=min(4, nw),
                            pin_memory=True, collate_fn=collate_fn)

    optimizer_cfg = resolve_optimizer_cfg(train_cfg)
    optimizer = build_optimizer(model, optimizer_cfg, device=device)

    lcfg = cfg.get('loss', {})
    loss_common = dict(
        img_size=img_size, strides=tuple(model.strides),
        obj_weight=lcfg.get('obj_weight', 8.0),
        box_weight=lcfg.get('box_weight', 4.0),
        qfl_beta=lcfg.get('qfl_beta', 2.0),
        hard_floor=lcfg.get('hard_floor', 0.0),
        obj_warmup_epochs=lcfg.get('obj_warmup_epochs', 10),
        prog_epochs=lcfg.get('prog_epochs', train_cfg['epochs']),
        k_tiny=lcfg.get('k_tiny', 6),
        k_normal=lcfg.get('k_normal', 3),
        search_r_tiny=lcfg.get('search_r_tiny', 3),
        search_r_normal=lcfg.get('search_r_normal', 2),
        tal_alpha=lcfg.get('tal_alpha', 1.0),
        tal_beta=lcfg.get('tal_beta', 6.0),
        nwd_area_thr=lcfg.get('nwd_area_thr', 1024.0),
        nwd_c=lcfg.get('nwd_c', 0.04),
        asl_area_ref=lcfg.get('asl_area_ref', 0.004),
        asl_max=lcfg.get('asl_max', 4.0),
        tiny_area_px=lcfg.get('tiny_area_px', 576.0),
        total_epochs=train_cfg['epochs'],
    )
    if model_name in ('v15', 'v17'):   # v17 dùng cùng contract head + loss v15
        criterion = PFDetLossV15(
            **loss_common,
            stal_gamma=lcfg.get('stal_gamma', 2.0),
            stal_area_ref=lcfg.get('stal_area_ref', 576.0),
            stal_min_area_px=lcfg.get('stal_min_area_px', 64.0),
            stal_k_min=lcfg.get('stal_k_min', 4),
            prog_loss_factor=lcfg.get('prog_loss_factor', 2.0),
        )
    elif model_name == 'v16':
        criterion = PFDetLossV16(
            **loss_common,
            stal_gamma=lcfg.get('stal_gamma', 2.0),
            stal_area_ref=lcfg.get('stal_area_ref', 576.0),
            stal_min_area_px=lcfg.get('stal_min_area_px', 64.0),
            stal_k_min=lcfg.get('stal_k_min', 4),
            prog_loss_factor=lcfg.get('prog_loss_factor', 2.0),
        )
    else:
        criterion = PFDetLossV14(**loss_common)

    # ---- DGS cơ chế 2: crowd-aware localization weighting ----
    # crowd_alpha=0 -> tắt (ablation A/B). Mặc định lấy từ config dgs, fallback 1.0.
    crowd_alpha = float(dgs_cfg.get('crowd_alpha', 0.0 if not use_density else 1.0))
    if hasattr(criterion, 'crowd_alpha'):
        criterion.crowd_alpha = crowd_alpha
        print(f"[DGS] crowd-loc weighting alpha={crowd_alpha} "
              f"({'BẬT' if crowd_alpha > 0 else 'TẮT'})")
    elif crowd_alpha > 0:
        print(f"[DGS] ⚠ criterion {type(criterion).__name__} không có crowd_alpha — bỏ qua crowd-loc.")

    # ---- DGS cơ chế 1: density head loss params ----
    lambda_density       = float(dgs_cfg.get('lambda_density', 0.5))
    density_scale        = float(dgs_cfg.get('density_scale', 100.0))
    density_sigma_factor = float(dgs_cfg.get('density_sigma_factor', 0.2))
    density_count_weight = float(dgs_cfg.get('density_count_weight', 0.1))

    # ---- Knowledge Distillation (booster, không phải đóng góp) ----
    teacher, kd_loss_fn = None, None
    lambda_kd = float(kd_cfg.get('lambda_kd', 1.0))
    if use_kd:
        teacher_ckpt = kd_cfg.get('teacher_ckpt')
        if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
            raise FileNotFoundError(f"KD bật nhưng teacher_ckpt không hợp lệ: {teacher_ckpt!r}")
        teacher = load_teacher(
            PFDetNanoV15, teacher_ckpt,
            profile=kd_cfg.get('teacher_profile', 'balanced'), device=device,
        )
        kd_loss_fn = DistillationLoss(
            student_neck_c=model.model_config['neck_c'],
            teacher_neck_c=teacher.model_config['neck_c'],
            n_scales=len(model.strides),
            use_feature_kd=kd_feat_on,           # need_aux đảm bảo có neck_feats
            temperature=float(kd_cfg.get('temperature', 2.0)),
            conf_thr=float(kd_cfg.get('conf_thr', 0.1)),
            w_cls=float(kd_cfg.get('w_cls', 1.0)),
            w_box=float(kd_cfg.get('w_box', 1.0)),
            w_feat=float(kd_cfg.get('w_feat', 0.5)),
        ).to(device)
        # adapters CÓ tham số -> phải đưa vào optimizer
        if kd_loss_fn.adapters is not None:
            optimizer.add_param_group({
                'params': list(kd_loss_fn.adapters.parameters()),
                'weight_decay': optimizer_cfg['weight_decay'],
            })
        print(f"[KD] BẬT — teacher={kd_cfg.get('teacher_profile','balanced')} "
              f"({sum(p.numel() for p in teacher.parameters())/1e6:.2f}M) | "
              f"feature_kd={kd_feat_on} | lambda_kd={lambda_kd}")

    # ---- Vibration-consistency (ĐÓNG GÓP CHÍNH): detector kháng rung/nhoè khi bay ----
    vibloss_fn = None
    lambda_vib = float(vib_cfg.get('lambda_vib', 0.5))
    vib_warmup = int(vib_cfg.get('warmup_epochs', 15))
    if use_vib:
        vibloss_fn = VibConsistencyLoss(
            temperature=float(vib_cfg.get('temperature', 2.0)),
            conf_thr=float(vib_cfg.get('conf_thr', 0.1)),
            w_cls=float(vib_cfg.get('w_cls', 1.0)),
            w_box=float(vib_cfg.get('w_box', 1.0)),
            w_feat=float(vib_cfg.get('w_feat', 0.5)),
            use_feature=vib_feat_on,
        ).to(device)
        print(f"[VIB] Vibration-consistency BẬT | feature={vib_feat_on} | "
              f"lambda_vib={lambda_vib} | warmup={vib_warmup} epoch")

    use_amp = train_cfg.get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    ema = ModelEMA(model, decay=train_cfg.get('ema_decay', 0.999))
    grad_accumulate = max(1, int(train_cfg.get('grad_accumulate', 1)))

    epochs = train_cfg['epochs']
    warmup_epochs = train_cfg.get('warmup_epochs', 5)
    lr_max, lr_min = optimizer_cfg['lr'], optimizer_cfg['lr_min']
    
    save_dir = train_cfg.get('save_dir', f"./runs/train_{model_name}")
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_ap = -1.0
    csv_path = os.path.join(save_dir, 'training_log.csv')
    
    # Pretrained weights (finetune: load weights only, reset optimizer/epoch)
    pretrained_path = train_cfg.get('pretrained', None)
    if pretrained_path and not args.resume:
        print(f"\n[INFO] Loading pretrained weights from {pretrained_path}")
        pt_ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
        state_key = 'ema' if 'ema' in pt_ckpt else 'model'
        # strict=False khi model là density wrapper (need_aux): checkpoint base không có density_head keys
        miss, unexp = model.load_state_dict(pt_ckpt[state_key], strict=not need_aux)
        if need_aux and miss:
            print(f"  -> aux head (density wrapper) khởi tạo mới ({len(miss)} keys thiếu — OK)")
        ema = ModelEMA(model, decay=train_cfg.get('ema_decay', 0.999))  # re-init EMA from loaded weights
        print(f"  -> Loaded {state_key.upper()} weights (AP={pt_ckpt.get('ap', '?')})")

    # SWA setup
    swa_cfg = train_cfg.get('swa', {})
    swa_enabled = swa_cfg.get('enabled', False)
    swa_start = swa_cfg.get('start_epoch', epochs)
    swa_lr = swa_cfg.get('lr', 0.0005)
    swa_model = None
    swa_n = 0
    if swa_enabled:
        swa_model = copy.deepcopy(model)
        swa_model.eval()
        print(f"[INFO] SWA enabled: start_epoch={swa_start}, lr={swa_lr}")

    if args.resume:
        print("\n[INFO] Đang Resume quá trình train...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema.ema.load_state_dict(ckpt['ema'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ap = ckpt.get('ap', -1.0)
    elif start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow([
                'epoch', 'lr', 'train_loss', 'obj_loss', 'box_loss', 'iou_loss',
                'n_pos', 'pos_s4', 'pos_s8', 'pos_s16', 'pos_s32',
                'ap', 'recall', 'precision'
            ])

    # Parse multiscale config
    ms_cfg = train_cfg.get('multiscale', {})
    ms_enabled = ms_cfg.get('enabled', False)
    ms_sizes = []
    ms_freq = 10
    if ms_enabled:
        ms_range = ms_cfg.get('range', [0.75, 1.25])
        ms_step  = ms_cfg.get('step', 32)
        ms_freq  = ms_cfg.get('freq', 10)
        lo = round(img_size * ms_range[0] / ms_step) * ms_step
        hi = round(img_size * ms_range[1] / ms_step) * ms_step
        ms_sizes = list(range(lo, hi + ms_step, ms_step))
        print(f"[INFO] Multi-scale training: {ms_sizes} (every {ms_freq} batches)")

    print(f"\nTraining started: {epochs} epochs, img_size={img_size}, batch={batch_size}")

    for epoch in range(start_epoch, epochs):
        model.train()
        # SWA phase: use fixed low LR instead of cosine schedule
        if swa_enabled and epoch >= swa_start:
            lr = swa_lr
            for pg in optimizer.param_groups:
                pg['lr'] = lr * pg.get('lr_ratio', 1.0)
        else:
            lr = cosine_lr(optimizer, epoch, epochs, warmup_epochs, lr_start=lr_max*0.1, lr_max=lr_max, lr_min=lr_min)

        # COCO enable / disable theo epoch schedule
        if extra_set is not None:
            want_coco = extra_enable_epoch <= epoch < extra_disable_epoch
            if want_coco != _coco_active:
                if want_coco:
                    train_set, n_extra = build_mixed_train_set(base_train_set, extra_set, extra_ratio)
                    print(f"  [DATA] Epoch {epoch+1}: BẬT COCO ({n_extra} ảnh, ratio={extra_ratio:.0%})")
                else:
                    train_set = base_train_set
                    print(f"  [DATA] Epoch {epoch+1}: TẮT COCO → chỉ VisDrone ({len(base_train_set)} ảnh)")
                train_loader = build_train_loader(train_set, batch_size, nw)
                _coco_active = want_coco

        if epoch >= max(0, epochs - train_cfg.get('mosaic_off_last', 15)):
            disable_train_aug_mix(train_set)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (lr={lr:.6f})")
        loss_accum = defaultdict(float)
        n_iter = 0

        for batch_idx, (imgs, labels_list, _) in enumerate(pbar):
            imgs = imgs.to(device, non_blocking=True)

            # Multi-scale resize (thay đổi resolution mỗi ms_freq batch)
            if ms_sizes and (batch_idx % ms_freq == 0):
                sz = random.choice(ms_sizes)
                if sz != imgs.shape[-1]:
                    imgs = F.interpolate(imgs, size=(sz, sz), mode='bilinear', align_corners=False)
                criterion.img_size = sz
            else:
                criterion.img_size = img_size

            if batch_idx % grad_accumulate == 0:
                optimizer.zero_grad(set_to_none=True)

            cur_img_size = criterion.img_size
            with torch.amp.autocast('cuda', enabled=use_amp):
                if need_aux:
                    preds, aux = model(imgs, return_aux=True)
                else:
                    preds, aux = model(imgs), None
                loss, loss_dict = criterion(preds, labels_list, epoch=epoch)

                # DGS cơ chế 1: density head loss (regularize feature, vá recall cảnh đông)
                if use_density:
                    boxes_px = _labels_to_pixel_boxes(labels_list, cur_img_size)
                    gt_density = build_density_targets(
                        boxes_px, img_size=cur_img_size, stride=model.density_stride,
                        sigma_factor=density_sigma_factor, density_scale=density_scale,
                        box_format="cxcywh", device=device, dtype=aux['density'].dtype,
                    )
                    d_loss = density_loss(aux['density'], gt_density,
                                          density_scale=density_scale,
                                          count_weight=density_count_weight)
                    loss = loss + lambda_density * d_loss
                    loss_dict['density'] = float(d_loss.detach())

                # KD (booster): teacher balanced -> student
                if use_kd:
                    cap = TeacherFeatCapture(teacher)
                    with torch.no_grad():
                        t_outs = teacher(imgs)
                    t_feats = list(cap.feats)
                    cap.remove()
                    s_feats = aux['neck_feats'] if aux is not None else None
                    kd_val, kd_logs = kd_loss_fn(
                        preds, t_outs,
                        student_feats=s_feats,
                        teacher_feats=t_feats if s_feats is not None else None,
                    )
                    loss = loss + lambda_kd * kd_val
                    loss_dict['kd'] = kd_logs['kd_total']

                # Vibration-consistency: forward ảnh RUNG, ép khớp dự đoán ảnh SẠCH (preds=anchor)
                if use_vib and epoch >= vib_warmup:
                    imgs_vib = corrupt_batch(imgs, severity=None)   # rung ngẫu nhiên sev 1-3
                    if vib_feat_on:
                        v_out, v_aux = model(imgs_vib, return_aux=True)
                        v_feats, c_feats = v_aux['neck_feats'], aux['neck_feats']
                    else:
                        v_out = model(imgs_vib)
                        v_feats, c_feats = None, None
                    v_loss, v_logs = vibloss_fn(v_out, preds, v_feats, c_feats)
                    loss = loss + lambda_vib * v_loss
                    loss_dict['vib'] = v_logs['vib_total']

                loss = loss / grad_accumulate

            if not torch.isfinite(loss): continue

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accumulate == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

            n_iter += 1
            for k, v in loss_dict.items():
                loss_accum[k] += v

            postfix = {
                'loss': f"{loss_accum['total']/n_iter:.4f}",
                'obj':  f"{loss_accum['obj']/n_iter:.4f}",
                'box':  f"{loss_accum['box']/n_iter:.4f}",
                'pos':  f"{loss_dict['n_pos']:.0f}",
            }
            if use_density: postfix['den'] = f"{loss_accum['density']/n_iter:.3f}"
            if use_kd:      postfix['kd']  = f"{loss_accum['kd']/n_iter:.3f}"
            if use_vib and loss_accum.get('vib', 0): postfix['vib'] = f"{loss_accum['vib']/n_iter:.3f}"
            pbar.set_postfix(**postfix)

        # ==========================================
        # SWA: accumulate model weights
        if swa_enabled and swa_model is not None and epoch >= swa_start:
            swa_n += 1
            with torch.no_grad():
                for swa_p, model_p in zip(swa_model.parameters(), model.parameters()):
                    swa_p.data.mul_(1 - 1.0 / swa_n).add_(model_p.data, alpha=1.0 / swa_n)
            if swa_n == 1:
                print(f"  [SWA] Started averaging at epoch {epoch+1}")
            elif (epoch + 1) % 10 == 0:
                print(f"  [SWA] Averaged {swa_n} models")

        # Validation & Visualization (ÉP CỨNG = 5)
        # ==========================================
        ap, recall, precision = 0.0, 0.0, 0.0
        
        val_interval = int(train_cfg.get('val_interval', 5))
        do_val = (epoch + 1) % val_interval == 0 or epoch == epochs - 1

        if do_val:
            print("\n  [VAL] Đang chạy Validation và xuất ảnh...")
            val_model = ema.ema if train_cfg.get('val_use_ema', True) else model
            
            # COCO breakdown (AP@.5:.95, AP75, AP_s, AR_s theo mật độ) cho bảng ablation —
            # luôn chạy ở epoch cuối; giữa chừng theo cờ eval.coco.
            run_coco = bool(eval_cfg.get('coco', False)) or epoch == epochs - 1
            ap, recall, precision = evaluate(
                val_model, val_loader, device, img_size, model.strides,
                conf_thr=eval_cfg.get('conf_thr', 0.01), iou_thr=eval_cfg.get('iou_thr', 0.5),
                min_box_area=eval_cfg.get('min_box_area', 0.0), max_det=int(eval_cfg.get('max_det', 300)),
                coco=run_coco,
            )
            print(f"  [VAL] AP@0.5={ap:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")

            sample_path = save_val_samples(
                val_model, val_set, device, img_size, model.strides, save_dir, epoch,
                conf_thr=train_cfg.get('val_vis_conf', 0.25), decode_fn=decode_fn,
                min_box_area=eval_cfg.get('min_box_area', 0.0), max_det=int(eval_cfg.get('max_det', 300))
            )
            print(f"  [VIS] Đã lưu ảnh Val tại: {sample_path}")

            plot_path = plot_training_curves(csv_path, save_dir, model_name)
            if plot_path: print(f"  [PLOT] Cập nhật biểu đồ tại: {plot_path}")

            ckpt = {
                'model': model.state_dict(), 'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(), 'epoch': epoch, 'ap': ap,
                'cfg': cfg, 'model_version': cfg.get('model', {}).get('version', 'v14'),
            }
            torch.save(ckpt, os.path.join(save_dir, 'last.pt'))
            if ap > best_ap:
                best_ap = ap
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
                print(f"  [SAVE] 🔥 MỚI: Best AP@0.5={best_ap:.4f} (Lưu tại epoch {epoch+1})")

        # Log CSV
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                epoch + 1, f'{lr:.6f}', f"{loss_accum['total']/n_iter:.4f}",
                f"{loss_accum['obj']/n_iter:.4f}", f"{loss_accum['box']/n_iter:.4f}",
                f"{loss_accum.get('iou', 0)/n_iter:.4f}", f"{loss_accum['n_pos']/n_iter:.0f}",
                f"{loss_accum.get('pos_s4', 0)/n_iter:.0f}", f"{loss_accum.get('pos_s8', 0)/n_iter:.0f}",
                f"{loss_accum.get('pos_s16', 0)/n_iter:.0f}", f"{loss_accum.get('pos_s32', 0)/n_iter:.0f}",
                f'{ap:.4f}' if ap > 0 else '', f'{recall:.4f}' if recall > 0 else '', f'{precision:.4f}' if precision > 0 else ''
            ])

    # SWA final: update BN stats and evaluate
    if swa_enabled and swa_model is not None and swa_n > 0:
        print(f"\n[SWA] Finalizing: averaged {swa_n} models. Updating BN statistics...")
        swa_model.train()
        with torch.no_grad():
            for batch_data in train_loader:
                imgs = batch_data[0].to(device, non_blocking=True)
                swa_model(imgs)
        swa_model.eval()

        swa_ap, swa_recall, swa_precision = evaluate(
            swa_model, val_loader, device, img_size, model.strides,
            conf_thr=eval_cfg.get('conf_thr', 0.01), iou_thr=eval_cfg.get('iou_thr', 0.5),
            min_box_area=eval_cfg.get('min_box_area', 0.0), max_det=int(eval_cfg.get('max_det', 300)),
            coco=True,
        )
        print(f"[SWA] AP@0.5={swa_ap:.4f} | Recall={swa_recall:.4f} | Precision={swa_precision:.4f}")
        print(f"[SWA] vs Best EMA: AP {swa_ap:.4f} vs {best_ap:.4f} ({'+' if swa_ap > best_ap else ''}{swa_ap - best_ap:.4f})")

        swa_ckpt = {
            'model': swa_model.state_dict(), 'ema': swa_model.state_dict(),
            'epoch': epochs - 1, 'ap': swa_ap,
            'cfg': cfg, 'model_version': cfg.get('model', {}).get('version', 'v14'),
        }
        torch.save(swa_ckpt, os.path.join(save_dir, 'swa.pt'))
        print(f"[SWA] Saved to {save_dir}/swa.pt")
        if swa_ap > best_ap:
            torch.save(swa_ckpt, os.path.join(save_dir, 'best.pt'))
            best_ap = swa_ap
            print(f"[SWA] New best! Saved as best.pt")

    print(f"\nTraining Complete! Best AP@0.5: {best_ap:.4f}")

if __name__ == "__main__":
    main()
