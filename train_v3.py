"""
Training script for PFDet-Nano v5.

Features:
  - Multi-positive assignment (3 cells per GT per scale)
  - Extended sigmoid decode (range [-0.5, 1.5])
  - IoU-aware objectness target
  - AMP (mixed precision)
  - Cosine LR with warmup
  - EMA with BatchNorm buffer copy
  - Mosaic + MixUp + affine augmentation
  - Val sample visualization every N epochs
  - Training curves (loss, AP, recall)
"""

import os
import sys
import math
import copy
import argparse
import yaml
import csv
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano, count_params
from datasets import VisDronePersonDataset, collate_fn
from utils import PFDetLoss, decode_predictions_np, nms_numpy, xywh2xyxy


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


# ============================================================
#  LR Scheduler
# ============================================================

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs=5,
              lr_start=1e-4, lr_max=1e-3, lr_min=1e-5):
    if epoch < warmup_epochs:
        lr = lr_start + (lr_max - lr_start) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ============================================================
#  Validation
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, img_size, strides, conf_thr=0.01, iou_thr=0.5):
    model.eval()
    all_scores = []
    all_correct = []
    total_gt = 0
    total_tp = 0
    total_fp = 0

    for imgs, labels_list, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)

        for b in range(imgs.shape[0]):
            gt_labels = labels_list[b]
            gt_person = gt_labels[gt_labels[:, 0] == 0] if len(gt_labels) > 0 else gt_labels
            gt_boxes_xywh = gt_person[:, 1:5].numpy() if len(gt_person) > 0 else np.zeros((0, 4))
            n_gt = len(gt_boxes_xywh)
            total_gt += n_gt

            dets = []
            for si, pred in enumerate(preds):
                raw = pred[b].cpu().numpy()
                scale_dets = decode_predictions_np(raw, strides[si], img_size)
                dets.extend(scale_dets)

            dets = [d for d in dets if d['score'] >= conf_thr]
            dets = nms_numpy(dets, iou_threshold=0.5)

            if n_gt == 0:
                for d in dets:
                    all_scores.append(d['score'])
                    all_correct.append(0)
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
                    all_scores.append(dets[di]['score'])
                    all_correct.append(1)
                    matched_gt.add(best_gi)
                    total_tp += 1
                else:
                    all_scores.append(dets[di]['score'])
                    all_correct.append(0)
                    total_fp += 1

    if len(all_scores) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0

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

    return ap, recall, precision


def _iou_single(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)


# ============================================================
#  Val Sample Visualization
# ============================================================

@torch.no_grad()
def save_val_samples(model, val_set, device, img_size, strides, save_dir, epoch,
                     n_samples=8, conf_thr=0.3):
    """Save detection results on random val images."""
    model.eval()
    out_dir = os.path.join(save_dir, 'val_samples')
    os.makedirs(out_dir, exist_ok=True)

    indices = random.sample(range(len(val_set)), min(n_samples, len(val_set)))

    grid_imgs = []
    for idx in indices:
        img_tensor, labels_tensor, img_path = val_set[idx]
        inp = img_tensor.unsqueeze(0).to(device)
        preds = model(inp)

        dets = []
        for si, pred in enumerate(preds):
            raw = pred[0].cpu().numpy()
            scale_dets = decode_predictions_np(raw, strides[si], img_size)
            dets.extend(scale_dets)
        dets = [d for d in dets if d['score'] >= conf_thr]
        dets = nms_numpy(dets, iou_threshold=0.5)

        img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Draw GT boxes (blue)
        if len(labels_tensor) > 0:
            for lb in labels_tensor:
                cx, cy, w, h = lb[1].item(), lb[2].item(), lb[3].item(), lb[4].item()
                x1 = int((cx - w/2) * img_size)
                y1 = int((cy - h/2) * img_size)
                x2 = int((cx + w/2) * img_size)
                y2 = int((cy + h/2) * img_size)
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (255, 100, 0), 1)

        # Draw predictions (green)
        for det in dets:
            x1, y1, x2, y2 = det['box']
            x1, y1 = int(x1 * img_size), int(y1 * img_size)
            x2, y2 = int(x2 * img_size), int(y2 * img_size)
            score = det['score']
            color = (0, 255, 0) if score > 0.5 else (0, 200, 0)
            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_bgr, f'{score:.2f}', (x1, y1-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        cv2.putText(img_bgr, f'Epoch {epoch+1}', (5, 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        grid_imgs.append(img_bgr)

    # Arrange in 2-row grid
    n_cols = 4
    n_rows = (len(grid_imgs) + n_cols - 1) // n_cols
    cell_h, cell_w = img_size, img_size

    grid = np.full((n_rows * cell_h, n_cols * cell_w, 3), 114, dtype=np.uint8)
    for i, img in enumerate(grid_imgs):
        r = i // n_cols
        c = i % n_cols
        h, w = img.shape[:2]
        grid[r*cell_h:r*cell_h+h, c*cell_w:c*cell_w+w] = img

    save_path = os.path.join(out_dir, f'epoch_{epoch+1:03d}.jpg')
    cv2.imwrite(save_path, grid, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return save_path


# ============================================================
#  Training Curves Plot
# ============================================================

def plot_training_curves(csv_path, save_dir):
    """Plot training curves from CSV log."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARN] matplotlib not installed, skip plotting")
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
    fig.suptitle('PFDet-Nano v5 Training', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    ax.plot(epochs, train_loss, 'b-', linewidth=1.5, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, obj_loss, 'r-', linewidth=1.5, label='Obj Loss')
    ax.plot(epochs, box_loss, 'g-', linewidth=1.5, label='Box Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Obj & Box Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    if val_epochs:
        ax.plot(val_epochs, ap_list, 'b-o', linewidth=2, markersize=4, label='AP@0.5')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AP@0.5')
        ax.set_title(f'AP@0.5 (best: {max(ap_list):.4f})')
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax = axes[1, 1]
    if val_epochs:
        ax.plot(val_epochs, recall_list, 'g-o', linewidth=2, markersize=4, label='Recall')
        ax.plot(val_epochs, precision_list, 'r-o', linewidth=2, markersize=4, label='Precision')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Recall & Precision')
        ax.grid(True, alpha=0.3)
        ax.legend()

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
    parser.add_argument("--config", default="configs/train_config.yaml")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    img_size = cfg['model']['img_size']
    base_c = cfg['model']['base_c']
    model = PFDetNano(base_c=base_c).to(device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    total_params, _ = count_params(model)
    print(f"Model: PFDetNano v5 (base_c={base_c}), {total_params/1e6:.2f}M params")

    # Dataset
    data_root = cfg['data']['root']
    train_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['train_images']),
        os.path.join(data_root, cfg['data']['train_labels']),
        img_size=img_size, augment=True,
        mosaic_prob=cfg['data'].get('mosaic_prob', 0.5),
        mixup_prob=cfg['data'].get('mixup_prob', 0.15),
        cache_ram=False,
    )
    val_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['val_images']),
        os.path.join(data_root, cfg['data']['val_labels']),
        img_size=img_size, augment=False, cache_ram=False,
    )

    nw = min(cfg['train']['num_workers'], 4)
    batch_size = cfg['train']['batch_size']
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
        prefetch_factor=2 if nw > 0 else None,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=max(1, batch_size // 2), shuffle=False,
        num_workers=min(2, nw), pin_memory=True,
        persistent_workers=min(2, nw) > 0,
        prefetch_factor=2 if min(2, nw) > 0 else None,
        collate_fn=collate_fn,
    )

    # Optimizer
    decay_params, no_decay_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith('.bias') or 'bn' in name.lower():
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': cfg['train']['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=cfg['train']['lr'])

    # Loss
    criterion = PFDetLoss(
        img_size=img_size, strides=model.strides,
        obj_weight=cfg['loss']['obj_weight'],
        box_weight=cfg['loss']['box_weight'],
    )

    # AMP
    use_amp = cfg['train'].get('amp', True) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # EMA
    ema = ModelEMA(model, decay=cfg['train'].get('ema_decay', 0.999))

    epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train'].get('warmup_epochs', 5)
    lr_max = cfg['train']['lr']
    lr_min = cfg['train'].get('lr_min', 1e-5)
    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    start_epoch = 0
    best_ap = -1.0
    best_epoch = 0

    # CSV log
    csv_path = os.path.join(save_dir, 'training_log.csv')
    csv_fields = ['epoch', 'lr', 'train_loss', 'obj_loss', 'box_loss',
                  'n_pos', 'ap', 'recall', 'precision']

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if 'ema' in ckpt:
            ema.ema.load_state_dict(ckpt['ema'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_ap = ckpt.get('ap', -1.0)
        print(f"Resumed from epoch {start_epoch}, best AP={best_ap:.4f}")

    # Init CSV (only if starting fresh)
    if start_epoch == 0:
        with open(csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(csv_fields)

    mosaic_off_epoch = max(0, epochs - cfg['train'].get('mosaic_off_last', 15))

    print(f"\n{'='*60}")
    print(f"Training: {epochs} epochs, img_size={img_size}, batch={batch_size}")
    print(f"Train: {len(train_set)} images, Val: {len(val_set)} images")
    print(f"Save dir: {save_dir}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        model.train()
        lr = cosine_lr(optimizer, epoch, epochs, warmup_epochs,
                       lr_start=lr_max * 0.1, lr_max=lr_max, lr_min=lr_min)

        if epoch >= mosaic_off_epoch:
            train_set.mosaic_prob = 0.0
            train_set.mixup_prob = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (lr={lr:.6f})")
        loss_accum = defaultdict(float)
        n_iter = 0

        for imgs, labels_list, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, labels_list)

            if not torch.isfinite(loss):
                print(f'  [WARN] skip non-finite loss')
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)

            n_iter += 1
            for k, v in loss_dict.items():
                loss_accum[k] += v

            pbar.set_postfix(
                loss=f"{loss_accum['total']/n_iter:.4f}",
                obj=f"{loss_accum['obj']/n_iter:.4f}",
                box=f"{loss_accum['box']/n_iter:.4f}",
                pos=f"{loss_dict['n_pos']:.0f}",
            )

        avg_loss = loss_accum['total'] / n_iter
        avg_obj = loss_accum['obj'] / n_iter
        avg_box = loss_accum['box'] / n_iter
        avg_pos = loss_accum['n_pos'] / n_iter

        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f} "
              f"(obj={avg_obj:.4f}, box={avg_box:.4f}, pos={avg_pos:.0f})")

        # Validation
        val_interval = cfg['train'].get('val_interval', 5)
        ap, recall, precision = 0.0, 0.0, 0.0
        do_val = (epoch + 1) % val_interval == 0 or epoch == epochs - 1

        if do_val:
            print("  Validating...")
            val_model = ema.ema if cfg['train'].get('val_use_ema', True) else model
            ap, recall, precision = evaluate(val_model, val_loader, device, img_size, model.strides)
            print(f"  [VAL] AP@0.5={ap:.4f} | Recall={recall:.4f} | Precision={precision:.4f}")

            # Save val sample images
            sample_path = save_val_samples(
                val_model, val_set, device, img_size, model.strides,
                save_dir, epoch, n_samples=8, conf_thr=0.3
            )
            print(f"  [VIS] Val samples saved: {sample_path}")

            # Save checkpoints
            ckpt = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch, 'ap': ap, 'cfg': cfg,
            }
            torch.save(ckpt, os.path.join(save_dir, 'last.pt'))

            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                torch.save(ckpt, os.path.join(save_dir, 'best.pt'))
                print(f"  [SAVE] New best AP={best_ap:.4f} @ epoch {epoch+1}")

        # Log to CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f'{lr:.6f}', f'{avg_loss:.4f}', f'{avg_obj:.4f}',
                f'{avg_box:.4f}', f'{avg_pos:.0f}',
                f'{ap:.4f}' if do_val else '',
                f'{recall:.4f}' if do_val else '',
                f'{precision:.4f}' if do_val else '',
            ])

        # Plot curves every val_interval
        if do_val:
            plot_path = plot_training_curves(csv_path, save_dir)
            if plot_path:
                print(f"  [PLOT] Training curves: {plot_path}")

    print(f"\n{'='*60}")
    print(f"Done! Best AP@0.5={best_ap:.4f} @ epoch {best_epoch+1}")
    print(f"Weights: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
