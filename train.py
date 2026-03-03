"""
Training script for PFDet-Nano.

Key improvements over original train.py:
  1. Cosine annealing learning rate schedule with warmup
  2. Exponential Moving Average (EMA) for stable weights
  3. Proper mAP evaluation (not just best-box IoU)
  4. Mixed precision (AMP) training
  5. Gradient clipping for stability
  6. Multi-positive SimOTA label assignment (via loss function)
  7. Strong augmentation (mosaic + color jitter + flip)
"""

import os
import sys
import time
import math
import copy
import argparse
import yaml
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano, count_params
from datasets import VisDronePersonDataset, collate_fn
from utils import PFDetLoss, decode_predictions_np, nms_numpy, xywh2xyxy, box_iou


# ============================================================
#  EMA (Exponential Moving Average)
# ============================================================

class ModelEMA:
    """
    Model Exponential Moving Average.
    Maintains shadow weights = decay * shadow + (1-decay) * model.
    Results in smoother, more stable model weights for inference.
    """
    def __init__(self, model, decay=0.9999):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        # Ramp up decay (start low, increase to target)
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(d).add_(model_p.data, alpha=1 - d)

    def state_dict(self):
        return self.ema.state_dict()


# ============================================================
#  Learning Rate Scheduler: Cosine with Warmup
# ============================================================

def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs=5,
              lr_start=1e-4, lr_max=1e-3, lr_min=1e-5):
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        lr = lr_start + (lr_max - lr_start) * epoch / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))

    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


# ============================================================
#  Validation: Compute mAP@0.5
# ============================================================

@torch.no_grad()
def evaluate(model, loader, device, img_size, strides, conf_thr=0.25, iou_thr=0.5):
    """
    Evaluate model using AP@0.5.
    Returns: AP@0.5, mean IoU of matched detections, recall
    """
    model.eval()
    all_tp = 0
    all_fp = 0
    all_fn = 0
    total_iou = 0.0
    n_matched = 0

    # Collect all predictions and GTs for AP calculation
    all_scores = []
    all_correct = []
    total_gt = 0

    for imgs, labels_list, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)

        for b in range(imgs.shape[0]):
            # Gather GT boxes for this image (person only)
            gt_labels = labels_list[b]
            gt_person = gt_labels[gt_labels[:, 0] == 0] if len(gt_labels) > 0 else gt_labels
            gt_boxes_xywh = gt_person[:, 1:5].numpy() if len(gt_person) > 0 else np.zeros((0, 4))
            n_gt = len(gt_boxes_xywh)
            total_gt += n_gt

            # Decode predictions from all scales
            dets = []
            for si, pred in enumerate(preds):
                raw = pred[b].cpu().numpy()
                scale_dets = decode_predictions_np(raw, strides[si], img_size)
                dets.extend(scale_dets)

            # Filter by confidence
            dets = [d for d in dets if d['score'] >= conf_thr]

            # NMS
            dets = nms_numpy(dets, iou_threshold=0.5)

            if n_gt == 0:
                for d in dets:
                    all_scores.append(d['score'])
                    all_correct.append(0)
                continue

            if len(dets) == 0:
                all_fn += n_gt
                continue

            # Match detections to GTs
            det_boxes = np.array([d['box'] for d in dets])  # xyxy
            gt_boxes_xyxy = np.array([xywh2xyxy(g) for g in gt_boxes_xywh])

            # IoU matrix
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
                    total_iou += best_iou
                    n_matched += 1
                else:
                    all_scores.append(dets[di]['score'])
                    all_correct.append(0)

    # Compute AP
    if len(all_scores) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0

    # Sort by score descending
    indices = np.argsort(-np.array(all_scores))
    correct = np.array(all_correct)[indices]

    # Cumulative TP and FP
    tp_cumsum = np.cumsum(correct)
    fp_cumsum = np.cumsum(1 - correct)

    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

    # AP (all-point interpolation)
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([1.], precisions, [0.]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    mean_iou = total_iou / max(1, n_matched)
    recall = tp_cumsum[-1] / total_gt if total_gt > 0 else 0.0

    return ap, mean_iou, recall


def _iou_single(box1, box2):
    """IoU between two xyxy boxes (numpy)."""
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
#  Main Training Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PFDet-Nano Training")
    parser.add_argument("--config", default="configs/train_config.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Device
    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Model
    img_size = cfg['model']['img_size']
    base_c = cfg['model']['base_c']
    model = PFDetNano(base_c=base_c, num_bifpn=cfg['model'].get('num_bifpn', 2)).to(device)

    total_params, trainable_params = count_params(model)
    print(f"Model: PFDetNano (base_c={base_c})")
    print(f"Parameters: {total_params/1e6:.2f}M")

    # Dataset
    data_root = cfg['data']['root']
    train_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['train_images']),
        os.path.join(data_root, cfg['data']['train_labels']),
        img_size=img_size,
        augment=True,
        mosaic_prob=cfg['data'].get('mosaic_prob', 0.5),
    )
    val_set = VisDronePersonDataset(
        os.path.join(data_root, cfg['data']['val_images']),
        os.path.join(data_root, cfg['data']['val_labels']),
        img_size=img_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['train']['batch_size'],
        shuffle=True,
        num_workers=cfg['train']['num_workers'],
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, cfg['train']['batch_size'] // 2),
        shuffle=False,
        num_workers=max(1, cfg['train']['num_workers'] // 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['train']['lr'],
        weight_decay=cfg['train']['weight_decay'],
    )

    # Loss
    criterion = PFDetLoss(
        img_size=img_size,
        strides=model.strides,
        obj_weight=cfg['loss']['obj_weight'],
        box_weight=cfg['loss']['box_weight'],
        foot_weight=cfg['loss']['foot_weight'],
    )

    # AMP scaler
    use_amp = cfg['train'].get('amp', True) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # EMA
    ema = ModelEMA(model, decay=0.9999)

    # Training params
    epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train'].get('warmup_epochs', 5)
    lr_max = cfg['train']['lr']
    lr_min = cfg['train'].get('lr_min', 1e-5)
    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    best_ap = -1.0
    best_epoch = 0

    # Disable mosaic in last N epochs (following YOLOX strategy)
    mosaic_off_epoch = max(0, epochs - cfg['train'].get('mosaic_off_last', 10))

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs, img_size={img_size}")
    print(f"Train: {len(train_set)} images, Val: {len(val_set)} images")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        model.train()

        # LR schedule
        lr = cosine_lr(optimizer, epoch, epochs, warmup_epochs,
                       lr_start=lr_max * 0.01, lr_max=lr_max, lr_min=lr_min)

        # Disable mosaic in final epochs
        if epoch >= mosaic_off_epoch:
            train_set.mosaic_prob = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} (lr={lr:.6f})")
        loss_accum = defaultdict(float)
        n_iter = 0

        for imgs, labels_list, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, labels_list)

            scaler.scale(loss).backward()

            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            scaler.step(optimizer)
            scaler.update()

            # EMA update
            ema.update(model)

            # Accumulate losses
            n_iter += 1
            for k, v in loss_dict.items():
                loss_accum[k] += v

            pbar.set_postfix(
                loss=f"{loss_accum['total']/n_iter:.4f}",
                obj=f"{loss_accum['obj']/n_iter:.4f}",
                box=f"{loss_accum['box']/n_iter:.4f}",
                pos=f"{loss_dict['n_pos']:.0f}",
            )

        # Print epoch summary
        print(f"  Epoch {epoch+1} avg loss: {loss_accum['total']/n_iter:.4f} "
              f"(obj={loss_accum['obj']/n_iter:.4f}, "
              f"box={loss_accum['box']/n_iter:.4f}, "
              f"foot={loss_accum['foot']/n_iter:.4f})")

        # Validation
        val_interval = cfg['train'].get('val_interval', 5)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            print("  Running validation...")
            ap, mean_iou, recall = evaluate(
                ema.ema, val_loader, device, img_size, model.strides,
                conf_thr=0.25, iou_thr=0.5
            )
            print(f"  [VAL] AP@0.5={ap:.4f} | mIoU={mean_iou:.4f} | Recall={recall:.4f}")

            # Save last
            last_ckpt = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'ap': ap,
                'cfg': cfg,
            }
            torch.save(last_ckpt, os.path.join(save_dir, 'last.pt'))

            # Save best
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                torch.save(last_ckpt, os.path.join(save_dir, 'best.pt'))
                print(f"  [SAVE] New best! AP@0.5={best_ap:.4f} @ epoch {epoch+1}")

    print(f"\n{'='*60}")
    print(f"Training complete! Best AP@0.5={best_ap:.4f} @ epoch {best_epoch+1}")
    print(f"Weights saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
