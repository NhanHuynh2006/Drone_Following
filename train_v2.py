"""
Training script for PFDet-Nano.
Stability-focused update for VisDrone person detection.
"""

import os
import sys
import math
import copy
import argparse
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import PFDetNano, count_params
from datasets import VisDronePersonDataset, collate_fn
from utils import PFDetLoss, decode_predictions_np, nms_numpy, xywh2xyxy


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

    def state_dict(self):
        return self.ema.state_dict()


def cosine_lr(optimizer, epoch, total_epochs, warmup_epochs=5, lr_start=1e-4, lr_max=1e-3, lr_min=1e-5):
    if epoch < warmup_epochs:
        lr = lr_start + (lr_max - lr_start) * epoch / max(1, warmup_epochs)
    else:
        denom = max(1, total_epochs - warmup_epochs)
        progress = (epoch - warmup_epochs) / denom
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def _iou_single(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-7)


@torch.no_grad()
def evaluate(model, loader, device, img_size, strides, conf_thr=0.01, iou_thr=0.5):
    model.eval()
    total_iou = 0.0
    n_matched = 0
    all_scores = []
    all_correct = []
    total_gt = 0

    for imgs, labels_list, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        preds = model(imgs)
        for b in range(imgs.shape[0]):
            gt_labels = labels_list[b]
            gt_person = gt_labels[gt_labels[:, 0] == 0] if len(gt_labels) > 0 else gt_labels
            gt_boxes_xywh = gt_person[:, 1:5].numpy() if len(gt_person) > 0 else np.zeros((0, 4), dtype=np.float32)
            n_gt = len(gt_boxes_xywh)
            total_gt += n_gt

            dets = []
            for si, pred in enumerate(preds):
                raw = pred[b].float().cpu().numpy()
                dets.extend(decode_predictions_np(raw, strides[si], img_size))
            dets = [d for d in dets if d['score'] >= conf_thr]
            dets = nms_numpy(dets, iou_threshold=0.5)

            if n_gt == 0:
                for d in dets:
                    all_scores.append(d['score'])
                    all_correct.append(0)
                continue
            if len(dets) == 0:
                continue

            det_boxes = np.array([d['box'] for d in dets], dtype=np.float32)
            gt_boxes_xyxy = np.array([xywh2xyxy(g) for g in gt_boxes_xywh], dtype=np.float32)
            iou_matrix = np.zeros((len(dets), n_gt), dtype=np.float32)
            for di in range(len(dets)):
                for gi in range(n_gt):
                    iou_matrix[di, gi] = _iou_single(det_boxes[di], gt_boxes_xyxy[gi])

            matched_gt = set()
            for di in range(len(dets)):
                best_gi = int(iou_matrix[di].argmax())
                best_iou = float(iou_matrix[di, best_gi])
                if best_iou >= iou_thr and best_gi not in matched_gt:
                    all_scores.append(dets[di]['score'])
                    all_correct.append(1)
                    matched_gt.add(best_gi)
                    total_iou += best_iou
                    n_matched += 1
                else:
                    all_scores.append(dets[di]['score'])
                    all_correct.append(0)

    if len(all_scores) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0

    indices = np.argsort(-np.array(all_scores))
    correct = np.array(all_correct)[indices]
    tp_cumsum = np.cumsum(correct)
    fp_cumsum = np.cumsum(1 - correct)
    recalls = tp_cumsum / total_gt
    precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))
    mean_iou = total_iou / max(1, n_matched)
    recall = float(tp_cumsum[-1] / total_gt) if total_gt > 0 else 0.0
    return ap, mean_iou, recall


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def make_loader(dataset, batch_size, shuffle, num_workers, pin_memory, drop_last=False):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )
    if num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2
    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser(description='PFDet-Nano Training')
    parser.add_argument('--config', default='configs/train_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['train']['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    img_size = cfg['model']['img_size']
    base_c = cfg['model']['base_c']
    model = PFDetNano(base_c=base_c, num_bifpn=cfg['model'].get('num_bifpn', 2)).to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)

    total_params, _ = count_params(model)
    print(f'Model: PFDetNano (base_c={base_c})')
    print(f'Parameters: {total_params/1e6:.2f}M')

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

    num_workers = cfg['train']['num_workers']
    pin_memory = device.type == 'cuda'
    train_loader = make_loader(train_set, cfg['train']['batch_size'], True, num_workers, pin_memory, drop_last=True)
    val_loader = make_loader(val_set, max(1, cfg['train']['batch_size'] // 2), False, max(1, num_workers // 2), pin_memory, drop_last=False)

    optimizer = build_optimizer(model, cfg['train']['lr'], cfg['train']['weight_decay'])
    criterion = PFDetLoss(
        img_size=img_size,
        strides=model.strides,
        obj_weight=cfg['loss']['obj_weight'],
        box_weight=cfg['loss']['box_weight'],
        foot_weight=cfg['loss']['foot_weight'],
    )

    use_amp = bool(cfg['train'].get('amp', True) and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu', enabled=use_amp)
    ema = ModelEMA(model, decay=0.9999)

    epochs = cfg['train']['epochs']
    warmup_epochs = cfg['train'].get('warmup_epochs', 5)
    lr_max = cfg['train']['lr']
    lr_min = cfg['train'].get('lr_min', 1e-5)
    save_dir = cfg['train']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    best_ap = -1.0
    best_epoch = 0
    mosaic_off_epoch = max(0, epochs - cfg['train'].get('mosaic_off_last', 10))

    print('\n' + '=' * 60)
    print(f'Starting training: {epochs} epochs, img_size={img_size}')
    print(f'Train: {len(train_set)} images, Val: {len(val_set)} images')
    print('=' * 60 + '\n')

    for epoch in range(epochs):
        model.train()
        lr = cosine_lr(optimizer, epoch, epochs, warmup_epochs, lr_start=lr_max * 0.01, lr_max=lr_max, lr_min=lr_min)
        if epoch >= mosaic_off_epoch:
            train_set.mosaic_prob = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} (lr={lr:.6f})')
        loss_accum = defaultdict(float)
        n_iter = 0

        for imgs, labels_list, _ in pbar:
            imgs = imgs.to(device, non_blocking=True)
            if device.type == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                preds = model(imgs)
                loss, loss_dict = criterion(preds, labels_list)

            if not torch.isfinite(loss):
                print(f'[WARN] Non-finite loss encountered. Skipping step. loss={loss.detach().item()}')
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
                loss=f"{loss_accum['total'] / max(1, n_iter):.4f}",
                obj=f"{loss_accum['obj'] / max(1, n_iter):.4f}",
                box=f"{loss_accum['box'] / max(1, n_iter):.4f}",
                pos=f"{loss_dict['n_pos']:.0f}",
            )

        if n_iter == 0:
            print('[ERROR] No optimizer steps were taken in this epoch.')
            continue

        print(
            f"  Epoch {epoch + 1} avg loss: {loss_accum['total'] / n_iter:.4f} "
            f"(obj={loss_accum['obj'] / n_iter:.4f}, "
            f"box={loss_accum['box'] / n_iter:.4f}, "
            f"foot={loss_accum['foot'] / n_iter:.4f})"
        )

        val_interval = cfg['train'].get('val_interval', 5)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            print('  Running validation...')
            ap, mean_iou, recall = evaluate(
                ema.ema, val_loader, device, img_size, model.strides,
                conf_thr=cfg['train'].get('val_conf_thr', 0.01),
                iou_thr=cfg['train'].get('val_iou_thr', 0.5),
            )
            print(f'  [VAL] AP@0.5={ap:.4f} | mIoU={mean_iou:.4f} | Recall={recall:.4f}')

            last_ckpt = {
                'model': model.state_dict(),
                'ema': ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'ap': ap,
                'cfg': cfg,
            }
            torch.save(last_ckpt, os.path.join(save_dir, 'last.pt'))
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                torch.save(last_ckpt, os.path.join(save_dir, 'best.pt'))
                print(f'  [SAVE] New best! AP@0.5={best_ap:.4f} @ epoch {epoch + 1}')

    print('\n' + '=' * 60)
    print(f'Training complete! Best AP@0.5={best_ap:.4f} @ epoch {best_epoch + 1}')
    print(f'Weights saved to: {save_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
