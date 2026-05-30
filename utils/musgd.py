"""
MuSGD Optimizer — Hybrid Muon + SGD
=====================================

Implementation faithful to Ultralytics YOLO26 (commit f2d3aed, Jan 2026).
Ref: ultralytics/optim/muon.py

Core design:
  - ALL 2D weight tensors → use_muon=True: blend of Muon (50%) + SGD (50%)
  - 1D tensors (bias, BN) → use_muon=False: plain SGD with momentum
  - NO separate AdamW path for head — the 50-50 blend handles thin matrices naturally
    (for a 1×48 head conv, Muon ≡ normalization → scale=1, SGD provides the update)

Newton-Schulz orthogonalization (5 iterations):
  φ(X) = aX + bAX + cAAX,  a=3.4445, b=-4.7750, c=2.0315
  - Always processes min-dimension matrix (transpose tall → wide, compute, transpose back)
  - Uses bfloat16 internally for stability

Momentum (Muon path):
  - Muon buffer: EMA via lerp_() → momentum.lerp_(grad, 1-beta)
  - Nesterov: update = grad.lerp(momentum, beta)
  - SGD buffer: standard mul_(beta).add_(grad) with optional Nesterov

Scale after NS: max(1, kH/kW)^0.5 using ORIGINAL grad's last 2 dims
  → for square kernels (3×3, 1×1): always 1.0
  → for asymmetric kernels (5×1): sqrt(5)

Hyperparameters (YOLO26 defaults):
  lr: 0.01    (hardcoded in Ultralytics auto optimizer for MuSGD)
  momentum: 0.9
  weight_decay: 0.0005
  muon: 0.2   (Muon fraction — lightweight regularizer)
  sgd:  1.0   (SGD fraction — dominates the update)
  nesterov: True
"""

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz orthogonalization (5 iterations).

    Computes the orthogonal factor of G (approximates UV^T from SVD G=USV^T).
    Always processes the min-dimension side for efficiency:
      - tall matrix (rows > cols): transpose → process wide → transpose back
      - wide/square matrix: process as-is
    """
    assert G.ndim == 2
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if G.size(0) > G.size(1):   # tall → make wide for smaller A = X @ X.T
        X = X.T
    for _ in range(5):
        a, b, c = 3.4445, -4.7750, 2.0315
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1):   # transpose back
        X = X.T
    return X


def muon_update(grad: torch.Tensor, momentum: torch.Tensor,
                beta: float = 0.95, nesterov: bool = True) -> torch.Tensor:
    """
    Compute Muon update for one parameter.

    Steps:
      1. Update momentum buffer via EMA: momentum ← beta*momentum + (1-beta)*grad
      2. Nesterov: update = (1-beta)*grad + beta*momentum  (= new momentum)
      3. Reshape 4D → 2D for NS
      4. NS orthogonalization
      5. Scale by sqrt(max(1, kH/kW)) using original grad's last two dims
    """
    momentum.lerp_(grad, 1 - beta)   # in-place EMA update
    update = grad.lerp(momentum, beta) if nesterov else momentum.clone()

    if update.ndim == 4:
        update = update.view(len(update), -1)   # (out_c, in_c*kH*kW)

    update = zeropower_via_newtonschulz5(update)

    # Scale using kernel dims of original grad (not reshaped dims).
    # For square kernels (1×1, 3×3): scale = 1.0.
    scale = max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5
    update = update * scale

    return update


class MuSGD(Optimizer):
    """
    MuSGD: Hybrid Muon + SGD optimizer from YOLO26 (Ultralytics, Jan 2026).

    Each param group must set 'use_muon':
      use_muon=True  → blend of Muon (muon_frac) + SGD (sgd_frac) updates
      use_muon=False → plain SGD with momentum (for bias, BN params)

    Usage in train config (v15 with MuSGD):
      optimizer:
        name: musgd
        lr: 0.002
        momentum: 0.95
        weight_decay: 0.0005
        lr_min: 0.00002

    Args:
        params:        model parameters
        lr:            learning rate (default: 0.002 — YOLO26 default for nc=1)
        momentum:      momentum beta for both Muon and SGD buffers (default: 0.95)
        weight_decay:  L2 regularization, applied to SGD component only (default: 5e-4)
        nesterov:      use Nesterov momentum (default: True)
        muon_frac:     fraction of update from Muon path (default: 0.5)
        sgd_frac:      fraction of update from SGD path (default: 0.5)
    """
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9,
                 weight_decay: float = 5e-4, nesterov: bool = True,
                 muon_frac: float = 0.2, sgd_frac: float = 1.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        nesterov=nesterov, use_muon=False, lr_ratio=1.0)
        super().__init__(params, defaults)
        self.muon_frac = muon_frac
        self.sgd_frac  = sgd_frac

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr       = group['lr']
            beta     = group['momentum']
            wd       = group['weight_decay']
            nesterov = group['nesterov']

            if group['use_muon']:
                # ---- Muon + SGD blend path (2D weight tensors) ----
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad  = p.grad
                    state = self.state[p]

                    if len(state) == 0:
                        state['momentum_muon'] = torch.zeros_like(p)
                        state['momentum_sgd']  = torch.zeros_like(p)

                    # 1. Muon update (Newton-Schulz orthogonalization)
                    update_muon = muon_update(
                        grad, state['momentum_muon'],
                        beta=beta, nesterov=nesterov,
                    )
                    p.add_(update_muon.reshape(p.shape), alpha=-(lr * self.muon_frac))

                    # 2. Weight decay applied to SGD component only
                    grad_wd = grad.add(p, alpha=wd) if wd != 0.0 else grad

                    # 3. SGD update with momentum
                    buf = state['momentum_sgd']
                    buf.mul_(beta).add_(grad_wd)
                    sgd_update = (grad_wd.add(buf, alpha=beta)
                                  if nesterov else buf)
                    p.add_(sgd_update, alpha=-(lr * self.sgd_frac))

            else:
                # ---- Plain SGD path (bias, BN scale/shift) ----
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad  = p.grad
                    state = self.state[p]

                    if wd != 0.0:
                        grad = grad.add(p, alpha=wd)

                    if len(state) == 0:
                        state['momentum_buffer'] = torch.zeros_like(p)

                    buf = state['momentum_buffer']
                    buf.mul_(beta).add_(grad)
                    update = (grad.add(buf, alpha=beta) if nesterov else buf)
                    p.add_(update, alpha=-lr)

        return loss
