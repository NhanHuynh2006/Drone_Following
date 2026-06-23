import os

from .pfdet_nano_v14 import (
    DEFAULT_EXPORT_OUTPUT_NAMES,
    MODEL_PROFILES,
    PFDetNanoV14,
    count_params,
)
from .pfdet_nano_v15 import PFDetNanoV15
from .pfdet_nano_v16 import PFDetNanoV16
from .pfdet_nano_v17 import PFDetNanoV17


def build_model(version='v14', **kwargs):
    """Build model by version string."""
    if version in ('v14', '14'):
        return PFDetNanoV14(**kwargs)
    if version in ('v15', '15'):
        return PFDetNanoV15(**kwargs)
    if version in ('v16', '16'):
        return PFDetNanoV16(**kwargs)
    if version in ('v17', '17'):
        return PFDetNanoV17(**kwargs)
    raise ValueError(f"Unknown model version: {version!r}. Supported: 'v14', 'v15', 'v16', 'v17'.")


def normalize_model_version(v):
    v = str(v).lower().strip()
    if not v.startswith('v'):
        v = 'v' + v
    return v


def _load_checkpoint(ckpt_or_path, device='cpu'):
    import torch
    if isinstance(ckpt_or_path, (str, os.PathLike)):
        return torch.load(ckpt_or_path, map_location=device, weights_only=False)
    if isinstance(ckpt_or_path, dict):
        return ckpt_or_path
    raise TypeError(f"Unsupported checkpoint input: {type(ckpt_or_path)!r}")


def build_model_from_checkpoint(ckpt_or_path, device='cpu', use_ema=False):
    """Build PFDet model and load weights from a checkpoint path or dict."""
    ckpt = _load_checkpoint(ckpt_or_path, device=device)
    version = normalize_model_version(ckpt.get('model_version', ckpt.get('cfg', {}).get('model', {}).get('version', 'v14')))
    cfg = ckpt.get('cfg', {})
    model_cfg = dict(cfg.get('model') or {})
    model_cfg.pop('version', None)
    model_cfg.pop('img_size', None)
    model = build_model(version, **model_cfg).to(device)
    state_key = 'ema' if use_ema and 'ema' in ckpt else 'model'
    model.load_state_dict(ckpt[state_key])
    model.eval()
    return model, ckpt, version, model_cfg


__all__ = [
    'DEFAULT_EXPORT_OUTPUT_NAMES',
    'MODEL_PROFILES',
    'PFDetNanoV14',
    'PFDetNanoV15',
    'PFDetNanoV16',
    'PFDetNanoV17',
    'count_params',
    'build_model',
    'normalize_model_version',
    'build_model_from_checkpoint',
]
