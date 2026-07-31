"""
Unified inference and checkpoint loading utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..data import build_dataloaders
from ..model import build_model
from . import _torch_compat  # noqa: F401
from .config import TrainConfig, load_train_config
from .utils import pick_device


def load_inference_state(
    checkpoint_path: str,
    data_dir: Optional[str] = None,
    subset: str = "val",
    config_path: Optional[str] = None,
    all_data: bool = False,
) -> Tuple[torch.nn.Module, TrainConfig, DataLoader, torch.device]:
    """
    Loads model weights, parses configuration, and constructs dataloaders for inference.
    """
    run_dir = str(Path(checkpoint_path).parent)
    cfg_path = config_path or str(Path(run_dir) / "config.yaml")
    cfg = load_train_config(cfg_path if Path(cfg_path).exists() else None)
    
    if data_dir is not None and data_dir != '':
        cfg.data.data_dir = data_dir
    cfg.data.augment = False
    
    if all_data:
        cfg.data.val_frac = 0.0
        cfg.data.test_frac = 0.0
        subset = "train"

    device = pick_device(cfg.model.device)
    model = build_model(cfg.model).to(device)
    
    # weights_only=True is used because this is an internal research checkpoint load
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ckpt.get("model_ema", ckpt["model"])
    
    # Strip torch.compile DDP prefixes if present
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()})
    model.eval()

    train_loader, val_loader, test_loader = build_dataloaders(
        cfg, device, seed=cfg.logging.seed
    )

    if subset == "test":
        if test_loader is None or len(test_loader) == 0:
            raise ValueError("Test set requested but test_frac=0 in config.")
        loader = test_loader
    elif subset == "val":
        if val_loader is None or len(val_loader) == 0:
            raise ValueError("Validation set requested but val_frac=0 in config.")
        loader = val_loader
    else:
        loader = train_loader

    return model, cfg, loader, device
