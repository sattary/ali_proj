"""
Real-world inference tool.
Loads arbitrary size inputs, pads dynamically, and runs inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

from ..core.config import load_train_config
from ..core.utils import pick_device
from ..model import EMA, build_model


@torch.no_grad()
def run_inference(
    input_path: str,
    checkpoint_path: str,
    device_str: str = "auto",
    config_path: Optional[str] = None,
) -> tuple[np.ndarray, int, int]:
    # 1. Load Data
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading input: {input_path}")
    if input_path_obj.suffix.lower() == ".npy":
        I_raw = np.load(input_path_obj).astype(np.float32)
    else:
        I_raw = iio.imread(str(input_path_obj)).astype(np.float32)
        if I_raw.ndim == 3:
            I_raw = I_raw.mean(axis=-1)  # Grayscale

    # Normalize input
    mean = I_raw.mean()
    std = max(I_raw.std(), 1e-6)
    I_norm = (I_raw - mean) / std

    H, W = I_norm.shape

    # 2. Setup Device & Model
    device = pick_device(device_str)
    cfg = load_train_config(config_path)

    model = build_model(cfg.model)
    ema = EMA(model, decay=0.0)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "ema_state_dict" in ckpt:
        dict_to_load = ckpt["ema_state_dict"]
    elif "model_state_dict" in ckpt:
        dict_to_load = ckpt["model_state_dict"]
    elif "model_ema" in ckpt:
        dict_to_load = ckpt["model_ema"]
    elif "model" in ckpt:
        dict_to_load = ckpt["model"]
    else:
        dict_to_load = ckpt
        
    state_dict = {}
    for k, v in dict_to_load.items():
        if k.startswith("_orig_mod."):
            state_dict[k.replace("_orig_mod.", "", 1)] = v
        else:
            state_dict[k] = v
            
    ema.m.load_state_dict(state_dict)

    ema.m.to(device)
    ema.m.eval()

    # 3. Dynamic Padding
    # UNet has 5 maxpools, so H and W must be multiples of 32
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    I_tensor = torch.from_numpy(I_norm).unsqueeze(0).unsqueeze(0).to(device)
    I_padded = F.pad(I_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")

    # Option B: zero phase hint — must match training hint_mode="zero"
    from ..data.augmentation import build_phi_hint

    hint_padded = build_phi_hint(I_padded, phi_gt=None, hint_mode="zero")
    x_in = torch.cat([I_padded, hint_padded], dim=1)

    # 4. Inference
    print(f"Running inference on {device.type}...")
    with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
        phi_raw, k_off = ema.m(x_in)
        phi_pred_padded = phi_raw + k_off

    phi_pred_padded = phi_pred_padded.cpu().float().squeeze()

    # 5. Strip Padding
    # phi_pred_padded has shape [H+pad_h, W+pad_w]
    slice_h = slice(pad_top, pad_top + H) if H > 0 else slice(None)
    slice_w = slice(pad_left, pad_left + W) if W > 0 else slice(None)
    phi_pred = phi_pred_padded[slice_h, slice_w].numpy()

    return phi_pred, H, W
