"""
Real-world inference tool for PCLCNModel.
Loads arbitrary size inputs, constructs reference beam gradient prior, and runs inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import imageio.v3 as iio
import numpy as np
import torch

from ..core.config import load_train_config
from ..core.ops import generate_reference_beam_gradients
from ..core.utils import pick_device
from ..model import EMA, build_model


@torch.no_grad()
def run_inference(
    input_path: str,
    checkpoint_path: str,
    device_str: str = "auto",
    config_path: Optional[str] = None,
) -> Tuple[np.ndarray, int, int]:
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading input: {input_path}")
    if input_path_obj.suffix.lower() == ".npy":
        I_raw = np.load(input_path_obj).astype(np.float32)
    else:
        I_raw = iio.imread(str(input_path_obj)).astype(np.float32)
        if I_raw.ndim == 3:
            I_raw = I_raw.mean(axis=-1)

    mean = I_raw.mean()
    std = max(I_raw.std(), 1e-6)
    I_norm = (I_raw - mean) / std

    H, W = I_norm.shape

    device = pick_device(device_str)
    cfg = load_train_config(config_path)

    model = build_model(cfg.model)
    ema = EMA(model, decay=0.0)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    dict_to_load = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt.get("model_ema", ckpt.get("model", ckpt))))

    state_dict = {}
    for k, v in dict_to_load.items():
        state_dict[k.replace("_orig_mod.", "", 1) if k.startswith("_orig_mod.") else k] = v

    ema.m.load_state_dict(state_dict)
    ema.m.to(device)
    ema.m.eval()

    I_tensor = torch.from_numpy(I_norm).unsqueeze(0).unsqueeze(0).to(device)
    grad_phi2_tensor = generate_reference_beam_gradients(
        H, W, z2=0.5, alpha=cfg.data.tilt_angle, wavelength=632.8e-9, dx=15.75e-6, device=device
    ).unsqueeze(0)

    print(f"Running PCLCN inference on {device.type}...")
    with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
        phi_abs, _, _, _, _ = ema.m(I_tensor, grad_phi2_tensor)

    phi_pred = phi_abs.cpu().float().squeeze().numpy()
    return phi_pred, H, W
