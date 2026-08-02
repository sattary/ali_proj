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

from ..core import _torch_compat  # noqa: F401
import math
from ..core.config import load_train_config
from ..core.utils import pick_device
from ..data.simulator import MatlabSimulator
from ..model import EMA, build_model


def generate_reference_beam_gradients(
    height: int, width: int, z2: float, alpha: float, wavelength: float, dx: float, device: torch.device
) -> torch.Tensor:
    x = torch.linspace(-dx * width / 2, dx * width / 2, width, dtype=torch.float32, device=device)
    y = torch.linspace(-dx * height / 2, dx * height / 2, height, dtype=torch.float32, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    r2 = torch.sqrt(xx.square() + yy.square() + z2**2)
    k = 2.0 * math.pi / wavelength
    grad_x = k * alpha * xx / r2
    grad_y = k * alpha * yy / r2
    return torch.stack((grad_x, grad_y), dim=0)

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
    checkpoint_cfg = Path(checkpoint_path).parent / "config.yaml"
    cfg = load_train_config(config_path or (str(checkpoint_cfg) if checkpoint_cfg.exists() else None))

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
    ema.m.observation_mode = "intensity"
    ema.m.reference_mode = getattr(cfg.model, "reference_mode", "reference_free")

    I_tensor = torch.from_numpy(I_norm).unsqueeze(0).unsqueeze(0).to(device)
    grad_phi2_tensor = generate_reference_beam_gradients(
        H, W, z2=0.5, alpha=getattr(cfg.data, "tilt_angle", 0.0),
        wavelength=632.8e-9, dx=15.75e-6, device=device
    ).unsqueeze(0)

    print(f"Running PCLCN inference on {device.type}...")
    with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
        phi_abs, _, _, _, _ = ema.m(I_tensor, grad_phi2_tensor)

    phi_pred = phi_abs.cpu().float().squeeze().numpy()
    return phi_pred, H, W


def _load_model(checkpoint_path: str, config_path: Optional[str], device: torch.device):
    checkpoint_cfg = Path(checkpoint_path).parent / "config.yaml"
    cfg = load_train_config(
        config_path
        or (str(checkpoint_cfg) if checkpoint_cfg.exists() else None)
    )
    model = build_model(cfg.model).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = state.get("model_ema", state.get("ema_state_dict", state.get("model", state)))
    state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.observation_mode = "two_frame"
    model.reference_mode = "reference_free"
    model.eval()
    return model, cfg


@torch.no_grad()
def run_otf_two_frame_inference(
    checkpoint_path: str,
    num_samples: int = 10,
    seed: int = 2026,
    device_str: str = "auto",
    config_path: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Generate and predict new two-frame samples without writing HDF5 files."""
    device = pick_device(device_str)
    model, cfg = _load_model(checkpoint_path, config_path, device)
    simulator = MatlabSimulator().to(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    batch = simulator.sample(
        num_samples, generator=generator, first_sample_id=0
    )
    wrapped = torch.atan2(2.0 - batch.I2, batch.I1 - 2.0)
    with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
        prediction, *_ = model(wrapped, None)
    prediction = prediction - prediction.amin(dim=(-2, -1), keepdim=True)
    return {
        "I1": batch.I1[:, 0].cpu().numpy(),
        "I2": batch.I2[:, 0].cpu().numpy(),
        "wrapped": wrapped[:, 0].cpu().numpy(),
        "prediction": prediction[:, 0].float().cpu().numpy(),
        "phi_gt": batch.phi_gt[:, 0].cpu().numpy(),
        "sample_ids": batch.sample_ids.cpu().numpy(),
    }


@torch.no_grad()
def run_two_frame_inference(
    input_i1: str,
    input_i2: str,
    checkpoint_path: str,
    device_str: str = "auto",
    config_path: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Predict a calibrated pair of physical I1/I2 .npy frames."""
    i1 = np.asarray(np.load(input_i1), dtype=np.float32)
    i2 = np.asarray(np.load(input_i2), dtype=np.float32)
    if i1.shape != i2.shape or i1.ndim != 2:
        raise ValueError("I1 and I2 must be matching 2-D .npy arrays")
    device = pick_device(device_str)
    model, cfg = _load_model(checkpoint_path, config_path, device)
    t1 = torch.from_numpy(i1)[None, None].to(device)
    t2 = torch.from_numpy(i2)[None, None].to(device)
    wrapped = torch.atan2(2.0 - t2, t1 - 2.0)
    with torch.autocast(device_type=device.type, enabled=cfg.model.use_amp):
        prediction, *_ = model(wrapped, None)
    prediction = prediction - prediction.amin(dim=(-2, -1), keepdim=True)
    return {
        "I1": i1[None], "I2": i2[None], "wrapped": wrapped[0, 0].cpu().numpy(),
        "prediction": prediction[0, 0].float().cpu().numpy(),
    }
