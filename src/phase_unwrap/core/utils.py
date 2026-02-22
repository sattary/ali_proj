from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | os.PathLike) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 1337) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_str: str) -> torch.device:
    """
    Resolve a device string into a torch.device.

    Accepts:
        - "auto": choose CUDA if available, else CPU.
        - "cuda" / "cpu" / explicit device strings understood by torch.device.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)
