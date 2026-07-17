from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: str | os.PathLike) -> None:
    """Create directory (and parents) if it does not already exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 1337, deterministic: bool = False) -> None:
    """Set Python, NumPy, and PyTorch RNG seeds for reproducibility.

    When ``deterministic`` is True, also enables deterministic cuDNN/cuBLAS
    algorithms. This makes GPU runs reproducible at some throughput cost and
    may raise if an op lacks a deterministic implementation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def pick_device(device_str: str) -> torch.device:
    """
    Resolve a device string into a torch.device.

    Accepts:
        - "auto": choose CUDA if available, else CPU.
        - "cuda" / "cpu" / explicit device strings understood by torch.device.
    """

    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)