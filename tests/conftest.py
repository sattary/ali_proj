"""
Test fixtures shared across the test suite.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from phase_unwrap.core.config import TrainConfig
from phase_unwrap.data.generate import _build_grid


@pytest.fixture
def default_config() -> TrainConfig:
    """Minimal config for unit tests (small model, CPU)."""
    cfg = TrainConfig()
    cfg.model.base = 8
    cfg.model.device = "cpu"
    cfg.model.use_amp = False
    cfg.optim.epochs = 1
    cfg.optim.batch_size = 2
    cfg.logging.seed = 42
    return cfg


@pytest.fixture
def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthetic batch: (I_input, phi_gt, I_raw), each shape [2, 1, 128, 128]."""
    rng = np.random.default_rng(42)
    x, y, r2 = _build_grid()

    from phase_unwrap.data.generate import generate_sample

    Is, phis = [], []
    for _ in range(2):
        I_s, dphi_s = generate_sample(x, y, r2, rng)
        Is.append(I_s)
        phis.append(dphi_s)

    I_raw = torch.from_numpy(np.stack(Is)).unsqueeze(1)  # [2, 1, 128, 128]
    phi_gt = torch.from_numpy(np.stack(phis)).unsqueeze(1)

    # normalize I for model input
    I_mean = I_raw.mean(dim=(2, 3), keepdim=True)
    I_std = I_raw.std(dim=(2, 3), keepdim=True) + 1e-6
    I_norm = (I_raw - I_mean) / I_std
    phi_hint = torch.angle(torch.exp(1j * I_raw))
    I_input = torch.cat([I_norm, phi_hint], dim=1)  # [2, 2, 128, 128]

    return I_input, phi_gt, I_raw
