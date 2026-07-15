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

    # Option B: zero hint channel (matches prepare_batch default / lab deploy)
    from phase_unwrap.data.augmentation import prepare_batch

    I_input, phi_gt, _, _ = prepare_batch(
        I_raw, phi_gt, noise_aug=None, hint_mode="zero"
    )

    return I_input, phi_gt, I_raw
