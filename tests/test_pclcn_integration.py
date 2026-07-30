"""
End-to-end integration tests for Physics-Constrained Latent Corrector Network (PCLCN).
Verifies model construction, 1-epoch training loop, and checkpointing on CPU.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from phase_unwrap.core.config import TrainConfig
from phase_unwrap.model import build_model
from phase_unwrap.model.unet import PCLCNModel
from phase_unwrap.training.train import train


def test_pclcn_build_model():
    """Verify build_model returns PCLCNModel when arch='pclcn'."""
    cfg = TrainConfig()
    cfg.model.arch = "pclcn"
    cfg.model.base = 8

    model = build_model(cfg.model)
    assert isinstance(model, PCLCNModel)


def test_pclcn_train_epoch(tmp_path):
    """Verify 1-epoch training loop with PCLCNModel on synthetic 3-tuple HDF5 shards."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create 3 dummy HDF5 shards containing (I, phi, grad_phi2)
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I = np.random.randn(2, 1, 128, 128).astype(np.float32)
            phi = np.random.randn(2, 1, 128, 128).astype(np.float32)
            grad_phi2 = np.random.randn(2, 2, 128, 128).astype(np.float32)
            f.create_dataset("I", data=I)
            f.create_dataset("phi", data=phi)
            f.create_dataset("grad_phi2", data=grad_phi2)

    run_dir = tmp_path / "run"

    cfg = TrainConfig()
    cfg.model.arch = "pclcn"
    cfg.model.base = 8
    cfg.model.device = "cpu"
    cfg.model.use_amp = False

    cfg.data.data_dir = str(data_dir)
    cfg.data.pattern = "*.h5"
    cfg.data.workers = 0
    cfg.data.augment = False
    cfg.data.val_frac = 0.25
    cfg.data.test_frac = 0.25

    cfg.optim.epochs = 1
    cfg.optim.batch_size = 2
    cfg.optim.compile = False

    cfg.logging.runs_root = str(tmp_path)
    cfg.logging.run_name = "run"
    cfg.logging.val_interval = 1

    # Run training for 1 epoch
    train(cfg)

    # Assert expected artifacts are produced
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "final.pth").exists()
    assert (run_dir / "test_metrics.csv").exists()
