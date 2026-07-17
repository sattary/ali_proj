import os
import h5py
import numpy as np
import pytest
import torch
from unittest.mock import patch

from phase_unwrap.core.config import TrainConfig
from phase_unwrap.training.train import train


def test_train_loop(tmp_path, default_config: TrainConfig):
    # Setup dummy HDF5 dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # Create 3 dummy HDF5 shards to satisfy train, val, and test splits
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I = np.random.randn(2, 1, 32, 32).astype(np.float32)
            phi = np.random.randn(2, 1, 32, 32).astype(np.float32)
            f.create_dataset("I", data=I)
            f.create_dataset("phi", data=phi)

    run_dir = tmp_path / "run"
    vis_dir = tmp_path / "vis"

    # Modify default_config
    cfg = default_config
    cfg.data.data_dir = str(data_dir)
    cfg.data.pattern = "*.h5"
    cfg.data.workers = 0
    cfg.data.augment = False
    cfg.data.val_frac = 0.25
    cfg.data.test_frac = 0.25
    
    cfg.logging.runs_root = str(tmp_path)
    cfg.logging.run_name = "run"
    cfg.logging.val_interval = 1
    
    # Run training for 1 epoch
    train(cfg)
    
    # Assert artifacts are created
    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "final.pth").exists()
    assert (run_dir / "test_metrics.csv").exists()


def test_train_oom_recovery(tmp_path, default_config: TrainConfig):
    # Setup dummy HDF5 dataset
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I = np.random.randn(4, 1, 32, 32).astype(np.float32)
            phi = np.random.randn(4, 1, 32, 32).astype(np.float32)
            f.create_dataset("I", data=I)
            f.create_dataset("phi", data=phi)

    run_dir = tmp_path / "run"
    
    cfg = default_config
    cfg.data.data_dir = str(data_dir)
    cfg.data.pattern = "*.h5"
    cfg.data.workers = 0
    cfg.optim.batch_size = 4
    cfg.optim.epochs = 2
    cfg.optim.compile = False
    cfg.logging.runs_root = str(tmp_path)
    cfg.logging.run_name = "run"
    cfg.logging.val_interval = 1
    
    oom_raised = False
    original_backward = torch.Tensor.backward
    
    def mock_backward(self, *args, **kwargs):
        nonlocal oom_raised
        if not oom_raised:
            oom_raised = True
            raise RuntimeError("CUDA out of memory")
        return original_backward(self, *args, **kwargs)

    torch.Tensor.backward = mock_backward
    try:
        train(cfg)
    finally:
        torch.Tensor.backward = original_backward
        
    assert (run_dir / "config_effective.yaml").exists()
    assert (run_dir / "recovery.log").exists()
    rec_log = (run_dir / "recovery.log").read_text()
    assert "old_bs=4 new_bs=2" in rec_log
    
    metrics = (run_dir / "metrics.csv").read_text().splitlines()
    header = metrics[0].split(",")
    idx_epoch = header.index("epoch")
    idx_partial = header.index("partial")
    
    # Check that we have a partial row for epoch 1
    row_1 = metrics[1].split(",")
    assert row_1[idx_epoch] == "1"
    assert row_1[idx_partial] == "1"
    
    # Epoch 2 should be successful and full
    row_2 = metrics[2].split(",")
    assert row_2[idx_epoch] == "2"
    assert row_2[idx_partial] == "0"
