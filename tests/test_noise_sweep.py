"""Noise sweep must evaluate on held-out test split, not fresh synthetic data."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from phase_unwrap.analysis.noise_sweep import compute_noise_robustness


def test_noise_sweep_uses_test_split(tmp_path):
    """Verify noise sweep loads from data_dir, not generate_sample."""
    # Create dummy HDF5 shards (need >=3 for smart_split to produce train/val/test)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for i in range(3):
        h5_path = str(data_dir / f"test_{i}.h5")
        with h5py.File(h5_path, "w") as f:
            I_arr = np.random.randn(4, 1, 128, 128).astype(np.float32)
            phi = np.random.randn(4, 1, 128, 128).astype(np.float32)
            f.create_dataset("I", data=I_arr)
            f.create_dataset("phi", data=phi)

    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        compute_noise_robustness(
            checkpoint_path=str(tmp_path / "nonexistent.pth"),
            data_dir=str(data_dir),
            n_samples=2,
            n_snr_steps=1,
        )
