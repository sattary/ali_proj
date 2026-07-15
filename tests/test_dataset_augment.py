"""Val/test geometric augmentation must be off for honest metrics."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from phase_unwrap.data.dataset import H5ShardDataset


def _write_one_sample_shard(path: Path) -> None:
    """Non-symmetric ramp so flips/rots change the tensor."""
    yy, xx = np.mgrid[0:32, 0:32]
    I = (xx + 2 * yy).astype(np.float32)
    phi = (0.1 * xx + 0.3 * yy).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("I", data=I[None, None, ...])
        f.create_dataset("phi", data=phi[None, None, ...])


def test_augment_false_is_deterministic(tmp_path: Path) -> None:
    p = tmp_path / "s0.h5"
    _write_one_sample_shard(p)
    ds = H5ShardDataset([str(p)], augment=False)
    a = ds[0]
    b = ds[0]
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])


def test_augment_true_can_transform(tmp_path: Path) -> None:
    p = tmp_path / "s0.h5"
    _write_one_sample_shard(p)
    ds = H5ShardDataset([str(p)], augment=True)
    base = ds[0][0].clone()
    changed = False
    for _ in range(40):
        if not torch.equal(ds[0][0], base):
            changed = True
            break
    assert changed, "expected geometric aug to alter at least one sample"
