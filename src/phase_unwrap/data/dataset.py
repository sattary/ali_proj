"""
Dataset and data loading for HDF5-sharded interferogram data.
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.config import DataConfig, TrainConfig


class H5ShardDataset(Dataset):
    """
    Dataset backed by multiple HDF5 shard files.

    Returns per sample:
        I_raw_t  [1, H, W]: raw clean interferogram
        phi_gt_t [1, H, W]: ground-truth phase
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        I_key: str = "I",
        phi_key: str = "phi",
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.I_key = I_key
        self.phi_key = phi_key

        self._shard_sizes: list[int] = []
        self._cumulative: list[int] = [0]
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                n = f[self.I_key].shape[0]
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)

        self._total = self._cumulative[-1]
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return self._total

    def _get_shard_handle(self, shard_idx: int) -> h5py.File:
        if shard_idx not in self._handles:
            self._handles[shard_idx] = h5py.File(self.shard_paths[shard_idx], "r", swmr=True)
        return self._handles[shard_idx]

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self._shard_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo, idx - self._cumulative[lo]

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_idx, local_idx = self._locate(idx)
        f = self._get_shard_handle(shard_idx)

        I_np = f[self.I_key][local_idx]
        phi_np = f[self.phi_key][local_idx]

        I_raw_t = torch.from_numpy(np.ascontiguousarray(I_np)).float()
        phi_gt_t = torch.from_numpy(np.ascontiguousarray(phi_np)).float()

        # Phase topology geometric augmentation
        if random.random() > 0.5:
            I_raw_t = I_raw_t.flip(-1)
            phi_gt_t = phi_gt_t.flip(-1)
        if random.random() > 0.5:
            I_raw_t = I_raw_t.flip(-2)
            phi_gt_t = phi_gt_t.flip(-2)
        k = random.randint(0, 3)
        if k > 0:
            I_raw_t = torch.rot90(I_raw_t, k, dims=(-2, -1))
            phi_gt_t = torch.rot90(phi_gt_t, k, dims=(-2, -1))

        return I_raw_t, phi_gt_t

    def close(self) -> None:
        """Close all open HDF5 file handles."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __enter__(self) -> "H5ShardDataset":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures all handles are closed."""
        self.close()


def smart_split(
    shard_paths: Sequence[str],
    seed: int = 1337,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """Shuffle shard paths and split into train/val/test at shard level."""
    n = len(shard_paths)
    if n <= 1:
        return list(shard_paths), [], []
    rng = random.Random(seed)
    paths = list(shard_paths)
    rng.shuffle(paths)

    val_count = max(1, int(round(val_frac * n)))
    test_count = max(1, int(round(test_frac * n)))

    # Fallback for very small datasets
    if val_count + test_count >= n:
        if n >= 3:
            val_count = max(1, n // 3)
            test_count = max(1, n // 3)
        else:
            return list(shard_paths), [], []

    train_paths = paths[: -(val_count + test_count)]
    val_paths = paths[-(val_count + test_count) : -test_count]
    test_paths = paths[-test_count:]

    return train_paths, val_paths, test_paths


def discover_h5_shards(cfg: DataConfig) -> List[str]:
    """Discover H5 shard files under data_dir matching the given pattern."""
    data_glob = os.path.join(cfg.data_dir, cfg.pattern)
    return sorted(glob.glob(data_glob))


def build_dataloaders(
    cfg: "TrainConfig",
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Construct training, validation, and test DataLoaders."""

    # Python 3.12+ explicitly deprecates fork() in multithreaded (PyTorch) environments.
    # Force 'spawn' to prevent hard deadlocks on process boundary.
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    paths = discover_h5_shards(cfg.data)
    if not paths:
        raise RuntimeError(f"No files match {cfg.data.data_dir}/{cfg.data.pattern}")

    # Use the 3-way split
    train_paths, val_paths, test_paths = smart_split(
        paths, seed=seed, val_frac=cfg.data.val_frac, test_frac=cfg.data.test_frac
    )

    train_ds = H5ShardDataset(
        train_paths,
        I_key=cfg.data.I_key,
        phi_key=cfg.data.phi_key,
    )
    val_ds = (
        H5ShardDataset(
            val_paths, I_key=cfg.data.I_key, phi_key=cfg.data.phi_key
        )
        if val_paths
        else None
    )
    test_ds = (
        H5ShardDataset(
            test_paths, I_key=cfg.data.I_key, phi_key=cfg.data.phi_key
        )
        if test_paths
        else None
    )

    use_cuda = device.type == "cuda"
    dl_kwargs = dict(
        batch_size=cfg.optim.batch_size,
        shuffle=True,
        num_workers=cfg.data.workers,
        pin_memory=use_cuda,
        drop_last=True,
    )
    if cfg.data.workers > 0:
        dl_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, **dl_kwargs)

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=False,
            num_workers=cfg.data.workers,
            pin_memory=use_cuda,
        )
        if val_ds is not None
        else None
    )

    test_loader = (
        DataLoader(
            test_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=False,
            num_workers=cfg.data.workers,
            pin_memory=use_cuda,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader
