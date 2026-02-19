"""
Dataset and data loading for HDF5-sharded interferogram data.

Each shard is an HDF5 file containing:
    - ``I``:   shape ``(N, 1, H, W)``  float32
    - ``phi``: shape ``(N, 1, H, W)``  float32

The dataset lazily opens shards and maps a global sample index to the
correct (shard, local_index) pair.
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

from .config import DataConfig, OptimizationConfig


class H5ShardDataset(Dataset):
    """
    Dataset backed by multiple HDF5 shard files.

    Returns per sample:
        I_input  [2, H, W]:
            - channel 0: normalized interferogram (per-sample z-score)
            - channel 1: phi_hint (broadcast scalar reference phase from GT center)
        phi_gt   [1, H, W]: ground-truth absolute/unwrapped phase (radians)
        I_raw    [1, H, W]: raw interferogram (for optional intensity weighting)
    """

    def __init__(
        self,
        shard_paths: Sequence[str],
        I_key: str = "I",
        phi_key: str = "phi",
        augment: bool = False,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.I_key = I_key
        self.phi_key = phi_key
        self.augment = augment

        # Build cumulative index: scan each shard for its sample count
        self._shard_sizes: list[int] = []
        self._cumulative: list[int] = [0]
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                n = f[self.I_key].shape[0]
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)

        self._total = self._cumulative[-1]

        # Lazy file handles (opened per-worker in __getitem__)
        self._handles: dict[int, h5py.File] = {}

    def __len__(self) -> int:
        return self._total

    def _get_shard_handle(self, shard_idx: int) -> h5py.File:
        """Lazy-open an HDF5 file (cached per worker process)."""
        if shard_idx not in self._handles:
            self._handles[shard_idx] = h5py.File(self.shard_paths[shard_idx], "r")
        return self._handles[shard_idx]

    def _locate(self, idx: int) -> Tuple[int, int]:
        """Map global index to (shard_idx, local_idx)."""
        # Binary search over cumulative boundaries
        lo, hi = 0, len(self._shard_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self._cumulative[mid + 1]:
                hi = mid
            else:
                lo = mid + 1
        return lo, idx - self._cumulative[lo]

    def __getitem__(self, idx: int):
        shard_idx, local_idx = self._locate(idx)
        f = self._get_shard_handle(shard_idx)

        I_np = f[self.I_key][local_idx]  # (1, H, W) float32
        phi_np = f[self.phi_key][local_idx]  # (1, H, W) float32

        I_raw_t = torch.from_numpy(np.ascontiguousarray(I_np)).float()
        phi_gt_t = torch.from_numpy(np.ascontiguousarray(phi_np)).float()

        # -- augmentation (applied jointly to preserve correspondence) --
        if self.augment:
            # random horizontal flip
            if random.random() > 0.5:
                I_raw_t = I_raw_t.flip(-1)
                phi_gt_t = phi_gt_t.flip(-1)
            # random vertical flip
            if random.random() > 0.5:
                I_raw_t = I_raw_t.flip(-2)
                phi_gt_t = phi_gt_t.flip(-2)
            # random 90-degree rotations (0, 90, 180, 270)
            k = random.randint(0, 3)
            if k > 0:
                I_raw_t = torch.rot90(I_raw_t, k, dims=(-2, -1))
                phi_gt_t = torch.rot90(phi_gt_t, k, dims=(-2, -1))

        # normalize interferogram per-sample
        mean = I_raw_t.mean(dim=(1, 2), keepdim=True)
        std = I_raw_t.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        I_norm_t = (I_raw_t - mean) / std

        # phi_hint: GT center-pixel value broadcast to full spatial extent
        _, H, W = phi_gt_t.shape
        cy, cx = H // 2, W // 2
        ref_val = float(phi_gt_t[0, cy, cx].item())
        phi_hint = torch.full_like(phi_gt_t, ref_val)

        # stack I_norm and phi_hint -> [2, H, W]
        I_input = torch.cat([I_norm_t, phi_hint], dim=0)

        return I_input, phi_gt_t, I_raw_t


def smart_split(
    shard_paths: Sequence[str],
    seed: int = 1337,
    val_frac: float = 0.1,
) -> Tuple[List[str], List[str]]:
    """
    Shuffle shard file paths and split into train / val sets.

    Splitting at the shard level (not sample level) to avoid reading
    the same shard in both train and val loaders.
    """
    n = len(shard_paths)
    if n <= 1:
        return list(shard_paths), []
    rng = random.Random(seed)
    paths = list(shard_paths)
    rng.shuffle(paths)
    val_count = max(1, int(round(val_frac * n)))
    return paths[:-val_count], paths[-val_count:]


def discover_h5_shards(cfg: DataConfig) -> List[str]:
    """Discover H5 shard files under data_dir matching the given pattern."""
    data_glob = os.path.join(cfg.data_dir, cfg.pattern)
    return sorted(glob.glob(data_glob))


def build_dataloaders(
    data_cfg: DataConfig,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader | None]:
    """Construct training and (optional) validation DataLoaders."""
    paths = discover_h5_shards(data_cfg)
    if not paths:
        raise RuntimeError(f"No files match {data_cfg.data_dir}/{data_cfg.pattern}")

    train_paths, val_paths = smart_split(paths, seed=seed, val_frac=data_cfg.val_frac)

    train_ds = H5ShardDataset(
        train_paths,
        I_key=data_cfg.I_key,
        phi_key=data_cfg.phi_key,
        augment=data_cfg.augment,
    )
    val_ds = (
        H5ShardDataset(
            val_paths, I_key=data_cfg.I_key, phi_key=data_cfg.phi_key, augment=False
        )
        if val_paths
        else None
    )

    use_cuda = device.type == "cuda"

    dl_kwargs = dict(
        batch_size=optim_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.workers,
        pin_memory=use_cuda,
        drop_last=True,
    )
    if data_cfg.workers > 0:
        dl_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, **dl_kwargs)

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=optim_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.workers,
            pin_memory=use_cuda,
        )
        if val_ds is not None
        else None
    )

    return train_loader, val_loader
