"""
Dataset and data loading for HDF5-sharded interferogram data.
"""

from __future__ import annotations

import glob
import os
import random
from typing import List, Sequence, Tuple
from collections import OrderedDict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ..core.config import DataConfig, TrainConfig


def _seed_worker(worker_id: int) -> None:
    # Derive a per-worker seed from torch's base seed so each worker's
    # random/numpy streams are distinct and reproducible.
    base = torch.initial_seed() % (2**32)
    seed = (base + worker_id) % (2**32)
    random.seed(seed)
    np.random.seed(seed)


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
        augment: bool = True,
    ) -> None:
        self.shard_paths = list(shard_paths)
        self.I_key = I_key
        self.phi_key = phi_key
        self.augment = bool(augment)

        self._shard_sizes: list[int] = []
        self._cumulative: list[int] = [0]
        for path in self.shard_paths:
            with h5py.File(path, "r") as f:
                n = f[self.I_key].shape[0]
            self._shard_sizes.append(n)
            self._cumulative.append(self._cumulative[-1] + n)

        self._total = self._cumulative[-1]
        self._handles: dict[int, h5py.File] = {}
        self._chunk_cache: OrderedDict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._cache_size = 128

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shard_idx, local_idx = self._locate(idx)
        f = self._get_shard_handle(shard_idx)

        block_start = (local_idx // self._cache_size) * self._cache_size
        shard_size = self._shard_sizes[shard_idx]
        block_end = min(block_start + self._cache_size, shard_size)

        cache_key = (shard_idx, block_start)
        if cache_key not in self._chunk_cache:
            if len(self._chunk_cache) >= 4:
                self._chunk_cache.popitem(last=False)
            
            I_block = f[self.I_key][block_start:block_end]
            phi_block = f[self.phi_key][block_start:block_end]
            self._chunk_cache[cache_key] = (I_block, phi_block)
        else:
            self._chunk_cache.move_to_end(cache_key)

        I_block, phi_block = self._chunk_cache[cache_key]
        idx_in_block = local_idx - block_start

        I_np = I_block[idx_in_block]
        phi_np = phi_block[idx_in_block]

        I_raw_t = torch.from_numpy(np.ascontiguousarray(I_np)).float()
        phi_gt_t = torch.from_numpy(np.ascontiguousarray(phi_np)).float()

        # Geometric aug is train-only; val/test must be orientation-stable for metrics.
        if self.augment:
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

    val_count = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    test_count = max(1, int(round(test_frac * n))) if test_frac > 0 else 0

    # Fallback for very small datasets
    if val_count + test_count >= n:
        if n >= 3:
            val_count = max(1, n // 3) if val_frac > 0 else 0
            test_count = max(1, n // 3) if test_frac > 0 else 0
        else:
            return list(shard_paths), [], []

    train_end = n - (val_count + test_count)
    val_end = n - test_count

    train_paths = paths[:train_end]
    val_paths = paths[train_end:val_end]
    test_paths = paths[val_end:]

    return train_paths, val_paths, test_paths


def discover_h5_shards(cfg: DataConfig) -> List[str]:
    """Discover H5 shard files under data_dir matching the given pattern."""
    data_glob = os.path.join(cfg.data_dir, cfg.pattern)
    return sorted(glob.glob(data_glob))


class HDF5BlockSampler(Sampler[int]):
    def __init__(self, dataset: H5ShardDataset, generator=None):
        self.dataset = dataset
        self.generator = generator

    def __iter__(self):
        n = len(self.dataset)
        cache_size = getattr(self.dataset, '_cache_size', 128)
        
        blocks = []
        for i in range(0, n, cache_size):
            blocks.append(range(i, min(i + cache_size, n)))
            
        if self.generator is not None:
            indices_list = torch.randperm(len(blocks), generator=self.generator).tolist()
        else:
            indices_list = list(range(len(blocks)))
            random.shuffle(indices_list)
        
        for idx in indices_list:
            block_indices = list(blocks[idx])
            if self.generator is not None:
                perm = torch.randperm(len(block_indices), generator=self.generator).tolist()
                yield from (block_indices[i] for i in perm)
            else:
                random.shuffle(block_indices)
                yield from block_indices

    def __len__(self) -> int:
        return len(self.dataset)


def build_dataloaders(
    cfg: "TrainConfig",
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | None]:
    """Construct training, validation, and test DataLoaders."""



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
        augment=cfg.data.augment,
    )
    val_ds = (
        H5ShardDataset(
            val_paths,
            I_key=cfg.data.I_key,
            phi_key=cfg.data.phi_key,
            augment=False,
        )
        if val_paths
        else None
    )
    test_ds = (
        H5ShardDataset(
            test_paths,
            I_key=cfg.data.I_key,
            phi_key=cfg.data.phi_key,
            augment=False,
        )
        if test_paths
        else None
    )

    import os
    safe_workers = min(cfg.data.workers, os.cpu_count() or 1)

    g = torch.Generator()
    g.manual_seed(seed)

    use_cuda = device.type == "cuda"
    dl_kwargs = dict(
        batch_size=cfg.optim.batch_size,
        num_workers=safe_workers,
        pin_memory=use_cuda,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    if safe_workers > 0 and getattr(cfg.data, "persistent_workers", False):
        dl_kwargs["persistent_workers"] = True
    if safe_workers > 0:
        # Rationale (perf): each worker prefetches this many batches ahead so
        # lzf-decompressed blocks are ready before the GPU asks, keeping the
        # tiny UNet from stalling on I/O between iterations.
        dl_kwargs["prefetch_factor"] = getattr(cfg.data, "prefetch_factor", 4)

    train_sampler = HDF5BlockSampler(train_ds, generator=g)
    train_loader = DataLoader(train_ds, sampler=train_sampler, **dl_kwargs)

    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=use_cuda,
            worker_init_fn=_seed_worker,
        )
        if val_ds is not None
        else None
    )

    test_loader = (
        DataLoader(
            test_ds,
            batch_size=cfg.optim.batch_size,
            shuffle=False,
            num_workers=safe_workers,
            pin_memory=use_cuda,
            worker_init_fn=_seed_worker,
        )
        if test_ds is not None
        else None
    )
    return train_loader, val_loader, test_loader
