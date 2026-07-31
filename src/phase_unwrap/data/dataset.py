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
from .simulator import MatlabSimulator

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

    def __len__(self) -> int:
        return self._total

    def _get_shard_handle(self, shard_idx: int) -> h5py.File:
        if shard_idx not in self._handles:
            self._handles[shard_idx] = h5py.File(
                self.shard_paths[shard_idx], "r", swmr=True
            )
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shard_idx, local_idx = self._locate(idx)
        f = self._get_shard_handle(shard_idx)

        I_raw_t = torch.from_numpy(
            np.ascontiguousarray(f[self.I_key][local_idx])
        ).float()
        phi_gt_t = torch.from_numpy(
            np.ascontiguousarray(f[self.phi_key][local_idx])
        ).float()

        if "grad_phi2" in f:
            grad_phi2_t = torch.from_numpy(
                np.ascontiguousarray(f["grad_phi2"][local_idx])
            ).float()
        else:
            grad_phi2_t = torch.zeros(
                (2, I_raw_t.shape[-2], I_raw_t.shape[-1]), dtype=torch.float32
            )

        if self.augment:
            if random.random() > 0.5:
                I_raw_t = I_raw_t.flip(-1)
                phi_gt_t = phi_gt_t.flip(-1)
                grad_phi2_t = grad_phi2_t.flip(-1)
                grad_phi2_t[0] *= -1.0  # Flip gx direction
            if random.random() > 0.5:
                I_raw_t = I_raw_t.flip(-2)
                phi_gt_t = phi_gt_t.flip(-2)
                grad_phi2_t = grad_phi2_t.flip(-2)
                grad_phi2_t[1] *= -1.0  # Flip gy direction

        return I_raw_t, phi_gt_t, grad_phi2_t

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
        cache_size = 128

        blocks = []
        for i in range(0, n, cache_size):
            blocks.append(range(i, min(i + cache_size, n)))

        if self.generator is not None:
            indices_list = torch.randperm(
                len(blocks), generator=self.generator
            ).tolist()
        else:
            indices_list = list(range(len(blocks)))
            random.shuffle(indices_list)

        for idx in indices_list:
            block_indices = list(blocks[idx])
            if self.generator is not None:
                perm = torch.randperm(
                    len(block_indices), generator=self.generator
                ).tolist()
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


class OTFLoader:
    """Small DataLoader-compatible iterable for device-resident simulation."""

    def __init__(
        self,
        simulator,
        batch_size,
        batches,
        generator,
        first_sample_id=0,
        fixed=False,
        observation="intensity",
    ):
        self.simulator, self.batch_size, self.batches = simulator, batch_size, batches
        self.observation = observation
        self.generator, self.next_sample_id, self.fixed = (
            generator,
            first_sample_id,
            fixed,
        )
        self._state, self._first_sample_id = generator.get_state(), first_sample_id

    def __len__(self):
        return self.batches

    def __iter__(self):
        if self.fixed:
            self.generator.set_state(self._state)
            self.next_sample_id = self._first_sample_id
        for _ in range(self.batches):
            b = self.simulator.sample(
                self.batch_size,
                generator=self.generator,
                first_sample_id=self.next_sample_id,
            )
            self.next_sample_id += self.batch_size
            x = b.wrapped_dp if self.observation == "wrapped_dp" else b.I_clean
            yield x, b.phi_gt, b.grad_phi2, b.sample_ids


def build_otf_loaders(cfg, device):
    sim = MatlabSimulator().to(device)
    train_g = torch.Generator(device=device)
    train_g.manual_seed(cfg.data.train_seed)
    val_g = torch.Generator(device=device)
    val_g.manual_seed(cfg.data.val_seed)
    test_g = torch.Generator(device=device)
    test_g.manual_seed(cfg.data.test_seed)
    observation = getattr(cfg.data, "observation", "intensity")
    return (
        OTFLoader(
            sim,
            cfg.optim.batch_size,
            cfg.data.steps_per_epoch,
            train_g,
            observation=observation,
        ),
        OTFLoader(
            sim,
            cfg.optim.batch_size,
            cfg.data.val_batches,
            val_g,
            fixed=True,
            observation=observation,
        ),
        OTFLoader(
            sim,
            cfg.optim.batch_size,
            cfg.data.test_batches,
            test_g,
            fixed=True,
            observation=observation,
        ),
    )
