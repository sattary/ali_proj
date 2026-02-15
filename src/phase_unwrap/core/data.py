from __future__ import annotations

import glob
import os
import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from phase_unwrap.config import DataConfig, OptimizationConfig

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

try:
    from scipy import io as spio  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    spio = None


def _to_chw(a: np.ndarray) -> np.ndarray:
    """
    Ensure an array has shape [C, H, W].

    Accepts:
      - [H, W]          -> [1, H, W]
      - [H, W, C_small] -> [C, H, W] (channels-last heuristic)
      - Already [C, H, W].
    """
    a = np.array(a)
    if a.ndim == 2:
        out = a[None, ...]
    elif a.ndim == 3:
        if a.shape[-1] <= 8 and a.shape[0] >= 16 and a.shape[1] >= 16:
            out = np.transpose(a, (2, 0, 1))
        else:
            out = a
    else:
        raise ValueError(f"Unsupported array ndim={a.ndim}")
    return np.ascontiguousarray(out)


class MatPhaseDataset(Dataset):
    """
    Dataset for interferograms and absolute phase from .mat files.

    Returns per sample:
        I_input  [2, H, W]:
            - channel 0: normalized interferogram (per-sample z-score)
            - channel 1: phi_hint (broadcast scalar reference phase)
        phi_gt   [1, H, W]: ground-truth absolute/unwrapped phase (radians)
        I_raw    [1, H, W]: raw interferogram (for optional intensity weighting)
    """

    def __init__(
        self, paths: Sequence[str], I_key: str = "I", phi_key: str = "dphi"
    ) -> None:
        self.paths = list(paths)
        self.I_key = I_key
        self.phi_key = phi_key

    def __len__(self) -> int:
        return len(self.paths)

    def _load_mat(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        # try h5py first
        if h5py is not None:
            try:
                with h5py.File(path, "r") as f:
                    I = np.array(f[self.I_key])
                    phi = np.array(f[self.phi_key])
                    return I, phi
            except Exception:
                pass
        # fallback scipy.io
        if spio is not None:
            d = spio.loadmat(path)
            return np.array(d[self.I_key]), np.array(d[self.phi_key])
        raise RuntimeError("Cannot read .mat file (need h5py or scipy.io)")

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        I_np, phi_np = self._load_mat(p)

        I_np = _to_chw(I_np).astype(np.float32, copy=False)
        phi_np = _to_chw(phi_np).astype(np.float32, copy=False)

        I_raw_t = torch.from_numpy(I_np).float()
        phi_gt_t = torch.from_numpy(phi_np).float()

        # normalize interferogram per-sample
        mean = I_raw_t.mean(dim=(1, 2), keepdim=True)
        std = I_raw_t.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        I_norm_t = (I_raw_t - mean) / std

        # build phi_hint channel from GT at one reference pixel (center)
        _, H, W = phi_gt_t.shape
        cy, cx = H // 2, W // 2
        ref_val = float(phi_gt_t[0, cy, cx].item())
        phi_hint = torch.full_like(phi_gt_t, ref_val)

        # stack I_norm and phi_hint -> [2, H, W]
        I_input = torch.cat([I_norm_t, phi_hint], dim=0)

        return I_input, phi_gt_t, I_raw_t


class AugmentedPhaseDataset(Dataset):
    """
    Wrapper around MatPhaseDataset that applies simple geometric and noise
    augmentations in a phase-aware way.
    """

    def __init__(self, base: MatPhaseDataset, cfg: DataConfig) -> None:
        self.base = base
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.base)

    def _maybe_flip_rotate(
        self, I_input: torch.Tensor, phi_gt: torch.Tensor, I_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # horizontal flip
        if random.random() < self.cfg.flip_prob:
            I_input = torch.flip(I_input, dims=[-1])
            phi_gt = torch.flip(phi_gt, dims=[-1])
            I_raw = torch.flip(I_raw, dims=[-1])
        # vertical flip
        if random.random() < self.cfg.flip_prob:
            I_input = torch.flip(I_input, dims=[-2])
            phi_gt = torch.flip(phi_gt, dims=[-2])
            I_raw = torch.flip(I_raw, dims=[-2])
        # rotate by k * 90 degrees
        if random.random() < self.cfg.rotate90_prob:
            k = random.randint(1, 3)
            for _ in range(k):
                I_input = I_input.transpose(-1, -2).flip(-1)
                phi_gt = phi_gt.transpose(-1, -2).flip(-1)
                I_raw = I_raw.transpose(-1, -2).flip(-1)
        return I_input, phi_gt, I_raw

    def _maybe_add_noise(
        self, I_input: torch.Tensor, I_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.noise_sigma <= 0.0:
            return I_input, I_raw
        sigma = self.cfg.noise_sigma
        # add noise to normalized interferogram channel (index 0)
        noise_I = torch.randn_like(I_input[0:1]) * sigma
        I_input = I_input.clone()
        I_input[0:1] = I_input[0:1] + noise_I
        # mirror the same noise pattern onto raw intensity for consistency
        if I_raw.numel() > 0:
            I_raw = I_raw.clone()
            I_raw = I_raw + noise_I
        return I_input, I_raw

    def __getitem__(self, idx: int):
        I_input, phi_gt, I_raw = self.base[idx]
        I_input, phi_gt, I_raw = self._maybe_flip_rotate(I_input, phi_gt, I_raw)
        I_input, I_raw = self._maybe_add_noise(I_input, I_raw)
        return I_input, phi_gt, I_raw


def smart_split(
    paths: Sequence[str], seed: int = 1337, val_frac: float = 0.1
) -> Tuple[List[str], List[str]]:
    """
    Shuffle filepaths and split into train / val sets.
    """
    n = len(paths)
    if n <= 1:
        return list(paths), []
    rng = random.Random(seed)
    paths = list(paths)
    rng.shuffle(paths)
    val_count = max(1, int(round(val_frac * n)))
    train_paths = paths[:-val_count]
    val_paths = paths[-val_count:]
    return train_paths, val_paths


def discover_mat_files(cfg: DataConfig) -> List[str]:
    """Discover .mat files under data_dir matching the given pattern."""
    data_glob = os.path.join(cfg.data_dir, cfg.pattern)
    paths = sorted(glob.glob(data_glob))
    return paths


def build_dataloaders(
    data_cfg: DataConfig,
    optim_cfg: OptimizationConfig,
    device: torch.device,
    seed: int,
) -> Tuple[DataLoader, DataLoader | None]:
    """
    Construct training and validation DataLoaders from configuration.
    """
    paths = discover_mat_files(data_cfg)
    if not paths:
        raise RuntimeError(f"No files match {data_cfg.data_dir}/{data_cfg.pattern}")

    train_paths, val_paths = smart_split(paths, seed=seed, val_frac=data_cfg.val_frac)

    train_ds = MatPhaseDataset(
        train_paths, I_key=data_cfg.I_key, phi_key=data_cfg.phi_key
    )
    val_ds = (
        MatPhaseDataset(val_paths, I_key=data_cfg.I_key, phi_key=data_cfg.phi_key)
        if len(val_paths) > 0
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

    if data_cfg.augment:
        train_ds = AugmentedPhaseDataset(train_ds, data_cfg)

    train_loader = DataLoader(train_ds, **dl_kwargs)

    if val_ds is None:
        val_loader = None
    else:
        val_loader = DataLoader(
            val_ds,
            batch_size=optim_cfg.batch_size,
            shuffle=False,
            num_workers=data_cfg.workers,
            pin_memory=use_cuda,
        )

    return train_loader, val_loader
