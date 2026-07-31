"""
Synthetic interferogram data generator.

Faithful Python port of ``src/image_generation.m``.  Generates pairs of
interferogram intensity images (interferogram) and ground-truth unwrapped phase
fields (dphi), writing them to chunked HDF5 shards.

Physics:
    Two coherent beams at lambda = 632.8 nm interfere on a 128x128 grid
    spanning [-1mm, 1mm] in both x and y.  Each sample randomizes the
    wavefront curvature and adds low-order aberration via bilinear
    interpolation of a small random vector.  The recorded intensity is
    ``|E1 + E2|^2`` and the supervision target is the unwrapped phase
    difference ``phi1 - phi2`` shifted so its minimum is zero.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import yaml
from scipy.ndimage import zoom
from tqdm.auto import tqdm


# ---------------------------------------------------------------------------
# Physical constants mirroring image_generation.m
# ---------------------------------------------------------------------------
NX, NY = 128, 128
X0_MIN, X0_MAX = -1e-3, 1e-3
Y0_MIN, Y0_MAX = -1e-3, 1e-3
LAMBDA = 632.8e-9  # metres
K = 2.0 * math.pi / LAMBDA
E0 = 1.0
Z2 = 0.5


def _build_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return meshgrid ``(x, y)`` and radial distance ``r2``."""
    x0 = np.linspace(X0_MIN, X0_MAX, NX)
    y0 = np.linspace(Y0_MIN, Y0_MAX, NY)
    # MATLAB meshgrid(x0, y0) == np.meshgrid(x0, y0, indexing='xy')
    x, y = np.meshgrid(x0, y0, indexing="xy")
    r2 = np.sqrt(x**2 + y**2 + Z2**2)
    return x, y, r2


def compute_reference_gradient(
    x: np.ndarray, y: np.ndarray, r2: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical gradient of spherical reference beam (grad_phi2_x, grad_phi2_y)."""
    grad_x = (K * alpha * x) / r2
    grad_y = (K * alpha * y) / r2
    return grad_x.astype(np.float32), grad_y.astype(np.float32)


def generate_sample(
    x: np.ndarray,
    y: np.ndarray,
    r2: np.ndarray,
    rng: np.random.Generator,
    speckle_noise: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate a single (interferogram, dphi, grad_phi2) tuple.

    Returns:
        interferogram: float32 array of shape (128, 128) -- interferogram intensity.
        dphi:          float32 array of shape (128, 128) -- unwrapped phase (min = 0).
        grad_phi2:     (grad_x, grad_y) analytical reference beam gradients.
    """
    # ---- phi1: randomized wavefront ----
    r1 = np.sqrt(
        rng.random() * 80.0
        + x / 50.0 * (rng.random() - 0.5)
        + y / 50.0 * (rng.random() - 0.5)
    )
    phi1 = K * r1

    # ---- low-order aberration via imresize ----
    n_dev = rng.integers(1, 4)  # 1, 2, or 3
    deviation = rng.random(n_dev) * 20.0

    scale_y = NY
    scale_x = NX / n_dev
    dev_2d = zoom(deviation.reshape(1, n_dev), (scale_y, scale_x), order=3)
    phi1 = phi1 + dev_2d

    # ---- phi2: reference beam ----
    alpha = float(rng.random() - 0.5)
    phi2 = K * r2 * alpha
    grad_phi2_x, grad_phi2_y = compute_reference_gradient(x, y, r2, alpha)

    # ---- interference ----
    E1 = E0 * np.exp(1j * phi1)
    E2 = E0 * np.exp(1j * phi2)
    interferogram = np.abs(E1 + E2) ** 2

    # Add optional multiplicative laser speckle
    if speckle_noise > 0.0:
        speckle = rng.gamma(
            shape=1.0 / speckle_noise, scale=speckle_noise, size=interferogram.shape
        )
        interferogram = interferogram * speckle

    # ---- ground truth unwrapped phase ----
    dp = phi1 - phi2
    dphi = dp - dp.min()

    return (
        interferogram.astype(np.float32),
        dphi.astype(np.float32),
        (grad_phi2_x, grad_phi2_y),
    )


def _generate_shard_worker(args: Tuple[int, int, int, Path]) -> int:
    shard_idx, n_in_shard, seed, out_path = args
    rng = np.random.default_rng(seed)
    x, y, r2 = _build_grid()

    samples = [generate_sample(x, y, r2, rng) for _ in range(n_in_shard)]
    I_buf = np.stack([s[0] for s in samples])[:, None]
    phi_buf = np.stack([s[1] for s in samples])[:, None]
    grad_buf = np.stack([np.stack(s[2]) for s in samples])

    shard_name = out_path / f"train_shard_{shard_idx:03d}.h5"
    chunk_n = min(128, n_in_shard)
    with h5py.File(shard_name, "w") as f:
        f.create_dataset(
            "I", data=I_buf, chunks=(chunk_n, 1, NY, NX), compression="lzf"
        )
        f.create_dataset(
            "phi", data=phi_buf, chunks=(chunk_n, 1, NY, NX), compression="lzf"
        )
        f.create_dataset(
            "grad_phi2", data=grad_buf, chunks=(chunk_n, 2, NY, NX), compression="lzf"
        )

    return n_in_shard


def generate_to_h5(
    out_dir: str | Path,
    num_samples: int = 180_000,
    shard_size: int = 1000,
    seed: int = 1337,
) -> None:
    """
    Generate ``num_samples`` interferograms and write HDF5 shards.

    Each shard contains datasets:
        - ``interferogram``:   shape ``(shard_size, 1, 128, 128)``  float32
        - ``phi``: shape ``(shard_size, 1, 128, 128)``  float32

    Args:
        out_dir:     Directory to write shard files into.
        num_samples: Total number of samples to generate.
        shard_size:  Number of samples per HDF5 shard file.
        seed:        RNG seed for reproducibility.
    """
    import multiprocessing as mp

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n_shards = math.ceil(num_samples / shard_size)
    args_list = []
    sample_idx = 0

    base_rng = np.random.default_rng(seed)
    shard_seeds = base_rng.integers(0, 2**31 - 1, size=n_shards)

    for shard_idx in range(n_shards):
        n_in_shard = min(shard_size, num_samples - sample_idx)
        args_list.append((shard_idx, n_in_shard, int(shard_seeds[shard_idx]), out_path))
        sample_idx += n_in_shard

    total_generated = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=max(1, mp.cpu_count() - 1)) as pool:
        for n_in_shard_done in tqdm(
            pool.imap_unordered(_generate_shard_worker, args_list),
            total=n_shards,
            desc="Shards",
            mininterval=2.0,
        ):
            total_generated += n_in_shard_done

    print(f"Generated {total_generated} samples across {n_shards} shards in {out_path}")

    data_config = {
        "num_samples": num_samples,
        "shard_size": shard_size,
        "seed": seed,
        "nx": NX,
        "ny": NY,
        "data_dir": str(out_path),
        "timestamp": datetime.now().isoformat(),
    }
    config_path = out_path / "data_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
    print(f"Saved data config: {config_path}")
