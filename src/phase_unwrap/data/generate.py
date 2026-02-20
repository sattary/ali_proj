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
from tqdm import tqdm


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


def generate_sample(
    x: np.ndarray,
    y: np.ndarray,
    r2: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a single (interferogram, dphi) pair.

    Returns:
        interferogram:    float32 array of shape (128, 128) -- interferogram intensity.
        dphi: float32 array of shape (128, 128) -- unwrapped phase (min = 0).
    """
    # ---- phi1: randomized wavefront ----
    # MATLAB: r1 = sqrt(rand(1)*80 + x/50*(rand(1)-.5) + y/50*(rand(1)-.5))
    r1 = np.sqrt(
        rng.random() * 80.0
        + x / 50.0 * (rng.random() - 0.5)
        + y / 50.0 * (rng.random() - 0.5)
    )
    phi1 = K * r1

    # ---- low-order aberration via imresize ----
    # MATLAB: deviation = rand(1, randi(3)) * 20
    n_dev = rng.integers(1, 4)  # 1, 2, or 3 (matches randi(3))
    deviation = rng.random(n_dev) * 20.0

    # MATLAB imresize(deviation, size(phi1)) uses bicubic interpolation (default).
    # deviation has shape (1, n_dev); resize to (NY, NX).
    scale_y = NY  # from 1 row to NY rows
    scale_x = NX / n_dev  # from n_dev cols to NX cols
    dev_2d = zoom(deviation.reshape(1, n_dev), (scale_y, scale_x), order=3)
    phi1 = phi1 + dev_2d

    # ---- phi2: reference beam ----
    # MATLAB: phi2 = k*r2*(rand(1)-.5)
    phi2 = K * r2 * (rng.random() - 0.5)

    # ---- interference ----
    E1 = E0 * np.exp(1j * phi1)
    E2 = E0 * np.exp(1j * phi2)
    interferogram = np.abs(E1 + E2) ** 2

    # ---- ground truth unwrapped phase ----
    dp = phi1 - phi2
    dphi = dp - dp.min()

    return interferogram.astype(np.float32), dphi.astype(np.float32)


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
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    x, y, r2 = _build_grid()

    n_shards = math.ceil(num_samples / shard_size)

    sample_idx = 0
    for shard_idx in tqdm(range(n_shards), desc="Shards"):
        n_in_shard = min(shard_size, num_samples - sample_idx)

        I_buf = np.empty((n_in_shard, 1, NY, NX), dtype=np.float32)
        phi_buf = np.empty((n_in_shard, 1, NY, NX), dtype=np.float32)

        for local in range(n_in_shard):
            I_sample, dphi_sample = generate_sample(x, y, r2, rng)
            I_buf[local, 0] = I_sample
            phi_buf[local, 0] = dphi_sample
            sample_idx += 1

        shard_name = out_path / f"train_shard_{shard_idx:03d}.h5"
        with h5py.File(shard_name, "w") as f:
            f.create_dataset("I", data=I_buf, compression="gzip", compression_opts=4)
            f.create_dataset(
                "phi", data=phi_buf, compression="gzip", compression_opts=4
            )

    print(f"Generated {sample_idx} samples across {n_shards} shards in {out_path}")

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


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate synthetic interferogram data.")
    p.add_argument("--num-samples", type=int, default=180_000)
    p.add_argument("--shard-size", type=int, default=1000)
    p.add_argument("--out-dir", type=str, default="data/full")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    generate_to_h5(
        out_dir=args.out_dir,
        num_samples=args.num_samples,
        shard_size=args.shard_size,
        seed=args.seed,
    )
