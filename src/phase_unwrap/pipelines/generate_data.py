import numpy as np
import scipy.io as spio
import os
import argparse
from tqdm import tqdm
import h5py


def generate_phase_map(N=128, num_blobs=5):
    """
    Generates a smooth base phase map using random Gaussian blobs.
    """
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    phi = np.zeros_like(X)

    for _ in range(num_blobs):
        # Random blob parameters
        x0, y0 = np.random.uniform(-0.8, 0.8, 2)
        sw = np.random.uniform(0.2, 0.5)  # width
        amp = np.random.uniform(-10, 10)  # amplitude (radians)

        blob = amp * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / sw**2)
        phi += blob

    return phi


def add_geometric_shapes(phi, num_shapes=2):
    """
    Adds regular geometric blocks (rects/circles) to the phase map.
    """
    N = phi.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    for _ in range(num_shapes):
        shape_type = np.random.choice(["rect", "circle"])
        x0, y0 = np.random.uniform(-0.6, 0.6, 2)
        step = np.random.uniform(5, 15)

        if shape_type == "rect":
            w, h = np.random.uniform(0.2, 0.4, 2)
            mask = (np.abs(X - x0) < w) & (np.abs(Y - y0) < h)
        else:
            r = np.random.uniform(0.15, 0.3)
            mask = ((X - x0) ** 2 + (Y - y0) ** 2) < r**2

        phi[mask] += step

    return phi


def add_discontinuities(phi, num_cuts=3):
    """
    Adds sharp discontinuities (shears/cliffs) to the phase map.
    """
    N = phi.shape[0]
    phi_cut = phi.copy()

    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    for _ in range(num_cuts):
        # Define a random line: ax + by + c > 0
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(-0.5, 0.5)

        a = np.cos(angle)
        b = np.sin(angle)

        mask = (a * X + b * Y + dist) > 0

        # Add a random step height (often > 2pi to cause wrapping ambiguities)
        step = np.random.uniform(3, 10) * np.sign(np.random.randn())

        phi_cut[mask] += step

    return phi_cut


def generate_interferogram(phi, noise_level=0.1, speckle=True, shadow_prob=0.2):
    """
    Simulates the interferogram with realistic illumination and noise.
    """
    N = phi.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)

    # 1. Non-uniform illumination (Gaussian beam, often off-center)
    xc, yc = np.random.uniform(-0.3, 0.3, 2)
    beam_width = np.random.uniform(1.0, 2.5)
    E0 = np.exp(-((X - xc) ** 2 + (Y - yc) ** 2) / beam_width**2)

    # 2. Add "Shadows" (regions of low reflectance)
    if np.random.rand() < shadow_prob:
        xs, ys = np.random.uniform(-0.7, 0.7, 2)
        sw, sh = np.random.uniform(0.1, 0.3, 2)
        shadow_mask = (np.abs(X - xs) < sw) & (np.abs(Y - ys) < sh)
        E0[shadow_mask] *= 0.1

    # 3. Complex field
    E = E0 * np.exp(1j * phi)

    # 4. Realistic Speckle (Salt & Pepper like intensity variations)
    if speckle:
        # Complex circular gaussian noise added to field
        # High noise level creates strong speckle/dropouts
        noise_field = (np.random.randn(N, N) + 1j * np.random.randn(N, N)) * noise_level
        E = E + E0 * noise_field

    # 5. Intensity
    I = np.abs(E) ** 2

    # 6. Additive sensor noise (thermal)
    I += np.random.randn(N, N) * 0.02

    # Normalize
    I = (I - I.min()) / (I.max() - I.min() + 1e-8)

    return I


import h5py


def generate_dataset(
    out_dir: str = "data/synthetic_v3",
    num_samples: int = 100,
    size: int = 128,
    fmt: str = "h5",
    shard_size: int = 10000,
):
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"Generating {num_samples} samples to {out_dir} (v3 - Stress Test) | format={fmt}..."
    )

    if fmt == "mat":
        # Legacy Mode
        for i in tqdm(range(num_samples)):
            phi_gt, I = _generate_single_sample(size)
            save_path = os.path.join(out_dir, f"sample_{i:04d}.mat")
            spio.savemat(save_path, {"I": I, "dphi": phi_gt})
    else:
        # HDF5 Mode (Sharded)
        num_shards = (num_samples + shard_size - 1) // shard_size
        sample_idx = 0

        for shard_id in range(num_shards):
            start_idx = sample_idx
            end_idx = min(sample_idx + shard_size, num_samples)
            current_shard_size = end_idx - start_idx

            shard_path = os.path.join(out_dir, f"train_shard_{shard_id:03d}.h5")
            print(f"  Writing shard {shard_id + 1}/{num_shards}: {shard_path}")

            # Pre-allocate arrays
            I_batch = np.zeros((current_shard_size, 1, size, size), dtype=np.float32)
            phi_batch = np.zeros((current_shard_size, 1, size, size), dtype=np.float32)

            for local_i in tqdm(range(current_shard_size), leave=False):
                phi_gt, I = _generate_single_sample(size)
                I_batch[local_i, 0] = I
                phi_batch[local_i, 0] = phi_gt

            with h5py.File(shard_path, "w") as f:
                f.create_dataset(
                    "I", data=I_batch, compression="gzip", compression_opts=4
                )
                f.create_dataset(
                    "phi", data=phi_batch, compression="gzip", compression_opts=4
                )

            sample_idx = end_idx

    print("Done.")


def _generate_single_sample(size: int):
    # 1. Base Phase
    phi_gt = generate_phase_map(N=size, num_blobs=np.random.randint(1, 4))

    # 2. Add Geometry (Blocks)
    if np.random.rand() > 0.4:
        phi_gt = add_geometric_shapes(phi_gt, num_shapes=np.random.randint(1, 3))

    # 3. Add Line Discontinuities (Shears)
    if np.random.rand() > 0.5:
        phi_gt = add_discontinuities(phi_gt, num_cuts=np.random.randint(1, 3))

    # 4. Generate Interferogram with noise & shadows
    I = generate_interferogram(
        phi_gt, noise_level=np.random.uniform(0.05, 0.3), shadow_prob=0.3
    )
    return phi_gt, I


if __name__ == "__main__":
    # Minimal CLI for standalone usage (optional, but good for testing)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/synthetic_v3")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--fmt", type=str, default="h5", choices=["mat", "h5"])
    parser.add_argument("--shard_size", type=int, default=10000)
    args = parser.parse_args()
    generate_dataset(
        args.out_dir, args.num_samples, args.size, args.fmt, args.shard_size
    )
