import numpy as np
import scipy.io as spio
import os
import argparse
from tqdm import tqdm


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


def generate_interferogram(phi, noise_level=0.1, speckle=True):
    """
    Simulates the interferogram I = |E_1 + E_2|^2.
    """
    # Amplitude variation (Gaussian beam profile)
    N = phi.shape[0]
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    E0 = np.exp(-(X**2 + Y**2) / 2.0)  # Gauss beam

    # Complex field
    E = E0 * np.exp(1j * phi)

    # Speckle noise (multiplicative complex noise)
    if speckle:
        # Complex circular gaussian noise
        n_real = np.random.randn(N, N)
        n_imag = np.random.randn(N, N)
        noise_field = (n_real + 1j * n_imag) * noise_level
        E = E + E * noise_field  # signal dependent speckle? Or additive to field?
        # Usually speckle is coherent subtraction, let's just add complex noise to field
        # which results in intensity speckle

    # Intensity
    I = np.abs(E) ** 2

    # Additive sensor noise (thermal)
    I += np.random.randn(N, N) * 0.05

    # Normalize
    I = (I - I.min()) / (I.max() - I.min() + 1e-8)

    return I


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/synthetic_v2")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--size", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples to {args.out_dir}...")

    for i in tqdm(range(args.num_samples)):
        # 1. Base Phase
        phi_base = generate_phase_map(N=args.size, num_blobs=np.random.randint(2, 6))

        # 2. Add Discontinuities (The "Novel" part)
        if np.random.rand() > 0.3:  # 70% chance of cuts
            phi_gt = add_discontinuities(phi_base, num_cuts=np.random.randint(1, 4))
        else:
            phi_gt = phi_base

        # 3. Generate Interferogram with noise
        I = generate_interferogram(phi_gt, noise_level=np.random.uniform(0.05, 0.2))

        # Save as .mat for compatibility
        save_path = os.path.join(args.out_dir, f"sample_{i:04d}.mat")
        spio.savemat(save_path, {"I": I, "dphi": phi_gt})

    print("Done.")


if __name__ == "__main__":
    main()
