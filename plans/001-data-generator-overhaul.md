# Plan 001: Data Generator Overhaul (Phase Diversity)

## Context & Motivation
The current MATLAB generator (`scripts/image_generation.m`) produces a "tiny latent family" comprising a fixed spherical curvature, a random constant offset, and low-frequency scaled random noise. A model trained on this dataset evaluates perfectly in-distribution but fails entirely on generalized phase surfaces because it learns the 7-parameter generative prior rather than the physics of phase unwrapping.

To prove that our new Phase Disambiguation Network actually works, we need a dataset with genuine structural diversity.

## Steps

1. **Migrate to a Python Generator**
   - Translate the MATLAB script to Python using NumPy/SciPy so we can integrate it seamlessly into PyTorch `Dataset` structures or a data generation pipeline without requiring a MATLAB license.
   - Command to verify: `python scripts/generate_dataset.py --n-samples 10`

2. **Introduce Zernike Polynomials & 2D Freeform Surfaces**
   - Replace the simplistic `imresize(rand(1,3))` deviation with a mixture of Zernike polynomials (up to 4th or 5th radial order).
   - Add localized phase defects (e.g., small Gaussian bumps or pits) to simulate realistic optical manufacturing defects or biological cells.

3. **Establish 4 Rigorous Data Splits**
   - **Train (IID)**: Standard random mixture of Zernike coefficients and defects.
   - **Val (IID)**: Same distribution as Train.
   - **Test 1 (Extrapolation)**: Disjoint parameter ranges (e.g., Zernike coefficients strictly higher amplitude than seen in training).
   - **Test 2 (OOD / Structural)**: Completely unseen phase families (e.g., Perlin noise or phase steps/discontinuities) to test out-of-distribution robustness.

4. **Preserve Ground Truth Sign and Wrap Fields**
   - The generator must explicitly export the ground truth `sign_field` $S(x,y) \in \{-1, +1\}$ and `wrap_field` $K(x,y) \in \mathbb{Z}$ for every pixel, since these will be the targets for the new architecture.

## Verification
- Output dataset must contain `I` (intensity), `phi_gt` (ground truth phase), `sign_gt`, and `wrap_gt`.
- Ensure $I(x,y) = 2 + 2\cos(\phi_{gt}(x,y))$ holds exactly (within floating point precision).
- Visually inspect samples from Test 2 (OOD) to confirm they look structurally distinct from the training set.

## Maintenance Notes
- Keep the random seed fixed during generation to ensure reproducibility of the 4 splits.
- If we later switch to off-axis interferometry, the carrier frequency must be explicitly parameterized here.
