# Project Handoff & Research History: PCLCN Pipeline

> **Branch:** `pclcn-pipeline`  
> **Date:** 2026-07-29  
> **Role:** Senior Principal Researcher  
> **Status:** All 57 Unit Tests Passing (100%) | Code Refactored & Committed  
> **Target Journal:** *Optics Express* (Optica Publishing Group)

---

## 1. EXECUTIVE SUMMARY & HANDOFF CONTEXT

This handoff document details the complete research, mathematical diagnostic, and engineering history of the `ali_proj` phase unwrapping pipeline. 

Over the course of multiple design iterations (Plan 020 $\to$ Plan 021 $\to$ Plan 022), we transformed an initial ad-hoc deep learning model into a scientifically rigorous, optics-native architecture: the **Physics-Constrained Latent Corrector Network (PCLCN)**.

All core modules have been refactored, fully unit-tested, and committed across 4 atomic commits on the `pclcn-pipeline` branch.

---

## 2. EVOLUTION OF PLANS & RESEARCH METHODOLOGY

### Phase 1: Plan 020 (Physics-Informed UNet & Failure Diagnosis)
* **Optical Forward Model Mapping:** Line-by-line extraction of physical constants from the professor's MATLAB script (`scripts/image_generation.m` $\to$ `generate.py`).
  - HeNe Laser ($\lambda = 632.8\text{ nm}$).
  - $128 \times 128$ sensor grid, $2\text{mm} \times 2\text{mm}$ area, pixel pitch $\Delta x \approx 15.75\,\mu\text{m}$ ($f_{\text{Nyquist}} \approx 31.7\text{ lp/mm}$).
  - Diverging spherical reference wave $\phi_2 \propto \sqrt{x^2+y^2+z_2^2}$ ($z_2 = 0.5\text{ m}$) with variable tilt $\alpha$.
  - Object wave $\phi_1$ with spherical curvature + low-order bicubic polynomial aberrations.
* **Leakage Diagnostics:**
  - Identified that $a_{\text{calib}}$ in preliminary plans was a "leaked" calibration metric requiring ground-truth $\phi_{\text{gt}}$, making it physically invalid for deployment.
  - Identified that `gt_center` passed an oracle point-reference hint into the network.
* **Gradient Pathology Proof:** Proved that PINN intensity re-synthesis ($\|\cos(\hat{\phi}) - I_{\text{obs}}\|_1$) yields vanishing gradients ($\frac{\partial}{\partial \hat{\phi}} \cos\hat{\phi} = -\sin\hat{\phi} = 0$) at every fringe peak and valley ($n\pi$).

### Phase 2: Plan 021 (Physics State Space Network - PSSN Concept)
* Proposed moving the neural network *inside* the physics solver rather than after it.
* Concept: Deploy a CNN to correct non-integrable gradient fields $(\delta g_x, \delta g_y)$ *before* a 2D DCT Poisson solver, and predict orthogonal non-Zernike residuals after integration.

### Phase 3: Plan 022 (Physics-Constrained Latent Corrector Network - PCLCN)
* **Red-Team Literature Audit & Refinements:**
  - **Naming Collision:** Renamed PSSN to **PCLCN** to avoid collision with *PI-VMamba* (Sept 2025: State Space Model for wavefront sensing).
  - **TIE Loss Purged:** Proved that single-shot planar intensity data lacks the axial derivative $\partial I / \partial z$, making the Transport of Intensity Equation mathematically invalid. Explicitly removed and documented.
  - **Complex-Domain Loss:** Replaced unstable `atan2` loss (which explodes at zero visibility $\nabla \text{atan2}(y,x) = \frac{1}{x^2+y^2}$) with a smooth, complex-domain $(\sin\hat{\phi}, \cos\hat{\phi})$ loss.
  - **De-Marketing Prior Art:** Positioned `AnalyticSignalStem` as adopting established best practices (FIN network, *Light: Sci. Appl.* 2022) rather than claiming it as a new invention.
  - **Core Novelty Established:** Differentiated our gradient corrector by **conditioning CNN 1 explicitly on the analytically known reference beam gradient $\nabla \phi_2(x,y)$**, injecting known optical curvature as a physical prior.

---

## 3. EMPIRICAL DIAGNOSTICS LOG

We executed `scratch/diagnostics.py` on $1,000$ synthetic samples from the optical forward model:

1. **Carrier Bandwidth Overlap (CRITICAL FINDING):**
   - **Result:** Max object spatial frequency reached up to $0.63\text{ cyc/px}$ (with $1.5 \times f_{\max} = 0.9455\text{ cyc/px} \gg f_0 = 0.125\text{ cyc/px}$).
   - **Implication:** A fixed-width Fourier bandpass filter **clips high-frequency object details** in high-curvature samples.
   - **Fix:** PCLCN retains `I_raw` as an explicit input channel to CNN 1 to preserve high-frequency features.

2. **Modal Representability:**
   - **Result:** Zernike modes 1–15 fit $99.99\%$ of phase variance on average for our simulator.
   - **Fix:** To prevent non-orthogonal boundary leakage across the square $128 \times 128$ grid, we masked the projection inner product strictly to the inscribed unit disk $\mathcal{D} = \{(x,y) \mid x^2+y^2 \le 1\}$.

---

## 4. CODEBASE IMPLEMENTATION SUMMARY

### Core Operators (`src/phase_unwrap/core/ops.py`)
- `AnalyticSignalStem`: Differentiable 2D FFT demodulation.
- `WrappedGradientOperator`: Computes wrapped finite-difference gradients $g_x, g_y \in [-\pi, \pi]$.
- `DifferentiablePoissonSolver`: 2D DCT-II Neumann Poisson integrator operating directly on continuous vector gradient fields $(\tilde{g}_x, \tilde{g}_y)$ via exact matrix transformations (`C_h @ rho @ C_w^T`).
- `MaskedZernikeProjection`: Disk-masked pseudoinverse projection onto 15 Zernike modes.

### Neural Network Models (`src/phase_unwrap/model/unet.py`)
- `ReferenceConditionedGradientCorrector` (CNN 1): 7-channel input (`I_raw`, `wrapped_phase`, `amplitude`, $g_x, g_y$, and $\nabla \phi_2$). Predicts gradient corrections $(\delta g_x, \delta g_y)$ to restore integrability ($\nabla \times \tilde{\mathbf{g}} = 0$).
- `OrthogonalResidualCNN` (CNN 2): Predicts non-Zernike orthogonal residual phase $r(x,y)$.
- `PCLCNModel`: Integrated 6-stage forward model pipeline.

### Loss Topology (`src/phase_unwrap/core/losses.py`)
- `ComplexDomainLoss`: $\mathcal{L}_{\text{phase}} = \|\sin\hat{\phi} - \sin\phi_{\text{gt}}\|_1 + \|\cos\hat{\phi} - \cos\phi_{\text{gt}}\|_1$.
- `GradientCurlLoss`: $\mathcal{L}_{\text{curl}} = \|\nabla \times \tilde{\mathbf{g}}\|_1 = \left|\frac{\partial \tilde{g}_y}{\partial x} - \frac{\partial \tilde{g}_x}{\partial y}\right|$.
- `CurvatureWeightedGradLoss`: Grad loss weighted by local fringe density $(1 + |\nabla \phi_{\text{gt}}|^2 / \pi^2)^{-1}$.
- `PCLCNLoss`: Combined loss manager supporting Option A (supervised) and Option B (unsupervised).

### Data Generator & Loader (`generate.py`, `dataset.py`, `train.py`)
- `generate.py`: Added `compute_reference_gradient` ($\nabla \phi_2$) and optional multiplicative laser speckle noise.
- `dataset.py`: Updated `H5ShardDataset` to load `grad_phi2` from HDF5 shards.
- `train.py`: Updated batch unpacking for 3-tuple loader compatibility.

---

## 5. GIT COMMIT LOG (Branch: `pclcn-pipeline`)

```
* cd43ff4 - Update training loop batch unpacking for 3-tuple datasets
* 069badf - Add reference beam gradients and speckle noise support
* 09d8782 - Add PCLCN network architecture and loss functions
* 9f4d8fc - Add PCLCN signal processing operators and tests
```

---

## 6. INSTRUCTIONS FOR NEXT AGENT / RESEARCHER

1. **Verify Unit Tests:**
   Run `uv run pytest tests/ -q` (all 57 tests should pass).
2. **Train PCLCN Model:**
   Execute `phase-unwrap train` using `PCLCNModel` and `PCLCNLoss`.
3. **Manuscript Benchmark:**
   Evaluate un-cheated `AbsMAE` on the test split using `piston_align` (mean offset removal only). Record performance in comparative tables against classical solvers (Itoh, 2D Least-Squares).
