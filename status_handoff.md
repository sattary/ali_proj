# Project Handoff & Research History: PCLCN Pipeline

> **Branch:** `pclcn-pipeline`  
> **Date:** 2026-07-30  
> **Role:** Senior Principal Researcher  
> **Status:** All 52 Unit Tests Passing (100%) | Code Refactored & Committed  
> **Target Journal:** *Optics Express* (Optica Publishing Group)

---

## 1. EXECUTIVE SUMMARY & HANDOFF CONTEXT

This handoff document details the complete research, mathematical diagnostic, and engineering history of the `ali_proj` phase unwrapping pipeline. 

Over the course of design iterations (Plan 020 $\to$ Plan 021 $\to$ Plan 022 $\to$ Plan 027 $\to$ Plan 028 $\to$ Plan 029), we transformed an initial ad-hoc deep learning model into a scientifically rigorous, optics-native architecture: the **Physics-Constrained Latent Corrector Network (PCLCN)**.

All legacy models (`UNetRes2`), obsolete data hints (`gt_center`), and unused facade shims have been purged from `src/`. The codebase strictly exposes PCLCN and provides a high-throughput, raw-data ablation and multi-seed framework designed for Kaggle GPU execution.

All unit tests are passing (52/52 tests, 100%).

---

## 2. EVOLUTION OF PLANS & RESEARCH METHODOLOGY

### Phase 1: Plan 020 (Physics-Informed UNet & Failure Diagnosis)
* **Optical Forward Model Mapping:** Line-by-line extraction of physical constants from the MATLAB script (`scripts/image_generation.m` $\to$ `generate.py`).
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

### Phase 4: Plan 027 (Ponytail Audit Refactoring)
* Replaced custom binary search while-loop in `dataset.py` with stdlib `bisect.bisect_right`.
* Converted `AddCoords` to native `torch.meshgrid(..., indexing="ij")`.
* Streamlined `_load_mapping` in `config.py`.

### Phase 5: Plan 028 (Purge Legacy `UNetRes2` & Oracle Leaks)
* Deleted `UNetRes2_AbsPhase`, `unetres2.py`, `unet.py` facade, `UpBlockRes2`, and `gt_center` oracle data leak options.
* Updated `build_model()` to directly construct `PCLCNModel`.

### Phase 6: Plan 029 (Modernize PCLCN Ablation & Multi-Seed Framework)
* Configured Option 1 Fast Core PCLCN Ablation Matrix (`no_reference_prior`, `no_curl_loss`, `unmasked_zernike`).
* Removed online LaTeX/plot compilation overhead during training runs, outputting raw numerical `ablation_summary.json` and `aggregate.csv` for Kaggle GPU speed.
* Added dynamic spatial resolution handling in `MaskedZernikeProjection` and `DifferentiablePoissonSolver`.

---

## 3. CODEBASE IMPLEMENTATION SUMMARY

### Core Operators (`src/phase_unwrap/core/ops.py`)
- `AnalyticSignalStem`: Differentiable 2D FFT demodulation.
- `WrappedGradientOperator`: Computes wrapped finite-difference gradients $g_x, g_y \in [-\pi, \pi]$.
- `DifferentiablePoissonSolver`: 2D DCT-II Neumann Poisson integrator with dynamic grid size support.
- `MaskedZernikeProjection`: Disk-masked pseudoinverse projection onto 15 Zernike modes with dynamic grid size support.

### Neural Network Models (`src/phase_unwrap/model/pclcn.py`)
- `ReferenceConditionedGradientCorrector` (CNN 1): 7-channel input (`I_raw`, `wrapped_phase`, `amplitude`, $g_x, g_y$, and $\nabla \phi_2$). Predicts gradient corrections $(\delta g_x, \delta g_y)$ to restore integrability ($\nabla \times \tilde{\mathbf{g}} = 0$).
- `OrthogonalResidualCNN` (CNN 2): Predicts non-Zernike orthogonal residual phase $r(x,y)$.
- `PCLCNModel`: Integrated 6-stage forward model pipeline with `zero_reference_prior` and `unmasked_zernike` ablation flags.

### Loss Topology (`src/phase_unwrap/core/losses.py`)
- `ComplexDomainLoss`: $\mathcal{L}_{\text{phase}} = \|\sin\hat{\phi} - \sin\phi_{\text{gt}}\|_1 + \|\cos\hat{\phi} - \cos\phi_{\text{gt}}\|_1$.
- `GradientCurlLoss`: $\mathcal{L}_{\text{curl}} = \|\nabla \times \tilde{\mathbf{g}}\|_1$.
- `PCLCNLoss`: Combined physics-constrained loss manager.

---

## 4. GIT COMMIT LOG (Branch: `pclcn-pipeline`)

```
* 12c9289 - Fix MaskedZernikeProjection height and width attribute initialization
* 6acae11 - Modernize PCLCN ablation and multiseed framework for Fast Core Matrix (Plan 029)
* dd2bdba - Purge legacy UNetRes2 architecture and gt_center oracle leak (Plan 028)
* c4259d8 - Refactor src over-engineering and simplify utilities (Plan 027)
* cd43ff4 - Update training loop batch unpacking for 3-tuple datasets
* 069badf - Add reference beam gradients and speckle noise support
* 09d8782 - Add PCLCN network architecture and loss functions
* 9f4d8fc - Add PCLCN signal processing operators and tests
```

---

## 5. INSTRUCTIONS FOR NEXT AGENT / RESEARCHER

1. **Verify Unit Tests:**
   Run `uv run pytest tests/ -q` (all 52 tests should pass).
2. **Train PCLCN Model:**
   Execute `phase-unwrap train` or launch Kaggle ablation matrix via `phase-unwrap ablation`.
3. **Manuscript Benchmark:**
   Evaluate un-cheated `AbsMAE` on the test split using `piston_align` (mean offset removal only). Record performance in comparative tables against classical solvers (Itoh, 2D Least-Squares).
