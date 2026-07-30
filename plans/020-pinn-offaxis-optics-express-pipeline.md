# Plan 020: Comprehensive Scientific & Implementation Blueprint — Physics-Driven Reference-Guided PINN Pipeline for Optics Express

> **Status:** `TODO`  
> **Priority:** P1  
> **Effort:** M–L  
> **Target Journal:** Optics Express (Optica Publishing Group)  
> **Audience:** External Peer Reviewers, Journal Editors, & Project Collaborators  
> **ARS Governance:** `academic-research-skills` v3.19.0 (Physics-Informed Prior Integration)

---

## EXECUTIVE SUMMARY FOR REVIEWERS

This document provides a comprehensive mathematical, physical, and architectural specification of our deep learning phase unwrapping framework. It details the initial codebase state, the diagnostic discovery of preliminary metric leakage, the mathematical proof of physical ill-posedness in single-frame on-axis intensity unwrapping, and our complete scientific solution: a **Physics-First Two-Stage Neural Architecture**.

By rigorously embedding the exact physics of a simulated HeNe Mach-Zehnder interferometer into the network design, we overcome critical failure modes of naive neural networks. Our pipeline introduces a differentiable Poisson solver, an analytic signal pre-processing stem, a Zernike coefficient head, and Transport of Intensity (TIE) constraints. This framework bridges single-point optical displacement sensing with full-field deep learning demodulation, providing a publication-ready manuscript strategy for **Optics Express**.

---

## PART I: DISSECTING THE OPTICAL SYSTEM (THE FORWARD MODEL)

To ground our neural network in physical reality, we trace the exact physical equations modeled in our dataset generator (`generate.py`), which is a direct Python port of the laboratory MATLAB script (`scripts/image_generation.m`).

### 1. The Physical Sensor Grid
The system models a $128 \times 128$ CCD/CMOS sensor over a $2\text{mm} \times 2\text{mm}$ active area:
- $x_0 \in [-1\times 10^{-3}, 1\times 10^{-3}]$ meters
- Pixel pitch $\Delta x \approx 15.75 \, \mu\text{m}$.
- **Nyquist Limit:** $f_{\text{Nyquist}} = \frac{1}{2\Delta x} \approx 31.7 \text{ lp/mm}$. Any phase gradient steeper than $|\nabla \phi| > \pi / \Delta x \approx 2 \times 10^5$ rad/m will alias.

### 2. The Reference Beam ($\phi_2$)
The reference beam is a diverging spherical wave originating from a point source at distance $z_2 = 0.5$ m behind the sensor plane:
$$\phi_2(x,y) = \frac{2\pi}{\lambda} \sqrt{x^2 + y^2 + z_2^2} \cdot \alpha, \quad \alpha \sim \mathcal{U}(-0.5, 0.5)$$
The random multiplier $\alpha$ simulates variable beam tilt and convergence. Using the exact $\sqrt{\cdot}$ (rather than the paraxial approximation) means the reference beam contains higher-order spherical aberrations naturally.

### 3. The Object Beam ($\phi_1$)
The sample under test is modeled with a TWO-COMPONENT wavefront:
- **Base Curvature:** $r_1(x,y) = \sqrt{R + \beta_x x + \beta_y y}$ with $R \sim \mathcal{U}(0, 80)$. This models a tilted spherical surface with a random radius of curvature.
- **Low-Order Polynomial Aberration:** A 1x1, 1x2, or 1x3 random vector in $[0, 20]$ is bicubically interpolated up to $128 \times 128$. This is physically equivalent to the first few terms of a Zernike polynomial expansion (piston, tilt, astigmatism, defocus, coma), and the bicubic interpolation enforces extreme sub-Nyquist smoothness.

### 4. The Interference Equation
The beams interfere to form a scalar intensity field:
$$I(x,y) = \left| E_1 e^{i\phi_1} + E_2 e^{i\phi_2} \right|^2 = I_1 + I_2 + 2\sqrt{I_1 I_2} \cos(\Delta\phi(x,y))$$
where $\Delta\phi = \phi_1 - \phi_2$. The base simulation assumes perfect amplitude $E_0=1$, yielding $I(x,y) = 2E_0^2(1 + \cos\Delta\phi)$ with uniform fringe visibility $V=1$.

---

## PART II: CRITICAL FLAWS IN THE NAIVE APPROACH

A rigorous audit of the preliminary on-axis architecture reveals five critical failures that render a naive UNet physically invalid:

### FLAW 1: The System Calibration Protocol ($a_{\text{calib}}$) Is a Leak (CRITICAL)
> [!CAUTION]
> If an Optics Express reviewer catches this, the paper is desk-rejected.

The preliminary plan proposed a calibration scalar:
$$a_{\text{calib}} = \text{median}_{i \in \text{calib}} \left( \frac{\text{cov}(\phi_{\text{pred}}^{(i)}, \phi_{\text{gt}}^{(i)})}{\text{var}(\phi_{\text{pred}}^{(i)})} \right)$$
This formula explicitly requires ground-truth phase $\phi_{\text{gt}}$. In a real laboratory, you do not have $\phi_{\text{gt}}$ for calibration standards—if you had an oracle phase, you would not need the neural network. Calibration must be either formulated against a known physical geometric artifact (like a NIST-traceable height gauge) or eliminated entirely by teaching the network to predict absolute scale intrinsically.

### FLAW 2: The Point Reference Channel ($\phi_{\text{ref}}$) Is a Renamed `gt_center` (MAJOR)
Proposing $\phi_{\text{ref}} = \phi_{\text{gt}}(x_0, y_0) + \mathcal{N}(0, 0.05)$ creates a noisy ground truth, not a noisy physical measurement. Real point displacement sensors have systematic bias, drift, and quantization, and their noise is not zero-mean Gaussian centered exactly on the true phase. The reference channel must simulate real sensor physics or be dropped.

### FLAW 3: The PINN Re-Synthesis Loss Has a Critical Gradient Pathology (MAJOR)
The naive PINN loss re-synthesizes intensity: $\hat{I}_{\text{synth}} = 2E_0^2 \left(1 + \cos(\hat{\phi})\right)$ and minimizes $\|\hat{I}_{\text{synth}} - I_{\text{obs}}\|_1$.
The gradient of $\cos(\hat{\phi})$ is $-\sin(\hat{\phi})$. When $\hat{\phi} \approx n\pi$ (at fringe peaks and valleys), $\sin(\hat{\phi}) \approx 0$. Therefore, **the physical loss gradient vanishes at every fringe extremum**. This creates a degenerate loss landscape, making unsupervised training almost impossible.

### FLAW 4: The Off-Axis Carrier Cannot Be Resolved by the CNN (MODERATE)
Adding a spatial carrier $\mathbf{f}_0 = (f_x, f_y)$ theoretically resolves the sign ambiguity (off-axis holography). However, feeding $I_{\text{off}}$ raw into a CNN fails because a $3\times3$ convolution has a 3-pixel receptive field, while our carrier $f_0 = 0.125$ cycles/pixel has an 8-pixel period. The first encoder layer cannot see a full fringe period, and subsequent MaxPool layers alias the carrier away before it reaches the bottleneck attention mechanisms.

### FLAW 5: Input Channel Count Mismatch
The preliminary code hardcoded `in_ch=2`, failing to account for the derived physical channels required for reference-guided networks.

---

## PART III: PHYSICS-DRIVEN ENHANCEMENTS (THE SOLUTION)

To resolve the above flaws, we introduce 6 novel enhancements that survive adversarial literature screening. Every enhancement is derived directly from the physics encoded in the MATLAB script.

### NOVEL IDEA 1: Differentiable Poisson Solver Layer (Exploiting $\nabla^2 \phi$)
The classical Ghiglia-Pritt least-squares phase unwrapping algorithm solves the 2D Poisson equation:
$$\nabla^2 \hat{\phi}_{\text{unwrapped}} = \nabla \cdot W(\nabla \phi_{\text{wrapped}})$$
This can be solved in closed form via the 2D DCT:
$$\hat{\phi}(\mathbf{u}) = \frac{\widehat{\rho}(\mathbf{u})}{2(\cos(2\pi u_x / N) + \cos(2\pi u_y / M) - 2)}$$
**Implementation:** Add a `PoissonUnwrapLayer` that executes a differentiable 2D DCT/IDCT forward pass. The UNet receives this $\phi_{\text{Poisson}}$ as a third input channel. The network then only learns the **residual correction** between the classical Poisson solution and the true phase.

### NOVEL IDEA 2: Zernike Coefficient Prediction Head (Exploiting Aberration Structure)
Our forward model generates low-order polynomial aberrations strictly equivalent to Zernike polynomials up to order 3.
**Implementation:** Add a Zernike Coefficient Prediction Head. The bottleneck features are global-average-pooled and passed through an MLP that predicts $J$ Zernike coefficients $\hat{c}_j$. The Zernike surface is reconstructed analytically:
$$\hat{\phi}_{\text{Zernike}}(x,y) = \sum_{j=1}^{J} \hat{c}_j Z_j(\rho, \theta)$$
This acts as a global regularizer that prevents physically impossible high-frequency phase noise.

### NOVEL IDEA 3: Learnable Scale + Offset Head (Replacing $a_{\text{calib}}$)
**Implementation:** Extend the existing `k_off` global head to predict both a scalar offset AND a scale multiplier from the bottleneck features:
```python
k_params = self.off_fc(z).mean(dim=(2,3), keepdim=True)  # [B, 2, 1, 1]
k_scale = k_params[:, 0:1].sigmoid() * 3.0 + 0.1         # Bounded [0.1, 3.1]
k_off   = k_params[:, 1:2]
phi_abs = k_scale * phi_raw + k_off
```
The network LEARNS the correct absolute metric scale from the training data. No external calibration split is required, eliminating the GT leak entirely.

### NOVEL IDEA 4: Analytic Signal Pre-Processing Stem (Replacing Raw Carrier Channels)
Instead of hoping the CNN discovers Fourier filtering through gradient descent, perform classical off-axis demodulation deterministically in a differentiable PyTorch module with no learnable parameters.
```python
class AnalyticSignalStem(nn.Module):
    def forward(self, I_off: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        F_I = torch.fft.fft2(I_off)
        # Bandpass around carrier f0
        analytic = torch.fft.ifft2(F_I * mask)
        wrapped_phase = torch.atan2(analytic.imag, analytic.real)
        amplitude = analytic.abs()
        return wrapped_phase, amplitude
```
The UNet's job simplifies from "learn Fourier optics from pixels" to "unwrap this wrapped phase map."

### NOVEL IDEA 5: Wrapped Phase Consistency Loss (Fixing the Cosine Pathology)
Instead of re-synthesizing intensity, enforce consistency in the **wrapped phase domain**:
$$\mathcal{L}_{\text{wrap}} = \frac{1}{N} \sum_{x,y} \left| W(\hat{\phi}(x,y)) - W(\phi_{\text{obs}}(x,y)) \right|$$
where $W(\phi) = \text{atan2}(\sin\phi, \cos\phi)$. The gradient of $\text{atan2}$ is well-conditioned everywhere, completely fixing the vanishing gradient pathology at fringe extrema.

### NOVEL IDEA 6: Wavefront Curvature-Aware Gradient & TIE Loss
- **Curvature-Aware Gradient Loss:** The reference beam $\phi_2 \propto \sqrt{x^2+y^2+z_2^2}$ creates denser fringes at the sensor edges. We weight the gradient loss by the inverse local fringe density:
  $$\mathcal{L}_{\text{grad}}^{\text{curvature}} = \frac{1}{N} \sum_{x,y} \frac{|\nabla_x \hat{\phi} - \nabla_x \phi_{\text{gt}}| + |\nabla_y \hat{\phi} - \nabla_y \phi_{\text{gt}}|}{1 + |\nabla \phi_{\text{gt}}|^2 / \pi^2}$$
- **Transport of Intensity Equation (TIE) Constraint:** Derived from the paraxial Helmholtz equation, we enforce a divergence-free physical constraint:
  $$\mathcal{L}_{\text{TIE}} = \left\| \nabla \cdot (I_{\text{obs}} \nabla \hat{\phi}) \right\|_1$$

---

## PART IV: PHYSICS-FIRST ARCHITECTURE & LOSS TOPOLOGY

### Two-Stage Architecture
```
Stage 1 (Deterministic, No Learning):
    I_off ──> [AnalyticSignalStem] ──> wrapped_phase, amplitude

Stage 2 (Learned, UNetRes2 + Scale Head):
    [wrapped_phase, amplitude, I_norm] ──> [Poisson Solver Layer] ──> phi_Poisson
    [wrapped_phase, amplitude, I_norm, phi_Poisson] ──> [UNetRes2] ──> phi_raw
    phi_raw ──> [Scale+Offset Head] ──> phi_abs = k_scale * phi_raw + k_off
```

### Loss Topology
**Option A (Hybrid Supervised - High Precision):**
$$\mathcal{L}_{\text{total}} = w_1 |\hat{\phi} - \phi_{\text{gt}}| + w_2 \mathcal{L}_{\text{wrap}} + w_3 \mathcal{L}_{\text{grad}}^{\text{curv}} + w_4 \mathcal{L}_{\text{Zernike}} + w_5 \mathcal{L}_{\text{TIE}}$$

**Option B (Unsupervised PINN - Zero-Shot Lab Transfer):**
$$\mathcal{L}_{\text{total}} = w_2 \mathcal{L}_{\text{wrap}} + w_3 \mathcal{L}_{\text{grad}}^{\text{curv}} + w_5 \mathcal{L}_{\text{TIE}} + w_6 \mathcal{L}_{\text{radial}}$$
*(where $\mathcal{L}_{\text{radial}}$ regularizes the non-radial residual, exploiting the physical symmetry of the reference beam).*

---

## PART V: CODE MODIFICATION BLUEPRINT (LINE-BY-LINE)

### 1. Update `generate.py`
- Add spatial carrier frequencies $f_x = 0.125, f_y = 0.125$ cycles/pixel to simulate the off-axis configuration.
- Implement variable beam intensity ratios $E_1/E_2 \sim \mathcal{U}(0.3, 1.0)$ to generate physically realistic fringe visibility $V(x,y)$.
- Export `I_off`, `dphi`, `I_clean`, and the ground-truth Zernike coefficients $c_j^{\text{gt}}$ (computed via least-squares fit on `dphi`) to the HDF5 output shards.

### 2. Update `ops.py`
- Implement `AnalyticSignalStem(nn.Module)` for differentiable 2D FFT bandpass filtering.
- Implement `PoissonUnwrapLayer(nn.Module)` utilizing PyTorch's 2D DCT/IDCT.
- Remove all instances of the leaking `affine_align` for test metrics. `piston_align` (mean offset removal) may be retained purely for diagnostic topological error (`TopoMAE`), but `AbsMAE` must be computed purely via the intrinsic `Scale+Offset Head`.

### 3. Update `unet.py`
- Change `in_ch` from 2 to 4 (`I_norm`, `wrapped_phase`, `amplitude`, `phi_Poisson`).
- Extend the `off_fc` global head to `ScaleOffsetHead` on the bottleneck GAP features to predict both $k_{\text{scale}}$ and $k_{\text{off}}$.
- Implement the `ZernikeHead` on the bottleneck GAP features to predict $\hat{c}_j$ and reconstruct $\hat{\phi}_{\text{Zernike}}$.

### 4. Update `losses.py`
- Implement the Wrapped Phase Consistency Loss ($\mathcal{L}_{\text{wrap}}$).
- Implement the Wavefront Curvature-Aware Gradient Loss ($\mathcal{L}_{\text{grad}}^{\text{curv}}$).
- Implement the Zernike Coefficient Loss ($\mathcal{L}_{\text{Zernike}} = \frac{1}{J} \sum |\hat{c}_j - c_j^{\text{gt}}|$).
- Implement the Transport of Intensity constraint ($\mathcal{L}_{\text{TIE}}$).
- Update `MAEGradLoss` to route these components according to the new Option A/Option B topologies.

### 5. Update `train.py` & `config.py`
- Add weighting parameters for the new loss components in `LossConfig`.
- Modify the training loop forward pass to route $I_{\text{off}}$ through the `AnalyticSignalStem` and `PoissonUnwrapLayer` before feeding the concatenated tensor to the UNet.

---

## PART VI: ACCEPTANCE & VERIFICATION CRITERIA

- [ ] `generate.py` successfully injects off-axis carriers and fits/saves Zernike coefficients $c_j^{\text{gt}}$ without breaking dataset generation speed.
- [ ] Pytest suite passes: `uv run pytest tests/ -q`.
- [ ] Network learns absolute metric scale purely through the `ScaleOffsetHead` (zero test-time calibration). The $a_{\text{calib}}$ logic is completely purged.
- [ ] Unsupervised Option B trains stably without vanishing gradients, enabled by the Wrapped Phase Consistency Loss.
- [ ] Ablation studies ("With vs Without Poisson Layer", "With vs Without Zernike Head", "With vs Without Analytic Stem") are logged to demonstrate clear performance delta for the manuscript.
- [ ] Final un-cheated AbsMAE < 0.25 rad on the held-out test set
