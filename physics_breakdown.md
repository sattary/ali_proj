# Physics and Mathematical Pipeline Breakdown

This document traces the physical approximations and mathematical topology of the absolute phase reconstruction pipeline in `ali_proj`.

## 1. Optical Wavefront Approximations ([data/generate.py](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/data/generate.py))

The dataset synthesizes interferograms representing the superposition of two coherent beams at $\lambda = 632.8$ nm over a $128 \times 128$ grid spanning $[-1, 1]$ mm.

**Target Wavefront ($\phi_1$):**
The primary wavefront is modeled with a randomized curvature base field:
$r_1(x, y) = \sqrt{\mathcal{U}(0, 80) + \frac{x}{50}\mathcal{U}(-0.5, 0.5) + \frac{y}{50}\mathcal{U}(-0.5, 0.5)}$
Where $\mathcal{K} = \frac{2\pi}{\lambda}$. This field is augmented with low-order aberrations via a bilinearly upsampled random vector ($N_{\text{dev}} \in \{1, 2, 3\}$) to simulate dynamic, non-uniform optical distortions.
$\phi_1(x, y) = \mathcal{K} r_1 + \text{zoom}(\delta_{aber})$

**Reference Wavefront ($\phi_2$):**
A spherical reference beam propagating from $Z = 0.5$ m, scaled by a random factor:
$r_2(x, y) = \sqrt{x^2 + y^2 + 0.5^2}$
$\phi_2(x, y) = \mathcal{K} r_2 \mathcal{U}(-0.5, 0.5)$

**Interference Intensity ($I$):**
The recorded intensity $I(x,y)$ represents the squared magnitude of the complex wavefields with base amplitude $E_0=1$:
$I(x,y) = |E_0 e^{i\phi_1} + E_0 e^{i\phi_2}|^2$

**Supervision Target ($\Delta\phi$):**
The regression target is the unwrapped phase difference shifted to a zero minimum:
$\Delta\phi(x,y) = (\phi_1 - \phi_2) - \min(\phi_1 - \phi_2)$

## 2. Global Ambiguity and [AddCoords](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/model/unet.py#25-46) ([model/unet.py](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/model/unet.py))

Interferograms inherently represent phase modulo $2\pi$. An absolute reconstruction requires removing this $2\pi k$ ambiguity.

The [AddCoords](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/model/unet.py#25-46) module explicitly embeds normalized spatial coordinates $(x,y) \in [-1, 1]^2$ as input channels. This breaks the strict translational invariance of convolutions, providing the [UNetRes2](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/model/unet.py#132-210) architecture with the spatial priors required to ground the dynamic unwrapping surface and synthesize the absolute scalar offset (`k_off`).

## 3. Loss Topology ([core/losses.py](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/core/losses.py) & [core/ops.py](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/core/ops.py))

The network is optimized through a multi-term objective balancing point-wise accuracy, structural fidelity, and topological smoothness.

**Phase Error (MAE):**
$L_{\text{MAE}} = w_{\text{mae}} |\phi_{\text{pred}} - \phi_{\text{gt}}|$

**Gradient Consistency Loss:**
Enforces structural continuity by taking depthwise Sobel derivatives ($\nabla_x, \nabla_y$):
$L_{\text{grad}} = w_{\text{grad}} \left( |\nabla_x\phi_{\text{pred}} - \nabla_x\phi_{\text{gt}}| + |\nabla_y\phi_{\text{pred}} - \nabla_y\phi_{\text{gt}}| \right)$
*Intensity Weighting:* When `intensity_weighted=True` is enabled, gradient errors are amplified in regions of high intensity using $\sqrt{I(x,y)}$.

**Curvature Regularization:**
To penalize noisy oscillations and enforce optical smoothness, a discrete 2D Laplacian applies an $L_1$ penalty to the unrolled surface:
$L_{\text{curv}} = w_{\text{curv}} |\nabla^2 \phi_{\text{pred}}|$

## 4. Evaluation & Affine Alignment ([core/ops.py](file:///home/sat/Code/Fun/ali_proj/src/phase_unwrap/core/ops.py) & `analysis/baselines.py`)

Because intensity measurements are scale- and shift-invariant relative to the absolute phase, direct point-wise evaluation metrics can be artificially inflated by global coordinate drift.

Prior to computing diagnostic metrics (MAE, RMSE, SSIM, PSNR), the validation branch performs a Least-Squares Affine Alignment on the raw prediction:
$a \cdot \phi_{\text{pred}} + c \approx \phi_{\text{gt}}$
Metrics are then calculated on the aligned prediction $\hat{\phi} = a\phi_{\text{pred}} + c$, cleanly separating local unwrapping accuracy from global scaling and macroscopic offset errors.
