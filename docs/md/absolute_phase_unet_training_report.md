### Technical Report on `src/try.py`: Absolute Phase Reconstruction with a UNet-Style Network

#### 1. Introduction

This document provides a detailed, academically styled description of the training script `src/try.py`. The script implements a supervised learning pipeline for recovering *absolute* (unwrapped) phase from interferometric measurements stored in MATLAB `.mat` files. It combines:

- A custom PyTorch `Dataset` for loading and preprocessing interferogram and phase data.
- A deep UNet-style convolutional neural network with Res2-inspired blocks and coordinate channels.
- A composite loss that supervises absolute phase and its spatial gradients, augmented with curvature regularization.
- An exponential moving average (EMA) teacher model, rigorous per-image affine alignment of predictions to ground truth, and visualization utilities for qualitative assessment.

The intent of this report is to summarize the design choices and data flow in a manner suitable for academic review or presentation.

#### 2. Dataset Construction and Preprocessing

##### 2.1 Data sources and file structure

The training data are MATLAB `.mat` files, each assumed to contain at least two arrays:

- `I`: an interferogram or intensity image.
- `dphi`: the corresponding ground-truth absolute (unwrapped) phase in radians.

The `MatPhaseDataset` class is responsible for loading these files. It first attempts to read them using `h5py` (appropriate for HDF5-based `.mat` files). If that fails, it falls back to `scipy.io.loadmat`. The specific keys for intensity and phase are configurable via command-line arguments (defaults are `I` and `dphi`).

##### 2.2 Channel layout normalization

The raw NumPy arrays obtained from the `.mat` files can appear in several shapes. To standardize this, the helper function `_to_chw` enforces a channel-first layout `[C, H, W]`:

- If the input is `[H, W]`, a singleton channel is added, producing `[1, H, W]`.
- If the input is `[H, W, C]` and the last dimension is relatively small (e.g., `C ≤ 8`) while height and width are at least moderately large, it is treated as channels-last and transposed to `[C, H, W]`.
- Otherwise, the array is assumed to already be in `[C, H, W]` format.

After this normalization, both the interferogram `I` and phase `dphi` are converted to contiguous `float32` arrays.

##### 2.3 Tensor conversion and intensity normalization

The normalized arrays are converted into PyTorch tensors:

- `I_raw_t ∈ ℝ^{1×H×W}`: raw interferogram.
- `phi_gt_t ∈ ℝ^{1×H×W}`: ground-truth absolute phase.

For each sample, the interferogram is z-score normalized:

\[
I_{\text{norm}} = \frac{I_{\text{raw}} - \mu}{\sigma},
\]

where \(\mu\) and \(\sigma\) are the per-image mean and standard deviation computed over spatial dimensions. The standard deviation is clamped below by a small constant to avoid numerical instabilities.

##### 2.4 Construction of the phase hint channel

To assist the network in resolving the global ambiguity of absolute phase, the dataset constructs a *phase hint* channel, denoted `phi_hint`. The ground-truth phase tensor `phi_gt_t` is sampled at a single reference pixel, chosen as the spatial centre:

- Let \((c_y, c_x)\) be the centre coordinates.
- Extract the scalar reference value:
  \[
  \phi_{\text{ref}} = \phi_{\text{gt}}[0, c_y, c_x].
  \]

This scalar is then broadcast to the full spatial extent, yielding a tensor:

\[
\phi_{\text{hint}}(y, x) = \phi_{\text{ref}}, \quad \forall (y, x).
\]

Intuitively, this provides the model with a single absolute phase anchor, replicated across the image. While the network still must recover the detailed spatial structure, it can use this anchor to stay globally consistent with the target phase scale.

##### 2.5 Final dataset outputs

For each sample, the dataset returns the triplet:

- `I_input ∈ ℝ^{2×H×W}`:
  - Channel 0: normalized interferogram \(I_{\text{norm}}\).
  - Channel 1: broadcast phase hint \(\phi_{\text{hint}}\).
- `phi_gt_t ∈ ℝ^{1×H×W}`: absolute ground-truth phase.
- `I_raw_t ∈ ℝ^{1×H×W}`: raw interferogram, retained for possible intensity-weighted loss terms and diagnostics.

These tensors form the core inputs and targets for the training loop.

#### 3. Model Architecture: UNetRes2_AbsPhase

##### 3.1 Coordinate augmentation

The model expects `I_input` with two channels but internally augments it with spatial coordinate information via the `AddCoords` module. This module constructs two 2D grids:

- \(x\)-coordinates linearly spaced in \([-1, 1]\) across width.
- \(y\)-coordinates linearly spaced in \([-1, 1]\) across height.

These are stacked to form two additional channels and concatenated with the original input, resulting in a 4-channel tensor:

\[
X_0 \in ℝ^{B×4×H×W},
\]

where the channels are \([I_{\text{norm}}, \phi_{\text{hint}}, x, y]\). This explicit coordinate encoding can improve the network’s ability to model spatially varying phase structures.

##### 3.2 Res2-style residual blocks

The core building block of the network is `Res2_DS_Block`, a Res2-inspired residual module with depthwise convolutional groups:

1. A \(1×1\) convolution expands the input to an intermediate number of channels.
2. The intermediate channels are partitioned into several groups.
3. Each group passes through a depthwise \(3×3\) convolution, where all but the first group also receive a residual connection from the previous group’s output (Res2-style hierarchy).
4. The outputs of all groups are concatenated and projected back to the desired output dimensionality via another \(1×1\) convolution and batch normalization.
5. If needed, an additional \(1×1\) projection aligns the shortcut’s channel dimensions, and the final output adds the shortcut followed by a non-linearity (ReLU or SiLU).

This design allows multi-scale feature interactions within a single block and improves representational capacity without excessive parameter growth.

##### 3.3 Encoder (contracting path)

The encoder is a UNet-like contracting path built from stacked `Res2_DS_Block`s and max-pooling:

- **Level 1 (full resolution)**:
  - Input: `X_0` with 4 channels.
  - Two consecutive Res2 blocks increase feature depth while retaining spatial resolution, producing `e1`.
- **Levels 2–5 (downsampling)**:
  - Each level applies `MaxPool2d(2)` to halve spatial resolution.
  - A subsequent Res2 block increases channel capacity, producing `e2`, `e3`, `e4`, and `e5` respectively.

Channels are scaled by a base factor (configurable via `--base`), with an upper cap to prevent excessive width.

##### 3.4 Bottleneck

At the deepest resolution, a single Res2 block processes the features from `e5`, producing `b`. This bottleneck representation aggregates high-level global information over a strongly downsampled spatial grid.

##### 3.5 Decoder (expanding path) with skip connections

The decoder mirrors the encoder using `UpBlockRes2` modules:

1. Bilinear upsampling restores the spatial resolution to match that of a corresponding encoder feature map.
2. The upsampled features are concatenated channel-wise with the encoder’s features (skip connection).
3. A Res2 block refines the combined representation.

This process is repeated across multiple levels, progressively reconstructing higher-resolution representations (`d4`, `d3`, `d2`, `d1`, `d0`) while reusing encoder information via skip connections. Immediately before the final output head, a spatial dropout layer (`Dropout2d`) is applied for regularization.

##### 3.6 Pixelwise output heads

The final decoder output `d0` passes through a \(1×1\) convolution (`head_pix`) to produce four channels per pixel:

- `phi_raw ∈ ℝ^{B×1×H×W}`: primary phase map capturing local phase structure.
- `a_pred ∈ ℝ^{B×1×H×W}`: auxiliary head (e.g., background term), currently unused in the loss.
- `b_pred_raw ∈ ℝ^{B×1×H×W}`: raw amplitude-like term, intended to be passed through a softplus nonlinearity if integrated into the training objective.
- `conf_logit ∈ ℝ^{B×1×H×W}`: confidence logit map, intended to be transformed via a sigmoid into a [0, 1] confidence weight if employed.

In the current script, only `phi_raw` is directly involved in constructing the supervised phase prediction; the remaining heads are provided as structured outputs and can be integrated into extended formulations (e.g., uncertainty weighting, amplitude-aware losses) in future work.

##### 3.7 Global scalar offset head

To resolve global phase offset at the image level, the model includes a dedicated scalar offset head:

1. The bottleneck tensor `b` passes through a \(1×1\) convolution and activation.
2. A second \(1×1\) convolution reduces channels to one per spatial location.
3. Global average pooling over spatial dimensions yields a single scalar per image:
   \[
   k_{\text{off}} ∈ ℝ^{B×1×1×1}.
   \]

This scalar is added uniformly to the pixelwise phase map:

\[
\phi_{\text{abs}} = \phi_{\text{raw}} + k_{\text{off}},
\]

resulting in an absolute phase prediction that combines detailed local structure with a learned global offset.

#### 4. Training Procedure and Loss Formulation

##### 4.1 Overview

The core training loop (`train(args)`) couples the dataset and model with a composite loss, EMA teacher, and learning rate scheduler. All major hyperparameters (batch size, learning rate, gradient clipping threshold, loss weights, etc.) are configurable via the command-line interface.

At a high level, each training iteration consists of:

1. Forward propagation through the UNetRes2 model to obtain `phi_raw` and `k_off`.
2. Construction of `phi_abs = phi_raw + k_off`.
3. Per-image affine alignment between `phi_abs` and `phi_gt`.
4. Computation of supervised phase loss and curvature regularization.
5. Backpropagation with optional mixed-precision (AMP), optimizer update, gradient clipping, and EMA update.

##### 4.2 Per-image affine alignment

Before applying the phase loss, the script performs a per-image affine alignment using the helper function `affine_align`. Given predicted and ground-truth phases (`pred`, `gt`), both in shape `[B, 1, H, W]`, the function computes:

- Flattened predictions and targets per image.
- Centered versions by subtracting their respective means.
- The variance of centered predictions and covariance between centered predictions and targets.

The optimal affine parameters \(a\) and \(c\) (in the least-squares sense) satisfy:

\[
gt \approx a \cdot pred + c.
\]

They are obtained via:

\[
a = \frac{\operatorname{cov}(pred, gt)}{\operatorname{var}(pred) + \varepsilon}, \quad
c = \mathbb{E}[gt] - a \cdot \mathbb{E}[pred],
\]

with a small \(\varepsilon\) added to stabilize the denominator. These parameters are then broadcast back to spatial resolution, and the aligned prediction is:

\[
\phi_{\text{align}} = a \cdot \phi_{\text{abs}} + c.
\]

This procedure is applied in three contexts:

- During training, prior to computing the supervised phase loss.
- During validation, before computing evaluation metrics.
- During visualization, before plotting prediction vs ground truth.

The alignment compensates for residual global scaling and offset differences, thereby focusing the loss and metrics on structural agreement between predicted and true phase fields.

##### 4.3 Supervised phase loss

The supervised loss is encapsulated in the `PhaseSupervisionLoss` module, which internally uses a core component `MAEGradCore`. Given aligned predictions \(\phi_{\text{align}}\), ground truth \(\phi_{\text{gt}}\), an optional raw intensity tensor \(I_{\text{raw}}\), and a confidence mask `conf` (currently uniform ones), the core loss comprises:

1. **Absolute phase error (MAE):**
   \[
   L_{\text{MAE}} = \mathbb{E}\left[ \, |\phi_{\text{align}} - \phi_{\text{gt}}| \, \right].
   \]
2. **Gradient mismatch term:**
   - Sobel filters extract horizontal and vertical gradients of both prediction and ground truth.
   - The gradient error is the sum of absolute differences of these components:
     \[
     L_{\nabla} = \mathbb{E}\left[ \, |\nabla_x \phi_{\text{align}} - \nabla_x \phi_{\text{gt}}|
     + |\nabla_y \phi_{\text{align}} - \nabla_y \phi_{\text{gt}}| \, \right].
     \]
   - Optionally, this term can be weighted by \(\sqrt{I_{\text{raw}}}\), emphasizing regions of higher intensity if the `--int-wgrad` flag is enabled.

These components are combined using user-specified weights \(w_{\text{mae}}\) and \(w_{\nabla}\):

\[
L_{\text{core}} = w_{\text{mae}} L_{\text{MAE}} + w_{\nabla} L_{\nabla}.
\]

The wrapper `PhaseSupervisionLoss` also defines a *wrap-consistent* term based on sine and cosine of the phases, which would be appropriate if focusing on periodic phase behavior. However, this term is given weight \(w_{\text{wrap}} = 0\) in the default configuration, since the goal is to supervise absolute (non-wrapped) phase directly.

##### 4.4 Curvature regularization

To encourage smoothness of the predicted phase fields, the training objective includes a curvature penalty via `adaptive_curvature_loss`. This term applies a discrete Laplacian operator to the aligned prediction \(\phi_{\text{align}}\), measuring the magnitude of second-order variations:

\[
L_{\text{curv}} = \mathbb{E}\left[ \, |\Delta \phi_{\text{align}}| \, \right],
\]

again optionally modulated by a confidence mask (currently all ones). The curvature loss is scaled by a user-specified weight \(w_{\text{curv}}\).

##### 4.5 Total loss and optimization

The overall loss for each batch takes the form:

\[
L_{\text{total}} = w_{\text{data}} L_{\text{core}} + w_{\text{curv}} L_{\text{curv}},
\]

where \(w_{\text{data}}\) is a global multiplier on the supervised phase loss. This scalar loss is backpropagated using AdamW optimization, with:

- Optional mixed-precision training (AMP).
- Global gradient norm clipping to increase numerical stability.
- Cosine annealing of the learning rate over the training epochs, down to a configurable minimum.

An EMA copy of the model is maintained throughout training and is used for evaluation and visualization, which tends to yield smoother validation curves and less noisy qualitative predictions.

#### 5. Evaluation, Visualization, and Logging

##### 5.1 Quantitative evaluation

Validation is performed periodically (by default, every epoch) using the EMA model. For each batch in the validation set:

1. The EMA model predicts `phi_raw` and `k_off`, from which `phi_abs` is formed.
2. `affine_align` aligns `phi_abs` to the ground truth.
3. Metrics are computed on the aligned predictions:
   - **Mean Absolute Error (MAE)**.
   - **Root Mean Squared Error (RMSE)**.

These metrics are aggregated across the entire validation dataset by summing per-batch contributions weighted by batch size and dividing by the total number of samples.

##### 5.2 Qualitative visualization

To complement quantitative metrics, the script periodically generates visual summaries of validation predictions. For up to a specified number of samples per epoch, the following are plotted:

- The normalized interferogram \(I_{\text{norm}}\) (grayscale).
- The ground-truth absolute phase \(\phi_{\text{gt}}\) (colored with a perceptually meaningful colormap and accompanied by a colorbar).
- The aligned predicted absolute phase \(\phi_{\text{align}}\) from the EMA model.
- The error map \(\Delta \phi = \phi_{\text{align}} - \phi_{\text{gt}}\), visualized with a diverging colormap, alongside textual annotations of the mean and standard deviation of the error.

These figures are saved to disk in the designated visualization directory and provide an intuitive view of where and how the model succeeds or fails.

##### 5.3 Checkpointing and training curves

The script maintains two forms of checkpoints:

- A *best* checkpoint, updated whenever the validation MAE improves.
- A *rolling* checkpoint saved at the end of every epoch, capturing the latest state of both the main and EMA models.

In addition, a live training curve is maintained using Matplotlib, displaying:

- Training loss versus epoch.
- Validation MAE versus epoch.

At the end of training, this curve is saved as an image file for later inspection, even in headless environments where interactive plotting is not available.

#### 6. Conclusion

The `src/try.py` script implements a comprehensive and carefully designed pipeline for supervised absolute phase reconstruction from interferometric data. The key characteristics include:

- A dataset pipeline that normalizes interferograms and injects a global phase hint derived from the ground truth.
- A UNet-style architecture with Res2-inspired residual blocks, coordinate channels, and a dedicated global scalar offset head, together producing detailed absolute phase predictions.
- A training objective that aligns predictions to ground truth via affine fitting, supervises both phase values and their spatial gradients, and incorporates curvature-based smoothness regularization.
- Robust evaluation and visualization machinery, including EMA-based validation, scalar performance metrics, qualitative plots, and checkpoint management.

This design provides a solid foundation for further research and experimentation in phase unwrapping and interferometric reconstruction, while remaining modular enough to accommodate extensions such as confidence-based weighting, amplitude-aware losses, and alternative architectural variants.

