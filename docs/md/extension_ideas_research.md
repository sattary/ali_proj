# Phase Unwrapping Project: Extension Ideas & Research Summary

## 1. State-of-the-Art Overview (2024–2026)

### Recent Advances

| Area | Method | Key idea |
|------|--------|----------|
| **Diffusion** | UnwrapDiff | DDPM for InSAR, conditional on SNAPHU; ~10% NRMSE reduction vs SNAPHU, robust in low-coherence regions |
| **Hybrid** | VOH-Net | CNN + LSTM encoder-decoder; composite loss (TV + variance + MSE) for phase continuity |
| **Unsupervised** | U³Net (CVPR 2024) | Recorruption-based self-reconstruction, wrapped-phase gradient loss, no GT needed |
| **Segmentation** | PhaseNet 2.0 | Wrap-count prediction via semantic segmentation; synthetic data with random shapes |
| **Attention** | PUnet | U-Net + attention + positional encoding; robustness to varying noise levels |
| **Gradient** | PGENet | Phase gradient estimation network; encoder-decoder for global phase features |

### Common Themes

- **Loss design**: Gradient continuity, TV regularization, wrapped-phase consistency (sin/cos), intensity-weighted gradients
- **Data**: Synthetic data with atmospheric effects, diverse noise, steep gradients
- **Robustness**: Low-coherence regions, sharp deformation gradients, 2π discontinuities

---

## 2. Extension Ideas (5–8 High-Impact, Colab-Friendly)

### Model / Loss / Research

#### 1. Wrapped-Phase Gradient Consistency Loss

**Description:** Add a loss term that constrains the *predicted* unwrapped phase so that its wrapped gradient matches the measured wrapped phase gradient. Uses the wrapping operator \(W(\nabla \phi_{pred})\) and compares to \(W(\nabla \phi_{wrapped})\). This encodes the physics that unwrapped gradients should agree with wrapped gradients modulo 2π.

**Rationale:** Literature (U³Net, PGENet) shows that gradient consistency in the wrapped domain improves robustness to noise and discontinuities. Your current gradient loss operates on absolute phase; this adds a complementary term that respects 2π periodicity.

**Implementation sketch:**
- **losses.py**: Add `wrapped_grad_loss(phi_pred, phi_wrapped)` using `FixedSobel`, wrapping operator `W(x) = atan2(sin(x), cos(x))`, and optional intensity weighting.
- **config.py**: Add `LossConfig.w_wrap_grad: float = 0.1`.
- **train.py**: Compute `phi_wrapped = torch.atan2(torch.sin(phi_gt), torch.cos(phi_gt))` from GT (or from interferogram if available), pass to loss.

**Risks / requirements:** Requires wrapped phase as input; current pipeline uses GT directly. Either derive wrapped from GT or add a wrapped-phase channel to the dataset. Minimal extra VRAM; no new deps.

---

#### 2. Total Variation (TV) Regularization

**Description:** Add a spatially adaptive TV term on the predicted phase: \(\sum_{i,j} |\nabla \phi|\) with optional weighting by intensity or a confidence map. Often used alongside phase gradient estimation in modern methods.

**Rationale:** TV promotes piecewise smoothness while allowing sharp discontinuities, which aligns well with phase maps that have step-like borders. Complements your existing curvature (Laplacian) term, which penalizes second-order variation.

**Implementation sketch:**
- **ops.py**: Add `tv_loss(phi: Tensor, conf_mask: Tensor | None) -> Tensor` using first-order finite differences.
- **losses.py**: Add optional TV term to `PhaseSupervisionLoss` or as a separate module.
- **config.py**: Add `LossConfig.w_tv: float = 0.0`.

**Risks / requirements:** Can over-smooth if weight is too high. `kornia` provides `total_variation` if desired; otherwise a few lines of `torch` suffice. No extra VRAM.

---

#### 3. Confidence-Weighted Loss (Use Existing Head)

**Description:** Your model already outputs `conf_logit`; use `sigmoid(conf_logit)` to weight the phase and gradient terms. Emphasize high-confidence regions and down-weight noisy/low-coherence areas.

**Rationale:** Enables the network to learn where it is uncertain and to focus supervision on reliable regions. Aligns with intensity-weighted gradient idea but makes it data-driven.

**Implementation sketch:**
- **losses.py**: In `MAEGradCore.forward`, replace `conf_used = torch.ones_like(...)` with `conf = torch.sigmoid(conf_logit)` passed from train; optionally add a KL or entropy regularizer so conf doesn’t collapse to 0.
- **train.py**: Pass `conf_logit` from model output into loss; optionally add `w_conf_reg` to discourage trivial conf maps.
- **config.py**: Add `LossConfig.use_conf_weight: bool = True`, `w_conf_reg: float = 0.01`.

**Risks / requirements:** Can collapse to uniform conf if regularizer is weak. Monitor conf distributions. No extra VRAM.

---

### Engineering / UX / Logging

#### 4. TensorBoard / Experiment Logging

**Description:** Log training loss, validation MAE/RMSE, learning rate, optional gradient norms, and sample images to TensorBoard. Optionally support Weights & Biases (wandb) for Colab-friendly cloud logging.

**Rationale:** Essential for research: compare runs, track hyperparameters, and debug training dynamics without polling files.

**Implementation sketch:**
- **config.py**: Add `LoggingConfig.use_tensorboard: bool = True`, `log_dir: str = "runs"`, optional `use_wandb: bool = False`.
- **train.py**: Create `SummaryWriter` (or `wandb.init`), log scalars each epoch, log a few validation images periodically.
- **cli.py**: Add `--no-tensorboard`, `--wandb` flags.

**Risks / requirements:** `tensorboard` is lightweight; `wandb` requires account and `wandb` package. Colab can run `%load_ext tensorboard` to view runs. Negligible VRAM.

---

#### 5. Config-Driven CLI Overrides

**Description:** Support overriding arbitrary config fields from the CLI (e.g. `--loss.w_curv 0.005`, `--optim.lr 1e-4`) using dot-notation. Useful for sweeps and Colab notebooks.

**Rationale:** Avoid editing JSON/YAML for quick experiments; keep config as single source of truth.

**Implementation sketch:**
- **cli.py**: Add `--override` / `-o` that accepts `key=value` pairs; parse nested keys like `loss.w_curv` and apply via `_update_dataclass` in `config.py`.
- **config.py**: Expose a helper `apply_overrides(cfg, overrides: list[str])` that parses and updates.

**Risks / requirements:** Type coercion (str→float, str→bool) must be handled; document supported keys. No extra deps.

---

### Data Pipeline & Augmentation

#### 6. Phase-Aware Augmentation

**Description:** Augment training data with geometric (flip, rotate 90°) and intensity (Gaussian noise on interferogram, contrast jitter) transforms. For phase: apply the same geometric transform to GT; optionally add synthetic wrapped-phase noise (small random phase shifts) to simulate decorrelation.

**Rationale:** PhaseNet and others use synthetic data with random shapes and noise to improve generalization. Your `.mat` data is limited; augmentation reduces overfitting and improves robustness to noise.

**Implementation sketch:**
- **data.py**: Add `TransformedMatPhaseDataset` wrapping `MatPhaseDataset` with `torchvision.transforms` or custom transforms. Apply `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(90)` (or k×90°); ensure I and phi are transformed together.
- **ops.py** or **data.py**: Add `add_interferogram_noise(I, sigma)` and `add_phase_noise(phi, sigma)` (wrapped).
- **config.py**: Add `DataConfig.augment: bool = True`, `aug_noise_sigma: float = 0.05`.

**Risks / requirements:** Geometric transforms are cheap; noise must be applied consistently to I and phi. No extra VRAM; `torchvision` is common.

---

#### 7. Additional Data Sources (HDF5 / NumPy)

**Description:** Extend `MatPhaseDataset` to also load from `.h5` / `.hdf5` and `.npy`/`.npz` with configurable keys. Allows integration of public benchmarks (e.g. InSAR-DLPU) and custom NumPy exports.

**Rationale:** Broader data compatibility supports reproducibility and comparison with published methods.

**Implementation sketch:**
- **data.py**: Add `discover_files(cfg)` that globs `*.mat`, `*.h5`, `*.hdf5`, `*.npz`; add `load_h5` and `load_npz` with key mapping; unify into a single `PhaseDataset` that dispatches by suffix.
- **config.py**: Add `DataConfig.h5_I_key`, `DataConfig.h5_phi_key`, `DataConfig.pattern` as list or extended glob.

**Risks / requirements:** Handle different array layouts (HDF5 often channel-last). `h5py` likely already in use for `.mat` v7.3. No extra VRAM.

---

### Inference & Visualization

#### 8. Inference Script with NRMSE & Residual Export

**Description:** Add a CLI command `infer` that loads a checkpoint, runs on a single file or directory, and writes predictions (and optional error maps) to disk. Extend metrics to NRMSE (standard in phase unwrapping papers) and optionally save residual maps (pred − GT) as `.npy` for external analysis.

**Rationale:** Enables evaluation on held-out data, demo notebooks, and deployment without the full training stack. NRMSE and residual maps support reproducibility and error analysis for papers.

**Implementation sketch:**
- **cli.py**: Add `infer` command with `--checkpoint`, `--input` (file or dir), `--output-dir`, `--format` (npy/mat/png), `--save-residuals`.
- **ops.py** or new **infer.py**: `run_inference(model, path, device)` returning `phi_pred`; optionally apply affine alignment if GT is available.
- **losses.py**: Add NRMSE to `compute_metrics`.
- **visualize.py**: Add `save_inference_comparison(phi_pred, phi_gt, path)`; optionally save residuals to `out_dir/residuals/`.
- **config.py**: Add `LoggingConfig.save_residuals: bool = False`.

**Risks / requirements:** Match preprocessing (z-score, phi_hint) with training. For inference without GT, phi_hint can use center pixel of a coarse unwrap or a user-provided value. Residual storage uses disk; make it optional. No extra deps.

---

## 3. Module Mapping Summary

| Idea | config.py | data.py | model.py | losses.py | ops.py | train.py | visualize.py | cli.py |
|------|-----------|---------|----------|-----------|--------|----------|--------------|--------|
| 1. Wrapped-phase grad loss | ✓ | - | - | ✓ | - | ✓ | - | - |
| 2. TV regularization | ✓ | - | - | ✓ | ✓ | ✓ | - | - |
| 3. Confidence-weighted loss | ✓ | - | - | ✓ | - | ✓ | - | - |
| 4. TensorBoard / wandb | ✓ | - | - | - | - | ✓ | - | ✓ |
| 5. Config CLI overrides | - | - | - | - | - | - | - | ✓ |
| 6. Phase-aware augmentation | ✓ | ✓ | - | - | ✓(?) | - | - | - |
| 7. HDF5 / NumPy support | ✓ | ✓ | - | - | - | - | - | - |
| 8. Inference + NRMSE & residuals | ✓ | ✓ | - | ✓ | - | ✓ | ✓ | ✓ |

---

## 4. Colab-Specific Notes

- **VRAM:** All ideas are compatible with single-GPU Colab (T4 ~15 GB). Diffusion (UnwrapDiff) is not recommended here; UNet-based extensions are fine.
- **Data:** `data/imgs` can be mounted from Drive; consider caching `.mat` in `/tmp` for faster epochs.
- **Logging:** TensorBoard in Colab: `%load_ext tensorboard` then `%tensorboard --logdir runs`. wandb works well with `wandb.init(project="phase-unwrap")`.
- **Checkpoints:** Save to Drive or download `best.pth` periodically to avoid losing work on disconnect.
