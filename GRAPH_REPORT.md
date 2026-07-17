# Graph Report - /home/sat/Code/Fun/ali_proj  (2026-07-14)

## Corpus Check
- 77 files · ~50,051 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 555 nodes · 1295 edges · 56 communities (33 shown, 23 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 56 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8
- Community 9
- Community 10
- Community 11
- Community 12
- Community 13
- Community 14
- Community 15
- Community 16
- Community 17
- Community 18
- Community 19
- Community 20
- Community 21
- Community 22
- Community 23
- Community 24
- Community 25
- Community 26
- Community 27
- Community 28
- Community 29
- Community 30
- Community 31
- Community 32
- Community 33
- Community 34
- Community 35
- Community 36
- Community 37
- Community 38
- Community 39
- Community 40
- Community 41
- Community 42
- Community 43
- Community 44
- Community 45
- Community 46
- Community 47
- Community 48
- Community 49
- Community 50
- Community 51
- Community 52
- Community 53
- Community 54
- Community 55

## God Nodes (most connected - your core abstractions)
1. `load_inference_state()` - 40 edges
2. `save_figure()` - 39 edges
3. `nature_style()` - 38 edges
4. `affine_align()` - 37 edges
5. `build_dataloaders()` - 35 edges
6. `train()` - 30 edges
7. `create_nature_palette()` - 26 edges
8. `prepare_batch()` - 25 edges
9. `TrainConfig` - 23 edges
10. `load_train_config()` - 22 edges

## Surprising Connections (you probably didn't know these)
- `UNetRes2 Architecture` --semantically_similar_to--> `UNetRes2-AbsPhase`  [INFERRED] [semantically similar]
  /home/sat/Code/Fun/ali_proj/docs/proposal/main.pdf → /home/sat/Code/Fun/ali_proj/README.md
- `Hybrid Two-Cycle Stochastic Curriculum` --semantically_similar_to--> `Dynamic Curriculum Noise`  [INFERRED] [semantically similar]
  /home/sat/Code/Fun/ali_proj/docs/proposal/main.pdf → /home/sat/Code/Fun/ali_proj/README.md
- `sample_batch()` --calls--> `_build_grid()`  [INFERRED]
  tests/conftest.py → src/phase_unwrap/data/generate.py
- `sample_batch()` --calls--> `generate_sample()`  [INFERRED]
  tests/conftest.py → src/phase_unwrap/data/generate.py
- `default_config()` --references--> `TrainConfig`  [EXTRACTED]
  tests/conftest.py → src/phase_unwrap/core/config.py

## Import Cycles
- None detected.

## Communities (56 total, 23 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.11
Nodes (36): Figure, plot_convergence(), Convergence diagnostics: learning rate schedule and gradient norms.  Reads metri, Plot LR schedule and per-epoch train loss components., Visualize sub-package: Nature-level publication figures with seaborn.  Provides, plot_method_comparison(), plot_method_comparison_bar(), Method comparison radar chart and grouped bar plots.  Compares DL model against (+28 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (32): Tensor, Per-epoch training visuals (quick sanity-check PNGs).  This module is the intern, Legacy function for backward compatibility., Save quick per-sample visualizations during training.      Refactored to a 1-row, save_epoch_visuals(), save_epoch_visuals_simple(), plot_noise_comparison_grid(), Noise comparison grid showing clean vs noisy inference. Ensures consistency and (+24 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (18): compute_metrics(), MAEGradLoss, Tensor, MAE and RMSE between absolute phase predictions and ground truth.      Both inpu, Supervised loss on absolute phase.      ``L = w_mae * |pred - gt| + w_grad * |gr, FixedSobel, Depthwise Sobel operator producing x and y gradients per channel., Tests for losses and ops. (+10 more)

### Community 3 - "Community 3"
Cohesion: 0.12
Nodes (23): config_to_yaml(), Serialize a TrainConfig to YAML string., Core sub-package: configuration, math ops, losses, and utilities., Loss functions for absolute phase supervision., curvature_loss(), laplacian(), Tensor, Signal-processing operators for phase supervision.  Contains:     - FixedSobel: (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.10
Nodes (21): ConfigPath, AugmentationConfig, DataConfig, _load_mapping(), load_train_config(), LoggingConfig, LossConfig, OptimizationConfig (+13 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (19): benchmark_inference(), Benchmark mean latency and throughput., Real-world inference tool. Loads arbitrary size inputs, pads dynamically, and ru, run_inference(), ModelConfig, Model and runtime configuration., Unified inference and checkpoint loading utilities., pick_device() (+11 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (20): PathLike, comparison_to_latex(), Multi-row comparison table across multiple runs.      Args:         run_dirs: Ma, Top-level configuration aggregating all sub-configs., TrainConfig, ensure_dir(), Create directory (and parents) if it does not already exist., _apply_overrides() (+12 more)

### Community 7 - "Community 7"
Cohesion: 0.14
Nodes (19): AxesImage, Colorbar, affine_align(), Per-image affine fit: ``a * pred + c ~ gt`` (least squares).      Args:, build_dataloaders(), DataLoader, device, Construct training, validation, and test DataLoaders. (+11 more)

### Community 8 - "Community 8"
Cohesion: 0.17
Nodes (19): GradScaler, Optimizer, SequentialLR, Phase unwrapping: UNetRes2-based absolute phase reconstruction.  Sub-package lay, _append_csv(), _build_warmup_scheduler(), _init_csv(), load_checkpoint() (+11 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (10): AddCoords, meshgrid_ij(), Module, Tensor, Bilinear upsample + skip concatenation + Res2 block., Compatibility wrapper for older PyTorch meshgrid API., Concatenate normalized (x, y) coordinate channels to the input., Res2-style residual block with depthwise-separated channel groups.      1x1 expa (+2 more)

### Community 10 - "Community 10"
Cohesion: 0.11
Nodes (16): Entry point for the phase unwrapping CLI., data_generate(), eval_tta_cmd(), plot_method_comparison_cmd(), plot_multiseed_comparison_cmd(), plot_noise_comparison_cmd(), plot_prediction_scatter_cmd(), plot_training_curve_cmd() (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.20
Nodes (12): NoiseAug, prepare_batch(), Optical curriculum noise augmentation module.  Faithfully implements physical in, Prepare a batch [B, 1, H, W] for training or evaluation natively on the GPU., Input-only augmentation with a schedulable 'level' in [0,1].     At level=0 -> n, Visualize the curriculum noise progression across epochs., plot_error_histogram(), Enhanced error distribution visualization with seaborn.  Features: - Violin plot (+4 more)

### Community 12 - "Community 12"
Cohesion: 0.15
Nodes (8): Dataset, File, H5ShardDataset, Tensor, Context manager entry., Context manager exit - ensures all handles are closed., Dataset backed by multiple HDF5 shard files.      Returns per sample:         I_, Close all open HDF5 file handles.

### Community 13 - "Community 13"
Cohesion: 0.21
Nodes (15): plot_loss_landscape_cmd(), Plot 2D loss surface contour (Li et al., 2018 filter-normalized)., _evaluate_loss(), _get_parameters(), _perturb(), plot_loss_landscape(), device, Module (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.17
Nodes (10): Shuffle shard paths and split into train/val/test at shard level., smart_split(), _generate_shard_worker(), generate_to_h5(), Path, Synthetic interferogram data generator.  Faithful Python port of ``src/image_gen, Generate ``num_samples`` interferograms and write HDF5 shards.      Each shard c, Data sub-package: HDF5 dataset and synthetic data generator. (+2 more)

### Community 15 - "Community 15"
Cohesion: 0.24
Nodes (12): Analysis sub-package: baselines, noise robustness, GradCAM, TTA, export, tables., evaluate_tta(), _hflip(), predict_tta(), device, Module, Tensor, Test-time augmentation (TTA) for phase prediction. (+4 more)

### Community 16 - "Community 16"
Cohesion: 0.21
Nodes (12): evaluate_baselines(), evaluate_dl_baseline(), ndarray, Classical phase unwrapping baselines., Evaluate the DL model on the same samples as the classical baselines., Least-squares 2D phase unwrapping via skimage., Itoh's 1D method: unwrap along rows, then columns., Evaluate classical methods on synthetic test data. (+4 more)

### Community 17 - "Community 17"
Cohesion: 0.18
Nodes (11): plot_gradcam(), GradCAM visualization for UNetRes2., Generate GradCAM overlay visualization (3-column: input, heatmap, overlay)., load_inference_state(), DataLoader, device, Module, Loads model weights, parses configuration, and constructs dataloaders for infere (+3 more)

### Community 18 - "Community 18"
Cohesion: 0.27
Nodes (8): _build_grid(), generate_sample(), ndarray, Return meshgrid ``(x, y)`` and radial distance ``r2``., Generate a single (interferogram, dphi) pair.      Returns:         interferogra, Interferogram intensity = |E1+E2|^2 must be non-negative., Ground truth phase should be shifted so min = 0., TestGenerateSample

### Community 19 - "Community 19"
Cohesion: 0.24
Nodes (11): ablation_cmd(), _apply_cli_overrides(), multiseed_cmd(), Path, Train the UNetRes2 absolute phase reconstruction model., Run Optuna hyperparameter search (TPE + MedianPruner)., Run N training runs with different seeds, aggregate results., Run ablation study and produce LaTeX comparison table. (+3 more)

### Community 20 - "Community 20"
Cohesion: 0.20
Nodes (9): export_onnx(), export_torchscript(), ONNX and TorchScript model export + inference benchmark., Export model to ONNX with dynamic spatial axes., Export model to TorchScript (traced) format., export_onnx_cmd(), export_torchscript_cmd(), Export model to ONNX format. (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.22
Nodes (9): _bold_best(), _fmt(), metrics_to_latex(), LaTeX table exporter from metrics CSV.  Reads metrics.csv or aggregate.csv and o, Format a float or return '--' for NaN., Wrap the best value in \\textbf{}., Convert metrics CSV to a LaTeX booktabs table.      Args:         run_dir:  Run, export_table_cmd() (+1 more)

### Community 22 - "Community 22"
Cohesion: 0.20
Nodes (5): Tests for model architecture., Model should handle any spatial size divisible by 16., Verify gradients propagate to all parameters., TestEMA, TestUNetRes2

### Community 23 - "Community 23"
Cohesion: 0.39
Nodes (8): Plan 001: Train Integration Tests, Plan 002: Fix EMA Update Condition, Plan 003: Prefetch Val Vis Batch, Plan 004: Optimize HDF5 DataLoader, Plan 005: Refactor CLI Layering, Plan 006: Allow Zero Splits, Plan 012: OOM Recovery Reproducibility, Plans Index

### Community 24 - "Community 24"
Cohesion: 0.25
Nodes (4): _GradCAM, Module, Tensor, GradCAM with forward/backward hooks on any target layer.

### Community 25 - "Community 25"
Cohesion: 0.38
Nodes (7): Affine Alignment, Phase Hint Channel, Master's Project: Phase Unwrapping, Plan 007: Affine Align Scale Leakage, Plan 008: Reproducibility Determinism CI, Plan 009: Phi Hint Input Leakage, Plan 010: Selection and Reporting Metric

### Community 26 - "Community 26"
Cohesion: 0.33
Nodes (6): _add_gaussian_noise(), noise_robustness_sweep(), ndarray, Enhanced noise robustness sweep with seaborn violin plots.  Shows full error dis, Add Gaussian noise to interferogram at a given SNR (dB)., Sweep SNR levels with enhanced seaborn visualization.      Features:     - Violi

### Community 27 - "Community 27"
Cohesion: 0.29
Nodes (6): default_config(), Tensor, Test fixtures shared across the test suite., Minimal config for unit tests (small model, CPU)., Synthetic batch: (I_input, phi_gt, I_raw), each shape [2, 1, 128, 128]., sample_batch()

### Community 28 - "Community 28"
Cohesion: 0.33
Nodes (3): NoiseScheduler, Piecewise noise level schedule relative to total epochs.     warmup_ratio: noise, Calculate the normalized [0, 1] noise strength for the given epoch.

### Community 29 - "Community 29"
Cohesion: 0.40
Nodes (5): plot_curriculum_noise_cmd(), Plot dynamic curriculum noise progression over epochs., discover_h5_shards(), Discover H5 shard files under data_dir matching the given pattern., plot_curriculum_noise()

### Community 30 - "Community 30"
Cohesion: 0.50
Nodes (4): Audit Playbook, Closing the Loop, Plan Template, Improve Skill

### Community 32 - "Community 32"
Cohesion: 0.67
Nodes (3): add_awgn(), Tensor, Add Additive White Gaussian Noise to a tensor at target SNR.

### Community 33 - "Community 33"
Cohesion: 0.67
Nodes (3): _compute_convergence_epoch(), ndarray, Find epoch where metric stabilizes within threshold.

## Knowledge Gaps
- **18 isolated node(s):** `ali-proj`, `Audit Playbook`, `Closing the Loop`, `Plan Template`, `UNetRes2-AbsPhase` (+13 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **23 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_inference_state()` connect `Community 17` to `Community 0`, `Community 1`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 11`, `Community 13`, `Community 15`, `Community 16`, `Community 20`, `Community 26`?**
  _High betweenness centrality (0.094) - this node is a cross-community bridge._
- **Why does `train()` connect `Community 8` to `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 10`, `Community 11`, `Community 28`?**
  _High betweenness centrality (0.082) - this node is a cross-community bridge._
- **Why does `build_model()` connect `Community 5` to `Community 3`, `Community 8`, `Community 17`, `Community 20`, `Community 22`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `affine_align()` (e.g. with `.test_batch_independence()` and `.test_identity()`) actually correct?**
  _`affine_align()` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `ali-proj`, `Audit Playbook`, `Closing the Loop` to the rest of the system?**
  _18 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.11074197120708748 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.12380952380952381 - nodes in this community are weakly interconnected._