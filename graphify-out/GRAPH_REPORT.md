# Graph Report - .  (2026-05-31)

## Corpus Check
- Corpus is ~38,439 words - fits in a single context window. You may not need a graph.

## Summary
- 841 nodes · 1969 edges · 38 communities (37 shown, 1 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 107 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Cli Cmd Plot|Cli Cmd Plot]]
- [[_COMMUNITY_Generate Analysis Tests|Generate Analysis Tests]]
- [[_COMMUNITY_Data Dataset Augmentation|Data Dataset Augmentation]]
- [[_COMMUNITY_Visualize Utils Grid|Visualize Utils Grid]]
- [[_COMMUNITY_Training Train Gpu|Training Train Gpu]]
- [[_COMMUNITY_Gpu Multi Tests|Gpu Multi Tests]]
- [[_COMMUNITY_Core Analysis Visualize|Core Analysis Visualize]]
- [[_COMMUNITY_Visualize Style Method|Visualize Style Method]]
- [[_COMMUNITY_Analysis Latex Tta|Analysis Latex Tta]]
- [[_COMMUNITY_Model Unet Init|Model Unet Init]]
- [[_COMMUNITY_Git Automation State|Git Automation State]]
- [[_COMMUNITY_Model Visualize Tests|Model Visualize Tests]]
- [[_COMMUNITY_Training Tune Core|Training Tune Core]]
- [[_COMMUNITY_Git Automation Cli|Git Automation Cli]]
- [[_COMMUNITY_Losses Core Ops|Losses Core Ops]]
- [[_COMMUNITY_Git Automation Pusher|Git Automation Pusher]]
- [[_COMMUNITY_Git Automation Callback|Git Automation Callback]]
- [[_COMMUNITY_Loss Landscape Visualize|Loss Landscape Visualize]]
- [[_COMMUNITY_Tests Losses Ops|Tests Losses Ops]]
- [[_COMMUNITY_Core Config Ops|Core Config Ops]]
- [[_COMMUNITY_Environment Git Automation|Environment Git Automation]]
- [[_COMMUNITY_Zip Git Automation|Zip Git Automation]]
- [[_COMMUNITY_Training Ablation Multiseed|Training Ablation Multiseed]]
- [[_COMMUNITY_Tests Git Automation|Tests Git Automation]]
- [[_COMMUNITY_Visualize Training Curve|Visualize Training Curve]]
- [[_COMMUNITY_Git Automation Tune|Git Automation Tune]]
- [[_COMMUNITY_Core Config Loggingconfig|Core Config Loggingconfig]]
- [[_COMMUNITY_Visualize Multiseed Comparison|Visualize Multiseed Comparison]]
- [[_COMMUNITY_Gradcam Visualize Runner|Gradcam Visualize Runner]]
- [[_COMMUNITY_Gradcam Analysis Hook|Gradcam Analysis Hook]]
- [[_COMMUNITY_Visualize Convergence Figure|Visualize Convergence Figure]]
- [[_COMMUNITY_Noise Degradation Visualize|Noise Degradation Visualize]]
- [[_COMMUNITY_Git Lfs Setup|Git Lfs Setup]]
- [[_COMMUNITY_Error Histogram Visualize|Error Histogram Visualize]]
- [[_COMMUNITY_Profile Visualize Plot|Profile Visualize Plot]]
- [[_COMMUNITY_Core Config Training|Core Config Training]]
- [[_COMMUNITY_Main Ali Proj|Main Ali Proj]]

## God Nodes (most connected - your core abstractions)
1. `load_train_config()` - 50 edges
2. `build_model()` - 46 edges
3. `save_figure()` - 40 edges
4. `pick_device()` - 39 edges
5. `build_dataloaders()` - 39 edges
6. `nature_style()` - 38 edges
7. `affine_align()` - 37 edges
8. `train()` - 35 edges
9. `str` - 29 edges
10. `create_nature_palette()` - 28 edges

## Surprising Connections (you probably didn't know these)
- `Tensor` --uses--> `TrainConfig`  [INFERRED]
  tests/conftest.py → src/phase_unwrap/core/config.py
- `TrainConfig` --uses--> `TrainConfig`  [INFERRED]
  tests/conftest.py → src/phase_unwrap/core/config.py
- `TestAffineAlign` --uses--> `FixedSobel`  [INFERRED]
  tests/test_losses_ops.py → src/phase_unwrap/core/ops.py
- `TestFixedSobel` --uses--> `FixedSobel`  [INFERRED]
  tests/test_losses_ops.py → src/phase_unwrap/core/ops.py
- `TestMAEGradLoss` --uses--> `FixedSobel`  [INFERRED]
  tests/test_losses_ops.py → src/phase_unwrap/core/ops.py

## Import Cycles
- None detected.

## Communities (38 total, 1 thin omitted)

### Community 0 - "Cli Cmd Plot"
Cohesion: 0.05
Nodes (71): benchmark_inference(), export_onnx(), export_torchscript(), Benchmark mean latency and throughput., Export model to ONNX with dynamic spatial axes., Export model to TorchScript (traced) format., ablation_cmd(), data_generate() (+63 more)

### Community 1 - "Generate Analysis Tests"
Cohesion: 0.05
Nodes (49): evaluate_baselines(), evaluate_dl_baseline(), Classical phase unwrapping baselines., Evaluate the DL model on the same samples as the classical baselines., Least-squares 2D phase unwrapping via skimage., Itoh's 1D method: unwrap along rows, then columns., Evaluate classical methods on synthetic test data., _unwrap_itoh() (+41 more)

### Community 2 - "Data Dataset Augmentation"
Cohesion: 0.06
Nodes (40): NoiseAug, NoiseScheduler, Optical curriculum noise augmentation module.  Faithfully implements physical in, Apply noise mathematically, modifying Raw/Norm tensors gracefully., Piecewise noise level schedule over epochs.     warmup_epochs: noise=0 during [1, Calculate the normalized [0, 1] noise strength for the given epoch., Input-only augmentation with a schedulable 'level' in [0,1].     At level=0 -> n, Manually push the curriculum strength. (+32 more)

### Community 3 - "Visualize Utils Grid"
Cohesion: 0.07
Nodes (51): AxesImage, Colorbar, float, int, str, Tensor, float, int (+43 more)

### Community 4 - "Training Train Gpu"
Cohesion: 0.09
Nodes (41): EMA, FixedSobel, GradScaler, Optimizer, Phase unwrapping: UNetRes2-based absolute phase reconstruction.  Sub-package lay, SequentialLR, int, Module (+33 more)

### Community 5 - "Gpu Multi Tests"
Cohesion: 0.06
Nodes (25): bool, Tests for multi-GPU training support., Should get state dict from DataParallel model., Should load state dict into regular model., Should load state dict into DataParallel model., Should detect Kaggle with multiple GPUs., Should not detect if not Kaggle., Should not detect if Kaggle but only 1 GPU. (+17 more)

### Community 6 - "Core Analysis Visualize"
Cohesion: 0.14
Nodes (20): ONNX and TorchScript model export + inference benchmark., GradCAM visualization for UNetRes2., Enhanced noise robustness sweep with seaborn violin plots.  Shows full error dis, Test-time augmentation (TTA) for phase prediction., ConfigPath, AugmentationConfig, load_train_config(), Structured configuration for training and data generation.  All hyperparameters (+12 more)

### Community 7 - "Visualize Style Method"
Cohesion: 0.11
Nodes (27): float, str, float, int, ndarray, str, bool, int (+19 more)

### Community 8 - "Analysis Latex Tta"
Cohesion: 0.11
Nodes (28): _bold_best(), comparison_to_latex(), _fmt(), metrics_to_latex(), LaTeX table exporter from metrics CSV.  Reads metrics.csv or aggregate.csv and o, Multi-row comparison table across multiple runs.      Args:         run_dirs: Ma, Format a float or return '--' for NaN., Wrap the best value in \\textbf{}. (+20 more)

### Community 9 - "Model Unet Init"
Cohesion: 0.13
Nodes (17): Model sub-package: UNetRes2 architecture and EMA., AddCoords, meshgrid_ij(), UNetRes2 model for absolute phase reconstruction., Bilinear upsample + skip concatenation + Res2 block., UNet encoder-decoder with Res2 blocks and a global offset head.      Input:  [B,, Compatibility wrapper for older PyTorch meshgrid API., Concatenate normalized (x, y) coordinate channels to the input. (+9 more)

### Community 10 - "Git Automation State"
Cohesion: 0.11
Nodes (17): Check if training was completed., Track push state for resume functionality., Validate that state matches expected configuration.          Args:             e, Initialize state tracker.          Args:             run_dir: Path to the traini, Read the current state.          Returns:             State dict or None if no s, Write the current state.          Args:             epoch: Current epoch., Check if we should push at this epoch.          Args:             current_epoch:, Get the epoch to resume from.          Returns:             Epoch number to resu (+9 more)

### Community 11 - "Model Visualize Tests"
Cohesion: 0.09
Nodes (21): build_model(), Construct the model from configuration., ModelConfig, bool, float, int, str, bool (+13 more)

### Community 12 - "Training Tune Core"
Cohesion: 0.14
Nodes (23): config_to_yaml(), Serialize a TrainConfig to YAML string., curvature_loss(), L1 Laplacian smoothness penalty.      Args:         phi_pred: [B, 1, H, W] predi, ensure_dir(), Create directory (and parents) if it does not already exist., EMA, Exponential moving average shadow model. (+15 more)

### Community 13 - "Git Automation Cli"
Cohesion: 0.15
Nodes (17): AutoPushCallback, AutoPushCallback, Auto-push callback for training loop integration., Callback for auto-pushing during training.      Integrates with the training loo, add_auto_push_args(), create_auto_push_callback(), CLI integration for auto-push arguments., Create an AutoPushCallback if auto-push is enabled.      Args:         run_dir: (+9 more)

### Community 14 - "Losses Core Ops"
Cohesion: 0.15
Nodes (14): compute_metrics(), MAEGradLoss, Loss functions for absolute phase supervision., MAE and RMSE between absolute phase predictions and ground truth.      Both inpu, Supervised loss on absolute phase.      ``L = w_mae * |pred - gt| + w_grad * |gr, FixedSobel, Depthwise Sobel operator producing x and y gradients per channel., bool (+6 more)

### Community 15 - "Git Automation Pusher"
Cohesion: 0.18
Nodes (12): GitPusher, Handle git operations for auto-push., Configure remote URL with PAT for authentication., Verify the push succeeded by checking git status., Return to original branch., Get the current remote URL (sanitized)., Initialize git pusher.          Args:             repo_dir: Path to the git repo, Create and checkout the results branch. (+4 more)

### Community 16 - "Git Automation Callback"
Cohesion: 0.15
Nodes (11): Get PAT from environment or Kaggle secrets., Initialize git pusher and setup branch., Call at the end of each epoch.          Args:             epoch: Current epoch n, Delete intermediate checkpoint files to save space.         Keeps only best.pth, Call at the end of training.          Args:             final_metrics: Final met, Get current auto-push status., Initialize auto-push callback.          Args:             run_dir: Path to train, Any (+3 more)

### Community 17 - "Loss Landscape Visualize"
Cohesion: 0.20
Nodes (17): MAEGradLoss, device, float, int, Module, str, Tensor, _evaluate_loss() (+9 more)

### Community 18 - "Tests Losses Ops"
Cohesion: 0.12
Nodes (8): Tests for losses and ops., If pred == gt, alignment should return a=1, c=0., If pred = 2*gt + 5, alignment should recover gt., Each batch element should be aligned independently., Gradient of constant field should be ~0., TestAffineAlign, TestFixedSobel, TestMAEGradLoss

### Community 19 - "Core Config Ops"
Cohesion: 0.12
Nodes (15): DataConfig, LossConfig, ModelConfig, OptimizationConfig, Data-related configuration., Model and runtime configuration., Optimization hyperparameters., Weights for supervised and regularization terms. (+7 more)

### Community 20 - "Environment Git Automation"
Cohesion: 0.21
Nodes (15): get_environment_name(), is_cloud_environment(), is_colab(), is_kaggle(), Environment detection for cloud platforms., Detect if running in Google Colab environment., Detect if running in Kaggle or Colab., Get the name of the cloud environment. (+7 more)

### Community 21 - "Zip Git Automation"
Cohesion: 0.17
Nodes (10): Get path to the most recent zip file in the run directory., Delete all existing zip files in run directory to save space.         Keeps only, Initialize zip packer.          Args:             run_dir: Path to the training, Check if file should be included in zip., Collect all files to include in the zip., Create a zip file of the experiment artifacts.          Args:             epoch:, bool, int (+2 more)

### Community 22 - "Training Ablation Multiseed"
Cohesion: 0.17
Nodes (16): Any, int, str, TrainConfig, int, str, TrainConfig, _apply_overrides() (+8 more)

### Community 23 - "Tests Git Automation"
Cohesion: 0.16
Nodes (9): str, Tests for git_automation module., Create a mock git repository for testing., Should not make actual git calls in dry-run mode., Should not push if interval is None., Should push on final epoch regardless of interval., Should return a boolean., TestAutoPushCallback (+1 more)

### Community 24 - "Visualize Training Curve"
Cohesion: 0.19
Nodes (13): bool, float, int, ndarray, str, Sequential visualization orchestrator for cloud payload extraction.  Rationale:, _compute_convergence_epoch(), plot_training_curve() (+5 more)

### Community 25 - "Git Automation Tune"
Cohesion: 0.21
Nodes (11): create_tune_auto_push_callback(), Optuna auto-push callback extraction.  Rationale:     ARCHITECTURE: The `cli.py`, Create callback for pushing optuna results to GitHub after HPO completes., Zip packing for experiment artifacts., Pack experiment artifacts into a zip file., ZipPacker, bool, int (+3 more)

### Community 26 - "Core Config Loggingconfig"
Cohesion: 0.25
Nodes (9): _load_mapping(), LoggingConfig, Recursively update a dataclass instance from a nested mapping., Load a configuration mapping from JSON or YAML., Logging, output, and reproducibility options., _update_dataclass(), Any, Path (+1 more)

### Community 27 - "Visualize Multiseed Comparison"
Cohesion: 0.24
Nodes (10): float, int, str, plot_ablation_radar(), plot_multiseed_comparison(), Multi-seed comparison with seaborn boxen plots and swarm overlays.  For comparin, Radar chart for ablation study visualization.      Args:         results: Dict o, Compare multiple runs/ablations with multi-seed statistics.      Args:         r (+2 more)

### Community 28 - "Gradcam Visualize Runner"
Cohesion: 0.22
Nodes (9): plot_gradcam(), Generate GradCAM overlay visualization (3-column: input, heatmap, overlay)., int, str, int, Path, str, Central sequence to render downloaded Kaggle/Colab payload locally on CPU. (+1 more)

### Community 29 - "Gradcam Analysis Hook"
Cohesion: 0.25
Nodes (4): _GradCAM, GradCAM with forward/backward hooks on any target layer., Module, Tensor

### Community 30 - "Visualize Convergence Figure"
Cohesion: 0.29
Nodes (7): Figure, str, plot_convergence(), Convergence diagnostics: learning rate schedule and gradient norms.  Reads metri, Plot LR schedule and per-epoch train loss components., Save figure in multiple formats: PNG (quick view), PDF (LaTeX), SVG (editable)., save_figure()

### Community 31 - "Noise Degradation Visualize"
Cohesion: 0.29
Nodes (8): float, int, str, Tensor, add_awgn(), plot_noise_degradation(), Add Additive White Gaussian Noise to a tensor at target SNR., Nature-style iterative noise degradation evaluation.     Rows: SNR levels (inf,

### Community 32 - "Git Lfs Setup"
Cohesion: 0.33
Nodes (6): Git LFS Setup Utility for Phase Unwrap.  Run this before starting training to co, Run a git command and return results., Setup Git LFS for the repository., run_git_command(), setup_git_lfs(), bool

### Community 33 - "Error Histogram Visualize"
Cohesion: 0.33
Nodes (5): bool, str, plot_error_histogram(), Enhanced error distribution visualization with seaborn.  Features: - Violin plot, Enhanced error distribution with seaborn violin plots and statistical analysis.

### Community 34 - "Profile Visualize Plot"
Cohesion: 0.33
Nodes (5): int, str, plot_phase_profile(), Enhanced phase profile with seaborn regression and confidence bands.  1D cross-s, Enhanced 1D cross-section with seaborn regression and confidence intervals.

### Community 35 - "Core Config Training"
Cohesion: 0.50
Nodes (3): Top-level configuration aggregating all sub-configs., TrainConfig, Automated ablation study runner.

## Knowledge Gaps
- **79 isolated node(s):** `Figure`, `AxesImage`, `Colorbar`, `ndarray`, `int` (+74 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `AutoPushCallback` connect `Git Automation Cli` to `Cli Cmd Plot`, `Training Train Gpu`, `Git Automation State`, `Git Automation Pusher`, `Git Automation Callback`, `Git Automation Tune`?**
  _High betweenness centrality (0.155) - this node is a cross-community bridge._
- **Why does `build_model()` connect `Model Visualize Tests` to `Cli Cmd Plot`, `Generate Analysis Tests`, `Error Histogram Visualize`, `Visualize Utils Grid`, `Training Train Gpu`, `Profile Visualize Plot`, `Core Analysis Visualize`, `Visualize Style Method`, `Analysis Latex Tta`, `Model Unet Init`, `Training Tune Core`, `Loss Landscape Visualize`, `Gradcam Visualize Runner`, `Noise Degradation Visualize`?**
  _High betweenness centrality (0.087) - this node is a cross-community bridge._
- **Why does `train()` connect `Training Train Gpu` to `Cli Cmd Plot`, `Data Dataset Augmentation`, `Visualize Utils Grid`, `Core Analysis Visualize`, `Model Visualize Tests`, `Training Tune Core`, `Losses Core Ops`, `Core Config Ops`, `Training Ablation Multiseed`?**
  _High betweenness centrality (0.086) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `build_model()` (e.g. with `.test_ema_diverges()` and `.test_deterministic()`) actually correct?**
  _`build_model()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Entry point for the phase unwrapping CLI.`, `Phase unwrapping: UNetRes2-based absolute phase reconstruction.  Sub-package lay`, `Phase Unwrapping CLI - Hierarchical Command Structure.  Usage:     phase-unwrap` to the rest of the system?**
  _363 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Cli Cmd Plot` be split into smaller, more focused modules?**
  _Cohesion score 0.05289193302891933 - nodes in this community are weakly interconnected._
- **Should `Generate Analysis Tests` be split into smaller, more focused modules?**
  _Cohesion score 0.05028248587570622 - nodes in this community are weakly interconnected._