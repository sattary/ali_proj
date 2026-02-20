# Phase Unwrapping via Absolute Phase Reconstruction

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **UNetRes2-based deep learning pipeline for two-dimensional absolute phase reconstruction from single interferograms.**

---

## Overview

This repository contains the full implementation of a supervised deep learning approach to 2D phase unwrapping. The model, **UNetRes2-AbsPhase**, uses a U-Net architecture augmented with Res2-style depthwise-separable residual blocks and a learned global offset head to directly regress absolute (unwrapped) phase from a normalized interferogram and a reference hint.

The framework includes:

- A physics-based synthetic data generator
- A complete training loop with AMP, EMA, cosine+warmup scheduling, and checkpoint resumption
- **Git auto-push for cloud training** (Kaggle/Colab) with configurable intervals
- Optuna-based hyperparameter optimisation
- Multi-seed aggregated evaluation
- Classical baseline comparison (Itoh 1D, least-squares 2D)
- Test-time augmentation (D4 symmetry group, 8 views)
- GradCAM interpretability visualisation
- Noise robustness sweep across SNR levels
- ONNX and TorchScript export
- Publication-quality LaTeX table generation
- Nature-style figure suite (qualitative grids, loss landscapes, error histograms, convergence curves)

---

## Repository Structure

```
src/phase_unwrap/
├── core/             # Configuration, mathematical ops, loss functions, utilities
│   ├── config.py     #   Dataclass-based config with YAML/JSON I/O
│   ├── ops.py        #   Fixed Sobel, affine alignment, curvature regularisation
│   ├── losses.py     #   MAE + gradient loss, RMSE/MAE metrics
│   └── utils.py      #   seed, device, directory helpers
│
├── data/             # Data pipeline
│   ├── dataset.py    #   HDF5-sharded lazy Dataset, DataLoader builder
│   └── generate.py   #   Physics-based synthetic interferogram generator
│
├── model/            # Neural network architecture
│   └── unet.py       #   UNetRes2-AbsPhase, Res2_DS_Block, EMA
│
├── training/         # Training orchestration
│   ├── train.py      #   Full training loop (AMP, EMA, warmup+cosine, resume)
│   ├── multiseed.py  #   N-seed runner with mean+/-std aggregation
│   ├── tune.py       #   Optuna TPE + MedianPruner HPO (SQLite resume)
│   └── ablation.py   #   Systematic ablation study runner
│
├── analysis/         # Evaluation and export tools
│   ├── baselines.py      #   Itoh 1D, least-squares 2D classical baselines
│   ├── noise_sweep.py    #   MAE-vs-SNR robustness sweep with error bars
│   ├── gradcam.py        #   GradCAM overlays (any encoder stage)
│   ├── tta.py            #   D4 test-time augmentation (8 views)
│   ├── export.py         #   ONNX + TorchScript export, inference benchmark
│   └── export_latex.py   #   LaTeX booktabs table generation
│
├── visualize/        # Publication-quality figures (Nature style)
│   ├── training_curve.py
│   ├── qualitative_grid.py
│   ├── phase_profile.py
│   ├── error_histogram.py
│   ├── loss_landscape.py
│   ├── convergence.py
│   └── style.py          #   Shared rcParams, colour palette, save helpers
│
├── git_automation/   # Cloud training with Git auto-push
│   ├── callback.py       #   AutoPushCallback for training integration
│   ├── git_pusher.py     #   Git operations with verification
│   ├── zip_packer.py     #   Artifact compression
│   ├── state_tracker.py  #   Resume state management
│   ├── environment.py    #   Kaggle/Colab detection
│   └── git_lfs_setup.py  #   One-time LFS configuration
│
└── cli.py            # Typer CLI entrypoint
```

---

## Method

### Synthetic Data Generation

Interferograms are generated analytically from a superposition of tilted planes, Gaussian bumps, and polynomial phase fields. The wrapped intensity field is:

```
I(x, y) = cos(phi(x, y) + noise)
```

The absolute phase `phi` serves as the regression target. This avoids the need for real annotated data and yields physically realistic fringe patterns.

### Model: UNetRes2-AbsPhase

- **Encoder**: 5 downsampling stages, each containing a `Res2_DS_Block` (depthwise-separable + hierarchical residual connections)
- **Bottleneck**: Deep Res2 block
- **Decoder**: Bilinear upsampling + skip concatenation + Res2 block per stage
- **Coordinate channels**: CoordConv (`AddCoords`) prepended to input for global position awareness
- **Output**: Per-pixel phase map (`phi_raw`) and a learned global offset scalar (`k_off`); combined as `phi = phi_raw + k_off`
- **EMA**: Exponential moving average shadow model (default decay 0.999) is used for all evaluations

### Loss Function

```
L = w_mae * |phi_pred - phi_gt| + w_grad * |grad(phi_pred) - grad(phi_gt)|
  + w_curv * Laplacian^2(phi_pred)
```

All predictions are affine-aligned to ground truth before metric computation to separate estimation from global scale/offset ambiguity.

---

## Installation

```bash
# Requires Python 3.10+, uv package manager
git clone https://github.com/sattary/ali_proj.git
cd ali_proj
uv sync
```

---

## Recommended Workflow

This workflow optimizes hyperparameters first, then trains the final model with tuned settings.

### Step 0: Generate Training Data

```bash
uv run phase-unwrap generate \
    --num-samples 180000 \
    --shard-size 1000 \
    --out-dir data/full
```

### Step 1: Hyperparameter Search

Find optimal learning rate, loss weights, batch size, and model capacity:

```bash
uv run phase-unwrap tune \
    --data-dir data/full \
    --n-trials 50 \
    --tune-epochs 15 \
    --n-workers 2 \
    --gpu-ids 0,1
```

**Parallel HPO:** Use `--n-workers 2 --gpu-ids 0,1` to run trials in parallel on multiple GPUs (~2x speedup on Kaggle).

This creates `runs/optuna/best_config.yaml` with the optimal configuration.

### Step 2: Train Final Model

Train with tuned hyperparameters for the full duration:

```bash
uv run phase-unwrap train \
    --data-dir data/full \
    --config runs/optuna/best_config.yaml \
    --run-name exp1
```

Resume from checkpoint if interrupted:

```bash
uv run phase-unwrap train \
    --data-dir data/full \
    --config runs/optuna/best_config.yaml \
    --resume runs/exp1/final.pth
```

### Step 3: Evaluate and Visualize

Generate publication-quality figures:

```bash
# Training progress
uv run phase-unwrap plot training-curve --run-dir runs/exp1

# Qualitative results
uv run phase-unwrap plot qualitative \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full

# Compare with classical baselines
uv run phase-unwrap baselines \
    --checkpoint runs/exp1/best.pth \
    --n-samples 500

# Test-time augmentation evaluation
uv run phase-unwrap tta \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full

# Noise robustness analysis
uv run phase-unwrap noise-sweep \
    --checkpoint runs/exp1/best.pth \
    --snr-min 5.0 --snr-max 40.0 --n-steps 8

# GradCAM interpretability
uv run phase-unwrap plot gradcam \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --layer enc5
```

### Step 4: Export for Production

```bash
# ONNX format
uv run phase-unwrap export-onnx \
    --checkpoint runs/exp1/best.pth \
    --out results/model.onnx

# TorchScript format
uv run phase-unwrap export-torchscript \
    --checkpoint runs/exp1/best.pth \
    --out results/model.pt

# Benchmark inference speed
uv run phase-unwrap benchmark \
    --checkpoint runs/exp1/best.pth \
    --device cpu --n-runs 200
```

---

## Cloud Training with Git Auto-Push

Train on Kaggle/Colab GPUs with automatic backups to GitHub. Perfect for long-running experiments (e.g., 10K epochs).

### Quick Setup

```bash
# 1. Setup Git LFS (one-time)
uv run python -m phase_unwrap.git_automation.git_lfs_setup

# 2. Configure Git
export GITHUB_PAT="your_personal_access_token"
git config user.email "you@example.com"
git config user.name "Your Name"

# 3. Train with auto-push (pushes every 1000 epochs)
#    Uses multi-GPU automatically on Kaggle (2x T4)
uv run phase-unwrap train \
    --epochs 10000 \
    --run-name exp_10k \
    --batch-size 40 \
    --auto-push-interval 1000 \
    --multi-gpu \
    --device cuda

# 4. Resume if interrupted
uv run phase-unwrap train \
    --resume runs/exp_10k/final.pth \
    --epochs 10000 \
    --run-name exp_10k \
    --batch-size 40 \
    --auto-push-interval 1000 \
    --multi-gpu \
    --device cuda
```

### Auto-Push Options

| Flag | Description | Default |
|------|-------------|---------|
| `--auto-push-interval N` | Push every N epochs (required to enable) | `None` (disabled) |
| `--auto-push-dry-run` | Test mode: no actual pushes | `False` |
| `--force-auto-push` | Enable outside Kaggle/Colab (for testing) | `False` |
| `--auto-push-pat TOKEN` | GitHub Personal Access Token | Uses `GITHUB_PAT` env var |

**Note:** Auto-push only works in Kaggle or Colab environments unless `--force-auto-push` is used.

### Multi-GPU Training (Kaggle 2x T4)

Train using both T4 GPUs on Kaggle for ~1.8x speedup:

```bash
# Use all available GPUs (auto-detected on Kaggle)
uv run phase-unwrap train \
    --epochs 10000 \
    --run-name exp_10k \
    --batch-size 40 \
    --multi-gpu

# Use specific GPUs
uv run phase-unwrap train \
    --epochs 10000 \
    --batch-size 40 \
    --multi-gpu \
    --gpu-ids "0,1"
```

**Batch Size Semantics:**
- `--batch-size 40` with `--multi-gpu` on 2 GPUs = 20 per GPU × 2 = 40 total
- The batch size you specify is the **total effective batch size**
- Each GPU processes `batch_size / num_gpus` samples

**Checkpoint Compatibility:**
- Single GPU → Multi-GPU: ✓ Works (state dict loads correctly)
- Multi-GPU → Single GPU: ✓ Works (unwraps automatically)
- Resume on different GPU count: ✓ Fully supported

---

## Complete CLI Reference

### Global Training Flags

```bash
phase-unwrap train \
    # Configuration
    --config PATH                    # YAML/JSON config file
    --data-dir PATH                  # Override data directory
    
    # Training parameters
    --epochs N                       # Number of training epochs
    --batch-size N                   # Batch size
    --device {auto,cuda,cpu}         # Device selection
    --run-name NAME                  # Run identifier
    --resume PATH                    # Resume from checkpoint
    
    # Auto-push (cloud training)
    --auto-push-interval N           # Push every N epochs
    --auto-push-dry-run              # Test auto-push setup
    --force-auto-push                # Force enable (testing)
    --auto-push-pat TOKEN            # GitHub PAT
    
    # Multi-GPU training
    --multi-gpu                      # Use all GPUs with DataParallel
    --gpu-ids "0,1"                  # Specific GPU IDs (default: all)
```

### Data Generation

```bash
phase-unwrap generate \
    --num-samples 180000             # Total samples to generate
    --shard-size 1000                # Samples per HDF5 file
    --out-dir data/full              # Output directory
    --seed 1337                      # RNG seed
```

### Hyperparameter Tuning

```bash
phase-unwrap tune \
    --data-dir data/full             # Training data location
    --n-trials 50                    # Number of Optuna trials
    --tune-epochs 15                 # Epochs per trial
    --study-name phase_unwrap_hpo    # Optuna study name
    --n-workers 2                    # Parallel workers (#GPUs)
    --gpu-ids 0,1                    # GPU IDs for parallel trials
```

**Note:** For multi-GPU HPO, use `--n-workers N --gpu-ids 0,1,...,N-1` instead of `--device cuda`. Each worker runs on its own GPU.

### Multi-Seed Evaluation

```bash
phase-unwrap multiseed \
    --config runs/optuna/best_config.yaml \
    --data-dir data/full \
    --run-name final_model \
    --seeds "42,1337,7,100,2024"     # Comma-separated seeds
    --num-seeds 5                    # Auto-generate N random seeds
    --epochs 200 \
    --device cuda
```

### Visualization Commands

```bash
# Training curves
phase-unwrap plot training-curve \
    --run-dir runs/exp1 \
    --out results/figs/training.png \
    --no-lr                          # Omit learning rate subplot

# Qualitative grid
phase-unwrap plot qualitative \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --out results/figs/grid.png \
    --n-samples 4

# Phase profile
phase-unwrap plot phase-profile \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --out results/figs/profile.png \
    --sample-idx 0

# Error histogram
phase-unwrap plot error-hist \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --out results/figs/errors.png

# Loss landscape
phase-unwrap plot loss-landscape \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --out results/figs/landscape.png \
    --grid-size 31 \
    --alpha-range 1.0 \
    --num-eval-samples 500

# Convergence
phase-unwrap plot convergence \
    --run-dir runs/exp1 \
    --out results/figs/convergence.png

# GradCAM
phase-unwrap plot gradcam \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --out results/figs/gradcam.png \
    --n-samples 4 \
    --layer enc5                     # Target encoder layer
```

### Analysis Commands

```bash
# Classical baselines comparison
phase-unwrap baselines \
    --checkpoint runs/exp1/best.pth \
    --n-samples 200 \
    --device cpu

# Test-time augmentation
phase-unwrap tta \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --n-augments 8                   # Number of augmentation views

# Noise robustness sweep
phase-unwrap noise-sweep \
    --checkpoint runs/exp1/best.pth \
    --out results/figs/noise.png \
    --snr-min 5.0 \
    --snr-max 40.0 \
    --n-steps 8 \
    --n-samples 100 \
    --device auto
```

### Export Commands

```bash
# ONNX export
phase-unwrap export-onnx \
    --checkpoint runs/exp1/best.pth \
    --out results/model.onnx \
    --opset 17

# TorchScript export
phase-unwrap export-torchscript \
    --checkpoint runs/exp1/best.pth \
    --out results/model.pt

# Benchmark inference
phase-unwrap benchmark \
    --checkpoint runs/exp1/best.pth \
    --device cpu \
    --batch-size 1 \
    --n-runs 200

# LaTeX table
phase-unwrap latex-table \
    --run-dir runs/final_model \
    --out results/tables/metrics.tex \
    --epoch -1                       # -1 = last epoch
```

---

## Alternative Workflows

### Quick Experiment (Skip Tuning)

Use default hyperparameters for rapid prototyping:

```bash
uv run phase-unwrap train --data-dir data/full --epochs 50 --run-name quicktest
```

### Publication-Ready Evaluation

Run multiple seeds and aggregate results:

```bash
uv run phase-unwrap multiseed \
    --data-dir data/full \
    --config runs/optuna/best_config.yaml \
    --run-name final_model \
    --seeds "42,1337,7,100,2024"

# Generate comparison table
uv run phase-unwrap latex-table \
    --run-dir runs/final_model \
    --out results/tables/metrics.tex
```

---

## Configuration

All hyperparameters are controlled via a single nested config object (`TrainConfig`). You can serialise and override it:

```bash
# Export current defaults
uv run phase-unwrap train --config path/to/config.yaml
```

Key fields:

| Group   | Field          | Default | Description                     |
| ------- | -------------- | ------- | ------------------------------- |
| `model` | `base`         | `32`    | Feature map base multiplier     |
| `model` | `activation`   | `relu`  | `relu` or `silu`                |
| `model` | `ema_decay`    | `0.999` | EMA decay                       |
| `optim` | `lr`           | `3e-4`  | Peak learning rate              |
| `optim` | `epochs`       | `200`   | Training epochs                 |
| `optim` | `warmup_steps` | `500`   | Linear warmup steps             |
| `loss`  | `w_mae`        | `1.0`   | MAE loss weight                 |
| `loss`  | `w_grad`       | `0.1`   | Gradient consistency weight     |
| `loss`  | `w_curv`       | `0.01`  | Curvature regularisation weight |

---

## Complete Example: Full Workflow with All Flags

Here's a comprehensive example showing a complete training workflow with all major features:

### Step 1: Environment Setup (Kaggle/Colab)

```python
# In your Kaggle/Colab notebook

# Clone and setup
!git clone -b alis_code https://github.com/sattary/ali_proj.git
%cd ali_proj
!pip install uv
!uv sync

# Setup Git LFS (one-time)
!uv run python -m phase_unwrap.git_automation.git_lfs_setup

# Configure Git with your PAT
import os
from getpass import getpass

# Option 1: Manual entry (less secure)
os.environ['GITHUB_PAT'] = getpass("Enter GitHub PAT: ")

# Option 2: Kaggle Secrets (recommended)
# from kaggle_secrets import UserSecretsClient
# os.environ['GITHUB_PAT'] = UserSecretsClient().get_secret("GITHUB_PAT")

!git config user.email "your-email@example.com"
!git config user.name "Your Name"
```

### Step 2: Generate Data with Custom Settings

```bash
uv run phase-unwrap generate \
    --num-samples 180000 \
    --shard-size 1000 \
    --out-dir data/kaggle_full \
    --seed 1337
```

### Step 3: Hyperparameter Search (Quick)

```bash
uv run phase-unwrap tune \
    --data-dir data/kaggle_full \
    --n-trials 30 \
    --tune-epochs 10 \
    --study-name kaggle_hpo \
    --n-workers 2 \
    --gpu-ids 0,1
```

**Parallel HPO:** Runs 2 trials simultaneously on 2 GPUs (~2x faster than single-GPU).

### Step 4: Full Training with All Flags

```bash
uv run phase-unwrap train \
    # Configuration
    --config runs/optuna/best_config.yaml \
    --data-dir data/kaggle_full \
    \
    # Training parameters
    --epochs 10000 \
    --batch-size 40 \
    --device cuda \
    --run-name exp_10k_full \
    \
    # Auto-push configuration (cloud training)
    --auto-push-interval 1000 \
    --auto-push-pat "$GITHUB_PAT"
```

### Step 5: Resume After Interruption

```bash
# If training is interrupted, resume from last checkpoint
uv run phase-unwrap train \
    --config runs/optuna/best_config.yaml \
    --data-dir data/kaggle_full \
    --epochs 10000 \
    --batch-size 40 \
    --device cuda \
    --run-name exp_10k_full \
    --resume runs/exp_10k_full/final.pth \
    --auto-push-interval 1000 \
    --auto-push-pat "$GITHUB_PAT"
```

### Step 6: Comprehensive Evaluation

```bash
# Generate all visualizations
uv run phase-unwrap plot training-curve --run-dir runs/exp_10k_full --out results/exp_10k/training.png
uv run phase-unwrap plot qualitative --checkpoint runs/exp_10k_full/best.pth --data-dir data/kaggle_full --out results/exp_10k/qualitative.png --n-samples 8
uv run phase-unwrap plot phase-profile --checkpoint runs/exp_10k_full/best.pth --data-dir data/kaggle_full --out results/exp_10k/profiles.png
uv run phase-unwrap plot error-hist --checkpoint runs/exp_10k_full/best.pth --data-dir data/kaggle_full --out results/exp_10k/errors.png
uv run phase-unwrap plot gradcam --checkpoint runs/exp_10k_full/best.pth --data-dir data/kaggle_full --out results/exp_10k/gradcam.png --layer enc5

# Compare with classical baselines
uv run phase-unwrap baselines --checkpoint runs/exp_10k_full/best.pth --n-samples 500 --device cuda

# Test-time augmentation
uv run phase-unwrap tta --checkpoint runs/exp_10k_full/best.pth --data-dir data/kaggle_full --n-augments 8

# Noise robustness
uv run phase-unwrap noise-sweep --checkpoint runs/exp_10k_full/best.pth --snr-min 5.0 --snr-max 40.0 --n-steps 10 --n-samples 200 --device cuda
```

### Step 7: Export for Production

```bash
# Export to multiple formats
uv run phase-unwrap export-onnx --checkpoint runs/exp_10k_full/best.pth --out results/exp_10k/model.onnx --opset 17
uv run phase-unwrap export-torchscript --checkpoint runs/exp_10k_full/best.pth --out results/exp_10k/model.pt

# Benchmark inference speed
uv run phase-unwrap benchmark --checkpoint runs/exp_10k_full/best.pth --device cpu --batch-size 1 --n-runs 1000
uv run phase-unwrap benchmark --checkpoint runs/exp_10k_full/best.pth --device cuda --batch-size 1 --n-runs 1000

# Generate LaTeX table for paper
uv run phase-unwrap latex-table --run-dir runs/exp_10k_full --out results/exp_10k/metrics.tex --epoch -1
```

### Testing Dry-Run Mode (Local Testing)

Before running on Kaggle, test the auto-push setup locally:

```bash
# Test with dry-run (no actual pushes)
uv run phase-unwrap train \
    --epochs 100 \
    --run-name dry_run_test \
    --batch-size 10 \
    --auto-push-interval 10 \
    --auto-push-dry-run \
    --force-auto-push \
    --data-dir data/kaggle_full
```

This will show you:
- Branch name that would be created
- When pushes would occur
- What would be committed
- Without actually pushing to GitHub

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -q

# Run specific test modules
uv run pytest tests/test_model.py -v
uv run pytest tests/test_git_automation.py -v

# Run with coverage
uv run pytest tests/ --cov=src/phase_unwrap --cov-report=html
```

31 tests covering data generation, model architecture, loss functions, ops, metrics, and git automation.

---

## Advanced Features

### Ablation Studies

Systematically evaluate the impact of architectural choices:

```python
from phase_unwrap.training.ablation import run_ablation
from phase_unwrap.core.config import TrainConfig

cfg = TrainConfig()
cfg.data.data_dir = "data/full"

ablations = {
    "no_curvature": {"loss.w_curv": 0.0},
    "no_gradient": {"loss.w_grad": 0.0},
    "double_base": {"model.base": 64},
    "silu_activation": {"model.activation": "silu"},
}

run_ablation(cfg, ablations, base_name="ablation_study")
```

### Custom Dataset

Use your own interferogram data by implementing a custom Dataset:

```python
from torch.utils.data import Dataset

class CustomInterferogramDataset(Dataset):
    def __getitem__(self, idx):
        # Load your interferogram I and ground truth phi
        I = ...  # [1, H, W] normalized interferogram
        phi = ...  # [1, H, W] unwrapped phase
        
        # Create hint from center value
        H, W = phi.shape[-2:]
        cy, cx = H // 2, W // 2
        phi_hint = torch.full_like(phi, phi[0, cy, cx].item())
        
        I_input = torch.cat([I, phi_hint], dim=0)  # [2, H, W]
        return I_input, phi, I
```

---

## Reproducibility

- All RNG states (Python, NumPy, PyTorch, CUDA) are saved and restored in every checkpoint
- Shard-level train/val split is deterministic for a given seed
- `uv.lock` pins all dependencies

---

## License

MIT
