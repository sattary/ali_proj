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

## Quick Start

### 1. Generate Synthetic Data

```bash
uv run phase-unwrap generate \
    --num-samples 180000 \
    --shard-size 1000 \
    --out-dir data/full
```

### 2. Train

```bash
uv run phase-unwrap train \
    --data-dir data/full \
    --epochs 200 \
    --device auto
```

Resume from checkpoint:

```bash
uv run phase-unwrap train \
    --data-dir data/full \
    --resume runs/exp1/final.pth
```

### 3. Hyperparameter Search

```bash
uv run phase-unwrap tune \
    --data-dir data/full \
    --n-trials 50 \
    --tune-epochs 15
```

### 4. Generate Publication Figures

```bash
# Training curve
uv run phase-unwrap plot training-curve --run-dir runs/exp1

# Qualitative grid
uv run phase-unwrap plot qualitative \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full

# GradCAM interpretability
uv run phase-unwrap plot gradcam \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --layer enc5
```

### 5. Export Model

```bash
# ONNX
uv run phase-unwrap export-onnx \
    --checkpoint runs/exp1/best.pth \
    --out results/model.onnx

# TorchScript
uv run phase-unwrap export-torchscript \
    --checkpoint runs/exp1/best.pth \
    --out results/model.pt

# Latency benchmark
uv run phase-unwrap benchmark \
    --checkpoint runs/exp1/best.pth \
    --device cpu --n-runs 200
```

### 6. Evaluate Baselines

```bash
uv run phase-unwrap baselines \
    --checkpoint runs/exp1/best.pth \
    --n-samples 500
```

### 7. Noise Robustness Sweep

```bash
uv run phase-unwrap noise-sweep \
    --checkpoint runs/exp1/best.pth \
    --snr-min 5.0 --snr-max 40.0 --n-steps 8
```

### 8. Test-Time Augmentation

```bash
uv run phase-unwrap tta \
    --checkpoint runs/exp1/best.pth \
    --data-dir data/full \
    --n-augments 8
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

## Testing

```bash
uv run pytest tests/ -q
```

23 tests covering data generation, model architecture, loss functions, ops, and metrics.

---

## Multi-Seed Evaluation

```bash
uv run phase-unwrap multiseed \
    --data-dir data/full \
    --run-name ablation/baseline \
    --seeds "42,1337,7,100,2024"
```

Produces `runs/ablation/baseline/aggregate.csv` with mean and standard deviation per epoch across all seeds.

---

## LaTeX Tables

```bash
uv run phase-unwrap latex-table \
    --run-dir runs/exp1 \
    --out results/tables/metrics.tex
```

---

## Reproducibility

- All RNG states (Python, NumPy, PyTorch, CUDA) are saved and restored in every checkpoint
- Shard-level train/val split is deterministic for a given seed
- `uv.lock` pins all dependencies

---

## License

MIT
