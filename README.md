# Phase Unwrapping via Absolute Phase Reconstruction

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **UNetRes2 pipeline for 2D continuous phase reconstruction from a single interferogram.**

**Protocol (Option B, locked):** training and inference use the same inputs — normalized intensity plus a **zero** second channel (`hint_mode=zero`). No ground-truth absolute piston is fed into the network. Checkpoint selection uses raw **AbsMAE**. Reported `TopoMAE` is **piston-only** aligned (no GT scale fit). Old GT-center runs are not comparable.

---

## Overview

Supervised deep learning for 2D phase reconstruction from synthetic interferograms:

- Physics-based HDF5 data generator (HeNe two-beam model)
- UNetRes2 + CoordConv + global offset head + multi-scale deep supervision (train)
- Curriculum optical noise (speckle, blur, photometric jitter, …)
- AMP, EMA, warmup + cosine LR, resume
- Optuna HPO, multi-seed, ablations
- Classical baselines (Itoh 1D, least-squares 2D)
- TTA, GradCAM, noise sweeps, ONNX / TorchScript export
- Nature-style figure helpers

CLI provides top-level commands for core tasks (`generate`, `train`, `tune`) and nested commands for tools (`eval`, `export`, `plot`).

---

## Repository Structure

```
src/phase_unwrap/
├── core/         # config, ops (piston_align), losses, utils, inference loaders
├── data/         # HDF5 dataset, generate, augmentation (hint_mode, curriculum noise)
├── model/        # UNetRes2_AbsPhase, EMA
├── training/     # train, tune, multiseed, ablation
├── analysis/     # baselines, noise_sweep, gradcam, tta, export, latex
├── visualize/    # publication figures
└── cli.py        # Typer entrypoint
```

---

## Method (current code)

### Data

Synthetic intensity / absolute-phase pairs written as HDF5 shards. Target phase is min-shifted so `min(phi)=0`.

### Model input (Option B)

`I_input = [I_norm, phi_hint]` with `phi_hint = 0` by default (`AugmentationConfig.hint_mode = "zero"`).  
`hint_mode="gt_center"` remains only as an explicit **ablation** of the old leak (not for deploy).

### Model

- U-Net + Res2 depthwise-separable blocks
- Default activation: **SiLU** (also supports relu / mish)
- CoordConv, multi-scale heads in train mode, `phi = phi_raw + k_off`
- EMA shadow model for eval

### Loss

```
L = w_mae * |pred - gt| + w_grad * |grad(pred) - grad(gt)| + w_curv * |Laplacian(pred)|
```

### Metrics

- **AbsMAE**: raw absolute error (primary; selects `best.pth`)
- **TopoMAE** (CSV name kept): **piston-only** mean alignment — does **not** fit scale to GT
- Val/test geometric flips/rots are **off**

---

## Installation

```bash
# Python 3.12+, uv
git clone https://github.com/sattary/ali_proj.git
cd ali_proj
uv sync
```

---

## Recommended Workflow

### 0. Generate data

```bash
uv run phun generate \
    --num-samples 180000 \
    --shard-size 1000 \
    --out-dir data/full
```

### 1. Hyperparameter search (optional)

```bash
uv run phun tune \
    --data-dir data/full \
    --n-trials 50 \
    --tune-epochs 15
```

Writes `runs/optuna/best_config.yaml` when configured by the tune path.

### 2. Train (Option B defaults)

```bash
uv run phun train \
    --data-dir data/full \
    --config runs/optuna/best_config.yaml \
    --run-name exp_option_b
```

Resume:

```bash
uv run phun train \
    --data-dir data/full \
    --config runs/optuna/best_config.yaml \
    --run-name exp_option_b \
    --resume runs/exp_option_b/final.pth
```

After training, check `runs/exp_option_b/test_metrics.csv` (held-out test) and `metrics.csv` (`val_abs_mae`).

### 3. Evaluate and plot

```bash
uv run phun plot training-curve --run-dir runs/exp_option_b
uv run phun plot qualitative \
    --checkpoint runs/exp_option_b/best.pth \
    --data-dir data/full
uv run phun eval baselines \
    --checkpoint runs/exp_option_b/best.pth \
    --n-samples 200
uv run phun eval tta \
    --checkpoint runs/exp_option_b/best.pth \
    --data-dir data/full
uv run phun eval noise \
    --checkpoint runs/exp_option_b/best.pth \
    --snr-min 5.0 --snr-max 40.0 --n-steps 8
uv run phun plot gradcam \
    --checkpoint runs/exp_option_b/best.pth \
    --data-dir data/full \
    --layer enc5
```

### 4. Export

```bash
uv run phun export onnx \
    --checkpoint runs/exp_option_b/best.pth \
    --out results/model.onnx
uv run phun export torchscript \
    --checkpoint runs/exp_option_b/best.pth \
    --out results/model.pt
uv run phun eval benchmark \
    --checkpoint runs/exp_option_b/best.pth \
    --device cpu --n-runs 200
```

### Lab / real interferogram

```bash
uv run phun infer \
    --checkpoint runs/exp_option_b/best.pth \
    --input path/to/lab_image.png \
    --out results/lab_phi.npy
```

Inference builds a **zero** hint channel — same as training under Option B.

---

## CLI surface

```bash
uv run phun --help
# generate | train | tune | multiseed | ablation | infer | eval | export | plot
```

Examples:

```bash
uv run phun multiseed --data-dir data/full --run-name ms --seeds "42,1337,7"
uv run phun ablation --config path/to/config.yaml --data-dir data/full --run-name abl
```

---

## Configuration

`TrainConfig` YAML/JSON. Important fields:

| Group | Field | Default | Notes |
|-------|--------|---------|--------|
| `model` | `base` | `32` | Channel multiplier |
| `model` | `activation` | `silu` | `relu` \| `silu` \| `mish` |
| `model` | `ema_decay` | `0.999` | |
| `model` | `use_coordconv` | `true` | |
| `optim` | `lr` | `3e-4` | Peak LR |
| `optim` | `epochs` | `200` | |
| `loss` | `w_mae` / `w_grad` / `w_curv` | `1.0` / `0.1` / `0.01` | |
| `data` | `augment` | `true` | Geometric D4 on **train only** |
| `aug` | `hint_mode` | `zero` | Option B; use `gt_center` only for leak ablations |
| `aug` | `enable` | `true` | Curriculum optical noise |

`configs/default.yaml` is a **smoke** config (smaller epochs/base), not a full match to code defaults. Prefer code defaults or Optuna `best_config.yaml` for real runs.

---

## Testing

```bash
uv run pytest tests/ -q
```

~34 tests: generate, losses/ops (incl. piston_align), model/EMA, hint (Option B), dataset augment, analysis CLI smoke.

---

## Plans and science notes

Implementation plans live in `plans/` (methodology 007–010, 013–014 done for Option B).  
**Option A** (lab-measurable reference channel) is deferred: `plans/015-option-a-reference-guided-fallback.md`.

Synthetic success does **not** guarantee lab transfer. Validate on real interferograms with re-wrap consistency before trusting numbers in a paper.

---

## Reproducibility

- Checkpointed RNG states (Python / NumPy / PyTorch / CUDA)
- Deterministic shard-level splits for a given seed
- `uv.lock` pins dependencies

---

## License

MIT
