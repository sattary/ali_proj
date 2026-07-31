# Plan 037: Calibrate robustness and transfer to real measurements

> **Executor instructions:** Do not claim real-image generalization without held-out instrument data and input-domain diagnostics.

## Status

- **Priority:** P1
- **Effort:** L
- **Risk:** HIGH
- **Depends on:** plans/032-data-provenance-generator-benchmark.md, plans/036-physics-guided-architecture-ablation.md
- **Category:** tests
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

The current noise curriculum is constructed but bypassed in the PCLCN training loop, and synthetic noise does not cover instrument mismatch. Real transfer is the strongest test of whether the model predicts new images rather than generator artifacts.

## Current state

`src/phase_unwrap/training/train.py` constructs `NoiseAug`/`NoiseScheduler` but the main PCLCN path calls the model directly; real-lab provenance and calibration protocol are not yet locked.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Locate augmentation path | `rg -n "NoiseAug|NoiseScheduler|prepare_batch" src/phase_unwrap` | training and visualization callers identified |
| Verify tests | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/ -q -p no:cacheprovider` | all relevant tests pass |

## Steps

### Step 1: Measure the sensor model

Estimate amplitude imbalance, visibility, shot/read noise, quantization, blur/MTF, saturation, background, and correlated speckle from calibration frames. Version the fitted ranges and use them only as deployment-plausible augmentation.

**Verify:** calibration report includes sample counts, fitted parameters, and held-out calibration error.

### Step 2: Re-enable and test augmentation

Ensure every training batch receives the configured noise/blur/photometric transforms; log effective distributions and seed them reproducibly.

**Verify:** a unit test changes the batch under nonzero augmentation and a training log records the applied policy.

### Step 3: Evaluate real and OOD data

Use held-out real frames, sensor/noise OOD, and structural OOD. Report uncertainty, abstention, rewrap residuals, and per-image failures. Add domain-shift diagnostics before inference.

**Verify:** real/OOD reports are generated without using real test labels for tuning.

## Done criteria

- [ ] Sensor model is measured and versioned.
- [ ] Noise curriculum is demonstrably active.
- [ ] Real/OOD performance and calibrated failure maps are reported.

## STOP conditions

- No ground-truth or trusted reference exists for real evaluation; report only unsupervised diagnostics and narrow the claim.
- Synthetic calibration cannot reproduce measured statistics.

## Scope

Calibration data, augmentation wiring, robustness evaluation, and domain diagnostics.

## Maintenance notes

Keep a frozen real-test set and calibration provenance; never tune on it.
