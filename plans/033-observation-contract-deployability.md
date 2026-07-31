# Plan 033: Enforce one observation contract from training to deployment

> **Executor instructions:** Align all callers and exports with the contract from Plan 031. Do not retain an oracle input behind a convenience flag.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** HIGH
- **Depends on:** plans/031-research-contract-identifiability.md, plans/032-data-provenance-generator-benchmark.md
- **Category:** bug
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

Training, inference, HPO, baselines, TTA, and export currently disagree: PCLCN expects `(I_raw, grad_phi2)` and returns five values while inference still uses a two-channel UNet and two outputs. This makes reported results non-reproducible and can hide unavailable-reference leakage.

## Current state

Audit targets include `src/phase_unwrap/models`, `src/phase_unwrap/training/train.py`, `training/tune.py`, `analysis/inference.py`, `analysis/baselines.py`, and `scripts/eval_10_samples.py`.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Find signatures | `rg -n "PCLCNModel|grad_phi2|ema\.m|destructur|UNetRes2" src scripts tests` | every caller listed |
| Smoke tests | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/ -q -p no:cacheprovider` | all relevant tests pass |

## Steps

### Step 1: Define typed sample/model schemas

Create one canonical batch object for each acquisition mode. Make deployment inputs explicit and reject training-only fields at runtime. Standardize normalization, output tuple, phase gauge, and checkpoint metadata.

**Verify:** a unit test fails when `phi_gt`, latent parameters, or unavailable `grad_phi2` are passed to deployment inference.

### Step 2: Migrate every caller

Update training, HPO, validation, inference, TTA, GradCAM, export, baselines, and sample scripts to the canonical signature. Remove stale UNet destructuring only after replacement paths are tested.

**Verify:** `rg -n "ema\.m\(x_in\)|UNetRes2|np\.angle\(np\.exp\(1j \* interferogram\)" src analysis scripts` returns no active-path matches.

### Step 3: Fix geometry and size assumptions

Correct vector-component swaps/signs for 90°/270° rotations; make masks and projections derive from tensor dimensions rather than hardcoded 128×128.

**Verify:** augmentation tests validate all eight dihedral transforms; 32×32 and 128×128 projection tests pass.

## Scope

Callers, schemas, inference/export paths, augmentation geometry, and tests only. No architecture novelty or data-distribution changes.

## Done criteria

- [ ] One signature is used by train/eval/export.
- [ ] Oracle fields are rejected in deployment mode.
- [ ] All stale caller patterns are removed.
- [ ] Geometry and variable-size tests pass.

## STOP conditions

- A required deployment input cannot be measured under Plan 031.
- Removing a stale path would break an explicitly supported public interface.
- Output semantics differ across checkpoints and cannot be versioned.

## Maintenance notes

Add a schema version to every checkpoint and dataset manifest; reject mismatches instead of guessing.
