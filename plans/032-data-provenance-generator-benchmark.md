# Plan 032: Build a provenance-controlled generator and split benchmark

> **Executor instructions:** Implement only after Plan 031 selects an acquisition mode. Preserve the professor generator; never silently replace it.

## Status

- **Priority:** P1
- **Effort:** L
- **Risk:** HIGH
- **Depends on:** plans/031-research-contract-identifiability.md
- **Category:** migration
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

The current dataset is a narrow latent family with random-shard IID splits, so 180k samples can look excellent while failing structural OOD. This plan creates auditable data versions, MATLAB parity, and splits that measure new-image generalization.

## Current state

- `src/phase_unwrap/data/generate.py` samples a few scalars and a mostly 1-D deviation.
- `smart_split()` produces random shards rather than latent-family holdouts.
- Samples include `grad_phi2`; provenance and all latent parameters are not consistently persisted.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Generator tests | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/ -q -p no:cacheprovider` | baseline recorded before changes |
| Locate schema | `rg -n "save|load|grad_phi2|smart_split" src/phase_unwrap/data tests` | all schema callers listed |

## Steps

### Step 1: Lock the legacy benchmark

Add an immutable professor-equivalent dataset manifest containing generator commit, MATLAB/Python version, seed policy, dimensions, and checksum. Add a parity test on fixed seeds comparing intensity and target arrays within documented tolerances.

**Verify:** parity test passes for at least 20 fixed seeds; manifest checksum is reproducible twice.

### Step 2: Add an expanded generator as a separate version

Add true 2-D diversity: independent smooth surfaces, Zernike mixtures beyond the current 15-mode easy family, localized defects/discontinuities, varying amplitude/visibility, blur, shot/read/quantization noise, and anti-aliasing. Save every latent parameter, forward-model parameter, `sample_id`, `split_key`, `generator_version`, `sign_gt`, `wrap_gt`, and ambiguity/visibility masks. Define `sign_gt` only where the contract makes it observable.

**Verify:** a schema validator rejects missing fields; no deployment input contains `phi_gt` or latent parameters.

### Step 3: Create frozen splits

Create professor-IID, parameter-extrapolation, structural-OOD, sensor/noise-OOD, and real-lab (if available) manifests. Split by latent family/scene ID before augmentation; deduplicate by latent hash. Keep train/validation/test manifests immutable and versioned.

**Verify:** a script reports zero latent-hash overlap and non-overlapping scene IDs across splits.

### Step 4: Validate physical invariants

Test intensity range, amplitude/visibility normalization, Nyquist limits, finite-difference gradient conventions, and forward re-rendering. Reject samples with aliasing or undefined labels unless explicitly placed in an ambiguity split.

**Verify:** invariant report exits 0 and writes a JSON summary with counts and thresholds.

## Scope

Only generator/data schema, manifests, parity/invariant tests, and documentation. Do not implement a new network or tune losses here.

## Done criteria

- [ ] Legacy parity passes and manifest is reproducible.
- [ ] Expanded data are versioned separately with complete provenance.
- [ ] Five split types exist with zero latent leakage.
- [ ] Invariant and schema tests pass.

## STOP conditions

- MATLAB parity cannot be established.
- A split requires random samples from the same latent scene.
- Labels are asserted observable when the measurement equation proves they are not.

## Maintenance notes

Never overwrite an existing dataset version. Any generator change increments the version and regenerates manifests.
