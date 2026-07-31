# Plan 031: Freeze the scientific contract and pass the identifiability gate

> **Executor instructions:** This is a design/research gate. Do not modify `src/` or run training. Record decisions and evidence in `plans/` only.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** HIGH
- **Depends on:** none
- **Category:** direction
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

The current single-frame on-axis measurement is `I=2+2 cos(Δφ)`, so `φ`, `-φ`, and `φ+2πk` are observationally identical. A high score on IID simulator samples can therefore be prior memorization rather than phase recovery. This plan freezes a defensible claim before any architecture work and prevents a publishability failure.

## Current state

- `scripts/image_generation.m` and `src/phase_unwrap/data/generate.py` generate on-axis intensity without a spatial carrier.
- `PCLCNModel` consumes `grad_phi2`, which is simulator metadata and is not available on ordinary deployment images.
- Plans 020–022 describe a reference-conditioned PCLCN, but call it final despite the above mismatch.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Confirm equation inputs | `rg -n "cos|interfer|grad_phi2|AnalyticSignalStem" scripts src` | matching generator/model lines |
| Inspect history | `git log -5 --oneline -- plans src` | commit history printed |

## Steps

### Step 1: Write the measurement and deployment contract

Create `plans/research-contract.md` containing the exact forward equation, units, phase gauge (piston/2π), available deployment inputs, and whether the experiment will use (a) calibrated off-axis carrier, (b) 3/4 phase-shifted frames, or (c) constrained-prior reference-free inversion. For each input mark `measured`, `known calibration`, or `training-only`; `phi_gt`, latent generator parameters, and `grad_phi2` must be training-only unless physically measured.

**Verify:** `test -s plans/research-contract.md` and `rg -n "measured|training-only|identifi" plans/research-contract.md`.

### Step 2: Run the novelty/failure-mode screen

Create `plans/literature-matrix.md` with at least Spoorthi et al., *PhaseNet 2.0* (IEEE TIP 2020, DOI `10.1109/TIP.2020.2977213`), Zuo et al., *Deep learning in optical metrology: a review* (Light: Science & Applications 2022, DOI `10.1038/s41377-022-00714-x`), and targeted searches for FIN/Fourier-inspired fringe analysis, off-axis digital holography, phase-shifting interferometry, uncertainty/OOD, and differentiable Poisson integration. For every candidate contribution record nearest prior art, failure mode, and what is genuinely new; do not claim novelty from an unsearched component.

**Verify:** `rg -n "DOI|failure mode|nearest prior|novel" plans/literature-matrix.md` returns all required columns.

### Step 3: Red-team the candidate claim

Create `plans/red-team-report.md` with adversarial tests: identical-intensity pairs `(φ,-φ)`, 2π-shift pairs, carrier/object-bandwidth overlap, unavailable reference gradients, and overconfident uncertainty. State the expected failure and the mitigation or abstention policy.

**Verify:** `rg -n "identical|bandwidth|abstain|counterexample" plans/red-team-report.md` finds each test.

## Scope

**In scope:** new research-contract and literature/red-team documents under `plans/`.

**Out of scope:** all implementation, dataset regeneration, model training, manuscript claims, and deleting Plans 020–030.

## Done criteria

- [ ] One acquisition mode is selected, or the project is explicitly narrowed to a constrained-prior claim.
- [ ] Every model input has a deployment provenance label.
- [ ] Literature matrix and red-team report contain URLs/DOIs and failure tests.
- [ ] A STOP decision is recorded if the task remains non-identifiable.

## STOP conditions

- No deployment-available information disambiguates sign and 2π branches.
- The proposed novelty is indistinguishable from an existing method after the literature screen.
- The professor generator cannot be preserved as a locked legacy benchmark.

## Maintenance notes

Treat this contract as an ADR. Any later architecture or loss change must cite the contract and update the red-team report.
