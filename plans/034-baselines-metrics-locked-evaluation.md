# Plan 034: Establish the locked baseline and evaluation protocol

> **Executor instructions:** No final architecture claims are allowed until this protocol is frozen and run on every split.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** MED
- **Depends on:** plans/032-data-provenance-generator-benchmark.md, plans/033-observation-contract-deployability.md
- **Category:** tests
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

The current classical baseline treats raw intensity as wrapped phase, and test evaluation can use final EMA instead of the validation-selected checkpoint. The paper needs valid baselines, per-image failures, confidence intervals, and a locked test protocol.

## Current state

`analysis/baselines.py` uses `np.angle(np.exp(1j * interferogram))`; current samples are `(I, phi, grad_phi2)`. Existing tests are mostly shape checks.

## Steps

### Step 1: Implement valid baselines

Use the actual wrapped-phase observation for classical quality-guided/unwrap and Fourier/Poisson baselines where acquisition supports them. Add a direct U-Net/ResUNet baseline with the same input contract and parameter budget. If reference-free data remain non-identifiable, label baselines as prior-conditioned rather than physical inversion.

**Verify:** baseline unit tests use synthetic known phases and recover the documented gauge within tolerance.

### Step 2: Freeze checkpoint selection

Select `best.pth` by validation metric only, record config hash/seed/data version, and evaluate the test set once with that checkpoint. Keep test manifests read-only.

**Verify:** an integration test proves changing test labels cannot change checkpoint selection.

### Step 3: Add metrics and statistics

Report circular/wrapped error, piston-aligned error where justified, gradient MAE, curl/residue count, rewrap consistency, P95/max error, per-image failure rate, calibration metrics, and bootstrap 95% CIs over images. Report mean±std over at least three seeds.

**Verify:** metric fixtures cover constant phase, 2π shifts, sign flips, residues, and NaNs.

## Done criteria

- [ ] Classical and direct neural baselines are valid for the observation mode.
- [ ] Best-checkpoint selection is test-independent.
- [ ] Locked metrics, CIs, and multi-seed aggregation are generated from manifests.

## STOP conditions

- A metric rewards a wrong 2π branch or sign; remove it from primary results.
- Test data influence tuning, normalization, or checkpoint selection.

## Scope

Baselines, metrics, statistics, evaluation scripts, and tests. No new physics module.

## Maintenance notes

The test manifest and metric definitions are part of the paper artifact; changes require a protocol version bump.
