# Plan 036: Gate and ablate the physics-guided architecture

> **Executor instructions:** Implement only if Plans 034–035 show a measurable gap that a physics module can address.

## Status

- **Priority:** P1
- **Effort:** L
- **Risk:** HIGH
- **Depends on:** plans/034-baselines-metrics-locked-evaluation.md, plans/035-ambiguity-uncertainty-spike.md
- **Category:** direction
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

PCLCN currently combines an unsupported fixed-carrier analytic stem, reference-gradient conditioning, Zernike projection, residual CNN, Poisson integration, and losses that can hide 2π errors. A publishable architecture must earn each component through ablation and deployment validity.

## Current state

Plan 022 describes `AnalyticSignalStem`, `grad_phi2`, masked Zernike projection, residual CNN, DCT-Poisson integration, complex loss, and curl loss as one bundle. The audit found each requires an independent validity test.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Locate modules | `rg -n "AnalyticSignalStem|Poisson|Zernike|curl|ComplexDomainLoss|grad_phi2" src` | module inventory printed |
| Run ablations | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/ -q -p no:cacheprovider` | tests pass before experiments |

## Steps

### Step 1: Start with the minimal candidate

Implement direct phase-gradient correction followed by deterministic integration. Use raw calibrated observations or measured off-axis features only. Do not pass `grad_phi2` unless Plan 031 marks it physically measurable.

**Verify:** deployment smoke test runs without ground-truth or hidden generator metadata.

### Step 2: Add one module at a time

Evaluate analytic demodulation only after sideband-separation diagnostics; otherwise use learned features. Add integrability regularization, Poisson/DCT integration, Zernike prior, and residual capacity as separate ablations. Enforce orthogonal residual by an explicit projection or rename it.

**Verify:** each ablation has a config, parameter count, seed list, and frozen-test result.

### Step 3: Correct the loss semantics

Use supervised unwrapped/piston-aligned loss only when observable and labeled. Combine circular consistency, gradient error, curl/integrability, and calibrated uncertainty terms as justified. Never call a loss unsupervised if it compares to `phi_gt`; complex sine/cosine loss alone cannot penalize 2π mistakes.

**Verify:** fixtures show a 2π-shifted prediction is not accepted as equal by the primary metric when absolute phase is required.

## Done criteria

- [ ] Minimal model is the reference point.
- [ ] Every physics component has a predeclared hypothesis and ablation.
- [ ] Fixed-carrier and unavailable-reference paths are rejected or justified by measurements.
- [ ] Losses and architecture names match their mathematics.

## STOP conditions

- A module improves only IID professor data, not structural OOD.
- The claimed physics input is unavailable at deployment.
- The architecture is not better than the direct baseline at matched budget.

## Scope

Model, losses, ablations, and related tests/configs. No generator redesign or manuscript rewrite.

## Maintenance notes

Keep the full ablation matrix and do not remove failed components; reviewers will need the negative evidence.
