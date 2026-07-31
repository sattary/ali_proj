# Plan 035: Red-team discrete ambiguity and calibrated uncertainty

> **Executor instructions:** Treat this as a falsifiable spike. It may end with rejection; do not promote it to the final model by default.

## Status

- **Priority:** P1
- **Effort:** M
- **Risk:** HIGH
- **Depends on:** plans/034-baselines-metrics-locked-evaluation.md
- **Category:** direction
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

Plans 002–003 correctly identify sign/wrap ambiguity, but `arccos(I)` is invalid without amplitude/visibility calibration, pixelwise classes ignore spatial integrability, and entropy is not calibrated uncertainty. This spike tests whether ambiguity-aware prediction adds scientific value.

## Current state

`plans/002-discrete-ambiguity-network.md` proposes sign/wrap heads; `plans/003-evaluation-and-ood.md` proposes entropy/OOD metrics. Both are hypotheses, not validated observability results.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Locate ambiguity code | `rg -n "arccos|sign_gt|wrap_gt|entropy|ECE|Brier" src plans` | all existing assumptions listed |
| Run spike tests | `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest tests/ -q -p no:cacheprovider` | baseline plus spike tests pass |

## Steps

### Step 1: Derive observable features

For the selected acquisition, derive the normalized visibility equation and valid wrapped-phase feature. Define sign/wrap targets, class range, and an ambiguity mask; never label indistinguishable pairs as uniquely correct.

**Verify:** symbolic/numeric fixtures show identical observations receive identical ambiguity masks and no invalid `arccos` inputs.

### Step 2: Compare three heads

Train continuous regression, discrete sign/wrap prediction, and structured gradient prediction with spatial consistency/integrability. Use ordinal wrap encoding if the range is ordered. Include an abstention head or multimodal output when labels are ambiguous.

**Verify:** all variants run with identical splits, seeds, parameter budgets, and checkpoint protocol.

### Step 3: Calibrate and adversarially evaluate

Apply temperature scaling or isotonic calibration on validation only. Report NLL, Brier, ECE, reliability diagrams, and selective risk/coverage on identical-intensity pairs, OOD splits, and low-visibility regions.

**Verify:** calibration artifacts are generated and selective risk decreases as coverage is reduced; otherwise record rejection.

## Done criteria

- [ ] Observable target definitions are mathematically documented.
- [ ] Three variants and a simple baseline are compared fairly.
- [ ] Uncertainty is calibrated and failure cases are reported.
- [ ] A written go/no-go decision is recorded.

## STOP conditions

- The head is overconfident on indistinguishable pairs.
- It only reproduces simulator labels without OOD improvement.
- `arccos` requires unavailable amplitude/visibility information.

## Scope

Spike models, target derivations, calibration, and evaluation. No claim that this is the final architecture.

## Maintenance notes

Keep rejected variants and their failure plots; negative results are manuscript evidence.
