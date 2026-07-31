# Plan 038: Produce reproducible multi-seed evidence and the Optics Express manuscript

> **Executor instructions:** This is the final evidence phase. Do not rewrite the proposal around unverified numbers.

## Status

- **Priority:** P1
- **Effort:** M–L
- **Risk:** MED
- **Depends on:** plans/034-baselines-metrics-locked-evaluation.md, plans/036-physics-guided-architecture-ablation.md, plans/037-robustness-real-lab-transfer.md
- **Category:** docs
- **Planned at:** commit `18c93f5`, 2026-07-31

## Why this matters

The existing proposal and Plans 020–022 overstate certainty and contain historical leakage assumptions. The paper must distinguish identifiable acquisition from prior-conditioned inversion, report negative results, and make every number reproducible.

## Current state

`docs/proposal/main.tex` is an initial proposal; Plans 020–022 are historical architecture records. No result may be copied into the manuscript until it has a frozen split, selected checkpoint, provenance record, and confidence interval.

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Validate artifacts | `python scripts/validate_artifacts.py` | exit 0; all records have SHA/config/split/seed |
| Build figures | `python scripts/render_paper_figures.py` | deterministic figure manifest written |
| Compile manuscript | `latexmk -pdf -interaction=nonstopmode docs/proposal/main.tex` | PDF builds without undefined references |

## Steps

### Step 1: Run the frozen experiment matrix

Run at least three seeds for every primary baseline and named ablation on every locked split. Store immutable JSON/CSV records containing git SHA, config hash, dataset version, seed, checkpoint path, and metrics.

**Verify:** an artifact validator rejects missing provenance or duplicate test evaluations.

### Step 2: Generate paper figures and statistics

Create figures for the forward model, ambiguity counterexample, split distributions, qualitative phase/error/uncertainty maps, calibration curves, ablation table, and real/OOD transfer. Use bootstrap CIs and state the phase gauge for every metric.

**Verify:** figure generation runs from artifacts only in a clean environment and produces a manifest of inputs.

### Step 3: Rewrite `docs/proposal/main.tex`

Remove unsupported claims (“reviewer-proof,” unrestricted recovery, unsupervised losses), report the identifiability limitation, acquisition contract, baselines, OOD results, uncertainty policy, and negative findings. Cite the literature matrix and preserve the professor generator as a benchmark, not proof of universality.

**Verify:** `rg -n "reviewer-proof|unsupervised|unrestricted|0\.20|0\.63" docs/proposal/main.tex` returns no stale unsupported claim unless explicitly qualified.

## Done criteria

- [ ] Multi-seed, split-locked artifacts exist for all primary comparisons.
- [ ] Every paper number traces to an immutable artifact.
- [ ] Manuscript claims match identifiability and deployment evidence.
- [ ] Limitations and rejected approaches are included.

## STOP conditions

- Results depend on test-set tuning or unavailable metadata.
- The final model does not beat the simple baseline on frozen OOD/real data.
- Confidence intervals overlap and do not support the claimed contribution; narrow the claim.

## Scope

Experiment orchestration, artifact validation, figures, and manuscript text. Do not alter source implementation during writing.

## Maintenance notes

Archive exact manifests, logs, environment lockfiles, and commit SHAs with the submission supplement.
