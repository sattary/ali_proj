# Plan 030: Modernize & Refactor CLI with Flat Architecture & Purge Plot Clutter (Ponytail)

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat 6bb8d6f..HEAD -- src/phase_unwrap/cli.py src/phase_unwrap/cli_cmds/ src/phase_unwrap/plots/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P1
- **Effort**: M
- **Risk**: LOW
- **Depends on**: plans/029-modernize-pclcn-ablation-and-multiseed-framework.md
- **Category**: tech-debt
- **Planned at**: commit `6bb8d6f`, 2026-07-30

## Why this matters

The existing CLI structure uses nested Typer sub-applications (`phase-unwrap train train`, `phase-unwrap data generate`, `phase-unwrap config show`) and contains 12+ redundant pre-PCLCN plot generation scripts with broken legacy imports. Flattening the CLI into direct top-level commands (`phase-unwrap generate`, `phase-unwrap train`, `phase-unwrap ablation`, `phase-unwrap multiseed`, `phase-unwrap eval`, `phase-unwrap export`, `phase-unwrap infer`, `phase-unwrap plot`) and purging redundant plot scripts eliminates technical debt, removes broken `UNetRes2` dependencies, and streamlines Kaggle GPU workflows.

---

## Double Adversarial Analysis Report

To ensure maximum scientific rigor for *Optics Express* while maintaining zero bloat, a two-pass adversarial analysis was performed on all 16 plot generators in `src/phase_unwrap/plots/`:

### Pass 1: Critical Analysis (Scientific Value & Code Health)
- **`fig1_architecture.py`**: Stateless numpy renderer for main text Fig. 1 (Interferograms, GT phase, wrapped phase, PCLCN prediction, error map). **(Kept)**
- **`fig2_baseline_comparison.py`**: Stateless numpy renderer for main text Fig. 2 (PCLCN vs Itoh 1D and 2D DCT Least-Squares). **(Kept)**
- **`fig3_noise_robustness.py`**: Statistical degradation curves (SNR vs MAE violins + line plots) and error shift grids. **(Kept)**
- **`fig4_ablation.py`**: Boxen plots and radar charts for PCLCN Fast Core Matrix (`no_reference_prior`, `no_curl_loss`, `unmasked_zernike`). **(Kept - Refactored)**
- **`phase_profile.py`**: Uses broken `UNetRes2` signature `phi_raw, k_off = model(...)` and `affine_align` (leaked gain). **(Flagged for Purge)**
- **`method_comparison.py`**: Duplicated by `fig2_baseline_comparison.py`. **(Flagged for Purge)**
- **`noise_degradation_grid.py` & `curriculum_noise_grid.py`**: Duplicated by `fig3_noise_robustness.py`. **(Flagged for Purge)**
- **`training_curve.py`, `convergence.py`, `loss_landscape.py`**: Exploratory debugging plots. **(Flagged for Purge)**
- **`gradcam_overlay.py`, `error_histogram.py`, `inference_output.py`, `prediction_scatter.py`, `residual_analysis.py`**: Ad-hoc scripts with UNetRes2 dependencies. **(Flagged for Purge)**

### Pass 2: Meta-Critical Challenge (Preserving Useful Optics Utilities)
- *Challenge:* Is 1D line-cut phase profiling useful for supplementary material?
- *Verdict:* Yes, but `phase_profile.py` is broken. Rather than retaining broken files, we integrate a clean, stateless `draw_line_profile_panel()` primitive into `src/phase_unwrap/plots/utils.py`.
- *Result:* **12 legacy files purged**. Exclusively 4 SOTA paper figures (`fig1`-`fig4`), `style.py`, `utils.py`, `_epoch_visuals.py`, and `__init__.py` survive.

---

## Current state

The CLI is currently structured as nested Typer sub-apps:
- `src/phase_unwrap/cli.py`: Imports and mounts 7 sub-apps (`DataApp`, `TrainApp`, `EvalApp`, `ExportApp`, `PlotApp`, `ConfigApp`, `InferApp`).
- `src/phase_unwrap/cli_cmds/`: Contains `config.py` (yagni), `data.py`, `eval.py`, `export.py`, `infer.py`, `plot.py`, `train.py`, and `_plot_runners.py`.
- `src/phase_unwrap/plots/`: Contains 20 plot files, 12 of which are legacy pre-PCLCN clutter.

Exemplar flat Typer CLI pattern:
Use single `typer.Typer(add_completion=False)` with top-level `@app.command()` decorators.

## Commands you will need

| Purpose   | Command                         | Expected on success |
|-----------|---------------------------------|---------------------|
| CLI test  | `uv run phase-unwrap --help`   | Displays flat commands list (exit 0) |
| Tests     | `uv run python -m pytest tests/ -q` | 52 passed in ~25s |

## Scope

**In scope** (files to modify or delete):
- `src/phase_unwrap/cli.py` (refactor to flat commands)
- `src/phase_unwrap/cli_cmds/` (purge obsolete `config.py` and `_plot_runners.py`, consolidate into clean flat modules)
- `src/phase_unwrap/plots/` (purge 12 legacy plot files, retain `fig1`-`fig4`, `style.py`, `utils.py`, `_epoch_visuals.py`)
- `tests/test_cli.py` (update CLI command tests for flat structure)

**Out of scope** (do NOT touch):
- `src/phase_unwrap/model/pclcn.py`
- `src/phase_unwrap/core/ops.py`
- `src/phase_unwrap/core/config.py`

## Git workflow

- Branch: `pclcn-pipeline`
- Commit per step or per logical unit; conventional commit messages.

## Steps

### Step 1: Purge Redundant Plot Files in `src/phase_unwrap/plots/`
Delete 12 redundant legacy plot files from `src/phase_unwrap/plots/`:
- `convergence.py`
- `curriculum_noise_grid.py`
- `error_histogram.py`
- `gradcam_overlay.py`
- `inference_output.py`
- `loss_landscape.py`
- `method_comparison.py`
- `noise_degradation_grid.py`
- `phase_profile.py`
- `prediction_scatter.py`
- `residual_analysis.py`
- `training_curve.py`

Update `src/phase_unwrap/plots/__init__.py` to export exclusively:
- `fig1_architecture`
- `fig2_baseline_comparison`
- `fig3_noise_robustness`
- `fig4_ablation`
- `style`
- `utils`
- `_epoch_visuals`

**Verify**: `ls src/phase_unwrap/plots/*.py` -> shows only active paper figure modules + style/utils/_epoch_visuals.

### Step 2: Add 1D Line Cut Profile Utility to `utils.py` & Refactor `fig4_ablation.py`
Add `draw_line_profile_panel()` in `src/phase_unwrap/plots/utils.py` for 1D cross-section plots.
Refactor `src/phase_unwrap/plots/fig4_ablation.py` to remove TTA functions and focus on PCLCN Fast Core Matrix boxen & radar plots.

**Verify**: `uv run python -c "import phase_unwrap.plots as p; print(dir(p))"` -> imports cleanly.

### Step 3: Refactor Plot Command & Purge `_plot_runners.py`
Delete `src/phase_unwrap/cli_cmds/_plot_runners.py`.
Update `src/phase_unwrap/cli_cmds/plot.py` to provide a single streamlined flat command:
`phase-unwrap plot --fig <1|2|3|4|all> --out-dir results/paper_figures`

**Verify**: `uv run phase-unwrap plot --help` -> shows `--fig` options.

### Step 4: Modernize `export` Command for PCLCN
Update `src/phase_unwrap/cli_cmds/export.py` to support exporting `PCLCNModel` to ONNX (`torch.onnx.export`) and TorchScript (`torch.jit.trace`) with 3-tuple inputs `(I_raw, grad_phi2)`.

**Verify**: `uv run phase-unwrap export --help` -> shows ONNX and TorchScript export options.

### Step 5: Flatten CLI in `src/phase_unwrap/cli.py` & Purge Obsolete `config.py`
Delete `src/phase_unwrap/cli_cmds/config.py` (yagni).
Update `src/phase_unwrap/cli.py` to register flat top-level commands directly:
- `generate` (from `cli_cmds/data.py`)
- `train` (from `cli_cmds/train.py`)
- `ablation` (from `cli_cmds/train.py`)
- `multiseed` (from `cli_cmds/train.py`)
- `eval` (from `cli_cmds/eval.py`)
- `export` (from `cli_cmds/export.py`)
- `infer` (from `cli_cmds/infer.py`)
- `plot` (from `cli_cmds/plot.py`)

**Verify**: `uv run phase-unwrap --help` -> displays single flat command list.

### Step 6: Update CLI Tests & Run Verification Suite
Update `tests/test_cli.py` to test flat CLI command invocations (`phase-unwrap --help`, `phase-unwrap train --help`, `phase-unwrap export --help`, `phase-unwrap plot --help`).

**Verify**: `uv run python -m pytest tests/ -q` -> 100% tests pass.

## Test plan

- Test flat CLI invocation for each command via Typer `CliRunner`.
- Verification: `uv run python -m pytest tests/test_cli.py -q` -> all pass.

## Done criteria

- [ ] `uv run phase-unwrap --help` displays flat top-level commands without nested sub-apps
- [ ] 12 legacy plot files deleted from `src/phase_unwrap/plots/`
- [ ] Obsolete `config.py` and `_plot_runners.py` deleted
- [ ] `uv run python -m pytest tests/ -q` exits 0 with 100% pass rate
- [ ] `plans/README.md` status row updated to DONE

## STOP conditions

- Stop and report back if any core model/loss import breaks.
- Stop if ONNX export fails on dynamic spatial grid shapes.

## Maintenance notes

- Future CLI additions should be added as top-level commands in `src/phase_unwrap/cli.py` without introducing nested sub-apps.
