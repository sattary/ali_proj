# Plan 025: Fix Undefined `subset` Variable in Noise Comparison Grid Visualization

> **Executor instructions**: Follow this plan step by step. Run every verification command and confirm the expected result before moving to the next step. If anything in the "STOP conditions" section occurs, stop and report — do not improvise. When done, update the status row for this plan in `plans/README.md`.
>
> **Drift check (run first)**: `git diff --stat ab82cfc..HEAD -- src/phase_unwrap/visualize/noise_comparison_grid.py src/phase_unwrap/cli_cmds/plot.py`
> If any in-scope file changed since this plan was written, compare the "Current state" excerpts against the live code before proceeding; on a mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: S
- **Risk**: LOW
- **Depends on**: none
- **Category**: bug
- **Planned at**: commit `ab82cfc`, 2026-07-30

## Why this matters

The plotting function `plot_noise_comparison_grid()` in `src/phase_unwrap/visualize/noise_comparison_grid.py` references a non-existent variable `subset` at lines 58, 62, 91, and 93. Executing `phase-unwrap plot noise-comparison` or calling this function raises a `NameError: name 'subset' is not defined`. This plan adds `subset: str = "val"` to the function parameters and fixes parameter handling in `src/phase_unwrap/cli_cmds/plot.py`.

## Current state

- `src/phase_unwrap/visualize/noise_comparison_grid.py` — `plot_noise_comparison_grid` signature (lines 35–45) omits parameter `subset`, but lines 58, 62, 91, 93 evaluate `if subset == "test":`.
- `src/phase_unwrap/cli_cmds/plot.py` — Calls `plot_noise_comparison_grid(...)` at line 239 with `subset=subset`, causing type check and runtime errors.

## Commands you will need

| Purpose   | Command | Expected on success |
|-----------|---------|---------------------|
| Lint Check | `uv run ruff check src/phase_unwrap/visualize/noise_comparison_grid.py` | exit 0, no errors |
| Typecheck | `uv run mypy src/phase_unwrap/visualize/noise_comparison_grid.py` | exit 0 |

## Scope

**In scope**:
- `src/phase_unwrap/visualize/noise_comparison_grid.py` — Add `subset: str = "val"` parameter to function definition and resolve loader selection logic.
- `src/phase_unwrap/cli_cmds/plot.py` — Ensure `subset` parameter is correctly forwarded.

**Out of scope**:
- Matplotlib theme or rendering styles in `src/phase_unwrap/visualize/style.py`.

## Git workflow

- Branch: `pclcn-pipeline`
- Commit message: `Fix undefined subset variable in plot_noise_comparison_grid`

## Steps

### Step 1: Add `subset` parameter to `plot_noise_comparison_grid` in `noise_comparison_grid.py`

Update function signature in `src/phase_unwrap/visualize/noise_comparison_grid.py`:

```python
def plot_noise_comparison_grid(
    clean_run_dir: str | Path,
    noisy_run_dir: str | Path,
    out_path: str | Path = "noise_comparison.pdf",
    subset: str = "val",
    device_str: str = "auto",
) -> Path:
```

### Step 2: Add validation for `subset` value

Validate that `subset` is either `"val"` or `"test"`:

```python
if subset not in ("val", "test"):
    raise ValueError(f"subset must be 'val' or 'test', got '{subset}'")
```

**Verify**: `uv run ruff check src/phase_unwrap/visualize/noise_comparison_grid.py` → 0 errors.

## Done criteria

- [ ] `plot_noise_comparison_grid()` parameter signature includes `subset: str = "val"`.
- [ ] No `NameError: name 'subset' is not defined` occurs during execution.
- [ ] `uv run ruff check src/phase_unwrap/visualize/noise_comparison_grid.py` exits 0.

## STOP conditions

- If caller signatures in `cli_cmds/plot.py` mismatch after update, stop and align parameter names.
