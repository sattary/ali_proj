# Plan 012: Make CUDA OOM recovery safe and reproducible

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/training/train.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P2
- **Effort**: M
- **Risk**: MED (touches the training loop's control flow)
- **Depends on**: 001 (integration test for train loop) — soft dependency
- **Category**: bug / repro
- **Planned at**: commit `967403f`, 2026-07-14

## Why this matters

The dynamic CUDA-OOM recovery (`train.py:423-482`) keeps a 48-hour run alive by halving
the batch size, but in doing so it silently corrupts reproducibility and metrics:

- It **overwrites the frozen `config.yaml`** (`train.py:455`) with the mutated batch size, so
  the recorded config no longer matches what the run was launched with — the run becomes
  unreproducible exactly when you most need the record.
- It **rebuilds and fast-forwards the scheduler** `global_step` times (`train.py:462-468`).
  The reconstructed cosine `T_max` (`new_total_steps`) differs from the original, so the LR
  trajectory changes shape mid-run, and the replay is O(global_step) synchronous steps.
- It `break`s the partial epoch (`train.py:481`) but still runs eval/checkpoint for that epoch;
  `train_loss = run_loss / max(1, cnt)` is a truncated-epoch average logged to `metrics.csv`
  as if it were a full epoch, contaminating any multiseed aggregate.

The goal is not to remove the safety net — it is to make recovery leave an honest, reproducible
record.

## Current state

`src/phase_unwrap/training/train.py:423-482` (OOM `except` block), key lines:
```python
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        torch.cuda.empty_cache()
        opt.zero_grad(set_to_none=True)
        if cfg.optim.batch_size <= 1:
            raise RuntimeError("Fatal CUDA OOM: Batch size is already 1.") from e
        cfg.optim.batch_size = max(1, cfg.optim.batch_size // 2)
        train_loader, val_loader, test_loader = build_dataloaders(cfg, device, seed=cfg.logging.seed)
        # ... rewrites config.yaml here (~line 455) ...
        # ... rebuilds scheduler and fast-forwards it global_step times (~462-468) ...
        break   # abandons the rest of the epoch (~481)
```
Downstream: eval + checkpoint still run for the abandoned epoch (`train.py:506`, `569`);
`train_loss` computed on partial `cnt` (~line 499) and written to CSV (`train.py:602`).

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Tests | `uv run pytest tests/ -q` | all pass |
| Grep | `grep -n "config.yaml\|batch_size //\|sched.step()" src/phase_unwrap/training/train.py` | shows recovery block |

(Real OOM cannot be triggered without a GPU; verification here is structural + unit-level.
Do not attempt to force a real CUDA OOM.)

## Scope

**In scope**:
- `src/phase_unwrap/training/train.py` — the OOM `except` block and the epoch-accounting around it.

**Out of scope**:
- `build_dataloaders` / `dataset.py` — do not change the loader API.
- The scheduler construction helper — reuse it; don't rewrite the schedule math.

## Steps

### Step 1: Stop overwriting the launch config
Write the mutated batch size to a **separate** file (e.g. `config_effective.yaml` or an
appended `recovery.log`), never overwriting the original `config.yaml`. The launch config
must remain the immutable record of intent.

**Verify**: `grep -n "config.yaml" src/phase_unwrap/training/train.py` → the OOM path no longer
writes `config.yaml`; it writes a distinctly named file.

### Step 2: Mark the recovered epoch as partial in the metrics
When an epoch is abandoned mid-way, either (a) do not write a `metrics.csv` row for that epoch,
or (b) write it with an explicit `partial=1` / `noise_level`-style marker column so multiseed
aggregation can exclude it. Do not log a truncated-epoch `train_loss` as a normal row.

**Verify**: add a unit test that simulates the partial-epoch path (inject a flag) and asserts the
row is either omitted or flagged. `uv run pytest tests/test_train.py -k oom -q` passes.

### Step 3: Record the recovery event
Emit a clear one-line record (stdout + a `recovery.log` in `run_dir`) capturing: epoch, old→new
batch size, and `global_step` at recovery, so the run's provenance is auditable.

**Verify**: unit test or smoke run shows `recovery.log` written with the batch-size transition.

### Step 4: Confirm scheduler continuity is documented
Add a comment at the fast-forward loop stating that `T_max` changes after recovery and the LR
trajectory is therefore only approximately continuous — or, preferred, reconstruct the scheduler
with the **original** `total_steps` so the annealing shape is preserved. Choose one and document it.

**Verify**: code review — the `new_total_steps` computation either matches the original schedule or
carries an explicit comment explaining the intended deviation.

## Done criteria

- [ ] Launch `config.yaml` is never overwritten by the OOM path
- [ ] Partial epochs are excluded or explicitly flagged in `metrics.csv`
- [ ] A `recovery.log` records each OOM downshift
- [ ] `uv run pytest tests/ -q` passes
- [ ] No files outside `src/phase_unwrap/training/train.py` (+ tests) modified
- [ ] `plans/README.md` status row updated

## STOP conditions

- Refactoring the OOM block requires changing `build_dataloaders`' signature → stop and report.
- The partial-epoch accounting turns out to be entangled with the visualization dispatch thread
  (`train.py:540-556`) such that a clean flag is not possible → stop and report the coupling.

## Maintenance notes

This plan intentionally preserves the OOM *safety net* (keeping long runs alive). It only fixes
the reproducibility and metric-honesty side effects. Reviewers: confirm the recovery path still
resumes training and does not raise on the `batch_size == 1` floor.
