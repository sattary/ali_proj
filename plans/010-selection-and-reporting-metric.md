# Plan 010: Fix checkpoint selection, HPO objective, and val-vs-test reporting

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/training/train.py src/phase_unwrap/training/tune.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P0
- **Effort**: S
- **Risk**: LOW (changes which checkpoint/trial wins; does not change training dynamics)
- **Depends on**: plans/007-affine-align-scale-leakage.md (hard); plans/013-disable-val-test-geometric-aug.md (recommended before re-eval)
- **Category**: bug / methodology
- **Planned at**: commit `967403f`, 2026-07-14
- **Refreshed**: 2026-07-15 (dependency order for Q1 path)

## Why this matters

Three coupled reporting problems make the published number more favorable than the
model's real capability:

1. **Checkpoint selection uses the lenient metric.** `train.py:569-570` saves `best.pth`
   on `TopoMAE` — the affine-*scale*-aligned metric (see plan 007). This selects the model
   whose shape is best *after* a GT-fitted scale correction, i.e. it rewards the exact
   failure mode (wrong absolute scale) that the aligned metric hides.
2. **HPO optimizes the same lenient metric.** `tune.py:216-244` computes the objective on
   `affine_align(...)` output and minimizes it. Hyperparameters are tuned to look good under
   the oracle correction.
3. **The paper reports validation numbers as final results.** `run_eval`'s per-epoch output
   is written only to `val_*` CSV columns (`train.py:611-617`); the held-out **test** eval
   (`train.py:622-632`) is `print`ed but never logged. The paper's Table `tab:metrics`
   (`docs/proposal/main.tex:159-176`) has per-epoch rows 1–100, so it is the **validation**
   curve — the same split checkpoints are selected on — presented as the final result.

## Current state

**Checkpoint selection, `src/phase_unwrap/training/train.py:569-570`:**
```python
if eval_stats.get("TopoMAE", float("inf")) < best_mae:
    best_mae = eval_stats["TopoMAE"]
    save_checkpoint(os.path.join(run_dir, "best.pth"), ...)
```

**HPO objective, `src/phase_unwrap/training/tune.py:216-244`:**
```python
aligned, _, _ = affine_align(phi_abs, phi_gt)
total_mae += float((aligned - phi_gt).abs().mean()) * I_input.size(0)
...
val_mae = total_mae / max(1, n)
best_mae = min(best_mae, val_mae)
...
return best_mae            # study direction="minimize" (tune.py:333)
```

**Test eval printed but not logged, `src/phase_unwrap/training/train.py:622-632`:**
```python
test_stats = run_eval(ema.m, test_loader, device, use_amp, eval_sobel)
print(f"[TEST] TopoMAE={test_stats.get('TopoMAE', 0):.4f} ...")   # never written to CSV
```

`run_eval` already computes **both** `AbsMAE` (raw) and `TopoMAE` (aligned) — see
`train.py:165-168` — so the honest metric is already available; this plan just switches
what drives selection/tuning/reporting.

## Commands you will need

| Purpose   | Command                    | Expected on success |
|-----------|----------------------------|---------------------|
| Tests     | `uv run pytest tests/ -q`  | all pass            |
| Grep      | `grep -n "TopoMAE\|AbsMAE" src/phase_unwrap/training/train.py` | shows updated selection line |

## Scope

**In scope**:
- `src/phase_unwrap/training/train.py` — selection metric + write test_stats to a `test_metrics.csv` (or a `[test]` row).
- `src/phase_unwrap/training/tune.py` — objective metric.

**Out of scope**:
- Do NOT remove `TopoMAE` from the logs — keep it as a secondary diagnostic column.
- Do NOT change `run_eval`'s computation (plan 007 owns that).
- Do NOT edit `docs/proposal/main.tex` numbers here — regenerating results is a separate,
  owner-driven run once 007+009+010 land.

## Steps

### Step 1: Select checkpoints on the honest metric
In `train.py`, change the selection key from `TopoMAE` to the primary honest metric defined
by plan 007. After 007, `TopoMAE` CSV key may still exist but means **piston-only** aligned
error; prefer selecting on **`AbsMAE`** (raw, no GT fit) as the strictest checkpoint criterion
unless the owner has documented piston-only as primary. Default for this plan: **`AbsMAE`**.
Keep logging the aligned diagnostic (`TopoMAE` key) alongside. Update `best_mae` consistently
in `save_checkpoint`.

**Verify**: `grep -n "best_mae = eval_stats" src/phase_unwrap/training/train.py` → references `AbsMAE`, not `TopoMAE`.

### Step 2: Tune HPO on the honest metric
In `tune.py:216-223`, compute the objective on the raw (or offset-only) prediction rather
than `affine_align(...)`. If plan 007 added an `align_mode`/offset-only helper, reuse it so
train and tune agree.

**Verify**: `grep -n "affine_align" src/phase_unwrap/training/tune.py` → no longer used to build the minimized objective (or uses offset-only mode).

### Step 3: Persist the held-out test result
After the final `test_stats = run_eval(...)`, write it to a machine-readable file
(`test_metrics.csv` in `run_dir`) with the full metric set, so the paper can cite a real
held-out number instead of the validation curve.

**Verify**: after a smoke run (or in a unit test with a fake `run_eval`), `test_metrics.csv` exists in `run_dir` and contains one row with `AbsMAE`/`TopoMAE`.

### Step 4: Add a selection regression test
In `tests/test_train.py` (created by plan 001) or a new `tests/test_selection.py`, feed two
synthetic `eval_stats` dicts where the model with lower `TopoMAE` has higher `AbsMAE`, and
assert the selection logic now prefers the lower-`AbsMAE` model.

**Verify**: `uv run pytest tests/test_selection.py -q` → passes.

## Done criteria

- [ ] `best.pth` selection uses the honest metric (grep confirms)
- [ ] HPO objective no longer minimizes the scale-aligned MAE
- [ ] `test_metrics.csv` written with the held-out test result
- [ ] Selection regression test passes
- [ ] `uv run pytest tests/ -q` green
- [ ] No files outside scope modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report (do not improvise) if:
- Plan 007 has not landed (the honest metric name/helper doesn't exist yet) — this plan
  depends on it.
- Excerpts don't match live code (drift).
- Changing the objective breaks an existing tune test.

## Maintenance notes

- After 007 + 009 + 010 land, the owner must **re-run training and HPO** and regenerate
  every number in `docs/proposal/main.tex` — the current table is validation data selected
  on the lenient metric with a leaked hint. All three effects compound.
- A reviewer should confirm the paper distinguishes "validation curve" (per-epoch) from
  "held-out test result" (single final number) and reports the latter as the headline.
- The non-monotone val curve (epoch 80 TopoMAE 0.4379 < epoch 100 0.5248) means `best.pth`
  ≠ the epoch-100 row the paper cites — another reason to report the persisted test metric.
