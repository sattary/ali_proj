# Plan 007: Report absolute-phase error without fitting a GT-derived scale factor

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/core/ops.py src/phase_unwrap/training/train.py src/phase_unwrap/training/tune.py`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P0
- **Effort**: M
- **Risk**: MED
- **Depends on**: plans/013-disable-val-test-geometric-aug.md (recommended first so re-eval is orientation-stable; soft if you only change code without re-running numbers)
- **Category**: bug (methodology / results validity)
- **Planned at**: commit `967403f`, 2026-07-14
- **Refreshed**: 2026-07-15 (title/index fix; Q1 order)

## Why this matters

The headline metric in the paper (`TopoMAE = 0.5248 rad`, `docs/proposal/main.tex:153,173`)
is computed on a prediction that has been fitted to the ground truth of the very
image being scored. `affine_align` (`src/phase_unwrap/core/ops.py:59-92`) solves a
per-image least-squares fit `a·pred + c ≈ gt` and applies it **before** the metric.

Removing the **offset** `c` is legitimate: absolute phase has a genuine global-piston
ambiguity (φ and φ+const yield the same fringes without a reference), and offset-invariant
error is standard in phase/depth literature. Removing the **scale** `a` is not: a radian
is a radian — there is no scale ambiguity in unwrapped phase. Fitting `a` to GT silently
absorbs systematic amplitude error. A model that outputs exactly `0.5·φ_gt` (perfect shape,
half amplitude) gets `a=2` and scores `TopoMAE ≈ 0` while its true error is large.

The size of the effect is visible in the paper's own table: `AbsMAE` (raw) is ~2× `TopoMAE`
(aligned) at every epoch (epoch 100: 0.9818 vs 0.5248). That gap is the combined piston+scale
correction, and the scale part is not reproducible at inference — on real data there is no GT
to fit `a` against. This plan makes the reported number an honest, offset-invariant error.

## Current state

- `src/phase_unwrap/core/ops.py:59-92` — `affine_align(pred, gt)` fits both `a` and `c`:

  ```python
  a = cov_pg / var_pred
  c = gt_mean - a * pred_mean
  ...
  pred_aligned = a_map * pred + c_map
  return pred_aligned, a, c
  ```

- `src/phase_unwrap/training/train.py:163-169` — `run_eval` computes the reported
  metrics on the aligned prediction:

  ```python
  phi_aligned, _, _ = affine_align(phi_abs, phi_gt)
  m_abs = compute_metrics(phi_abs, phi_gt)      # AbsMAE (raw)
  m_topo = compute_metrics(phi_aligned, phi_gt) # TopoMAE (aligned) -> headline
  sums["AbsMAE"] += float(m_abs["MAE"]) * bs
  sums["TopoMAE"] += float(m_topo["MAE"]) * bs
  sums["RMSE"] += float(m_topo["RMSE"]) * bs    # RMSE also uses aligned
  ```
  Lines 171-190 compute `MaxErr`, `GradMAE`, `SSIM`, `PSNR` **all** on `phi_aligned`.

- `src/phase_unwrap/training/tune.py:216` — HPO also aligns before its objective MAE
  (handled in plan 003; this plan only changes the alignment function's meaning).

- Convention: metrics live in `core/losses.py::compute_metrics` (MAE/RMSE on raw tensors);
  alignment is a separate `core/ops.py` concern. Match that separation — do not fold
  alignment into `compute_metrics`.

**Design constraint from the paper** (`docs/proposal/main.tex:146-150`), which the executor
has not read — quote it in the code comment you add:
> "the validation pipeline performs a Least-Squares Affine Alignment on the raw prediction:
> a·φ_pred + c ≈ φ_gt ... This explicitly isolates the network's topological unwrapping
> capability from trivial global drift errors."
The paper claims only *drift* (piston) is removed. Offset-only alignment makes the code match
that claim; the current scale term contradicts it.

## Commands you will need

| Purpose | Command | Expected on success |
|---|---|---|
| Drift check | `git diff --stat 967403f..HEAD -- src/phase_unwrap/core/ops.py` | empty or reviewed |
| Unit tests | `uv run pytest tests/test_losses_ops.py -q` | all pass |
| Targeted test | `uv run pytest tests/test_losses_ops.py -k affine -q` | all pass |

(If `uv run pytest` triggers a long dependency sync or torch is missing, see plan 001 —
establish the verification baseline first. Do not proceed on an unrunnable suite.)

## Scope

**In scope** (the only files you should modify):
- `src/phase_unwrap/core/ops.py` — add offset-only alignment
- `src/phase_unwrap/training/train.py` — use offset-only for reported metrics; keep AbsMAE raw
- `tests/test_losses_ops.py` — add/adjust alignment tests

**Out of scope** (do NOT touch):
- `src/phase_unwrap/training/tune.py` — HPO objective is plan 010 (not this plan).
- Any `visualize/` file that calls `affine_align` — those are figures, not reported metrics;
  changing them is a separate cosmetic pass. Leave them on the old function.
- `docs/proposal/main.tex` — paper rewrite is plan 011. Do not edit prose or numbers here.

## Git workflow

- Branch: `advisor/002-affine-align-scale-leakage`
- Commit style matches repo (`git log` shows `Fix: ...` / `Architectural ...` prefixes).
  Example: `Fix: report offset-invariant phase MAE instead of GT-fitted affine`
- Do NOT push or open a PR unless the operator instructed it.

## Steps

### Step 1: Add an offset-only alignment function, keep `affine_align` intact

In `src/phase_unwrap/core/ops.py`, add a new function next to `affine_align` (do not delete
`affine_align` — visualize/ still imports it):

```python
def piston_align(
    pred: torch.Tensor, gt: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Per-image offset-only alignment: ``pred + c ~ gt`` where ``c`` removes the
    global piston (mean) difference. Unlike ``affine_align`` this does NOT fit a
    scale ``a`` to the ground truth, so the resulting error is reproducible at
    inference (no GT-derived gain). Absolute phase is defined only up to a global
    additive constant, so removing ``c`` is physically justified; removing scale
    is not. See docs/proposal/main.tex:146-150.

    Returns:
        pred_aligned [B,1,H,W], c [B,1].
    """
    B = pred.shape[0]
    pred_flat = pred.reshape(B, -1)
    gt_flat = gt.reshape(B, -1)
    c = (gt_flat.mean(dim=1, keepdim=True) - pred_flat.mean(dim=1, keepdim=True))
    c_map = c.unsqueeze(-1).unsqueeze(-1)
    return pred + c_map, c
```

**Verify**: `uv run pytest tests/test_losses_ops.py -q` → all existing pass (no regressions;
you only added a function).

### Step 2: Switch reported metrics in `run_eval` to offset-only alignment

In `src/phase_unwrap/training/train.py`, `run_eval` (around lines 159-190):

- Change the import at the top of the file to also import `piston_align`
  (line 28 currently: `from ..core.ops import FixedSobel, affine_align, curvature_loss`).
- Replace `phi_aligned, _, _ = affine_align(phi_abs, phi_gt)` (line 163) with
  `phi_aligned, _ = piston_align(phi_abs, phi_gt)`.
- Keep `AbsMAE` exactly as-is (raw `phi_abs`, line 165/167) — it remains the strictest number.
- Rename the aligned key from `TopoMAE` to `PistonMAE` **only in a follow-up** — NOT here,
  because `multiseed.py` and `train.py:569` read `TopoMAE` by name. Keep the key string
  `"TopoMAE"` for now so nothing downstream breaks; the value now reflects offset-only error.
  Add a comment: `# TopoMAE key retained for CSV compat; now offset-only (piston) aligned.`

**Verify**: `uv run python -c "from phase_unwrap.training.train import run_eval, piston_align" 2>&1 | head`
→ no ImportError. (If torch is absent per plan 001, STOP and report.)

### Step 3: Add a regression test that scale error is no longer hidden

In `tests/test_losses_ops.py`, add:

```python
def test_piston_align_does_not_hide_scale_error():
    import torch
    from phase_unwrap.core.ops import piston_align, affine_align
    gt = torch.randn(2, 1, 16, 16)
    half = 0.5 * gt  # perfect shape, half amplitude
    # affine_align fits a=2 and hides the error:
    aff, a, _ = affine_align(half, gt)
    assert (aff - gt).abs().mean() < 1e-3
    assert torch.allclose(a, torch.full_like(a, 2.0), atol=1e-2)
    # piston_align only removes the mean, so the scale error survives:
    pist, _ = piston_align(half, gt)
    assert (pist - gt).abs().mean() > 0.1


def test_piston_align_removes_offset():
    import torch
    from phase_unwrap.core.ops import piston_align
    gt = torch.randn(2, 1, 16, 16)
    shifted = gt + 3.7
    aligned, c = piston_align(shifted, gt)
    assert (aligned - gt).abs().mean() < 1e-5
    assert torch.allclose(c.flatten(), torch.full((2,), -3.7), atol=1e-4)
```

**Verify**: `uv run pytest tests/test_losses_ops.py -k "piston" -q` → 2 passed.

## Test plan

- New tests: `test_piston_align_does_not_hide_scale_error`, `test_piston_align_removes_offset`
  in `tests/test_losses_ops.py`. Model them after the existing `affine_align` tests in the
  same file (structural pattern).
- Cases covered: scale-error is preserved (the leakage regression), offset is removed (correctness).
- Verification: `uv run pytest tests/test_losses_ops.py -q` → all pass including 2 new.

## Done criteria

Machine-checkable. ALL must hold:

- [ ] `uv run pytest tests/test_losses_ops.py -q` exits 0, 2 new tests present and passing
- [ ] `grep -n "piston_align" src/phase_unwrap/core/ops.py` → function defined
- [ ] `grep -n "affine_align(phi_abs, phi_gt)" src/phase_unwrap/training/train.py` → **no match**
      in `run_eval` (replaced by `piston_align`); the `visualize/` matches are untouched
- [ ] `affine_align` still exists in `ops.py` (visualize/ depends on it)
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report (do not improvise) if:

- The code at `ops.py:59-92` or `train.py:163-169` does not match the "Current state" excerpts.
- The test suite cannot run because torch is not installed (do plan 001 first).
- Removing scale from `run_eval` would require touching `multiseed.py` or `tune.py` to keep the
  suite green — that means the coupling is deeper than assumed; report it.
- You discover `visualize/` figures import `run_eval` (they should not) — that would widen scope.

## Maintenance notes

- After this lands, `TopoMAE` in `metrics.csv` means offset-only error and will be **higher**
  than the old scale-fitted value. That is expected and correct — the number got honest, not worse.
- Plan 005 rewrites the paper to (a) report this number, (b) rename it clearly, and (c) drop the
  claim that alignment removes only "drift" if any scale language remains.
- A reviewer should confirm no `visualize/` metric is presented as a headline result; those still
  use `affine_align` and would need the same treatment if ever promoted to a reported number.
- Follow-up deferred: renaming the CSV column `TopoMAE → PistonMAE` repo-wide (touches
  `multiseed.py`, `train.py:569`, plotting) — deferred to avoid breaking CSV compatibility mid-stream.
