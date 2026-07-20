# Plan 009: Unify the `phi_hint` input channel and resolve GT leakage

> **Executor instructions**: Follow this plan step by step. Run every
> verification command and confirm the expected result before moving to the
> next step. If anything in the "STOP conditions" section occurs, stop and
> report — do not improvise. When done, update the status row for this plan
> in `plans/README.md` — unless a reviewer dispatched you and told you they
> maintain the index.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/data/augmentation.py src/phase_unwrap/analysis/ src/phase_unwrap/training/`
> If any in-scope file changed since this plan was written, compare the
> "Current state" excerpts against the live code before proceeding; on a
> mismatch, treat it as a STOP condition.

## Status

- **Priority**: P0
- **Effort**: M
- **Risk**: MED (changes model input distribution; requires retraining to publish new numbers)
- **Depends on**: plans/013-disable-val-test-geometric-aug.md (soft); plans/007 and 010 for honest reporting after retrain
- **Category**: bug / methodology
- **Planned at**: commit `967403f`, 2026-07-14
- **Refreshed**: 2026-07-15
- **Owner decision (locked)**: **Option B** — reference-free. Default `hint_mode="zero"`.
  Option A is deferred to `plans/015-option-a-reference-guided-fallback.md` (do not implement A here).

## Why this matters

The model input is `[B, 2, H, W] = (I_norm, phi_hint)` (`src/phase_unwrap/model/unet.py:171`).
The second channel — `phi_hint` — is constructed **four mutually incompatible ways**
across the codebase:

1. **Training / val / test** (`data/augmentation.py:265-269`): the ground-truth
   unwrapped-phase value at the **center pixel**, broadcast over the whole image.
   This is a direct injection of GT absolute phase into the input.
2. **Baselines eval** (`analysis/baselines.py:129`): `np.angle(np.exp(1j * I_clean))`
   — the wrapped phase of the intensity image, a completely different signal.
3. **Noise sweep** (`analysis/noise_sweep.py:84`): same wrapped-intensity construction.
4. **Inference** (`analysis/inference.py:96`): all zeros.

This causes **two distinct problems**:

- **GT leakage**: feeding the GT center-pixel absolute phase hands the network the
  global piston it is supposedly learning to infer. This partially short-circuits
  the `k_off` head and undermines any "reference-free absolute phase reconstruction"
  claim. The val/test metrics are computed with GT leaked into the input.
- **Train/eval distribution mismatch**: the model is trained on the GT-center hint but
  evaluated in `baselines.py`/`noise_sweep.py` with wrapped-intensity and in
  `inference.py` with zeros — inputs it never saw. Those reported numbers measure the
  model on an out-of-distribution channel and are not comparable to the val/test table.

This plan does **not** unilaterally decide the science. It (a) centralizes hint
construction into one function so all paths agree, and (b) forces an explicit,
documented choice of hint semantics via a config flag. The default preserves current
training behavior so nothing silently changes until the owner decides.

## Current state

**Training hint (canonical), `src/phase_unwrap/data/augmentation.py:256-283`:**
```python
def prepare_batch(I_raw, phi_gt, noise_aug=None):
    _, _, H, W = phi_gt.shape
    cy, cx = H // 2, W // 2
    ref_vals = phi_gt[:, :, cy : cy + 1, cx : cx + 1]   # <-- GT center pixel
    phi_hint = ref_vals.expand_as(phi_gt).clone()
    I_raw_clean = I_raw.clone()
    if noise_aug is not None:
        I_raw_noisy, I_norm_noisy, phi_hint, delta = noise_aug(I_raw, phi_hint)
        phi_gt = phi_gt + delta          # label shifted to match corrupted hint
    else:
        I_raw_noisy = I_raw
        mean = I_raw.mean(dim=(-2, -1), keepdim=True)
        std = I_raw.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        I_norm_noisy = (I_raw - mean) / std
    I_input = torch.cat([I_norm_noisy, phi_hint], dim=1)
    return I_input, phi_gt, I_raw_noisy, I_raw_clean
```

**Divergent eval constructions:**
- `src/phase_unwrap/analysis/baselines.py:128-132`:
  ```python
  I_norm = (I_clean - I_mean) / I_std
  phi_hint = np.angle(np.exp(1j * I_clean))
  inp_t = torch.from_numpy(np.stack([I_norm, phi_hint], 0)).unsqueeze(0).to(device)
  ```
- `src/phase_unwrap/analysis/noise_sweep.py:83-87`: same `np.angle(np.exp(1j * I_noisy))`.
- `src/phase_unwrap/analysis/inference.py:95-98`:
  ```python
  hint_padded = torch.zeros_like(I_padded)   # "Create blank phase hint"
  x_in = torch.cat([I_padded, hint_padded], dim=1)
  ```

**Config**: `src/phase_unwrap/core/config.py` `AugmentationConfig` has `hint_std: float = 0.2`
(line 117) but no field selecting *what the hint is*.

## Commands you will need

| Purpose   | Command                       | Expected on success |
|-----------|-------------------------------|---------------------|
| Tests     | `uv run pytest tests/ -q`     | all pass            |
| Import    | `uv run python -c "import phase_unwrap.data.augmentation"` | exit 0 |

(If `uv run` triggers a long dependency sync and times out, see plan 008 — torch is not
installed in a fresh checkout. Report it as a STOP condition rather than waiting >10 min.)

## Scope

**In scope**:
- `src/phase_unwrap/core/config.py` — add `hint_mode` field to `AugmentationConfig`.
- `src/phase_unwrap/data/augmentation.py` — extract a `build_phi_hint(...)` helper honoring `hint_mode`.
- `src/phase_unwrap/analysis/baselines.py`, `analysis/noise_sweep.py`, `analysis/inference.py` — call the shared helper.
- `tests/test_hint.py` (create).

**Out of scope**:
- Do NOT change `hint_std` semantics or the `phi_gt + delta` label-coupling.
- Do NOT retrain or regenerate paper numbers — that is the owner's call once the mode is chosen.
- Do NOT touch `model/unet.py` input arity.

## The decision this plan surfaces (do not resolve it yourself)

Add `hint_mode: str = "gt_center"` to `AugmentationConfig` with legal values:

- `"gt_center"` — current behavior (GT center pixel). **Keeps the absolute-piston leak**,
  preserves existing training. Paper must say **reference-anchored**, not reference-free.
- `"zero"` — zeros (reference-free; matches production `inference.py` blank hint).
- `"wrapped"` — **do not implement as `angle(exp(1j * intensity))`**. That is what
  `baselines.py` / `noise_sweep.py` do today and it is **not** optical phase (intensity is
  photometric, not radians). If you need a third mode for API compatibility, implement
  `"wrapped"` as an alias that raises `NotImplementedError` with a clear message, **or**
  omit it until a real wrapped-phase source exists (complex field / `atan2` on known
  quadrature). Prefer shipping only `"gt_center"` and `"zero"` unless the owner specifies
  a correct wrapped-phase formula.

**Default is `"zero"` (Option B, owner-locked 2026-07-15).**  
Keep `"gt_center"` only as an explicit ablation of the old leak (never default).  
Option A (lab-measurable reference) is **not** this plan — see plan 015.

When `hint_mode="zero"`, do **not** apply `hint_std` delta corruption (that would re-inject
a random absolute channel and re-teach the model to read piston from channel 2).

## Steps

### Step 1: Add the `hint_mode` config field
In `src/phase_unwrap/core/config.py`, `AugmentationConfig`, add after `hint_std`:
```python
hint_mode: str = "gt_center"  # "gt_center" | "zero"  (see plans/009 — no intensity-wrap mode)
```

**Verify**: `uv run python -c "from phase_unwrap.core.config import AugmentationConfig; print(AugmentationConfig().hint_mode)"` → `gt_center`

### Step 2: Extract `build_phi_hint` in `augmentation.py`
Add a module-level function that produces the hint channel `[B,1,H,W]` given a reference
tensor for shape/device (use `I_norm` or `phi_gt`), optional `phi_gt`, and `hint_mode`:
```python
def build_phi_hint(
    ref: torch.Tensor,
    phi_gt: torch.Tensor | None = None,
    hint_mode: str = "gt_center",
) -> torch.Tensor:
    """Build the second input channel [B,1,H,W]. ref is any [B,1,H,W] for shape/device."""
    B, _, H, W = ref.shape
    if hint_mode == "gt_center":
        if phi_gt is None:
            raise ValueError(
                "hint_mode='gt_center' requires phi_gt (unavailable at inference)"
            )
        cy, cx = H // 2, W // 2
        return phi_gt[:, :, cy : cy + 1, cx : cx + 1].expand(B, 1, H, W).clone()
    if hint_mode == "zero":
        return torch.zeros(B, 1, H, W, device=ref.device, dtype=ref.dtype)
    if hint_mode == "wrapped":
        # Intentionally unsupported: angle(exp(i*I)) is not optical phase.
        raise NotImplementedError(
            "hint_mode='wrapped' is not supported; use 'gt_center' or 'zero'"
        )
    raise ValueError(f"unknown hint_mode: {hint_mode}")
```
Wire `prepare_batch` to accept `hint_mode: str = "gt_center"` (or read from a passed
config) and call `build_phi_hint(I_raw, phi_gt, hint_mode=...)` before noise. Preserve
the existing `phi_gt + delta` coupling when noise is applied.

Note: `"gt_center"` cannot be used at real inference without GT — that is intentional.

**Verify**: `uv run pytest tests/test_hint.py -q` → passes (after Step 4).

### Step 3: Route eval/inference paths through the helper
Replace ad-hoc constructions in `analysis/baselines.py:129`, `analysis/noise_sweep.py:84`,
and `analysis/inference.py:96` with `build_phi_hint(ref, phi_gt=None, hint_mode="zero")`
for production-like paths that lack GT.

For **training-matched** offline eval when GT is available (optional path in baselines if
GT exists), you may use `hint_mode="gt_center"` only when measuring the same distribution
as train — but document it. Default replacement for the old intensity-wrap lines: **`zero`**.

Delete `np.angle(np.exp(1j * I_*))` hint construction entirely.

**STOP and report** to the owner: eval with `zero` does not match train with `gt_center`.
Owner must choose Option A (document leak + keep gt_center everywhere GT exists) or
Option B (retrain with `zero`). Do not retrain in this plan.

### Step 4: Add `tests/test_hint.py`
Model after `tests/test_losses_ops.py`. Cover:
- `build_phi_hint(..., "gt_center")` returns the center-pixel value broadcast
  (every element equals `phi_gt[..., cy, cx]`).
- `build_phi_hint(..., "gt_center")` with `phi_gt=None` raises `ValueError`.
- `"zero"` returns all-zeros of the right shape/device/dtype.
- `"wrapped"` raises `NotImplementedError` (guards against reintroducing intensity-wrap).
- Regression: `conftest.py` center-pixel logic agrees with `build_phi_hint(..., "gt_center")`
  if that fixture still duplicates the logic — or update conftest to call the helper.

**Verify**: `uv run pytest tests/test_hint.py -q` → all pass.

## Done criteria

- [ ] `AugmentationConfig.hint_mode` exists, defaults to `"gt_center"`
- [ ] `build_phi_hint` is the only place any code path constructs the hint channel (`grep -rn "expand_as(phi_gt)\|angle(np.exp\|zeros_like(I_padded)" src/` returns only the helper or nothing)
- [ ] `uv run pytest tests/test_hint.py -q` passes
- [ ] `uv run pytest tests/ -q` still green
- [ ] No files outside the in-scope list modified (`git status`)
- [ ] A STOP-and-report note delivered to the owner on the `gt_center` leak / retrain decision
- [ ] `plans/README.md` status row updated

## STOP conditions

Stop and report (do not improvise) if:
- The code at the excerpts above doesn't match live code (drift).
- Switching a mode would require retraining to keep the paper honest — that is the owner's call.
- The `phi_gt + delta` label-coupling breaks any existing test.
- `uv run` cannot import torch (fresh env) — report, don't wait.

## Maintenance notes

- This is the single highest-severity finding for publication validity: the val/test
  numbers in `docs/proposal/main.tex` were produced with GT leaked into the input via
  `gt_center`. If the owner keeps `gt_center`, the paper must (a) drop any reference-free
  claim and (b) describe the hint as a known reference measurement.
- Interacts with plan 007 (affine scale leakage) and 010 (selection metric) — all three
  concern how favorably the reported number is produced. A reviewer should read all three together.
- If a reference-free mode is chosen, all of `docs/proposal/main.tex`'s quantitative
  results must be regenerated; coordinate with plan 010.
