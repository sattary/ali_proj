# Plan 015: Option A fallback — reference-guided absolute phase (future)

> **Executor instructions**: Do **not** implement this plan until the owner
> explicitly schedules Option A. The project default is **Option B**
> (reference-free / `hint_mode=zero`). This plan is a frozen design so a future
> agent can implement A without rediscovering the science.
>
> **Drift check (run first)**: `git diff --stat 967403f..HEAD -- src/phase_unwrap/data/augmentation.py src/phase_unwrap/core/config.py`
> Re-read live code; this plan was written against Option B landing first.

## Status

- **Priority**: P3 (future / fallback only)
- **Effort**: M–L (includes retrain + lab protocol)
- **Risk**: MED
- **Depends on**: Option B stack stable (013, 007, 009-B, 010); real lab reference protocol defined by owner
- **Category**: direction / methodology
- **Planned at**: commit `967403f`, 2026-07-15
- **Owner decision**: **Deferred**. Primary path is Option B.

## Why this exists

Option B trains without any absolute anchor so train/deploy match a bare lab intensity.
If B proves insufficient for absolute metrology (ill-posed piston, lab needs SI-scale
absolute phase), Option A adds a **lab-available** reference scalar — not GT center
leakage. This plan defines A so it is never confused with the old GT-center cheat.

## Science (locked definitions)

| Mode | Input channel 2 | Allowed in lab? | Paper claim |
|------|-----------------|-----------------|-------------|
| **B (current)** | zeros | Yes | Reference-free continuous phase from \(I\) (piston may be conventional) |
| **A (this plan)** | measurable reference \(r\) | Only if lab can measure \(r\) without unwrapping | Reference-guided absolute phase |
| **Forbidden** | GT center / any GT-derived absolute value used only in sim | No | Cheating |

### Valid lab references for A (examples — owner picks one)

1. Known piston on a flat mirror region (mean phase from PSI / dual-wavelength on a ROI).
2. Dual-wavelength coarse absolute estimate at one pixel.
3. Calibrated actuator / stage height at a fiducial.
4. Operator-supplied constant from an independent instrument.

**Not valid:** GT center from synthetic labels; `angle(exp(1j * intensity))`.

### Simulation of A

In the generator / `prepare_batch`, when `hint_mode="reference"`:

- Sample or compute a scalar \(r\) that **mimics the lab protocol** (e.g. true phase at a
  fixed fiducial pixel **only if** the lab always knows that fiducial; or true mean of a
  simulated “flat” ROI; or \(r = \phi_{\text{gt}}[c_y,c_x] + \mathcal{N}(0,\sigma)\) with
  \(\sigma\) matching lab reference noise).
- Broadcast \(r\) as channel 2.
- Do **not** call this `gt_center` in configs or papers — name it after the protocol
  (`fiducial`, `roi_mean`, `dual_wl`).

## Current state (after Option B)

- `AugmentationConfig.hint_mode` supports `"zero"` (default) and may still accept
  `"gt_center"` only for ablation of the old leak — do not use for production A.
- Inference uses zero hint.
- Metrics are piston-only / AbsMAE (no GT scale fit).

## Scope (when activated)

**In scope**:
- `src/phase_unwrap/core/config.py` — add `hint_mode="reference"` + `reference_protocol` fields
- `src/phase_unwrap/data/augmentation.py` — `build_phi_hint` branch for lab-like \(r\)
- `src/phase_unwrap/analysis/inference.py` — accept CLI/config reference scalar or ROI
- Lab SOP markdown under `docs/` (how to measure \(r\) on the instrument)
- Retrain + ablation table: B vs A vs old GT-center leak

**Out of scope**:
- Reverting B as default without owner approval
- Intensity-wrap as reference

## Steps (future executor)

### Step 1: Owner freezes the lab protocol

Document in `docs/lab_reference_protocol.md`: what is measured, units (rad), uncertainty,
per-frame vs per-session. **STOP** if the owner cannot measure \(r\) without already
unwrapping — then A is invalid and B remains the only path.

### Step 2: Implement `hint_mode="reference"`

`build_phi_hint` produces constant map of \(r\) with optional `reference_noise_std`.
Train/val/test/infer all use the **same** construction. Lab inference takes `--ref-rad`
or reads a sidecar file.

### Step 3: Ablations

| Run | hint | Purpose |
|-----|------|---------|
| B | zero | Current main |
| A | reference (lab-like) | Fallback product |
| Leak | gt_center | Oracle upper bound only; never deploy |

### Step 4: Paper / thesis wording

Claim **reference-guided** absolute reconstruction. Compare to B. Disclose reference noise.

## Done criteria (when this plan is executed)

- [ ] Lab protocol document exists and is physically implementable
- [ ] Train/deploy use identical reference construction
- [ ] No GT-only shortcuts in the A production path
- [ ] Ablation table B vs A published or logged
- [ ] Default config remains B unless owner flips product default

## STOP conditions

- Lab cannot provide \(r\) without GT unwrapping → reject A, keep B.
- Implementer reintroduces GT center as production default → STOP, report.

## Maintenance notes

- Option A is a **product fork**, not a metric hack. Budget a full retrain.
- Reviewer: verify inference CLI can supply \(r\) without loading GT phase maps.
