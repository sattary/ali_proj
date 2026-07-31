# Implementation Plans

Generated and reconciled by the improve skill on 2026-07-31 (`18c93f5`).

Each executor: read the plan fully before starting, honor STOP conditions, and update your row when done.

Status values: `TODO` | `IN PROGRESS` | `DONE` | `BLOCKED` | `REJECTED` | `SUPERSEDED`

---

## Execution Order & Status

| Plan | Title | Priority | Effort | Depends on | Status |
|------|-------|----------|--------|------------|--------|
| **001** | Data generator overhaul | P1 | L | 031 | SUPERSEDED (by 032) |
| **002** | Discrete ambiguity network | P1 | M | 034 | SUPERSEDED (by 035) |
| **003** | Evaluation and OOD | P1 | M | 034 | SUPERSEDED (by 034, 035, 037) |
| **015** | Option A reference-guided fallback (**future only**) | P3 | M–L | B stable + lab protocol | TODO |
| **019** | Fix gzip codec + dataloader throughput (GPU starvation) | P1 | S | none | DONE |
| **020** | Reference-Guided Dual-Sensor PINN & Calibration Pipeline | P1 | M–L | none | SUPERSEDED (by 031–036) |
| **021** | Physics State Space Network (PSSN) | P1 | M–L | 020 | SUPERSEDED (by 031–036) |
| **022** | Physics-Constrained Latent Corrector Network (PCLCN) | P1 | M–L | 020, 021 | SUPERSEDED (by 031–036) |
| **023** | Integrate PCLCN Architecture Config & Model Dispatch | P1 | M | 022 | DONE |
| **024** | Unpack `grad_phi2` Batch Tuple in Training & Evaluation Loops | P1 | S | 023 | DONE |
| **025** | Fix Undefined `subset` Variable in Noise Comparison Grid | P2 | S | none | DONE |
| **026** | Add PCLCN End-to-End Integration Tests | P1 | S | 023, 024 | DONE |
| **027** | Refactor `src/` Over-Engineering & Simplify Utilities (Ponytail Audit) | P2 | S | none | DONE |
| **028** | Purge Legacy `UNetRes2` Models & Oracle Data Leaks from `src/` | P1 | M | none | DONE |
| **029** | Modernize PCLCN Ablation & Multi-Seed Framework (Fast Core Matrix) | P1 | M | 028 | DONE |
| **030** | Modernize & Refactor CLI with Flat Architecture & Purge Plot Clutter (Ponytail) | P1 | M | 029 | DONE |
| **031** | Freeze the scientific contract and pass the identifiability gate | P1 | M | none | TODO |
| **032** | Build a provenance-controlled generator and split benchmark | P1 | L | 031 | TODO |
| **033** | Enforce one observation contract from training to deployment | P1 | M | 031, 032 | TODO |
| **034** | Establish the locked baseline and evaluation protocol | P1 | M | 032, 033 | TODO |
| **035** | Red-team discrete ambiguity and calibrated uncertainty | P1 | M | 034 | TODO |
| **036** | Gate and ablate the physics-guided architecture | P1 | L | 034, 035 | TODO |
| **037** | Calibrate robustness and transfer to real measurements | P1 | L | 032, 036 | TODO |
| **038** | Produce reproducible multi-seed evidence and the Optics Express manuscript | P1 | M–L | 034, 036, 037 | TODO |

---

## Dependency Notes

- **023** unblocks PCLCN model and loss instantiation in `train.py`.
- **024** depends on **023** because `grad_phi2` must be forwarded into `PCLCNModel`.
- **026** depends on **023** and **024** to execute end-to-end integration tests on the completed PCLCN pipeline.
- **025** is an independent visualization bug fix.
- **027** is an independent refactoring plan derived from the ponytail audit.
- **028** purges legacy baseline code (`UNetRes2`, `gt_center`) so `src/` contains PCLCN exclusively.
- **029** depends on **028** to modernize `ablation.py` and `multiseed.py` for Option 1 Fast Core PCLCN Matrix.
- **031** is a hard scientific gate; do not implement a final architecture before its acquisition contract and red-team report are accepted.
- **032–034** establish provenance, deployability, and evaluation before model novelty.
- **035** is a falsifiable ambiguity/uncertainty spike and may end in rejection.
- **036** is conditional on a measurable gap over the simple baseline; every physics module requires an ablation.
- **037–038** are evidence and publication phases, not substitutes for an identifiable measurement.

---

## Findings Considered and Rejected

- **TIE Loss Integration**: Rejected in Plan 022 because single-plane 2D interferometry lacks axial intensity derivative $\partial I / \partial z$.
- **$a_{\text{calib}}$ System Gain Fitting**: Purged as a data leak requiring ground-truth phase $\phi_{\text{gt}}$.
- **Reference-free unrestricted single-frame phase recovery**: rejected as non-identifiable under $I=2+2\cos(\Delta\phi)$; only a constrained-prior or ambiguity-aware claim is permissible.
- **Pixelwise sign/wrap classification as a complete solution**: rejected; it does not enforce spatial consistency and cannot resolve identical observations.
- **Fixed-carrier Fourier demodulation on the current on-axis generator**: rejected until a measured carrier and sideband-separation test exist.
- **Single-plane TIE loss**: rejected because $\partial I/\partial z$ is unavailable.
- **Entropy without calibration**: rejected as a publishable uncertainty claim; require ECE/Brier/selective-risk evidence.

## Recommended execution order

Run **031 → 032 → 033 → 034** first. Only then run the **035** ambiguity spike. Promote **036** only if the spike and baselines identify a real, deployment-valid gap. Finish with **037 → 038**. Plans **020–022** remain historical design records and are superseded by this sequence; they must not be treated as scientific validation.

---

## Executor Quick Start

```bash
# Verify suite before/after each plan
uv run pytest tests/ -q
```
