w# Implementation Plans

Generated and reconciled by the improve skill on 2026-07-30 (`ab82cfc`).

Each executor: read the plan fully before starting, honor STOP conditions, and update your row when done.

Status values: `TODO` | `IN PROGRESS` | `DONE` | `BLOCKED` | `REJECTED`

---

## Execution Order & Status

| Plan | Title | Priority | Effort | Depends on | Status |
|------|-------|----------|--------|------------|--------|
| **015** | Option A reference-guided fallback (**future only**) | P3 | M–L | B stable + lab protocol | TODO |
| **019** | Fix gzip codec + dataloader throughput (GPU starvation) | P1 | S | none | DONE |
| **020** | Reference-Guided Dual-Sensor PINN & Calibration Pipeline | P1 | M–L | none | SUPERSEDED (by 022) |
| **021** | Physics State Space Network (PSSN) | P1 | M–L | 020 | SUPERSEDED (by 022) |
| **022** | Physics-Constrained Latent Corrector Network (PCLCN) | P1 | M–L | 020, 021 | DONE |
| **023** | Integrate PCLCN Architecture Config & Model Dispatch | P1 | M | 022 | DONE |
| **024** | Unpack `grad_phi2` Batch Tuple in Training & Evaluation Loops | P1 | S | 023 | DONE |
| **025** | Fix Undefined `subset` Variable in Noise Comparison Grid | P2 | S | none | DONE |
| **026** | Add PCLCN End-to-End Integration Tests | P1 | S | 023, 024 | DONE |

---

## Dependency Notes

- **023** unblocks PCLCN model and loss instantiation in `train.py`.
- **024** depends on **023** because `grad_phi2` must be forwarded into `PCLCNModel`.
- **026** depends on **023** and **024** to execute end-to-end integration tests on the completed PCLCN pipeline.
- **025** is an independent visualization bug fix.

---

## Findings Considered and Rejected

- **TIE Loss Integration**: Rejected in Plan 022 because single-plane 2D interferometry lacks axial intensity derivative $\partial I / \partial z$.
- **$a_{\text{calib}}$ System Gain Fitting**: Purged as a data leak requiring ground-truth phase $\phi_{\text{gt}}$.

---

## Executor Quick Start

```bash
# Verify suite before/after each plan
uv run pytest tests/ -q
```
