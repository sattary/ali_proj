"""Small PNG/NPY writer for inference outputs."""

from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_inference_outputs(
    outputs: dict[str, np.ndarray], out_dir: str, *, mode: str = "unknown"
) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, values in outputs.items():
        np.save(root / f"{name}.npy", values)
    names = [n for n in ("I1", "I2", "wrapped", "prediction") if n in outputs]

    summary: dict[str, object] = {
        "mode": mode,
        "num_samples": int(len(outputs[names[0]])) if names else 0,
        "ground_truth_available": "phi_gt" in outputs,
    }
    if "phi_gt" in outputs and "prediction" in outputs:
        pred = np.asarray(outputs["prediction"], dtype=np.float64)
        gt = np.asarray(outputs["phi_gt"], dtype=np.float64)
        offset = (gt - pred).mean(axis=(1, 2), keepdims=True)
        error = pred + offset - gt
        per_mae = np.abs(error).mean(axis=(1, 2))
        per_rmse = np.sqrt((error**2).mean(axis=(1, 2)))
        summary["piston_aligned_mae"] = float(per_mae.mean())
        summary["piston_aligned_rmse"] = float(np.sqrt((error**2).mean()))
        summary["worst_sample_mae"] = float(per_mae.max())
        summary["worst_sample_rmse"] = float(per_rmse.max())
        summary["per_sample_mae"] = per_mae.tolist()
        summary["per_sample_rmse"] = per_rmse.tolist()
        print(
            f"[inference] piston-aligned MAE={summary['piston_aligned_mae']:.4f} "
            f"RMSE={summary['piston_aligned_rmse']:.4f} "
            f"worst_MAE={summary['worst_sample_mae']:.4f}"
        )
    else:
        print("[inference] ground truth unavailable; saved predictions only")
    (root / "summary.json").write_text(json.dumps(summary, indent=2))

    if not names:
        return
    count = len(outputs[names[0]])
    for i in range(count):
        fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, names):
            image = outputs[name][i] if outputs[name].ndim == 3 else outputs[name]
            cmap = "twilight" if name == "wrapped" else "viridis"
            ax.imshow(image, cmap=cmap)
            ax.set_title(name)
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(root / f"sample_{i:03d}.png", dpi=150)
        plt.close(fig)
