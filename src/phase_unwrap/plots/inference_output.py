"""Small PNG/NPY writer for inference outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_inference_outputs(outputs: dict[str, np.ndarray], out_dir: str) -> None:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    for name, values in outputs.items():
        np.save(root / f"{name}.npy", values)

    names = [n for n in ("I1", "I2", "wrapped", "prediction") if n in outputs]
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

