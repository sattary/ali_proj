"""
Visualizer for real-world inference outputs.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_inference_result(phi_pred: np.ndarray, H: int, W: int, out_prefix: str) -> None:
    """
    Save the unwrapped phase to .npy and render a heatmap visualization.
    """
    out_dir = Path(out_prefix).parent
    if str(out_dir) != "" and str(out_dir) != ".":
        out_dir.mkdir(parents=True, exist_ok=True)

    npy_out = f"{out_prefix}.npy"
    png_out = f"{out_prefix}.png"

    np.save(npy_out, phi_pred)

    plt.figure(figsize=(10, 8))
    plt.imshow(phi_pred, cmap="viridis")
    plt.colorbar(label="Absolute Phase (rad)")
    plt.title(f"Unwrapped Phase (H={H}, W={W})")
    plt.tight_layout()
    plt.savefig(png_out, dpi=150, bbox_inches="tight")
    plt.close()

    print("Inference complete.")
    print(f" -> Raw Phase Matrix: {npy_out}")
    print(f" -> Visualization:    {png_out}")
