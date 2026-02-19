"""
Shared matplotlib style for Nature-level publication figures.

Sets up sans-serif fonts, LaTeX math rendering, perceptually uniform
colormaps following optics publishing conventions, and standard figure
dimensions matching Nature journal requirements.
"""

from __future__ import annotations

import contextlib
from typing import Generator

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Figure dimensions (Nature: 89mm single-col, 183mm double-col)
# ---------------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
SINGLE_COL = 89 * MM_TO_INCH  # ~3.504 inches
DOUBLE_COL = 183 * MM_TO_INCH  # ~7.205 inches
DPI = 300

# ---------------------------------------------------------------------------
# Colormaps (optics convention, perceptually uniform, colorblind-safe)
# ---------------------------------------------------------------------------
CMAP_PHASE = "twilight"  # cyclic for wrapped phase
CMAP_INTENSITY = "gray"  # interferogram
CMAP_ERROR_SIGNED = "RdBu_r"  # signed error (diverging)
CMAP_ERROR_ABS = "inferno"  # unsigned / absolute error
CMAP_CONTINUOUS = "cividis"  # general continuous (colorblind-safe)

# ---------------------------------------------------------------------------
# Color palette for line plots (6 distinguishable colors)
# ---------------------------------------------------------------------------
LINE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # teal
    "#CC79A7",  # pink
    "#F0E442",  # yellow
    "#56B4E9",  # sky blue
]

# ---------------------------------------------------------------------------
# Nature-style rcParams
# ---------------------------------------------------------------------------
NATURE_RC = {
    # fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    # math rendering (LaTeX-style without requiring TeX installation)
    "mathtext.fontset": "dejavusans",
    # figure defaults
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    # axes
    "axes.linewidth": 0.5,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # ticks
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # lines
    "lines.linewidth": 1.0,
    "lines.markersize": 3,
    # legend
    "legend.frameon": False,
    "legend.borderpad": 0.2,
    "legend.handlelength": 1.5,
    # grid (when used)
    "grid.linewidth": 0.3,
    "grid.alpha": 0.4,
}


@contextlib.contextmanager
def nature_style() -> Generator[None, None, None]:
    """Context manager that temporarily applies Nature-style rcParams."""
    old_rc = mpl.rcParams.copy()
    mpl.rcParams.update(NATURE_RC)
    try:
        yield
    finally:
        mpl.rcParams.update(old_rc)


def save_figure(fig: plt.Figure, path: str, also_pdf: bool = True) -> None:
    """Save figure as PNG (raster) and optionally PDF (vector)."""
    fig.savefig(path, dpi=DPI)
    if also_pdf and path.lower().endswith(".png"):
        fig.savefig(path.replace(".png", ".pdf"))
    plt.close(fig)


def add_colorbar(
    ax: plt.Axes,
    im: mpl.image.AxesImage,
    label: str = "",
    **kwargs,
) -> mpl.colorbar.Colorbar:
    """Add a slim colorbar to an image axis."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.04)
    cb = plt.colorbar(im, cax=cax, **kwargs)
    if label:
        cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=5)
    return cb
