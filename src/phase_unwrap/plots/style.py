"""
Shared matplotlib/seaborn style for Nature-level publication figures.

Combines seaborn's statistical elegance with Nature journal requirements:
- Serif fonts for academic publication
- Nature color palette (colorblind-safe)
- Statistical annotations support
- Multi-format export (PNG, PDF, SVG)
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Generator, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# ---------------------------------------------------------------------------
# Figure dimensions (Nature: 89mm single-col, 183mm double-col)
# ---------------------------------------------------------------------------
MM_TO_INCH = 1.0 / 25.4
SINGLE_COL = 89 * MM_TO_INCH  # ~3.504 inches
DOUBLE_COL = 183 * MM_TO_INCH  # ~7.205 inches
DPI = 300

# ---------------------------------------------------------------------------
# Nature Color Palette (colorblind-safe, from Nature's guidelines)
# ---------------------------------------------------------------------------
NATURE_PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    "purple": "#9C27B0",
    "green": "#4CAF50",
    "orange": "#FF9800",
    "gray": "#7F7F7F",
}

LINE_COLORS = list(NATURE_PALETTE.values())[:6]

# ---------------------------------------------------------------------------
# Colormaps (perceptually uniform, colorblind-safe)
# ---------------------------------------------------------------------------
CMAP_PHASE = "twilight"  # cyclic for wrapped phase
CMAP_INTENSITY = "gray"  # interferogram
CMAP_ERROR_SIGNED = "RdBu_r"  # signed error (diverging)
CMAP_ERROR_ABS = "inferno"  # unsigned / absolute error
CMAP_CONTINUOUS = "cividis"  # general continuous
CMAP_DIVERGING = "coolwarm"  # for signed differences

# ---------------------------------------------------------------------------
# Nature-style rcParams with SERIF fonts
# ---------------------------------------------------------------------------
NATURE_RC = {
    # fonts - SERIF for academic publication
    "font.family": "serif",
    "font.serif": [
        "Computer Modern",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "font.size": 10,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    # math rendering
    "mathtext.fontset": "dejavuserif",
    "mathtext.rm": "serif",
    "mathtext.it": "serif:italic",
    # figure defaults
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    # axes
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    # ticks
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # lines
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    # legend
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "gray",
    "legend.borderpad": 0.3,
    "legend.handlelength": 1.5,
    # grid
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
}


@contextlib.contextmanager
def nature_style() -> Generator[None, None, None]:
    """Context manager applying Nature-style rcParams with seaborn integration."""
    old_rc = mpl.rcParams.copy()
    mpl.rcParams.update(NATURE_RC)
    sns.set_context("paper", font_scale=1.0)
    sns.set_palette(LINE_COLORS)
    try:
        yield
    finally:
        mpl.rcParams.update(old_rc)
        sns.set()


def save_figure(fig: plt.Figure, path: str) -> None:
    """Save figure in multiple formats: PNG (quick view), PDF (LaTeX), SVG (editable)."""
    path_obj = Path(path)
    base_path = path_obj.with_suffix("")

    # Always save PNG for quick viewing
    fig.savefig(f"{base_path}.png", dpi=DPI, bbox_inches="tight")

    # Save PDF for LaTeX inclusion
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")

    # Save SVG for vector editing
    fig.savefig(f"{base_path}.svg", bbox_inches="tight")

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
        cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return cb


def annotate_significance(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    pvalue: float,
    h: float = 0.02,
    text_offset: float = 0.01,
) -> None:
    """
    Draw significance bracket with p-value annotation.

    Args:
        ax: matplotlib Axes
        x1, x2: x-coordinates of the two groups being compared
        y: y-coordinate for the bracket
        pvalue: p-value from statistical test
        h: height of bracket arms
        text_offset: vertical offset for text
    """
    # Determine significance stars
    if pvalue < 0.001:
        stars = "***"
    elif pvalue < 0.01:
        stars = "**"
    elif pvalue < 0.05:
        stars = "*"
    else:
        stars = "ns"

    # Draw bracket
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=0.8, color="black")

    # Add text
    text = f"{stars}\np={pvalue:.3f}" if stars != "ns" else "ns"
    ax.text(
        (x1 + x2) / 2, y + h + text_offset, text, ha="center", va="bottom", fontsize=7
    )


def compute_statistical_test(
    group1: np.ndarray,
    group2: np.ndarray,
    test: str = "ttest",
) -> Tuple[float, float]:
    """
    Compute statistical test between two groups.

    Args:
        group1, group2: Arrays of samples
        test: "ttest" (parametric) or "mannwhitney" (non-parametric)

    Returns:
        (statistic, pvalue)
    """
    if test == "ttest":
        stat, pval = stats.ttest_ind(group1, group2)
    elif test == "mannwhitney":
        stat, pval = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    elif test == "wilcoxon":
        stat, pval = stats.wilcoxon(group1, group2)
    else:
        raise ValueError(f"Unknown test: {test}")

    return stat, pval


def format_pvalue(pval: float) -> str:
    """Format p-value for display."""
    if pval < 0.001:
        return "< 0.001"
    elif pval < 0.01:
        return f"{pval:.3f}"
    else:
        return f"{pval:.3f}"


def create_nature_palette(n_colors: int = 6) -> list[str]:
    """Generate Nature-compliant color palette with n colors."""
    base_colors = list(NATURE_PALETTE.values())[:n_colors]
    return base_colors