# Master Prompt: Publication-Quality Visualization Suite

Implement a comprehensive visualization system for Deep Learning experiments producing publication-ready figures. Combines seaborn's statistical visualization with Nature journal aesthetics.

## Architecture Philosophy

This module follows **Self-Contained Plot Pattern** where each visualization function:

1. **Handles Own Data Loading**: Loads metrics, checkpoints, or runs inference as needed
2. **Uses Shared Style System**: Context manager applies consistent aesthetics
3. **Exports Multi-Format**: PNG (quick view), PDF (LaTeX), SVG (editable)
4. **Supports Multi-Seed**: Aggregates runs with confidence intervals
5. **Statistical Annotations**: Significance testing and brackets
6. **Domain-Agnostic Core**: Style system reusable; plot functions domain-specific

## Design Decisions

1. **Style Module Separation**: Central `style.py` with rcParams, palettes, utilities
2. **Context Manager**: `nature_style()` temporarily applies publication settings
3. **Nature Standards**: Serif fonts, specific color palette, 300 DPI
4. **Colorblind Safety**: All palettes verified colorblind-friendly
5. **Perceptual Colormaps**: `twilight` (cyclic), `cividis` (continuous), `RdBu_r` (diverging)
6. **Figure Dimensions**: 89mm (single column), 183mm (double column)

## Integration Pattern

```python
from your_project.visualize import plot_training_curve, nature_style

# Single run visualization
plot_training_curve("runs/experiment_001", out_path="results/training_curve")

# Multi-seed visualization (aggregate.csv auto-detected)
plot_training_curve("runs/multiseed_experiment/", out_path="results/training_curve")

# Manual style context (for custom plots)
with nature_style():
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.6))
    sns.lineplot(data=df, x="epoch", y="loss", ax=ax)
    save_figure(fig, "results/custom_plot")
```

## Directory Structure

```
src/your_project/visualize/
├── __init__.py              # Exports all plot functions
├── style.py                 # Nature style system and utilities
├── training_curve.py        # Loss/MAE convergence with confidence bands
├── error_histogram.py       # Error distribution with KDE
├── prediction_scatter.py    # Predicted vs ground truth scatter
├── qualitative_grid.py      # Sample predictions grid
├── method_comparison.py     # Bar charts with significance testing
├── multiseed_comparison.py  # Multi-run aggregation
├── convergence.py           # Convergence rate analysis
├── phase_profile.py         # 1D phase line profiles
├── loss_landscape.py        # 2D loss surface visualization
├── residual_analysis.py     # Residual error patterns
├── tta_benefit.py          # Test-time augmentation analysis
└── _epoch_visuals.py       # Quick per-epoch training samples
```

## Style System Components

### Constants
- `SINGLE_COL`: 89mm (3.504 inches)
- `DOUBLE_COL`: 183mm (7.205 inches)
- `DPI`: 300
- `MM_TO_INCH`: Conversion factor

### Nature Color Palette
```python
NATURE_PALETTE = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "teal": "#009E73",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "sky_blue": "#56B4E9",
    # ... more
}
```

### Colormaps
- `CMAP_PHASE`: "twilight" (cyclic for wrapped phase)
- `CMAP_INTENSITY`: "gray" (interferogram)
- `CMAP_ERROR_SIGNED`: "RdBu_r" (diverging)
- `CMAP_ERROR_ABS`: "inferno" (sequential)

### rcParams (NATURE_RC)
- Serif fonts: Computer Modern, Times New Roman, DejaVu Serif
- Font sizes: 10pt base, 9pt labels/titles, 7pt ticks/legend
- Clean axes: No top/right spines
- Professional grids: Dashed, subtle

## Utility Functions in style.py

- `nature_style()`: Context manager applying Nature rcParams
- `save_figure(fig, path)`: Save as PNG/PDF/SVG simultaneously
- `add_colorbar(ax, im, label)`: Slim colorbar with proper sizing
- `annotate_significance(ax, x1, x2, y, pvalue)`: Statistical bracket
- `compute_statistical_test(group1, group2, test)`: t-test/Mann-Whitney
- `format_pvalue(pval)`: "< 0.001" or formatted value
- `create_nature_palette(n_colors)`: Generate palette

## Plot Function Signatures

```python
# Training convergence
def plot_training_curve(run_dir, out_path=None, show_lr=True, confidence=0.95)

# Error distribution
def plot_error_histogram(run_dir, out_path=None, bins=50, kde=True)

# Predictions vs targets
def plot_prediction_scatter(run_dir, out_path=None, max_samples=1000)

# Qualitative samples grid
def plot_qualitative_grid(run_dir, checkpoint_path, out_path=None, n_samples=8)

# Method comparison bars
def plot_method_comparison(results_dict, out_path=None, test="ttest")

# Multi-seed aggregation
def plot_multiseed_comparison(run_dirs, out_path=None)

# Convergence rate
def plot_convergence(run_dir, out_path=None, window=10)
```

## Multi-Seed Support

When `aggregate.csv` exists (from multi-seed runs), plots automatically:
- Show mean line with shaded confidence band
- Use `_mean` and `_std` column suffixes
- Compute convergence epoch from stability

## Statistical Testing

Built-in support for:
- Student's t-test (parametric)
- Mann-Whitney U (non-parametric)
- Wilcoxon signed-rank (paired)
- Automatic significance stars: *** (p<0.001), ** (p<0.01), * (p<0.05)

---

```json
{
  "module_name": "visualization_suite",
  "package_path": "src/{PROJECT_NAME}/visualize/",
  "style_system": {
    "file": "style.py",
    "constants": {
      "SINGLE_COL": "89 * MM_TO_INCH (~3.504 inches)",
      "DOUBLE_COL": "183 * MM_TO_INCH (~7.205 inches)",
      "DPI": "300",
      "MM_TO_INCH": "1.0 / 25.4"
    },
    "color_palette": {
      "name": "NATURE_PALETTE",
      "type": "dict[str, str]",
      "colors": {
        "blue": "#0072B2",
        "vermillion": "#D55E00",
        "teal": "#009E73",
        "pink": "#CC79A7",
        "yellow": "#F0E442",
        "sky_blue": "#56B4E9"
      },
      "description": "Colorblind-safe palette from Nature guidelines"
    },
    "colormaps": {
      "CMAP_PHASE": "'twilight' (cyclic)",
      "CMAP_INTENSITY": "'gray' (grayscale)",
      "CMAP_ERROR_SIGNED": "'RdBu_r' (diverging)",
      "CMAP_ERROR_ABS": "'inferno' (sequential)",
      "CMAP_CONTINUOUS": "'cividis' (perceptually uniform)",
      "CMAP_DIVERGING": "'coolwarm' (centered differences)"
    },
    "rcparams": {
      "font.family": "'serif'",
      "font.serif": "['Computer Modern', 'Times New Roman', 'DejaVu Serif']",
      "font.size": "10",
      "axes.labelsize": "9",
      "axes.titlesize": "9",
      "axes.linewidth": "0.6",
      "axes.spines.top": "False",
      "axes.spines.right": "False",
      "figure.dpi": "300",
      "savefig.dpi": "300"
    }
  },
  "components": [
    {
      "name": "style",
      "file": "style.py",
      "exports": [
        {
          "name": "nature_style",
          "type": "contextmanager",
          "signature": "nature_style() -> Generator[None, None, None]",
          "description": "Temporarily apply Nature publication style"
        },
        {
          "name": "save_figure",
          "signature": "save_figure(fig: plt.Figure, path: str) -> None",
          "description": "Save as PNG/PDF/SVG simultaneously, then close fig"
        },
        {
          "name": "add_colorbar",
          "signature": "add_colorbar(ax, im, label, **kwargs) -> Colorbar",
          "description": "Add slim colorbar with proper axis sizing"
        },
        {
          "name": "annotate_significance",
          "signature": "annotate_significance(ax, x1, x2, y, pvalue, h=0.02, text_offset=0.01) -> None",
          "description": "Draw significance bracket with stars and p-value"
        },
        {
          "name": "compute_statistical_test",
          "signature": "compute_statistical_test(group1, group2, test='ttest') -> Tuple[float, float]",
          "description": "Run t-test, Mann-Whitney, or Wilcoxon"
        },
        {
          "name": "format_pvalue",
          "signature": "format_pvalue(pval: float) -> str",
          "description": "Format p-value for display"
        },
        {
          "name": "create_nature_palette",
          "signature": "create_nature_palette(n_colors: int = 6) -> list[str]",
          "description": "Generate Nature-compliant color list"
        }
      ],
      "constants": ["SINGLE_COL", "DOUBLE_COL", "DPI", "NATURE_PALETTE", "*CMAP_*"]
    },
    {
      "name": "training_curve",
      "file": "training_curve.py",
      "signature": "plot_training_curve(run_dir: str, out_path: str | None = None, show_lr: bool = True, confidence: float = 0.95)",
      "description": "Plot training loss and validation metric with confidence bands",
      "features": ["Multi-seed aggregation", "Convergence epoch detection", "Dual y-axes", "Learning rate subplot"],
      "data_source": "metrics.csv or aggregate.csv"
    },
    {
      "name": "error_histogram",
      "file": "error_histogram.py",
      "signature": "plot_error_histogram(run_dir: str, out_path: str | None = None, bins: int = 50, kde: bool = True)",
      "description": "Error distribution with seaborn KDE overlay",
      "features": ["Rug plot option", "Mean/median annotations", "Normal fit comparison"]
    },
    {
      "name": "prediction_scatter",
      "file": "prediction_scatter.py",
      "signature": "plot_prediction_scatter(run_dir: str, out_path: str | None = None, max_samples: int = 1000)",
      "description": "Predicted vs ground truth scatter with identity line",
      "features": ["Hexbin option for large N", "R² annotation", "MAE/RMSE stats"]
    },
    {
      "name": "qualitative_grid",
      "file": "qualitative_grid.py",
      "signature": "plot_qualitative_grid(run_dir: str, checkpoint_path: str, out_path: str | None = None, n_samples: int = 8)",
      "description": "Grid of sample predictions showing input, prediction, target, error",
      "features": ["Automatic sample selection", "Colorbar per column", "Error heatmaps"]
    },
    {
      "name": "method_comparison",
      "file": "method_comparison.py",
      "signature": "plot_method_comparison(results_dict: dict, out_path: str | None = None, test: str = 'ttest')",
      "description": "Bar chart comparing multiple methods with significance",
      "features": ["Grouped bars", "Error bars", "Significance brackets", "Multiple metrics"]
    },
    {
      "name": "multiseed_comparison",
      "file": "multiseed_comparison.py",
      "signature": "plot_multiseed_comparison(run_dirs: list[str], out_path: str | None = None)",
      "description": "Compare multiple independent runs with confidence intervals",
      "features": ["Aggregation across seeds", "Mean ± std bands", "Convergence comparison"]
    },
    {
      "name": "convergence",
      "file": "convergence.py",
      "signature": "plot_convergence(run_dir: str, out_path: str | None = None, window: int = 10)",
      "description": "Analyze convergence rate and stability",
      "features": ["First to target", "Time to convergence", "Stability metrics"]
    }
  ],
  "integration_points": [
    {
      "location": "After training completes",
      "code": "from your_project.visualize import plot_training_curve, plot_error_histogram; plot_training_curve(cfg.logging.run_dir); plot_error_histogram(cfg.logging.run_dir)",
      "reason": "Generate standard visualizations for every run"
    },
    {
      "location": "After multi-seed training",
      "code": "plot_multiseed_comparison(seed_run_dirs)",
      "reason": "Visualize statistical significance across seeds"
    },
    {
      "location": "For paper figures",
      "code": "with nature_style(): # custom plot for specific figure",
      "reason": "Ensure publication quality"
    },
    {
      "location": "In training loop (per epoch)",
      "code": "from visualize._epoch_visuals import save_epoch_visuals; save_epoch_visuals(model, batch, epoch, out_dir)",
      "reason": "Quick training progress visualization"
    }
  ],
  "customization_variables": {
    "PROJECT_NAME": "Your package name",
    "PRIMARY_METRIC": "Main metric name (e.g., 'MAE', 'Accuracy')",
    "METRIC_UNITS": "Units for metric (e.g., 'rad', 'dB', '%')",
    "INPUT_KEY": "Dataset key for input (e.g., 'I', 'image')",
    "TARGET_KEY": "Dataset key for target (e.g., 'phi', 'label')",
    "PLOT_FUNCTIONS": "List of domain-specific plots to implement"
  },
  "dependencies": ["matplotlib", "seaborn", "numpy", "scipy", "pandas (optional)"],
  "smoke_tests": [
    {
      "name": "Style Context Manager",
      "test": "with nature_style(): fig, ax = plt.subplots(); ax.plot([1,2,3]); plt.close(fig); # No error"
    },
    {
      "name": "Save Figure Multi-Format",
      "test": "with nature_style(): fig, ax = plt.subplots(); save_figure(fig, 'test_plot'); assert Path('test_plot.png').exists(); assert Path('test_plot.pdf').exists()"
    },
    {
      "name": "Statistical Test",
      "test": "stat, pval = compute_statistical_test(np.random.randn(100), np.random.randn(100)); assert isinstance(pval, float)"
    },
    {
      "name": "Color Palette",
      "test": "colors = create_nature_palette(6); assert len(colors) == 6; assert all(isinstance(c, str) for c in colors)"
    }
  ]
}
```
