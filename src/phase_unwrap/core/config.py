"""
Structured configuration for training and data generation.

All hyperparameters are organized into typed dataclass groups.
Configuration can be loaded from JSON or YAML files, with missing
fields falling back to built-in defaults.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    """Data-related configuration."""

    data_dir: str = "data"
    pattern: str = "*.h5"
    I_key: str = "I"
    phi_key: str = "phi"
    val_frac: float = 0.1
    test_frac: float = 0.1
    workers: int = 4
    persistent_workers: bool = False
    augment: bool = True


@dataclass
class ModelConfig:
    """Model and runtime configuration."""

    base: int = 32
    activation: str = "silu"  # "relu" or "silu"
    final_dropout: float = 0.3
    ema_decay: float = 0.999
    device: str = "auto"  # "auto", "cuda", "cpu"
    use_amp: bool = False
    use_coordconv: bool = True


@dataclass
class OptimizationConfig:
    """Optimization hyperparameters."""

    epochs: int = 200
    batch_size: int = 40
    lr: float = 3e-4
    eta_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    warmup_steps: int = 500
    compile: bool = True


@dataclass
class LossConfig:
    """Weights for supervised and regularization terms."""

    w_mae: float = 1.0
    w_grad: float = 0.1
    int_wgrad: bool = False
    w_curv: float = 0.01
    w_data: float = 1.0


@dataclass
class LoggingConfig:
    """Logging, output, and reproducibility options."""

    run_name: str = "default"
    runs_root: str = "runs"
    vis_max: int = 8
    val_interval: int = 1
    seed: int = 1337

    @property
    def run_dir(self) -> str:
        return str(Path(self.runs_root) / self.run_name)

    @property
    def vis_dir(self) -> str:
        return str(Path(self.runs_root) / self.run_name / "visuals")


@dataclass
class AugmentationConfig:
    """Optical curriculum noise hyperparameters."""

    enable: bool = True
    warmup_ratio: float = 0.1
    full_ratio: float = 0.4
    profile: str = "cosine"
    cycles: float = 2.0
    stochastic_std: float = 0.05
    gauss_std: float = 0.02
    speckle_std: float = 0.05
    poisson_scale: float = 0.0
    lowfreq_amp: float = 0.05
    blur_prob: float = 0.2
    blur_min: float = 0.5
    blur_max: float = 1.0
    dropout_prob: float = 0.02
    sap_prob: float = 0.0
    gain_min: float = 0.9
    gain_max: float = 1.1
    off_min: float = -0.05
    off_max: float = 0.05
    hint_std: float = 0.2
    # Option B (default): zero second channel — train/deploy match lab (no GT).
    # "gt_center" is ablation-only (GT absolute leak). Option A = plan 015 later.
    hint_mode: str = "zero"  # "zero" | "gt_center"


@dataclass
class TrainConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    aug: AugmentationConfig = field(default_factory=AugmentationConfig)


ConfigPath = Union[str, Path]


def _update_dataclass(dc: Any, values: Dict[str, Any]) -> Any:
    """Recursively update a dataclass instance from a nested mapping."""
    if not is_dataclass(dc):
        return dc
    for f in fields(dc):
        if f.name not in values:
            continue
        current = getattr(dc, f.name)
        new_val = values[f.name]
        if is_dataclass(current) and isinstance(new_val, dict):
            setattr(dc, f.name, _update_dataclass(current, new_val))
        else:
            setattr(dc, f.name, new_val)
    return dc


def _load_mapping(path: Path) -> Dict[str, Any]:
    """Load a configuration mapping from JSON or YAML."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text()
    if suffix in {".json"}:
        return json.loads(text)
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install `pyyaml` or use JSON instead."
            )
        return yaml.safe_load(text) or {}
    try:
        return json.loads(text)
    except Exception:
        if yaml is not None:
            return yaml.safe_load(text) or {}
        raise


def load_train_config(path: Optional[ConfigPath] = None) -> TrainConfig:
    """Load a TrainConfig from an optional JSON/YAML file."""
    cfg = TrainConfig()
    if path is None:
        return cfg
    p = Path(path)
    data = _load_mapping(p)
    return _update_dataclass(cfg, data)


def config_to_yaml(cfg: TrainConfig) -> str:
    """Serialize a TrainConfig to YAML string."""
    d = asdict(cfg)
    if yaml is not None:
        return yaml.dump(d, default_flow_style=False, sort_keys=False)
    return json.dumps(d, indent=2)
