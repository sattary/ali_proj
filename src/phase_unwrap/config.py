from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


@dataclass
class DataConfig:
    """Data-related configuration."""

    data_dir: str = "data"
    pattern: str = "*.mat"
    I_key: str = "I"
    phi_key: str = "dphi"
    val_frac: float = 0.1
    workers: int = 4
    augment: bool = True
    flip_prob: float = 0.5
    rotate90_prob: float = 0.5
    noise_sigma: float = 0.0


@dataclass
class ModelConfig:
    """Model and runtime configuration."""

    base: int = 16
    model_type: str = "unet"  # "unet" or "swin"
    activation: str = "relu"  # "relu" or "silu"
    final_dropout: float = 0.3
    ema_decay: float = 0.999
    device: str = "auto"  # "auto", "cuda", "cpu"
    use_amp: bool = False


@dataclass
class OptimizationConfig:
    """Optimization hyperparameters."""

    epochs: int = 40
    batch_size: int = 40
    lr: float = 3e-4
    eta_min: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 5.0


@dataclass
class LossConfig:
    """Weights for supervised and regularization terms."""

    w_mae: float = 1.0
    w_grad: float = 0.1
    w_wrapped_grad: float = 0.0
    w_res: float = 0.0  # New: Weight for residue detection loss
    w_wrap: float = 0.0
    int_wgrad: bool = False
    w_curv: float = 0.003
    w_tv: float = 0.0
    w_data: float = 1.0
    use_conf_weight: bool = False
    w_conf_reg: float = 0.0


@dataclass
class LoggingConfig:
    """Logging, output, and reproducibility options."""

    out_dir: str = "runs/affine_align"
    vis_dir: str = "viz/affine_align"
    vis_max: int = 8
    val_interval: int = 1
    seed: int = 1337
    use_tensorboard: bool = True
    log_dir: str = "runs"
    log_csv: bool = True
    use_wandb: bool = False
    wandb_project: str = "phase_unwrap"
    wandb_entity: Optional[str] = None
    resume_from: Optional[str] = None  # Path to checkpoint to resume from


@dataclass
class TrainConfig:
    """Top-level configuration aggregating all sub-configs."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimizationConfig = field(default_factory=OptimizationConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


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
    # Default: try JSON first, then YAML if available.
    try:
        return json.loads(text)
    except Exception:
        if yaml is not None:
            return yaml.safe_load(text) or {}
        raise


def load_train_config(path: Optional[ConfigPath] = None) -> TrainConfig:
    """
    Load a TrainConfig from an optional JSON/YAML file.

    If `path` is None, returns a config populated with built-in defaults.
    """
    cfg = TrainConfig()
    if path is None:
        return cfg
    p = Path(path)
    data = _load_mapping(p)
    return _update_dataclass(cfg, data)


def apply_overrides(cfg: "TrainConfig", overrides: list[str]) -> "TrainConfig":
    """
    Apply dot-notation overrides of the form `section.field=value` to a TrainConfig.

    Types are inferred from the current value in the config.
    """

    def _cast_value(current: Any, raw: str) -> Any:
        if isinstance(current, bool):
            return raw.lower() in {"1", "true", "yes", "y"}
        if isinstance(current, int) and raw.isdigit():
            return int(raw)
        if isinstance(current, float):
            try:
                return float(raw)
            except ValueError:
                return current
        return raw

    for ov in overrides:
        if "=" not in ov:
            continue
        key, raw_val = ov.split("=", 1)
        parts = key.split(".")
        if not parts:
            continue
        target = cfg
        for name in parts[:-1]:
            if not hasattr(target, name):
                target = None
                break
            target = getattr(target, name)
        if target is None:
            continue
        leaf = parts[-1]
        if not hasattr(target, leaf):
            continue
        current = getattr(target, leaf)
        new_val = _cast_value(current, raw_val)
        setattr(target, leaf, new_val)

    return cfg
