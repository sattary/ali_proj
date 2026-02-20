# Master Prompt: Hierarchical Configuration System

Implement a type-safe, hierarchical configuration system for Deep Learning experiments using Python dataclasses. Supports JSON and YAML loading with automatic nested merging and sensible defaults.

## Architecture Philosophy

This module follows **Structured Configuration Pattern** using dataclasses to ensure:

1. **Type Safety**: All config fields are typed; IDEs provide autocomplete and validation
2. **Hierarchical Organization**: Configs are grouped by domain (data, model, optimization, loss, logging)
3. **Sensible Defaults**: Every field has a default value; configs work out-of-the-box
4. **File Loading**: JSON and YAML support with graceful fallback
5. **Nested Merging**: Partial config files only override specified fields
6. **Computed Properties**: Dynamic values (e.g., run_dir, vis_dir) derived from other fields

## Design Decisions

1. **Dataclasses over Dictionaries**: Compile-time type checking, IDE support, immutability options
2. **Hierarchical Grouping**: Separate dataclasses for Data, Model, Optimization, Loss, Logging
3. **Top-Level Aggregation**: Single `TrainConfig` that composes all sub-configs
4. **File Format Agnostic**: Auto-detect JSON vs YAML from file extension
5. **Partial Updates**: Config files only need to specify non-default values
6. **Path Safety**: All path operations use `pathlib.Path`

## Integration Pattern

```python
# Option 1: Default config (all defaults)
from your_project.core.config import TrainConfig
cfg = TrainConfig()

# Option 2: Load from file (JSON or YAML)
cfg = load_train_config("configs/experiment.yaml")

# Option 3: Programmatic modification
cfg.optim.epochs = 500
cfg.logging.run_name = "experiment_v2"

# Access nested values
print(cfg.data.data_dir)
print(cfg.optim.lr)
print(cfg.logging.run_dir)  # Computed property
```

## Configuration File Example (YAML)

```yaml
data:
  data_dir: "data/custom"
  val_frac: 0.15
  workers: 8

model:
  base: 64
  activation: "silu"
  final_dropout: 0.5

optim:
  epochs: 300
  batch_size: 64
  lr: 1.0e-4

loss:
  w_mae: 1.0
  w_grad: 0.2

logging:
  run_name: "experiment_001"
  seed: 42
```

## Directory Structure

```
src/your_project/core/
├── __init__.py
└── config.py          # All configuration classes and loading functions
```

## Config Groups

### DataConfig
- `data_dir`: Dataset location
- `pattern`: File glob pattern
- `I_key`/`phi_key`: Dataset key names for inputs/targets
- `val_frac`: Validation split fraction
- `workers`: DataLoader workers
- `augment`: Enable augmentation flag

### ModelConfig
- `base`: Base channel width multiplier
- `activation`: "relu" or "silu"
- `final_dropout`: Dropout probability
- `ema_decay`: EMA model decay rate
- `device`: "auto", "cuda", or "cpu"
- `use_amp`: Automatic Mixed Precision flag

### OptimizationConfig
- `epochs`: Training epochs
- `batch_size`: Per-GPU batch size (total = batch_size × num_gpus)
- `lr`: Peak learning rate
- `eta_min`: Minimum LR for scheduler
- `weight_decay`: L2 regularization
- `grad_clip`: Gradient clipping threshold
- `warmup_steps`: Linear warmup steps

### LossConfig
- `w_mae`: L1 loss weight
- `w_grad`: Gradient loss weight
- `int_wgrad`: Integer gradient mode
- `w_curv`: Curvature loss weight
- `w_data`: Data consistency weight

### LoggingConfig
- `run_name`: Experiment name (determines run_dir)
- `runs_root`: Root directory for all runs
- `vis_max`: Max visualizations per epoch
- `val_interval`: Validate every N epochs
- `seed`: Random seed for reproducibility
- `run_dir` (computed): Full path to run directory
- `vis_dir` (computed): Full path to visualization directory

## Loading Behavior

1. **No file provided**: Returns `TrainConfig()` with all defaults
2. **Partial file**: Only specified fields are overridden; others use defaults
3. **Nested override**: Can override nested fields (e.g., only `data.val_frac`)
4. **Unknown fields**: Silently ignored (allows config forward-compatibility)

## Serialization

```python
# To YAML (for saving configs alongside runs)
yaml_str = config_to_yaml(cfg)
Path("run_dir/config.yaml").write_text(yaml_str)

# To dict (for logging frameworks)
from dataclasses import asdict
cfg_dict = asdict(cfg)
```

---

```json
{
  "module_name": "hierarchical_config",
  "package_path": "src/{PROJECT_NAME}/core/config.py",
  "components": [
    {
      "name": "DataConfig",
      "type": "dataclass",
      "fields": [
        {"name": "data_dir", "type": "str", "default": "\"data\""},
        {"name": "pattern", "type": "str", "default": "\"*.h5\""},
        {"name": "I_key", "type": "str", "default": "\"I\""},
        {"name": "phi_key", "type": "str", "default": "\"phi\""},
        {"name": "val_frac", "type": "float", "default": "0.1"},
        {"name": "workers", "type": "int", "default": "4"},
        {"name": "augment", "type": "bool", "default": "true"}
      ]
    },
    {
      "name": "ModelConfig",
      "type": "dataclass",
      "fields": [
        {"name": "base", "type": "int", "default": "32"},
        {"name": "activation", "type": "str", "default": "\"relu\""},
        {"name": "final_dropout", "type": "float", "default": "0.3"},
        {"name": "ema_decay", "type": "float", "default": "0.999"},
        {"name": "device", "type": "str", "default": "\"auto\""},
        {"name": "use_amp", "type": "bool", "default": "false"}
      ]
    },
    {
      "name": "OptimizationConfig",
      "type": "dataclass",
      "fields": [
        {"name": "epochs", "type": "int", "default": "200"},
        {"name": "batch_size", "type": "int", "default": "40"},
        {"name": "lr", "type": "float", "default": "3e-4"},
        {"name": "eta_min", "type": "float", "default": "1e-6"},
        {"name": "weight_decay", "type": "float", "default": "1e-4"},
        {"name": "grad_clip", "type": "float", "default": "5.0"},
        {"name": "warmup_steps", "type": "int", "default": "500"}
      ]
    },
    {
      "name": "LossConfig",
      "type": "dataclass",
      "fields": [
        {"name": "w_mae", "type": "float", "default": "1.0"},
        {"name": "w_grad", "type": "float", "default": "0.1"},
        {"name": "int_wgrad", "type": "bool", "default": "false"},
        {"name": "w_curv", "type": "float", "default": "0.01"},
        {"name": "w_data", "type": "float", "default": "1.0"}
      ]
    },
    {
      "name": "LoggingConfig",
      "type": "dataclass",
      "fields": [
        {"name": "run_name", "type": "str", "default": "\"default\""},
        {"name": "runs_root", "type": "str", "default": "\"runs\""},
        {"name": "vis_max", "type": "int", "default": "8"},
        {"name": "val_interval", "type": "int", "default": "1"},
        {"name": "seed", "type": "int", "default": "1337"}
      ],
      "computed_properties": [
        {
          "name": "run_dir",
          "return_type": "str",
          "implementation": "str(Path(self.runs_root) / self.run_name)"
        },
        {
          "name": "vis_dir",
          "return_type": "str",
          "implementation": "str(Path(self.runs_root) / self.run_name / 'visuals')"
        }
      ]
    },
    {
      "name": "TrainConfig",
      "type": "dataclass",
      "description": "Top-level configuration aggregating all sub-configs",
      "fields": [
        {"name": "data", "type": "DataConfig", "default_factory": "DataConfig"},
        {"name": "model", "type": "ModelConfig", "default_factory": "ModelConfig"},
        {"name": "optim", "type": "OptimizationConfig", "default_factory": "OptimizationConfig"},
        {"name": "loss", "type": "LossConfig", "default_factory": "LossConfig"},
        {"name": "logging", "type": "LoggingConfig", "default_factory": "LoggingConfig"}
      ]
    }
  ],
  "utility_functions": [
    {
      "name": "load_train_config",
      "signature": "load_train_config(path: Optional[Union[str, Path]] = None) -> TrainConfig",
      "description": "Load config from JSON/YAML file or return defaults if no path provided",
      "logic": [
        "Start with TrainConfig() defaults",
        "If path is None, return defaults",
        "Detect file type by extension (.json, .yml, .yaml)",
        "Load file content",
        "Recursively merge into dataclass (only override specified fields)",
        "Return updated config"
      ]
    },
    {
      "name": "config_to_yaml",
      "signature": "config_to_yaml(cfg: TrainConfig) -> str",
      "description": "Serialize config to YAML string for saving",
      "logic": [
        "Convert dataclass to dict using asdict()",
        "Use yaml.dump() with default_flow_style=False",
        "Return YAML string"
      ]
    },
    {
      "name": "_update_dataclass",
      "signature": "_update_dataclass(dc: Any, values: Dict[str, Any]) -> Any",
      "visibility": "private",
      "description": "Recursively update dataclass from nested mapping",
      "logic": [
        "Iterate over dataclass fields",
        "If field not in values, skip (keep default)",
        "If field is dataclass and value is dict, recurse",
        "Otherwise, set field to value"
      ]
    },
    {
      "name": "_load_mapping",
      "signature": "_load_mapping(path: Path) -> Dict[str, Any]",
      "visibility": "private",
      "description": "Load dict from JSON or YAML file",
      "logic": [
        "Check file exists",
        "Parse by extension: json.loads() or yaml.safe_load()",
        "Handle missing yaml dependency gracefully",
        "Return dict"
      ]
    }
  ],
  "integration_points": [
    {
      "location": "Training script entry point",
      "code": "cfg = load_train_config(config_path)",
      "reason": "Single source of configuration"
    },
    {
      "location": "Saving run artifacts",
      "code": "Path(cfg.logging.run_dir).mkdir(parents=True, exist_ok=True)",
      "reason": "Ensure run directory exists"
    },
    {
      "location": "CLI argument parsing",
      "code": "config_path: Optional[str] = typer.Option(None, '--config', '-c')",
      "reason": "Allow users to specify config file"
    },
    {
      "location": "Model initialization",
      "code": "model = build_model(cfg.model)",
      "reason": "Pass model config to model builder"
    },
    {
      "location": "DataLoader setup",
      "code": "train_loader = DataLoader(dataset, batch_size=cfg.optim.batch_size, num_workers=cfg.data.workers)",
      "reason": "Use data and optimization configs"
    },
    {
      "location": "After training starts",
      "code": "config_yaml = config_to_yaml(cfg); Path(cfg.logging.run_dir, 'config.yaml').write_text(config_yaml)",
      "reason": "Save full config with run for reproducibility"
    }
  ],
  "customization_variables": {
    "PROJECT_NAME": "Your Python package name",
    "DATA_FIELDS": "Domain-specific data config fields (keys, paths, etc.)",
    "MODEL_FIELDS": "Model architecture hyperparameters",
    "OPTIM_FIELDS": "Optimizer and scheduler settings",
    "LOSS_FIELDS": "Loss function weights and modes",
    "LOGGING_FIELDS": "Experiment tracking and output settings"
  },
  "dependencies": ["pyyaml (optional, for YAML support)"],
  "smoke_tests": [
    {
      "name": "Default Config Creation",
      "test": "cfg = TrainConfig(); assert cfg.data.data_dir == 'data'; assert cfg.optim.lr == 3e-4"
    },
    {
      "name": "Load JSON Config",
      "test": "Create temp JSON with {'optim': {'epochs': 500}}; cfg = load_train_config(temp_path); assert cfg.optim.epochs == 500; assert cfg.optim.lr == 3e-4 (default)"
    },
    {
      "name": "Load YAML Config",
      "test": "Create temp YAML with 'logging:\n  run_name: test'; cfg = load_train_config(temp_path); assert cfg.logging.run_name == 'test'"
    },
    {
      "name": "Computed Properties",
      "test": "cfg = TrainConfig(); cfg.logging.run_name = 'exp1'; assert 'runs/exp1' in cfg.logging.run_dir"
    },
    {
      "name": "Serialization Roundtrip",
      "test": "cfg = TrainConfig(); cfg.optim.epochs = 999; yaml_str = config_to_yaml(cfg); assert 'epochs: 999' in yaml_str"
    }
  ]
}
```
