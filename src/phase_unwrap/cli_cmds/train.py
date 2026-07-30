import typer
from pathlib import Path
from typing import Optional, Annotated
from ..core.config import TrainConfig, load_train_config

ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="Path to YAML or JSON config file.")]
RunNameOpt = Annotated[Optional[str], typer.Option("--run-name", help="Name for this training run.")]
UseAmpOpt = Annotated[bool, typer.Option("--use-amp", help="Enable Automatic Mixed Precision (AMP).")]
DataDirOpt = Annotated[Optional[str], typer.Option("--data-dir", help="Override dataset directory.")]
BatchSizeOpt = Annotated[Optional[int], typer.Option("--batch-size", help="Override config batch size.")]
EpochsOpt = Annotated[Optional[int], typer.Option("--epochs", help="Override config training epochs.")]
SeedsOpt = Annotated[Optional[str], typer.Option("--seeds", help="Comma-separated seeds (e.g., '1337,42,7').")]

TrainApp = typer.Typer(help="Training commands.")

def _apply_cli_overrides(
    cfg: TrainConfig,
    run_name: Optional[str] = None,
    use_amp: Optional[bool] = None,
    data_dir: Optional[str] = None,
    batch_size: Optional[int] = None,
    epochs: Optional[int] = None,
) -> None:
    """Apply common CLI overrides to the configuration."""
    if run_name is not None:
        cfg.logging.run_name = run_name
    if use_amp:
        cfg.model.use_amp = True
    if data_dir is not None:
        cfg.data.data_dir = data_dir
    if batch_size is not None:
        cfg.optim.batch_size = batch_size
    if epochs is not None:
        cfg.optim.epochs = epochs



# ============================================================================
# TRAINING COMMANDS
# ============================================================================


@TrainApp.command("train")
def train_cmd(
    config: ConfigOpt = None,
    resume: Annotated[Optional[Path], typer.Option("--resume", help="Path to checkpoint to resume from.")] = None,
    run_name: RunNameOpt = None,
    use_amp: UseAmpOpt = False,
    data_dir: DataDirOpt = None,
    batch_size: BatchSizeOpt = None,
    epochs: EpochsOpt = None,
) -> None:
    """Train the UNetRes2 absolute phase reconstruction model."""
    from phase_unwrap.training.train import train as run_train

    cfg: TrainConfig = load_train_config(config)

    _apply_cli_overrides(
        cfg,
        run_name=run_name,
        use_amp=use_amp,
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
    )



    run_train(
        cfg,
        resume_path=str(resume) if resume else None,
    )


@TrainApp.command("tune")
def tune_cmd(
    config: ConfigOpt = None,
    n_trials: Annotated[int, typer.Option("--n-trials", help="Number of Optuna trials.")] = 50,
    tune_epochs: Annotated[int, typer.Option("--tune-epochs", help="Epochs per trial.")] = 15,
    study_name: Annotated[str, typer.Option("--study-name", help="Optuna study name.")] = "phase_unwrap_hpo",
    n_workers: Annotated[int, typer.Option("--n-workers", "-j", help="Number of parallel workers.")] = 1,
    use_amp: UseAmpOpt = False,
    batch_size: BatchSizeOpt = None,
    data_dir: DataDirOpt = None,
) -> None:
    """Run Optuna hyperparameter search (TPE + MedianPruner)."""
    import torch
    from phase_unwrap.training.tune import run_tuning

    cfg: TrainConfig = load_train_config(config)

    _apply_cli_overrides(
        cfg,
        use_amp=use_amp,
        data_dir=data_dir,
        batch_size=batch_size,
    )

    gpu_id_list: Optional[list[int]] = None
    if n_workers > 1 and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id_list = [i % n_gpus for i in range(n_workers)]
        typer.echo(f"Auto-detected {n_gpus} GPUs, mapping {n_workers} workers: {gpu_id_list}")

    run_tuning(
        cfg,
        n_trials=n_trials,
        tune_epochs=tune_epochs,
        study_name=study_name,
        n_workers=n_workers,
        gpu_ids=gpu_id_list,
        batch_size_override=batch_size,
    )


@TrainApp.command("multiseed")
def multiseed_cmd(
    config: ConfigOpt = None,
    run_name: Annotated[str, typer.Option("--run-name", help="Base name for the run group.")] = "multiseed",
    seeds: SeedsOpt = None,
    num_seeds: Annotated[int, typer.Option("--num-seeds", help="Auto-generate N seeds.")] = 3,
    use_amp: UseAmpOpt = False,
    data_dir: DataDirOpt = None,
    batch_size: BatchSizeOpt = None,
    epochs: EpochsOpt = None,
) -> None:
    """Run N training runs with different seeds, aggregate results."""
    from phase_unwrap.training.multiseed import run_multiseed

    import random as _rnd

    cfg: TrainConfig = load_train_config(config)

    seed_list: list[int]
    if seeds is not None:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    else:
        seed_list = [_rnd.randint(0, 2**31) for _ in range(num_seeds)]

    _apply_cli_overrides(
        cfg,
        use_amp=use_amp,
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
    )



    run_multiseed(cfg, base_run_name=run_name, seeds=seed_list, use_amp=use_amp)


@TrainApp.command("ablation")
def ablation_cmd(
    config: ConfigOpt = None,
    run_name: Annotated[str, typer.Option("--run-name", help="Base name for ablation run group.")] = "ablation",
    seeds: SeedsOpt = None,
    num_seeds: Annotated[int, typer.Option("--num-seeds", help="Number of seeds per ablation.")] = 3,
    out_table: Annotated[str, typer.Option("--out-table", help="Output LaTeX table path.")] = "results/tables/ablation.tex",
    use_amp: UseAmpOpt = False,
    data_dir: DataDirOpt = None,
    batch_size: BatchSizeOpt = None,
    epochs: EpochsOpt = None,
) -> None:
    """Run ablation study and produce LaTeX comparison table."""
    from phase_unwrap.training.ablation import run_ablation

    import random as _rnd

    cfg: TrainConfig = load_train_config(config)

    seed_list: list[int]
    if seeds is not None:
        seed_list = [int(s.strip()) for s in seeds.split(",")]
    else:
        seed_list = [_rnd.randint(0, 2**31) for _ in range(num_seeds)]

    ablations = {
        "no_reference_prior": {"model.zero_reference_prior": True},
        "no_curl_loss": {"loss.w_curl": 0.0},
        "unmasked_zernike": {"model.unmasked_zernike": True},
    }

    _apply_cli_overrides(
        cfg,
        use_amp=use_amp,
        data_dir=data_dir,
        batch_size=batch_size,
        epochs=epochs,
    )

    run_ablation(
        base_cfg=cfg,
        ablations=ablations,
        base_name=run_name,
        seeds=seed_list,
        use_amp=use_amp,
    )


