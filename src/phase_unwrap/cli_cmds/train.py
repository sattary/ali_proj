import random
import typer
from pathlib import Path
from typing import Optional
from ..core.config import TrainConfig, load_train_config

TrainApp = typer.Typer(help="Training commands.")

@TrainApp.command("train")
def train_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    resume: Optional[Path] = typer.Option(None, "--resume", help="Path to checkpoint to resume from."),
) -> None:
    """Train the UNetRes2 absolute phase reconstruction model."""
    from phase_unwrap.training.train import train as run_train
    cfg = load_train_config(config)
    run_train(cfg, resume_path=str(resume) if resume else None)

@TrainApp.command("tune")
def tune_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    n_trials: int = typer.Option(50, "--n-trials", help="Number of Optuna trials."),
    tune_epochs: int = typer.Option(15, "--tune-epochs", help="Epochs per trial."),
    study_name: str = typer.Option("phase_unwrap_hpo", "--study-name", help="Optuna study name."),
    n_workers: int = typer.Option(1, "-j", "--n-workers", help="Number of parallel workers."),
) -> None:
    """Run Optuna hyperparameter search (TPE + MedianPruner)."""
    import torch
    from phase_unwrap.training.tune import run_tuning
    cfg = load_train_config(config)
    
    gpu_id_list = None
    if n_workers > 1 and torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_id_list = [i % n_gpus for i in range(n_workers)]
        typer.echo(f"Auto-detected {n_gpus} GPUs, mapping {n_workers} workers: {gpu_id_list}")
        
    run_tuning(cfg, n_trials=n_trials, tune_epochs=tune_epochs, study_name=study_name, n_workers=n_workers, gpu_ids=gpu_id_list)

@TrainApp.command("multiseed")
def multiseed_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    run_name: str = typer.Option("multiseed", "--run-name", help="Base name for the run group."),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Comma-separated seeds."),
    num_seeds: int = typer.Option(3, "--num-seeds", help="Auto-generate N seeds."),
) -> None:
    """Run N training runs with different seeds, aggregate results."""
    from phase_unwrap.training.multiseed import run_multiseed
    cfg = load_train_config(config)
    seed_list = [int(s.strip()) for s in seeds.split(",")] if seeds else [random.randint(0, 2**31) for _ in range(num_seeds)]
    run_multiseed(cfg, base_run_name=run_name, seeds=seed_list, use_amp=cfg.model.use_amp)

@TrainApp.command("ablation")
def ablation_cmd(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file."),
    run_name: str = typer.Option("ablation", "--run-name", help="Base name for ablation run group."),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Comma-separated seeds."),
    num_seeds: int = typer.Option(3, "--num-seeds", help="Number of seeds per ablation."),
) -> None:
    """Run ablation study and produce LaTeX comparison table."""
    from phase_unwrap.training.ablation import run_ablation
    cfg = load_train_config(config)
    seed_list = [int(s.strip()) for s in seeds.split(",")] if seeds else [random.randint(0, 2**31) for _ in range(num_seeds)]
    
    ablations = {
        "no_reference_prior": {"model.zero_reference_prior": True},
        "no_curl_loss": {"loss.w_curl": 0.0},
        "unmasked_zernike": {"model.unmasked_zernike": True},
    }
    run_ablation(base_cfg=cfg, ablations=ablations, base_name=run_name, seeds=seed_list, use_amp=cfg.model.use_amp)

