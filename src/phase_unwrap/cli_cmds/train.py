import random
import typer
from pathlib import Path
from typing import Optional
from ..core.config import TrainConfig, load_train_config

TrainApp = typer.Typer(help="Training commands.")


@TrainApp.command("train")
def train_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Path to checkpoint to resume from."
    ),
) -> None:
    """Train the PCLCN absolute phase reconstruction model."""
    from phase_unwrap.training.train import train as run_train

    cfg = load_train_config(config)
    run_train(cfg, resume_path=str(resume) if resume else None)


@TrainApp.command("multiseed")
def multiseed_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
    run_name: str = typer.Option(
        "multiseed", "--run-name", help="Base name for the run group."
    ),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Comma-separated seeds."),
    num_seeds: int = typer.Option(3, "--num-seeds", help="Auto-generate N seeds."),
    auto_resume: bool = typer.Option(False, "--auto-resume", help="Resume crashed runs automatically."),
) -> None:
    """Run N training runs with different seeds, aggregate results."""
    from phase_unwrap.training.multiseed import run_multiseed

    cfg = load_train_config(config)
    seed_list = (
        [int(s.strip()) for s in seeds.split(",")]
        if seeds
        else [random.randint(0, 2**31) for _ in range(num_seeds)]
    )
    run_multiseed(
        cfg, base_run_name=run_name, seeds=seed_list, auto_resume=auto_resume
    )


@TrainApp.command("ablation")
def ablation_cmd(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to config file."
    ),
    run_name: str = typer.Option(
        "ablation", "--run-name", help="Base name for ablation run group."
    ),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Comma-separated seeds."),
    num_seeds: int = typer.Option(
        3, "--num-seeds", help="Number of seeds per ablation."
    ),
    auto_resume: bool = typer.Option(False, "--auto-resume", help="Resume crashed runs automatically."),
) -> None:
    """Run ablation study and output JSON summary."""
    from phase_unwrap.training.ablation import run_ablation

    cfg = load_train_config(config)
    seed_list = (
        [int(s.strip()) for s in seeds.split(",")]
        if seeds
        else [random.randint(0, 2**31) for _ in range(num_seeds)]
    )

    ablations = {
        "no_reference_prior": {"model.zero_reference_prior": True},
        "no_curl_loss": {"loss.w_curl": 0.0},
        "unmasked_zernike": {"model.unmasked_zernike": True},
    }
    run_ablation(
        base_cfg=cfg,
        ablations=ablations,
        base_name=run_name,
        seeds=seed_list,
        auto_resume=auto_resume,
    )
