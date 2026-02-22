"""
Optuna auto-push callback extraction.

Rationale:
    ARCHITECTURE: The `cli.py` entry point orchestrator must not contain
    complex git automation or filesystem traversal logic.
    By encapsulating this in `git_automation/tune_push.py`, we isolate
    the GitHub payload dispatch mechanism, ensuring `cli.py` remains
    a purely declarative Typer definition.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .git_pusher import GitPusher
from .zip_packer import ZipPacker


def create_tune_auto_push_callback(
    optuna_dir: str,
    data_dir: str,
    push_interval: int,
    pat: Optional[str],
    dry_run: bool,
) -> Callable[[], None]:
    """Create callback for pushing optuna results to GitHub after HPO completes."""

    def callback() -> None:
        """Push optuna results after HPO finishes."""
        from .environment import is_colab, is_kaggle

        if not is_kaggle() and not is_colab() and not dry_run:
            print("[auto-push] Not on Kaggle/Colab, skipping push")
            return

        print("\n" + "=" * 60)
        print("Pushing Optuna results to GitHub...")
        print("=" * 60)

        data_config_path = Path(data_dir) / "data_config.yaml"
        if data_config_path.exists():
            import yaml

            with open(data_config_path) as f:
                data_config = yaml.safe_load(f)
            data_name = Path(data_config["data_dir"]).name
            num_samples = data_config.get("num_samples", 0)
            seed = data_config.get("seed", 0)
        else:
            data_name = Path(data_dir).name
            num_samples = 0
            seed = 0

        optuna_path = Path(optuna_dir) / "optuna"
        if not optuna_path.exists():
            print(f"Optuna directory not found: {optuna_path}")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if num_samples > 0:
            zip_name = f"optuna_{data_name}_n{num_samples}_s{seed}_{timestamp}.zip"
        else:
            zip_name = f"optuna_{data_name}_{timestamp}.zip"

        zip_path = optuna_path / zip_name

        if data_config_path.exists():
            import shutil

            dest_data_config = optuna_path / "data_config.yaml"
            shutil.copy2(data_config_path, dest_data_config)

        # Rationale: Fixing unused `packer` assignment bug by replacing raw OS
        # traversals with the unified `ZipPacker` abstraction.
        packer = ZipPacker(
            run_dir=str(optuna_path),
            include_checkpoints=False,
        )
        try:
            packer.pack(output_path=str(zip_path))
            print(f"Created zip: {zip_path.name}")
        except Exception as e:
            print(f"Failed to create zip: {e}")
            return

        branch_name = "artifacts"
        pusher = GitPusher(
            repo_dir=str(Path.cwd()),
            branch_name=branch_name,
            pat=pat,
            dry_run=dry_run,
        )

        try:
            pusher.setup_branch()
            success = pusher.push_artifact(
                zip_path=str(zip_path),
                epoch=0,
                total_epochs=0,
                metrics={"n_trials": 0},
                is_final=True,
            )
            if success:
                print("=" * 60)
                print("✓ Optuna results pushed successfully!")
                print("=" * 60)
            else:
                print("✗ Push failed")
        except Exception as e:
            print(f"✗ Push error: {e}")

    return callback