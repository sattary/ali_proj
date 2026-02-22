"""
Auto-push callback for training loop integration.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .environment import verify_cloud_environment
from .git_pusher import GitPusher
from .state_tracker import StateTracker
from .zip_packer import ZipPacker


class AutoPushCallback:
    """
    Callback for auto-pushing during training.

    Integrates with the training loop to periodically push results to GitHub.
    """

    def __init__(
        self,
        run_dir: str,
        push_interval: int,
        pat: Optional[str] = None,
        dry_run: bool = False,
        force: bool = False,
        include_checkpoints: bool = True,
        repo_dir: Optional[str] = None,
    ):
        """
        Initialize auto-push callback.

        Args:
            run_dir: Path to training run directory.
            push_interval: Push every N epochs.
            pat: GitHub Personal Access Token.
            dry_run: If True, don't actually push.
            force: If True, skip cloud environment check.
            include_checkpoints: Whether to include .pth files in zip.
            repo_dir: Path to git repo (default: parent of run_dir).
        """
        self.run_dir = Path(run_dir)
        self.push_interval = push_interval
        self.dry_run = dry_run
        self.force = force

        # Verify environment
        if not dry_run:
            verify_cloud_environment(force=force)

        # Initialize components
        self.state_tracker = StateTracker(run_dir)
        self.zip_packer = ZipPacker(
            run_dir=run_dir,
            compression_level=6,
            include_checkpoints=include_checkpoints,
        )

        # Determine repo directory
        if repo_dir is None:
            # Search strategy: check multiple starting points
            search_paths = [
                self.run_dir,  # Start from run directory
                Path.cwd(),  # Current working directory
                Path(__file__).parent.parent.parent.parent,  # Package root
            ]

            for start_path in search_paths:
                current = start_path
                # Search up to 5 levels up to avoid infinite loops
                for _ in range(5):
                    if current == current.parent:
                        break
                    if (current / ".git").exists():
                        repo_dir = str(current)
                        break
                    current = current.parent

                if repo_dir is not None:
                    break

        if repo_dir is None:
            raise RuntimeError(
                "Could not find git repository. "
                "Please specify repo_dir or run from within a git repo."
            )

        self.repo_dir = repo_dir

        # Generate unique branch name
        run_name = self.run_dir.name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.branch_name = f"results/{run_name}_{timestamp}"

        # Get PAT
        self.pat = pat or self._get_pat_from_env()

        # Initialize git pusher (but don't setup branch yet)
        self.git_pusher: Optional[GitPusher] = None
        self._initialized = False

    def _get_pat_from_env(self) -> Optional[str]:
        """Get PAT from environment or Kaggle secrets."""
        # Try environment variable first
        pat = os.environ.get("GITHUB_PAT")
        if pat:
            return pat

        # Try Kaggle secrets
        try:
            from kaggle_secrets import UserSecretsClient

            return UserSecretsClient().get_secret("GITHUB_PAT")
        except Exception:
            pass

        return None

    def initialize(self) -> None:
        """Initialize git pusher and setup branch."""
        if self._initialized:
            return

        if self.dry_run:
            print("[DRY-RUN] Mode enabled - no actual pushes will occur")
            print(f"[DRY-RUN] Would create branch: {self.branch_name}")
            print(f"[DRY-RUN] Would push every {self.push_interval} epochs")

        self.git_pusher = GitPusher(
            repo_dir=self.repo_dir,
            branch_name=self.branch_name,
            pat=self.pat,
            dry_run=self.dry_run,
        )

        if not self.dry_run:
            self.git_pusher.setup_branch()
            print("✓ Auto-push initialized")
            print(f"  Branch: {self.branch_name}")
            print(f"  Interval: every {self.push_interval} epochs")
            print(f"  Remote: {self.git_pusher.get_remote_url()}")

        self._initialized = True

    def on_epoch_end(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Call at the end of each epoch.

        Args:
            epoch: Current epoch number.
            total_epochs: Total number of epochs.
            metrics: Current metrics dict.

        Returns:
            True if push was attempted, False otherwise.
        """
        if not self._initialized:
            self.initialize()

        # Check if we should push
        if epoch % self.push_interval != 0 and epoch != total_epochs:
            return False

        is_final = epoch == total_epochs

        if self.dry_run:
            print(f"[DRY-RUN] Would push at epoch {epoch}/{total_epochs}")
            return True

        if self.git_pusher is None:
            raise RuntimeError("Git pusher not initialized")

        try:
            # Create zip
            print(f"Creating zip for epoch {epoch}...")
            zip_path = self.zip_packer.create_zip(
                epoch=epoch,
                total_epochs=total_epochs,
                metrics=metrics,
            )
            print(f"✓ Created: {Path(zip_path).name}")

            # Push
            success = self.git_pusher.push_artifact(
                zip_path=zip_path,
                epoch=epoch,
                total_epochs=total_epochs,
                metrics=metrics,
                is_final=is_final,
            )

            if success:
                # Update state
                self.state_tracker.write_state(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    push_interval=self.push_interval,
                    last_push_epoch=epoch,
                    best_mae=metrics.get("MAE", float("inf"))
                    if metrics
                    else float("inf"),
                    zip_path=zip_path,
                )

                # Cleanup old checkpoints to save space
                ckpt_deleted = self._cleanup_old_checkpoints()
                if ckpt_deleted > 0:
                    print(
                        f"  Cleaned up {ckpt_deleted} old checkpoint(s) to save space"
                    )

            return success

        except Exception as e:
            print(f"✗ Auto-push failed at epoch {epoch}: {e}")
            # Don't raise - let training continue
            return False

    def _cleanup_old_checkpoints(self) -> int:
        """
        Delete intermediate checkpoint files to save space.
        Keeps only best.pth and latest.pth (or checkpoint_epoch_N.pth if latest not present).

        Returns:
            Number of deleted checkpoint files.
        """
        checkpoints_dir = self.run_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return 0

        # Patterns to keep
        keep_patterns = ["best.pth", "latest.pth"]

        # Find all .pth files
        all_checkpoints = list(checkpoints_dir.glob("*.pth"))
        if not all_checkpoints:
            return 0

        deleted_count = 0

        for ckpt_file in all_checkpoints:
            # Check if file should be kept
            should_keep = any(pattern in ckpt_file.name for pattern in keep_patterns)

            if not should_keep:
                try:
                    ckpt_file.unlink()
                    print(f"  Deleted old checkpoint: {ckpt_file.name}")
                    deleted_count += 1
                except OSError as e:
                    print(f"  Warning: Could not delete {ckpt_file.name}: {e}")

        return deleted_count

    def on_train_end(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Call at the end of training.

        Args:
            final_metrics: Final metrics dict.
        """
        if not self._initialized:
            return

        if self.git_pusher and not self.dry_run:
            self.git_pusher.cleanup()

    def get_status(self) -> Dict[str, Any]:
        """Get current auto-push status."""
        state = self.state_tracker.read_state()

        return {
            "initialized": self._initialized,
            "branch": self.branch_name if self._initialized else None,
            "push_interval": self.push_interval,
            "dry_run": self.dry_run,
            "state": state,
        }