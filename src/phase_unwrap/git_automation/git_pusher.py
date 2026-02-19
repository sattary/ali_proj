"""
Git operations for pushing to GitHub.
"""

import subprocess
from pathlib import Path
from typing import Optional, Tuple


class GitPusher:
    """Handle git operations for auto-push."""

    def __init__(
        self,
        repo_dir: str,
        branch_name: str,
        pat: Optional[str] = None,
        dry_run: bool = False,
    ):
        """
        Initialize git pusher.

        Args:
            repo_dir: Path to the git repository.
            branch_name: Name of the branch to push to.
            pat: GitHub Personal Access Token.
            dry_run: If True, don't actually push.
        """
        self.repo_dir = Path(repo_dir)
        self.branch_name = branch_name
        self.pat = pat
        self.dry_run = dry_run
        self._original_branch: Optional[str] = None

    def _run_git(self, args: list, check: bool = True) -> Tuple[int, str, str]:
        """Run a git command."""
        cmd = ["git"] + args

        if self.dry_run and args[0] in ["push", "commit", "checkout", "add"]:
            print(f"[DRY-RUN] Would run: {' '.join(cmd)}")
            return 0, "", ""

        result = subprocess.run(
            cmd,
            cwd=self.repo_dir,
            capture_output=True,
            text=True,
        )

        if check and result.returncode != 0:
            raise RuntimeError(f"Git command failed: {result.stderr}")

        return result.returncode, result.stdout, result.stderr

    def setup_branch(self) -> None:
        """Create and checkout the results branch."""
        # Save original branch
        _, stdout, _ = self._run_git(["branch", "--show-current"], check=False)
        self._original_branch = stdout.strip() or "alis_code"

        if self.dry_run:
            print(f"[DRY-RUN] Would create branch: {self.branch_name}")
            return

        # Check if branch exists
        returncode, _, _ = self._run_git(
            ["rev-parse", "--verify", self.branch_name], check=False
        )

        if returncode != 0:
            # Create new branch
            self._run_git(["checkout", "-b", self.branch_name])
        else:
            # Checkout existing branch
            self._run_git(["checkout", self.branch_name])

    def push_artifact(
        self,
        zip_path: str,
        epoch: int,
        total_epochs: int,
        metrics: Optional[dict] = None,
        is_final: bool = False,
    ) -> bool:
        """
        Push the artifact to GitHub.

        Args:
            zip_path: Path to the zip file.
            epoch: Current epoch.
            total_epochs: Total epochs.
            metrics: Metrics dict for commit message.
            is_final: Whether this is the final push.

        Returns:
            True if push succeeded, False otherwise.
        """
        zip_file = Path(zip_path)
        if not zip_file.exists():
            raise FileNotFoundError(f"Zip file not found: {zip_path}")

        # Create artifacts directory
        artifacts_dir = self.repo_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Copy zip to artifacts directory (overwrite if exists)
        dest_path = artifacts_dir / zip_file.name
        if self.dry_run:
            print(f"[DRY-RUN] Would copy {zip_file.name} to artifacts/")
        else:
            import shutil

            shutil.copy2(zip_path, dest_path)

        # Build commit message
        progress = f"{epoch}/{total_epochs}"
        progress_pct = f"({100 * epoch / total_epochs:.1f}%)"
        mae_str = ""
        if metrics and "MAE" in metrics:
            mae_str = f" - MAE: {metrics['MAE']:.6f}"

        final_tag = " [FINAL]" if is_final else ""
        commit_msg = f"Auto-push: Epoch {progress} {progress_pct}{mae_str}{final_tag}"

        # Stage, commit, push
        try:
            self._run_git(["add", "artifacts/"])
            self._run_git(["add", ".gitattributes"])
            self._run_git(["commit", "-m", commit_msg])

            # Configure remote with PAT if provided
            if self.pat and not self.dry_run:
                self._configure_remote_with_pat()

            self._run_git(["push", "-u", "origin", self.branch_name])

            # Verify push succeeded
            if not self.dry_run:
                success, status_msg = self._verify_push()
                if not success:
                    print(f"Warning: Push verification failed: {status_msg}")
                    return False

            print(f"✓ Successfully pushed to {self.branch_name}")
            print(f"  Commit: {commit_msg}")
            return True

        except RuntimeError as e:
            print(f"✗ Push failed: {e}")
            return False

    def _configure_remote_with_pat(self) -> None:
        """Configure remote URL with PAT for authentication."""
        # Get current remote URL
        _, stdout, _ = self._run_git(["remote", "get-url", "origin"])
        current_url = stdout.strip()

        # If URL already has PAT, skip
        if "@" in current_url and self.pat in current_url:
            return

        # Parse URL and add PAT
        if current_url.startswith("https://"):
            # Extract username/repo
            parts = current_url.replace("https://", "").split("/")
            if len(parts) >= 2:
                domain = parts[0]
                username = parts[1]
                repo = "/".join(parts[2:])
                new_url = f"https://{username}:{self.pat}@{domain}/{username}/{repo}"
                self._run_git(["remote", "set-url", "origin", new_url])

    def _verify_push(self) -> Tuple[bool, str]:
        """Verify the push succeeded by checking git status."""
        returncode, stdout, stderr = self._run_git(["status", "-sb"], check=False)

        if returncode != 0:
            return False, f"git status failed: {stderr}"

        # Check if we're ahead of origin
        if "ahead" in stdout:
            return False, "Commits not pushed (still ahead of origin)"

        return True, "Push verified"

    def cleanup(self) -> None:
        """Return to original branch."""
        if self._original_branch and not self.dry_run:
            self._run_git(["checkout", self._original_branch], check=False)

    def get_remote_url(self) -> str:
        """Get the current remote URL (sanitized)."""
        _, stdout, _ = self._run_git(["remote", "get-url", "origin"], check=False)
        url = stdout.strip()
        # Sanitize PAT from URL
        if self.pat and self.pat in url:
            url = url.replace(f":{self.pat}@", ":***@")
        return url
