"""
Git LFS Setup Utility for Phase Unwrap.

Run this before starting training to configure Git LFS for large files.
"""

import subprocess
import sys
from pathlib import Path


def run_git_command(args: list, check: bool = True) -> tuple:
    """Run a git command and return results."""
    result = subprocess.run(
        ["git"] + args,
        capture_output=True,
        text=True,
    )

    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)

    return result.returncode, result.stdout, result.stderr


def setup_git_lfs():
    """Setup Git LFS for the repository."""
    print("Setting up Git LFS for phase_unwrap...")
    print("-" * 50)

    # Check if we're in a git repo
    if not Path(".git").exists():
        print("Error: Not in a git repository!")
        print("Please run this from the root of your git repository.")
        sys.exit(1)

    # Initialize LFS
    print("1. Initializing Git LFS...")
    run_git_command(["lfs", "install"])
    print("   ✓ Git LFS initialized")

    # Track patterns
    print("\n2. Setting up LFS tracking...")
    patterns = [
        "artifacts/*.zip",
        "*.pth",
        "*.onnx",
        "*.pt",
        "data/**/*.h5",
    ]

    for pattern in patterns:
        run_git_command(["lfs", "track", pattern], check=False)
        print(f"   ✓ Tracking: {pattern}")

    # Stage .gitattributes
    print("\n3. Staging .gitattributes...")
    run_git_command(["add", ".gitattributes"])

    # Check if there are changes to commit
    _, stdout, _ = run_git_command(["status", "--porcelain"], check=False)
    if ".gitattributes" in stdout:
        run_git_command(["commit", "-m", "Configure Git LFS for artifacts"])
        print("   ✓ Committed .gitattributes")
    else:
        print("   ℹ .gitattributes already up to date")

    # Push if we have a remote
    _, stdout, _ = run_git_command(["remote"], check=False)
    if stdout.strip():
        print("\n4. Pushing to remote...")
        _, branch_stdout, _ = run_git_command(["branch", "--show-current"])
        branch = branch_stdout.strip()
        run_git_command(["push", "origin", branch])
        print(f"   ✓ Pushed to origin/{branch}")

    print("\n" + "=" * 50)
    print("Git LFS setup complete!")
    print("\nYou can now use auto-push with:")
    print("  phase-unwrap train --auto-push-interval 1000 ...")
    print("\nTo verify LFS is working:")
    print("  git lfs status")


if __name__ == "__main__":
    setup_git_lfs()
