"""
Tests for flat CLI command interface.
"""

from __future__ import annotations

from typer.testing import CliRunner

from phase_unwrap.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "PCLCN" in result.stdout or "phase" in result.stdout.lower()


def test_cli_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0


def test_cli_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0


def test_cli_ablation_help():
    result = runner.invoke(app, ["ablation", "--help"])
    assert result.exit_code == 0


def test_cli_plot_help():
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0


def test_cli_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0


def test_cli_export_help():
    result = runner.invoke(app, ["onnx", "--help"])
    assert result.exit_code == 0
