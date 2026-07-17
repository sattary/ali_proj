from typer.testing import CliRunner
from phase_unwrap.cli import app

runner = CliRunner()

def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Phase unwrap" in result.stdout or "phase" in result.stdout.lower()

def test_cli_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "generate" in result.stdout.lower()

def test_cli_train_help():
    result = runner.invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    assert "train" in result.stdout.lower()

def test_cli_tune_help():
    result = runner.invoke(app, ["tune", "--help"])
    assert result.exit_code == 0
    assert "tune" in result.stdout.lower()

def test_cli_plot_help():
    result = runner.invoke(app, ["plot", "--help"])
    assert result.exit_code == 0

def test_cli_eval_help():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
