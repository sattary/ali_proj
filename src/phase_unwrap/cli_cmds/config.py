import typer
from pathlib import Path
from ..core.config import TrainConfig, config_to_yaml

ConfigApp = typer.Typer(help="Configuration commands.")

@ConfigApp.command("dump")
def dump_cmd(
    out: str = typer.Option("config_template.yaml", "--out", help="Output path for config.")
) -> None:
    """Dump default configuration to a YAML file."""
    cfg = TrainConfig()
    yaml_str = config_to_yaml(cfg)
    Path(out).write_text(yaml_str)
    typer.echo(f"Dumped default configuration to {out}")
