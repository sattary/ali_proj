import typer
from typing import Annotated, Optional

app = typer.Typer()
ConfigOpt = Annotated[Optional[str], typer.Option(None, "--config", "-c", help="Path to config")]

@app.command()
def main(config: ConfigOpt = None):
    print(config)

if __name__ == "__main__":
    app()
