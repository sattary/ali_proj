import typer

DataApp = typer.Typer(help="Data commands.")

# ============================================================================
# DATA COMMANDS
# ============================================================================


@DataApp.command("generate")
def data_generate(
    num_samples: int = typer.Option(
        180_000, "--num-samples", help="Total number of samples to generate."
    ),
    shard_size: int = typer.Option(
        1000, "--shard-size", help="Number of samples per HDF5 shard."
    ),
    out_dir: str = typer.Option(
        "data/full", "--out-dir", help="Output directory for HDF5 shards."
    ),
    seed: int = typer.Option(1337, "--seed", help="Random seed for reproducibility."),
) -> None:
    """Generate synthetic interferogram data to HDF5 shards."""
    from .data.generate import generate_to_h5

    generate_to_h5(
        out_dir=out_dir,
        num_samples=num_samples,
        shard_size=shard_size,
        seed=seed,
    )


