# opensloth/cli/os_data.py
"""
CLI for preparing datasets for OpenSloth training (SFT, DPO, GRPO).
"""

import json
from pathlib import Path

import typer

from opensloth.api import run_prepare_data
from opensloth.cli.autogen import cli_from_pydantic
from opensloth.dataset.config_schema import DatasetPrepConfig

app = typer.Typer(
    name="os-data",
    help="Prepare datasets for OpenSloth training.",
    add_completion=False,
)


@app.command()
@cli_from_pydantic(DatasetPrepConfig)
def main(
    # Accept either a local file path OR a remote dataset name (do NOT enforce exists=True)
    input_file: str = typer.Argument(
        ..., help="Path to an input .jsonl file OR a HuggingFace dataset name."
    ),
    method: str = typer.Argument(
        ...,
        help="Preparation method (sft, dpo, grpo). Affects required data format.",
    ),
    output: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for the processed dataset."
    ),
    force: bool = typer.Option(
        ..., "--force", "-f", help="Force overwrite of existing output directory."
    ),
    dry_run: bool = typer.Option(
        ..., "--dry-run", help="Print the configuration and exit without running."
    ),
    cli_overrides: dict | None = None,
):
    """
    Prepares a raw dataset for OpenSloth training by tokenizing, formatting,
    and sharding it for single or multi-GPU use.

    All configuration options are dynamically generated from the DatasetPrepConfig Pydantic model.
    """

    # Defensive fallback for Typer help/CLI introspection
    if not isinstance(method, str):
        return app()

    # Validate method
    method_l = method.lower()
    if method_l not in {"sft", "dpo", "grpo"}:
        typer.secho(
            f"Error: Invalid method '{method}'. Must be 'sft', 'dpo', or 'grpo'.",
            fg="red",
        )
        raise typer.Exit(1)

    # Build configuration from CLI args using the new approach
    dataset_prep_config = (
        cli_overrides.get("dataset_prep_config", {}) if cli_overrides else {}
    )

    # Create the base config with essential parameters
    # Prioritize explicit parameters over autogen defaults
    config_dict = {
        **dataset_prep_config,  # Start with autogen values
        "training_type": method_l,  # Override with explicit values
        "dataset_name": input_file,
    }

    # Create the DatasetPrepConfig object directly
    config = DatasetPrepConfig(**config_dict)

    # Determine output directory
    if not output:
        # Derive a friendly base name whether it's a path or remote dataset identifier
        base_name = (
            Path(input_file).stem
            if Path(input_file).suffix
            else input_file.replace("/", "_")
        )
        num_samples = config.num_samples
        output_dir = Path(
            f"data/{method_l}_{base_name}_n{num_samples if num_samples > 0 else 'all'}"
        )
    else:
        output_dir = output

    # Update the config with the final output directory
    config.output_dir = str(output_dir)

    if output_dir.exists() and not force:
        typer.secho(
            f"Error: Output directory '{output_dir}' already exists. Use --force to overwrite.",
            fg="red",
        )
        raise typer.Exit(1)

    if dry_run:
        typer.echo("DRY RUN: Configuration that would be used:")
        typer.echo(json.dumps(config.model_dump(), indent=2))
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    typer.secho(f"🚀 Starting data preparation for '{method.upper()}'...", fg="green")
    final_output_dir = run_prepare_data(config)
    typer.secho(
        f"✅ Data preparation complete. Processed dataset saved to: {final_output_dir}",
        fg="green",
    )


if __name__ == "__main__":
    app()
