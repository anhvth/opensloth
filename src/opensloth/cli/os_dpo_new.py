# opensloth/cli/os_dpo.py
"""
CLI for running Direct Preference Optimization (DPO) with OpenSloth.
"""
import json
from datetime import datetime
from pathlib import Path

import typer

from opensloth.api import run_training
from opensloth.cli.autogen import cli_from_pydantic
from opensloth.config.builder import TrainingConfigBuilder
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

app = typer.Typer(
    name="os-dpo",
    help="Run Direct Preference Optimization (DPO).",
    add_completion=False,
)

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/dpo_{model_short}_{dataset_short}_{timestamp}"

@app.command()
@cli_from_pydantic(OpenSlothConfig, TrainingArguments)
def train(
    dataset: Path = typer.Argument(..., help="Path to the processed DPO dataset.", exists=True),
    cli_overrides: dict | None = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
):
    """
    Trains a model using Direct Preference Optimization (DPO) on a prepared dataset.
    All configuration options are dynamically generated from the Pydantic models.
    """
    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="dpo")
        .with_cli_args(cli_overrides or {})
    )

    opensloth_cfg, train_args = builder.build()
    
    if not train_args.output_dir:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, str(dataset))

    if dry_run:
        typer.echo("DRY RUN: DPO training configuration:")
        summary = {
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.secho("🚀 Starting DPO training...", fg="green")
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"✅ DPO Training complete. Model saved to: {train_args.output_dir}", fg="green")

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
