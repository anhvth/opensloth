# opensloth/cli/os_sft.py
"""
CLI for running Supervised Fine-Tuning (SFT) with OpenSloth.
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
    name="os-sft",
    help="Run Supervised Fine-Tuning (SFT).",
    add_completion=False,
)

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/sft_{model_short}_{dataset_short}_{timestamp}"

@app.command()
@cli_from_pydantic(OpenSlothConfig, TrainingArguments)
def train(
    dataset: Path = typer.Argument(..., help="Path to the processed and sharded dataset.", exists=True),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
    cli_overrides: dict | None = None,
):
    """
    Trains a model using Supervised Fine-Tuning (SFT) on a prepared dataset.
    All configuration options are dynamically generated from the Pydantic models.
    """
    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="sft")
        .with_cli_args(cli_overrides or {})
    )

    opensloth_cfg, train_args = builder.build()
    
    if not train_args.output_dir:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, str(dataset))

    if dry_run:
        typer.echo("DRY RUN: SFT training configuration:")
        summary = {
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.secho("🚀 Starting SFT training...", fg="green")
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"✅ SFT Training complete. Model saved to: {train_args.output_dir}", fg="green")

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
