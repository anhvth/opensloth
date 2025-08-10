# opensloth/cli/os_grpo.py
"""
CLI for running Group Relative Policy Optimization (GRPO) with OpenSloth.
"""
import json
from datetime import datetime
from pathlib import Path

import typer

from opensloth.api import run_training
from opensloth.cli.autogen import cli_from_pydantic
from opensloth.config.builder import TrainingConfigBuilder
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

try:
    from opensloth.grpo_rewards import create_reward_preset
except ImportError:
    create_reward_preset = None

app = typer.Typer(
    name="os-grpo",
    help="Run Group Relative Policy Optimization (GRPO).",
    add_completion=False,
)

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/grpo_{model_short}_{dataset_short}_{timestamp}"

@app.command()
@cli_from_pydantic(OpenSlothConfig, TrainingArguments)
def train(
    dataset: Path = typer.Argument(..., help="Path to the processed and sharded dataset.", exists=True),
    cli_overrides: dict | None = None,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
):
    """
    Trains a model using Group Relative Policy Optimization (GRPO) on a prepared dataset.
    All configuration options are dynamically generated from the Pydantic models.
    """
    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="grpo")
        .with_cli_args(cli_overrides or {})
    )

    opensloth_cfg, train_args = builder.build()
    
    if not train_args.output_dir:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, str(dataset))

    if (
        opensloth_cfg.training_type == "grpo"
        and opensloth_cfg.grpo_args
        and not opensloth_cfg.grpo_args.reward_functions
        and create_reward_preset
    ):
        try:
            preset_names = create_reward_preset(opensloth_cfg.grpo_args.task_type)
            opensloth_cfg.grpo_args.reward_functions = list(preset_names)
        except Exception:
            pass  # Silently ignore; trainer_factory will handle it

    if opensloth_cfg.training_type == "grpo" and opensloth_cfg.grpo_args:
        model_max_seq = opensloth_cfg.fast_model_args.max_seq_length
        grpo_max_new = opensloth_cfg.grpo_args.max_new_tokens
        grpo_max_prompt = opensloth_cfg.grpo_args.max_prompt_length
        
        if grpo_max_new + grpo_max_prompt > model_max_seq:
            typer.secho(
                f"⚠️  Warning: GRPO configuration may cause truncation!\n"
                f"max_new_tokens ({grpo_max_new}) + max_prompt_length ({grpo_max_prompt}) "
                f"= {grpo_max_new + grpo_max_prompt} exceeds model max_seq_length ({model_max_seq}).\n"
                f"Consider using --max-seq-length to increase the limit.", 
                fg="yellow"
            )

    if dry_run:
        typer.echo("DRY RUN: GRPO training configuration:")
        summary = {
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.secho("🚀 Starting GRPO training...", fg="green")
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"✅ GRPO Training complete. Model saved to: {train_args.output_dir}", fg="green")

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    main()
