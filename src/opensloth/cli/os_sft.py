# opensloth/cli/os_sft.py
"""
CLI for running Supervised Fine-Tuning (SFT) with OpenSloth.
"""
import json
from pathlib import Path
from typing import Optional
from datetime import datetime
import typer

from opensloth.api import run_training
import typer as _typer_internal
from opensloth.config.builder import TrainingConfigBuilder

app = typer.Typer(name="os-sft", help="Run Supervised Fine-Tuning (SFT).", add_completion=True)

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = dataset_path.split("/")[-1].replace("-", "_").lower()
    return f"outputs/{model_short}_{dataset_short}_{timestamp}"

def _build_cli_overrides(
    output: Optional[Path],
    epochs: Optional[int],
    batch_size: Optional[int],
    grad_accum: Optional[int],
    lr: Optional[float],
    lora_r: Optional[int],
    lora_alpha: Optional[int]
) -> dict:
    """Helper to construct a nested dictionary of CLI overrides."""
    training_args = {
        k: v for k, v in {
            "output_dir": str(output) if output else None,
            "num_train_epochs": epochs,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": lr,
        }.items() if v is not None
    }
    lora_args = {
        k: v for k, v in {
            "r": lora_r,
            "lora_alpha": lora_alpha,
        }.items() if v is not None
    }
    
    overrides = {}
    if training_args:
        overrides["training_args"] = training_args
    if lora_args:
        overrides["opensloth_config"] = {"lora_args": lora_args}
        
    return overrides

@app.command()
def main(
    dataset: Path = typer.Argument(..., help="Path to the processed and sharded dataset.", exists=True),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override the base model specified in the dataset config."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for LoRA weights and logs."),
    preset: Optional[str] = typer.Option(None, "--preset", help="Training preset (e.g., 'quick', 'small', 'large')."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Per-device batch size."),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum", help="Gradient accumulation steps."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    lora_r: Optional[int] = typer.Option(None, "--lora-r", help="LoRA `r` parameter (rank)."),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha", help="LoRA `alpha` parameter."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the configuration and exit without running."),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
):
    """
    Trains a model using Supervised Fine-Tuning (SFT) on a prepared dataset.
    """
    # Defensive: if Typer option objects leaked through (entrypoint misconfigured), re-route to app()
    if isinstance(preset, _typer_internal.models.OptionInfo):  # type: ignore
        # Running via console script calling main() directly instead of Typer parser
        return app()

    cli_overrides = _build_cli_overrides(output, epochs, batch_size, grad_accum, lr, lora_r, lora_alpha)
    if model:
        cli_overrides.setdefault("opensloth_config", {}).setdefault("fast_model_args", {})["model_name"] = model

    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="sft")
        .with_preset(preset)
        .infer_from_dataset()
        .with_cli_args(cli_overrides) # Use the new structured override
        .finalise()
    )

    opensloth_cfg, train_args = builder.build()
    
    # Auto-generate output dir if not provided
    if not train_args.output_dir:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, str(dataset))

    summary = {
        "opensloth_config": opensloth_cfg.model_dump(exclude_unset=True),
        "training_args": train_args.model_dump(exclude_unset=True),
    }

    if dry_run:
        typer.echo("DRY RUN: SFT training configuration:")
        typer.echo(json.dumps(summary, indent=2))
        return

    typer.secho("🚀 Starting SFT training...", fg="green")
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"✅ SFT Training complete. Model saved to: {train_args.output_dir}", fg="green")

if __name__ == "__main__":
    app()