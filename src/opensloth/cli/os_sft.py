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
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/sft_{model_short}_{dataset_short}_{timestamp}"

def _build_cli_overrides(
    # Model and LoRA
    model: Optional[str],
    max_seq_length: Optional[int],
    load_in_4bit: Optional[bool],
    load_in_8bit: Optional[bool],
    full_finetuning: Optional[bool],
    lora_r: Optional[int],
    lora_alpha: Optional[int],
    lora_dropout: Optional[float],
    lora_targets: Optional[str],
    use_rslora: Optional[bool],
    lora_bias: Optional[str],
    # Training
    output: Optional[Path],
    epochs: Optional[int],
    max_steps: Optional[int],
    batch_size: Optional[int],
    grad_accum: Optional[int],
    lr: Optional[float],
    lr_scheduler_type: Optional[str],
    warmup_steps: Optional[int],
    logging_steps: Optional[int],
    save_steps: Optional[int],
    save_total_limit: Optional[int],
    optim: Optional[str],
    weight_decay: Optional[float],
    seed: Optional[int],
    report_to: Optional[str],
    use_gradient_checkpointing: Optional[str],
    # Devices
    devices: Optional[str],
) -> dict:
    """Helper to construct a nested dictionary of CLI overrides from authentic param names."""
    overrides: dict = {"opensloth_config": {}, "training_args": {}}

    # FastModelArgs
    fast_model_args = {k: v for k, v in {
        "model_name": model,
        "max_seq_length": max_seq_length,
        "load_in_4bit": load_in_4bit,
        "load_in_8bit": load_in_8bit,
        "full_finetuning": full_finetuning,
        "use_gradient_checkpointing": use_gradient_checkpointing,
    }.items() if v is not None}
    if fast_model_args:
        overrides["opensloth_config"]["fast_model_args"] = fast_model_args

    # LoraArgs
    lora_args = {k: v for k, v in {
        "r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_rslora": use_rslora,
        "bias": lora_bias,
    }.items() if v is not None}
    if lora_targets and isinstance(lora_targets, str):
        targets = [t.strip() for t in lora_targets.split(",") if t.strip()]
        if targets:
            lora_args["target_modules"] = targets
    if lora_args:
        overrides["opensloth_config"]["lora_args"] = lora_args

    # TrainingArguments
    training_args = {k: v for k, v in {
        "output_dir": str(output) if output else None,
        "num_train_epochs": epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_steps": warmup_steps,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "optim": optim,
        "weight_decay": weight_decay,
        "seed": seed,
        "report_to": report_to,
    }.items() if v is not None}
    if training_args:
        overrides["training_args"] = training_args
        
    # Device override
    if devices and isinstance(devices, str):
        try:
            dev_list = [int(d) for d in devices.split(",") if d.strip()]
            if dev_list:
                overrides["opensloth_config"]["devices"] = dev_list
        except ValueError:
            raise typer.BadParameter("--devices must be a comma-separated list of integers, e.g. 0,1")

    return overrides

@app.command()
def train(
    dataset: Path = typer.Argument(..., help="Path to the processed and sharded dataset.", exists=True),
    
    # Model and Tokenizer
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override base model from dataset config."),
    max_seq_length: Optional[int] = typer.Option(None, "--max-seq-length", help="Override model's max sequence length."),
    load_in_4bit: Optional[bool] = typer.Option(None, "--load-in-4bit/--no-4bit", help="Load in 4-bit (QLoRA). Default: True."),
    load_in_8bit: Optional[bool] = typer.Option(None, "--load-in-8bit/--no-8bit", help="Load in 8-bit. Default: False."),
    full_finetuning: Optional[bool] = typer.Option(None, "--full-finetuning/--lora", help="Perform full fine-tuning instead of LoRA."),
    use_gradient_checkpointing: Optional[str] = typer.Option(None, help="Gradient checkpointing strategy ('unsloth' or 'True')."),

    # LoRA Specific
    lora_r: Optional[int] = typer.Option(None, "--lora-r", help="LoRA `r` parameter (rank)."),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha", help="LoRA `alpha` parameter."),
    lora_dropout: Optional[float] = typer.Option(None, "--lora-dropout", help="LoRA dropout."),
    lora_targets: Optional[str] = typer.Option(None, "--lora-targets", help="Comma-separated list of LoRA target modules."),
    use_rslora: Optional[bool] = typer.Option(None, "--use-rslora/--no-rslora", help="Use RSLoRA (Rank-Stabilized LoRA)."),
    lora_bias: Optional[str] = typer.Option(None, "--lora-bias", help="LoRA bias configuration ('none', 'all', 'lora_only')."),

    # Training Hyperparameters
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for LoRA weights and logs."),
    epochs: Optional[int] = typer.Option(None, "--num-train-epochs", "-e", help="Number of training epochs."),
    max_steps: Optional[int] = typer.Option(None, "--max-steps", help="Max training steps (overrides epochs)."),
    batch_size: Optional[int] = typer.Option(None, "--per-device-train-batch-size", "-b", help="Per-device batch size."),
    grad_accum: Optional[int] = typer.Option(None, "--gradient-accumulation-steps", help="Gradient accumulation steps."),
    lr: Optional[float] = typer.Option(None, "--learning-rate", help="Learning rate."),
    lr_scheduler_type: Optional[str] = typer.Option(None, "--lr-scheduler-type", help="LR scheduler type (e.g., 'linear', 'cosine')."),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps", help="Warmup steps for LR scheduler."),
    logging_steps: Optional[int] = typer.Option(None, "--logging-steps", help="Log metrics every N steps."),
    save_steps: Optional[int] = typer.Option(None, "--save-steps", help="Save a checkpoint every N steps."),
    save_total_limit: Optional[int] = typer.Option(None, "--save-total-limit", help="Maximum number of checkpoints to keep."),
    optim: Optional[str] = typer.Option(None, "--optim", help="Optimizer to use (e.g., 'adamw_8bit', 'adamw_torch')."),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay", help="Weight decay."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for reproducibility."),
    report_to: Optional[str] = typer.Option(None, "--report-to", help="Reporting integration ('tensorboard', 'wandb', 'none')."),

    # System
    devices: Optional[str] = typer.Option(None, "--devices", help="Comma-separated GPU indices (e.g., '0,1'). Overrides dataset config."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running."),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
):
    """
    Trains a model using Supervised Fine-Tuning (SFT) on a prepared dataset.
    """
    # Check if we're getting OptionInfo objects (indicates Typer help/autocomplete mode)
    if isinstance(dataset, _typer_internal.models.OptionInfo):
        # This happens during help generation or when missing required args
        # Don't try to build config, just let Typer handle it
        return

    cli_overrides = _build_cli_overrides(
        model=model, max_seq_length=max_seq_length, load_in_4bit=load_in_4bit, load_in_8bit=load_in_8bit,
        full_finetuning=full_finetuning, lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        lora_targets=lora_targets, use_rslora=use_rslora, lora_bias=lora_bias, output=output, epochs=epochs, 
        max_steps=max_steps, batch_size=batch_size, grad_accum=grad_accum, lr=lr, 
        lr_scheduler_type=lr_scheduler_type, warmup_steps=warmup_steps, logging_steps=logging_steps, 
        save_steps=save_steps, save_total_limit=save_total_limit, optim=optim, weight_decay=weight_decay, 
        seed=seed, report_to=report_to, use_gradient_checkpointing=use_gradient_checkpointing, devices=devices,
    )

    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="sft")
        .with_cli_args(cli_overrides)
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