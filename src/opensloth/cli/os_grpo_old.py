# opensloth/cli/os_grpo.py
"""
CLI for running Group Relative Policy Optimization (GRPO) with OpenSloth.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
import typer as _typer_internal

from opensloth.api import run_training
from opensloth.config.builder import TrainingConfigBuilder
from opensloth.grpo_rewards import list_reward_functions

try:  # Local import for optional preset resolution in summary
    from opensloth.grpo_rewards import create_reward_preset
except Exception:  # pragma: no cover - defensive, preset resolution is best-effort
    create_reward_preset = None  # type: ignore

app = typer.Typer(name="os-grpo", help="Run Group Relative Policy Optimization (GRPO).", add_completion=True)

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = dataset_path.split("/")[-1].replace("-", "_").lower()
    return f"outputs/{model_short}_{dataset_short}_{timestamp}"

def _build_cli_overrides(
    output: Path | None,
    epochs: int | None,
    batch_size: int | None,
    grad_accum: int | None,
    lr: float | None,
    lora_r: int | None,
    lora_alpha: int | None,
    lora_dropout: float | None,
    lora_targets: str | None,
    pretrained_lora: str | None,
    beta: float | None,
    group_size: int | None,
    task: str | None,
    rewards: str | None,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None,
    max_new: int | None,
    max_prompt_len: int | None,
    max_seq_length: int | None,
    prompt_length_pct: float | None,
    eval_interval: int | None,
    save_interval: int | None,
    print_sample_every: int | None,
    stop_sequences: str | None,
    no_custom_chat_template: bool,
    logging_steps: int | None,
    warmup_steps: int | None,
    weight_decay: float | None,
    max_steps: int | None,
    save_total_limit: int | None,
    report_to: str | None,
    seed: int | None,
    devices: str | None,
    comm_backend: str | None,
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
    
    lora_args = {k: v for k, v in {"r": lora_r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout}.items() if v is not None}
    if lora_targets and isinstance(lora_targets, str):
        targets = [t.strip() for t in lora_targets.split(",") if t.strip()]
        if targets:
            lora_args["target_modules"] = targets
    
    grpo_args = {}
    # Map GRPO specific hyperparameters
    mapping_vals = {
        "kl_coef": beta,
        "group_size": group_size,
        "task_type": task,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "max_new_tokens": max_new,
        "max_prompt_length": max_prompt_len,
        "prompt_length_percentile": prompt_length_pct,
        "eval_interval": eval_interval,
        "save_interval": save_interval,
        "print_sample_every": print_sample_every,
    }
    for k, v in mapping_vals.items():
        if v is not None:
            grpo_args[k] = v
    if no_custom_chat_template:
        grpo_args["use_custom_chat_template"] = False
    if stop_sequences and isinstance(stop_sequences, str):
        stops = [s.strip() for s in stop_sequences.split(",") if s.strip()]
        grpo_args["stop_sequences"] = stops
    if rewards and isinstance(rewards, str):
        available = set(list_reward_functions())
        parsed = [r.strip() for r in rewards.split(",") if r.strip()]
        unknown = [r for r in parsed if r not in available]
        if unknown:
            raise typer.BadParameter(f"Unknown reward functions: {unknown}. Available: {sorted(available)}")
        grpo_args["reward_functions"] = parsed
    
    overrides = {}
    # Handle training_args overrides - both explicit and additional args
    ta_extra = {}
    if logging_steps is not None:
        ta_extra["logging_steps"] = logging_steps
    if warmup_steps is not None:
        ta_extra["warmup_steps"] = warmup_steps
    if weight_decay is not None:
        ta_extra["weight_decay"] = weight_decay
    if max_steps is not None:
        ta_extra["max_steps"] = max_steps
    if save_total_limit is not None:
        ta_extra["save_total_limit"] = save_total_limit
    if report_to is not None:
        ta_extra["report_to"] = report_to
    if seed is not None:
        ta_extra["seed"] = seed
    
    # Merge training_args (core) with ta_extra (additional)
    all_training_args = {**training_args, **ta_extra}
    if all_training_args:
        overrides["training_args"] = all_training_args
    
    # Handle opensloth_config overrides
    opensloth_overrides = {}
    if lora_args:
        opensloth_overrides["lora_args"] = lora_args
    if grpo_args:
        opensloth_overrides["grpo_args"] = grpo_args
    if pretrained_lora:
        opensloth_overrides["pretrained_lora"] = pretrained_lora
    if devices and isinstance(devices, str):
        try:
            dev_list = [int(d) for d in devices.split(",") if d.strip()]
            if dev_list:
                opensloth_overrides["devices"] = dev_list
        except ValueError:
            raise typer.BadParameter("--devices must be a comma-separated list of integers, e.g. 0,1")

    if comm_backend:
        opensloth_overrides["comm_backend"] = comm_backend
    
    # Always ensure opensloth_config exists even if empty for downstream builder logic
    overrides["opensloth_config"] = opensloth_overrides
    
    # Handle fast_model_args overrides (like max_seq_length)
    if max_seq_length is not None:
        if "fast_model_args" not in overrides["opensloth_config"]:
            overrides["opensloth_config"]["fast_model_args"] = {}
        overrides["opensloth_config"]["fast_model_args"]["max_seq_length"] = max_seq_length
        
    return overrides

@app.command()
def train(
    dataset: Path = typer.Argument(..., help="Path to the processed and sharded dataset.", exists=True),
    model: str | None = typer.Option(None, "--model", "-m", help="Override the base model specified in the dataset config."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output directory for LoRA weights and logs."),
    epochs: int | None = typer.Option(None, "--epochs", help="Number of training epochs."),
    batch_size: int | None = typer.Option(None, "--batch-size", help="Per-device batch size."),
    grad_accum: int | None = typer.Option(None, "--grad-accum", help="Gradient accumulation steps."),
    lr: float | None = typer.Option(None, "--lr", help="Learning rate."),
    lora_r: int | None = typer.Option(None, "--lora-r", help="LoRA `r` parameter (rank)."),
    lora_alpha: int | None = typer.Option(None, "--lora-alpha", help="LoRA `alpha` parameter."),
    pretrained_lora: str | None = typer.Option(None, "--pretrained-lora", help="Path to pretrained LoRA adapters directory (for continuing from SFT checkpoint)."),
    beta: float | None = typer.Option(None, "--beta", help="KL coefficient beta (policy divergence strength)."),
    group_size: int | None = typer.Option(None, "--group-size", help="GRPO group size / num_generations."),
    task: str | None = typer.Option(None, "--task", help="Task type preset for rewards/chat template (math, code, general, reasoning)."),
    rewards: str | None = typer.Option(None, "--rewards", help="Comma-separated reward function names (overrides task preset)."),
    # GRPO sampling + reward controls
    temperature: float | None = typer.Option(None, "--temperature", help="Sampling temperature."),
    top_p: float | None = typer.Option(None, "--top-p", help="Nucleus sampling top_p."),
    top_k: int | None = typer.Option(None, "--top-k", help="Top-k sampling."),
    min_p: float | None = typer.Option(None, "--min-p", help="Minimum probability threshold (min_p)."),
    max_new: int | None = typer.Option(None, "--max-new", help="Max new tokens (completion length)."),
    max_prompt_len: int | None = typer.Option(None, "--max-prompt-len", help="Max prompt token length."),
    max_seq_length: int | None = typer.Option(None, "--max-seq-length", help="Override model's max sequence length."),
    prompt_length_pct: float | None = typer.Option(None, "--prompt-length-pct", help="Percentile for prompt length filtering (0-1)."),
    eval_interval: int | None = typer.Option(None, "--eval-interval", help="Evaluation / preview interval (steps)."),
    save_interval: int | None = typer.Option(None, "--save-interval", help="Save interval (steps)."),
    print_sample_every: int | None = typer.Option(None, "--print-sample-every", help="Print sample generations every N reward calls."),
    stop_sequences: str | None = typer.Option(None, "--stop", help="Comma-separated stop sequences."),
    no_custom_chat_template: bool = typer.Option(False, "--no-custom-chat-template", help="Disable task-specific custom chat template."),
    # TrainingArguments extras
    logging_steps: int | None = typer.Option(None, "--logging-steps", help="Logging steps."),
    warmup_steps: int | None = typer.Option(None, "--warmup-steps", help="Warmup steps."),
    weight_decay: float | None = typer.Option(None, "--weight-decay", help="Weight decay."),
    max_steps: int | None = typer.Option(None, "--max-steps", help="Max training steps (overrides epochs)."),
    save_total_limit: int | None = typer.Option(None, "--save-total-limit", help="Max number of checkpoints to keep."),
    report_to: str | None = typer.Option(None, "--report-to", help="Reporting target: tensorboard, wandb, none."),
    seed: int | None = typer.Option(None, "--seed", help="Random seed."),
    # LoRA extras
    lora_dropout: float | None = typer.Option(None, "--lora-dropout", help="LoRA dropout."),
    lora_targets: str | None = typer.Option(None, "--lora-targets", help="Comma list of LoRA target module names."),
    # Device override
    devices: str | None = typer.Option(None, "--devices", help="Comma-separated GPU indices to use (overrides dataset)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the configuration and exit without running."),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
    comm_backend: str | None = typer.Option(None, "--comm-backend", help="Communication backend: allreduce (default) or async-ps"),
):
    """
    Trains a model using Group Relative Policy Optimization (GRPO) on a prepared dataset.
    """
    # Check if we're getting OptionInfo objects (indicates Typer help/autocomplete mode)
    if isinstance(dataset, _typer_internal.models.OptionInfo):
        # This happens during help generation or when missing required args
        # Don't try to build config, just let Typer handle it
        return
    cli_overrides = _build_cli_overrides(
        output, epochs, batch_size, grad_accum, lr,
        lora_r, lora_alpha, lora_dropout, lora_targets, pretrained_lora,
        beta, group_size, task, rewards,
        temperature, top_p, top_k, min_p, max_new, max_prompt_len, max_seq_length,
        prompt_length_pct, eval_interval, save_interval, print_sample_every,
    stop_sequences, no_custom_chat_template,
    logging_steps, warmup_steps, weight_decay, max_steps,
    save_total_limit, report_to, seed, devices, comm_backend,
    )
    if model:
        cli_overrides.setdefault("opensloth_config", {}).setdefault("fast_model_args", {})["model_name"] = model

    builder = (
        TrainingConfigBuilder(dataset_path=str(dataset), method="grpo")
        .with_cli_args(cli_overrides)
    )

    opensloth_cfg, train_args = builder.build()
    
    # Auto-generate output dir if not provided
    if not train_args.output_dir:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, str(dataset))

    # If GRPO and reward_functions empty, resolve preset early so user sees actual functions in summary
    if (
        opensloth_cfg.training_type == "grpo"
        and getattr(opensloth_cfg, "grpo_args", None) is not None  # has grpo_args
        and not opensloth_cfg.grpo_args.reward_functions  # type: ignore[attr-defined]
        and create_reward_preset is not None
    ):
        try:
            preset_names = create_reward_preset(opensloth_cfg.grpo_args.task_type)  # type: ignore[attr-defined]
            # Assign so downstream training uses same resolved list (idempotent with trainer_factory logic)
            opensloth_cfg.grpo_args.reward_functions = list(preset_names)  # type: ignore[attr-defined]
        except Exception:
            # Silently ignore; trainer_factory will resolve later
            pass

    summary = {
        "opensloth_config": opensloth_cfg.model_dump(),
        "training_args": train_args.model_dump(),
    }

    # Early validation for GRPO-specific requirements (show even in dry-run)
    if opensloth_cfg.training_type == "grpo" and opensloth_cfg.grpo_args:
        model_max_seq_length = opensloth_cfg.fast_model_args.max_seq_length
        grpo_max_new_tokens = opensloth_cfg.grpo_args.max_new_tokens
        grpo_max_prompt_length = opensloth_cfg.grpo_args.max_prompt_length
        
        # Warn if max_new_tokens + max_prompt_length exceeds model's max_seq_length
        # But don't block execution - user may have intentionally overridden max_seq_length
        if grpo_max_new_tokens + grpo_max_prompt_length > model_max_seq_length:
            typer.secho(
                f"⚠️  Warning: GRPO configuration may cause truncation!\n"
                f"max_new_tokens ({grpo_max_new_tokens}) + max_prompt_length ({grpo_max_prompt_length}) "
                f"= {grpo_max_new_tokens + grpo_max_prompt_length} exceeds model max_seq_length ({model_max_seq_length}).\n"
                f"Sequences will be truncated. Consider using --max-seq-length to increase the limit.", 
                fg="yellow"
            )

    if dry_run:
        typer.echo("DRY RUN: GRPO training configuration:")
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
