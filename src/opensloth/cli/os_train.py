# opensloth/cli/os_train.py
"""
Unified CLI for end-to-end OpenSloth training workflows.

This command orchestrates both data preparation and training in a single step,
ensuring consistency across both phases.
"""
import json
from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(
    name="os-train",
    help="A unified command to prepare a dataset AND run a training job in one step.",
    add_completion=False
)

def run_workflow(method: str, cli_overrides: dict, **static_kwargs):
    """Orchestrates the data prep and training workflow."""
    
    # Lazy imports to avoid slow startup
    from opensloth.api import run_prepare_data, run_training
    from opensloth.config.builder import TrainingConfigBuilder
    from opensloth.dataset.config_schema import DatasetPrepConfig
    
    # Extract DatasetPrepConfig arguments from cli_overrides
    dataset_prep_config = cli_overrides.get('dataset_prep_config', {})
    
    # 1. Handle the shared arguments that apply to both prep and training
    shared_args = {}
    if 'fast_model_args' in cli_overrides.get('opensloth_config', {}):
        fast_model_args = cli_overrides['opensloth_config']['fast_model_args']
        if 'model_name' in fast_model_args:
            shared_args['model'] = fast_model_args['model_name']
        if 'max_seq_length' in fast_model_args:
            shared_args['max_seq_length'] = fast_model_args['max_seq_length']
    
    # Override with explicit dataset prep values if provided
    if 'tokenizer_name' in dataset_prep_config:
        shared_args['model'] = dataset_prep_config['tokenizer_name']
    if 'max_seq_length' in dataset_prep_config:
        shared_args['max_seq_length'] = dataset_prep_config['max_seq_length']
    
    # 2. Build dataset prep config 
    prep_kwargs = {
        'training_type': method,
        **dataset_prep_config,
        **shared_args,
        'dataset_name': static_kwargs['input'],
    }
    
    # Determine the main output directory for the entire experiment
    if static_kwargs.get('output'):
        main_output_dir = Path(static_kwargs['output'])
    else:
        # Auto-generate a unique experiment directory name
        model_name_safe = prep_kwargs['tokenizer_name'].split("/")[-1] if 'tokenizer_name' in prep_kwargs else 'unknown'
        dataset_name_safe = Path(prep_kwargs['dataset_name']).stem if Path(prep_kwargs['dataset_name']).suffix else prep_kwargs['dataset_name'].replace('/', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        main_output_dir = Path(f"outputs/{method}_{model_name_safe}_{dataset_name_safe}_{timestamp}")

    # 3. Define the data cache path within the main output directory
    prepared_data_path = main_output_dir / "dataset_cache"

    # 4. Update the prep_config with the new cache path
    prep_kwargs['output_dir'] = str(prepared_data_path)
    prep_config = DatasetPrepConfig(**prep_kwargs)

    # 5. Handle data preparation
    if prepared_data_path.exists():
        if not typer.confirm(f"⚠️  Processed data already exists at '{prepared_data_path}'.\nDo you want to overwrite it and re-prepare the data?"):
            typer.echo("Reusing existing data. Skipping data preparation.")
        else:
            typer.secho(f"🚀 Overwriting and starting data preparation for '{method.upper()}'...", fg="cyan")
            run_prepare_data(prep_config)
    else:
        typer.secho(f"🚀 Starting data preparation for '{method.upper()}'...", fg="cyan")
        run_prepare_data(prep_config)

    typer.secho(f"✅ Data preparation complete. Using data from: {prepared_data_path}", fg="green")

    # 6. Build training config, ensuring the main output dir is used for training artifacts
    final_overrides = {
        **cli_overrides,
        'opensloth_config': {
            **cli_overrides.get('opensloth_config', {}),
            'data_cache_path': str(prepared_data_path),
            'training_type': method,
        },
        'training_args': {
            **cli_overrides.get('training_args', {}),
            'output_dir': str(main_output_dir)
        }
    }
    
    builder = TrainingConfigBuilder(dataset_path=str(prepared_data_path), method=method).with_cli_args(final_overrides)
    opensloth_cfg, train_args = builder.build()

    # 7. Run Training
    typer.secho(f"🚀 Starting {method.upper()} training...", fg="bright_blue")
    use_tmux = static_kwargs.get('tmux', False) or static_kwargs.get('use_tmux', False)
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"🎉🎉🎉 Workflow complete! Final model saved to: {train_args.output_dir}", fg="magenta")

def _generate_output_dir(model_name: str, input_name: str, method: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    input_short = Path(input_name).name.replace("-", "_").lower()
    return f"outputs/{method}_{model_short}_{input_short}_{timestamp}"

# SFT Command with autogen
@app.command("sft")
def sft_command(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
    cli_overrides: dict | None = None,
):
    """
    Prepare data and run SFT training in a single command.
    All configuration options are dynamically generated from the Pydantic models.
    """
    # Set defaults for missing model_name if needed
    if not cli_overrides:
        cli_overrides = {}
    
    # Extract model name for output directory generation
    model_name = "unknown"
    if 'dataset_prep_config' in cli_overrides and 'tokenizer_name' in cli_overrides['dataset_prep_config']:
        model_name = cli_overrides['dataset_prep_config']['tokenizer_name']
    elif 'opensloth_config' in cli_overrides and 'fast_model_args' in cli_overrides['opensloth_config']:
        if 'model_name' in cli_overrides['opensloth_config']['fast_model_args']:
            model_name = cli_overrides['opensloth_config']['fast_model_args']['model_name']
    
    if not output:
        output = _generate_output_dir(model_name, input, "sft")

    if dry_run:
        typer.echo("DRY RUN: SFT training configuration:")
        summary = {
            "input": input,
            "output": output,
            "cli_overrides": cli_overrides,
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    run_workflow("sft", cli_overrides, input=input, output=output, tmux=use_tmux)

# DPO Command with autogen
@app.command("dpo")  
def dpo_command(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
    cli_overrides: dict | None = None,
):
    """
    Prepare data and run DPO training in a single command.
    All configuration options are dynamically generated from the Pydantic models.
    """
    # Set defaults for missing model_name if needed
    if not cli_overrides:
        cli_overrides = {}
    
    # Extract model name for output directory generation
    model_name = "unknown"
    if 'dataset_prep_config' in cli_overrides and 'tokenizer_name' in cli_overrides['dataset_prep_config']:
        model_name = cli_overrides['dataset_prep_config']['tokenizer_name']
    elif 'opensloth_config' in cli_overrides and 'fast_model_args' in cli_overrides['opensloth_config']:
        if 'model_name' in cli_overrides['opensloth_config']['fast_model_args']:
            model_name = cli_overrides['opensloth_config']['fast_model_args']['model_name']
    
    if not output:
        output = _generate_output_dir(model_name, input, "dpo")

    if dry_run:
        typer.echo("DRY RUN: DPO training configuration:")
        summary = {
            "input": input,
            "output": output,
            "cli_overrides": cli_overrides,
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    run_workflow("dpo", cli_overrides, input=input, output=output, tmux=use_tmux)

# GRPO Command with autogen
@app.command("grpo")
def grpo_command(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print configuration and exit without running.", rich_help_panel="System Options"),
    use_tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training.", rich_help_panel="System Options"),
    cli_overrides: dict | None = None,
):
    """
    Prepare data and run GRPO training in a single command.
    All configuration options are dynamically generated from the Pydantic models.
    """
    # Set defaults for missing model_name if needed
    if not cli_overrides:
        cli_overrides = {}
    
    # Extract model name for output directory generation
    model_name = "unknown"
    if 'dataset_prep_config' in cli_overrides and 'tokenizer_name' in cli_overrides['dataset_prep_config']:
        model_name = cli_overrides['dataset_prep_config']['tokenizer_name']
    elif 'opensloth_config' in cli_overrides and 'fast_model_args' in cli_overrides['opensloth_config']:
        if 'model_name' in cli_overrides['opensloth_config']['fast_model_args']:
            model_name = cli_overrides['opensloth_config']['fast_model_args']['model_name']
    
    if not output:
        output = _generate_output_dir(model_name, input, "grpo")

    if dry_run:
        typer.echo("DRY RUN: GRPO training configuration:")
        summary = {
            "input": input,
            "output": output,
            "cli_overrides": cli_overrides,
        }
        typer.echo(json.dumps(summary, indent=2))
        return

    run_workflow("grpo", cli_overrides, input=input, output=output, tmux=use_tmux)

# Apply autogen decorators when module is imported
def _apply_autogen():
    """Apply autogen decorators to all commands."""
    # Lazy import
    from opensloth.cli.autogen import cli_from_pydantic
    from opensloth.dataset.config_schema import DatasetPrepConfig
    from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
    
    global sft_command, dpo_command, grpo_command
    sft_command = cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)(sft_command)
    dpo_command = cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)(dpo_command)
    grpo_command = cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)(grpo_command)

# Comment out for now to test if this is causing the issue
# _apply_autogen()

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    app()
