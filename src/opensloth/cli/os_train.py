# opensloth/cli/os_train.py
"""
Unified CLI for end-to-end OpenSloth training workflows.

This command orchestrates both data preparation and training in a single step,
ensuring consistency across both phases.
"""
import json
from pathlib import Path
from typing import Optional

import typer
from opensloth.dataset.config_schema import DatasetPrepConfig
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

from .autogen import cli_from_pydantic

app = typer.Typer(
    name="os-train",
    help="A unified command to prepare a dataset AND run a training job in one step.",
    add_completion=False
)

def run_workflow(method: str, cli_overrides: dict, **static_kwargs):
    """Orchestrates the data prep and training workflow."""
    
    # Lazy imports to avoid slow startup
    from datetime import datetime

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
    if 'tok_name' in dataset_prep_config:
        shared_args['model'] = dataset_prep_config['tok_name']
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
        model_name_safe = prep_kwargs['model'].split("/")[-1] if 'model' in prep_kwargs else 'unknown'
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

# Create simplified commands that leverage dynamic CLI generation

@app.command("sft")
@cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)
def sft(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
    cli_overrides: dict | None = None,
):
    """Prepare data and run SFT in a single command."""
    run_workflow("sft", cli_overrides or {}, input=input, output=output, tmux=tmux)

@app.command("dpo")
@cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)
def dpo(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
    cli_overrides: dict | None = None,
):
    """Prepare data and run DPO in a single command."""
    run_workflow("dpo", cli_overrides or {}, input=input, output=output, tmux=tmux)

@app.command("grpo")
@cli_from_pydantic(DatasetPrepConfig, OpenSlothConfig, TrainingArguments)
def grpo(
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    output: str | None = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
    cli_overrides: dict | None = None,
):
    """Prepare data and run GRPO in a single command."""
    run_workflow("grpo", cli_overrides or {}, input=input, output=output, tmux=tmux)

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    app()
