# opensloth/cli/os_train.py
"""
Unified CLI for end-to-end OpenSloth training workflows.

This command orchestrates both data preparation and training in a single step,
ensuring consistency across both phases.
"""
import typer
import json
from pathlib import Path
from typing import Optional

app = typer.Typer(
    name="os-train",
    help="A unified command to prepare a dataset AND run a training job in one step.",
    add_completion=False
)

def run_workflow(method: str, **kwargs):
    """Orchestrates the data prep and training workflow."""
    
    # Lazy imports to avoid slow startup
    from opensloth.api import run_prepare_data, run_training
    from opensloth.config.builder import TrainingConfigBuilder
    from opensloth.dataset.config_schema import DatasetPrepConfig
    from datetime import datetime
    
    # 1. Separate kwargs for data prep and training
    prep_fields = set(DatasetPrepConfig.model_fields.keys())
    prep_kwargs = {k: v for k, v in kwargs.items() if k in prep_fields and v is not None}
    
    # Set training type for data prep
    prep_kwargs['training_type'] = method
    
    # 2. Determine the main output directory for the entire experiment
    if 'output' in kwargs and kwargs['output']:
        main_output_dir = Path(kwargs['output'])
    else:
        # Auto-generate a unique experiment directory name
        model_name_safe = prep_kwargs['tok_name'].split("/")[-1]
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
    cli_overrides = {
        "opensloth_config": {"devices": list(range(prep_config.gpus))},
        "training_args": {"output_dir": str(main_output_dir)}
    }
    
    # Map common training parameters
    if 'epochs' in kwargs and kwargs['epochs']:
        cli_overrides["training_args"]["num_train_epochs"] = kwargs['epochs']
    if 'batch_size' in kwargs and kwargs['batch_size']:
        cli_overrides["training_args"]["per_device_train_batch_size"] = kwargs['batch_size']
    if 'lr' in kwargs and kwargs['lr']:
        cli_overrides["training_args"]["learning_rate"] = kwargs['lr']
    
    builder = TrainingConfigBuilder(dataset_path=str(prepared_data_path), method=method).with_cli_args(cli_overrides)
    opensloth_cfg, train_args = builder.build()

    # 7. Run Training
    typer.secho(f"🚀 Starting {method.upper()} training...", fg="bright_blue")
    use_tmux = kwargs.get('tmux', False) or kwargs.get('use_tmux', False)
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux)
    typer.secho(f"🎉🎉🎉 Workflow complete! Final model saved to: {train_args.output_dir}", fg="magenta")

@app.command("sft")
def sft(
    # Data preparation parameters
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    model: str = typer.Option(..., "--model", "-m", help="Base model/tokenizer to use for tokenization."),
    samples: int = typer.Option(-1, "--samples", "-n", help="Number of samples to process (-1 for all)."),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPU shards to create for the dataset."),
    data_output: Optional[str] = typer.Option(None, "--data-output", help="Output directory for processed data."),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Maximum sequence length."),
    
    # Training parameters
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Per-device batch size."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
):
    """Prepare data and run SFT in a single command."""
    run_workflow("sft", 
                 dataset_name=input, tok_name=model, num_samples=samples, 
                 gpus=gpus, output_dir=data_output, max_seq_length=max_seq_length,
                 output=output, epochs=epochs, batch_size=batch_size, lr=lr, tmux=tmux)

@app.command("dpo")
def dpo(
    # Data preparation parameters
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    model: str = typer.Option(..., "--model", "-m", help="Base model/tokenizer to use for tokenization."),
    samples: int = typer.Option(-1, "--samples", "-n", help="Number of samples to process (-1 for all)."),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPU shards to create for the dataset."),
    data_output: Optional[str] = typer.Option(None, "--data-output", help="Output directory for processed data."),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Maximum sequence length."),
    
    # Training parameters
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Per-device batch size."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
):
    """Prepare data and run DPO in a single command."""
    run_workflow("dpo", 
                 dataset_name=input, tok_name=model, num_samples=samples, 
                 gpus=gpus, output_dir=data_output, max_seq_length=max_seq_length,
                 output=output, epochs=epochs, batch_size=batch_size, lr=lr, tmux=tmux)

@app.command("grpo")
def grpo(
    # Data preparation parameters
    input: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    model: str = typer.Option(..., "--model", "-m", help="Base model/tokenizer to use for tokenization."),
    samples: int = typer.Option(-1, "--samples", "-n", help="Number of samples to process (-1 for all)."),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPU shards to create for the dataset."),
    data_output: Optional[str] = typer.Option(None, "--data-output", help="Output directory for processed data."),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Maximum sequence length."),
    
    # Training parameters
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory for training results."),
    epochs: Optional[int] = typer.Option(None, "--epochs", help="Number of training epochs."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Per-device batch size."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Learning rate."),
    tmux: bool = typer.Option(False, "--tmux", help="Use tmux for multi-GPU training."),
):
    """Prepare data and run GRPO in a single command."""
    run_workflow("grpo", 
                 dataset_name=input, tok_name=model, num_samples=samples, 
                 gpus=gpus, output_dir=data_output, max_seq_length=max_seq_length,
                 output=output, epochs=epochs, batch_size=batch_size, lr=lr, tmux=tmux)

def main():
    """Entry point for the CLI."""
    app()

if __name__ == "__main__":
    app()
