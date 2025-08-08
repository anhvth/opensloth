#!/usr/bin/env python3
"""
OpenSloth Dataset Preparation CLI

A modern, intuitive command-line interface for preparing datasets for fine-tuning.
Built with Typer for excellent user experience and type safety.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Annotated
from datetime import datetime

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Lazy imports to avoid slow startup
def _get_preparers():
    """Lazy import of dataset preparers to avoid slow CLI startup."""
    from opensloth.dataset import (
        DatasetPrepConfig,
        BaseDatasetPreparer,
        QwenDatasetPreparer,
        GemmaDatasetPreparer
    )
    return DatasetPrepConfig, BaseDatasetPreparer, QwenDatasetPreparer, GemmaDatasetPreparer

app = typer.Typer(
    name="opensloth-dataset",
    help="🦥 OpenSloth Dataset Preparation - Prepare datasets for fine-tuning",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

# Dataset preparation presets
PRESETS = {
    "qwen_chat": {
        "description": "Optimized for Qwen models with chat templates",
        "defaults": {
            "tok_name": "unsloth/Qwen2.5-0.5B-Instruct",
            "chat_template": "qwen-2.5",
            "train_on_target_only": True,
            "instruction_part": "<|im_start|>user\\n",
            "response_part": "<|im_start|>assistant\\n",
        }
    },
    "llama_chat": {
        "description": "Optimized for Llama models with chat templates", 
        "defaults": {
            "tok_name": "unsloth/llama-3.2-1b-instruct",
            "chat_template": "llama-3",
            "train_on_target_only": True,
            "instruction_part": "<|start_header_id|>user<|end_header_id|>\\n",
            "response_part": "<|start_header_id|>assistant<|end_header_id|>\\n",
        }
    },
    "gemma_chat": {
        "description": "Optimized for Gemma models with chat templates",
        "defaults": {
            "tok_name": "unsloth/gemma-2-2b-it",
            "chat_template": "gemma",
            "train_on_target_only": True,
            "instruction_part": "<start_of_turn>user\\n",
            "response_part": "<start_of_turn>model\\n",
        }
    },
    "mistral_chat": {
        "description": "Optimized for Mistral models with chat templates",
        "defaults": {
            "tok_name": "unsloth/Mistral-7B-Instruct-v0.3",
            "chat_template": "mistral",
            "train_on_target_only": True,
            "instruction_part": "[INST]",
            "response_part": "[/INST]",
        }
    },
}


def _generate_dataset_name(model_name: str, dataset_name: str, num_samples: int) -> str:
    """Auto-generate dataset directory name based on model and config"""
    today = datetime.now().strftime("%m%d")
    
    # Extract model family from model name
    model_family = "unknown"
    if "qwen" in model_name.lower():
        model_family = "qwen"
    elif "gemma" in model_name.lower():
        model_family = "gemma" 
    elif "llama" in model_name.lower():
        model_family = "llama"
    elif "mistral" in model_name.lower():
        model_family = "mistral"
    
    # Extract dataset name (remove path/organization)
    dataset_short = dataset_name.split('/')[-1].replace('-', '_').lower()
    
    # Format: model-dataset-samples-mmdd
    name = f"{model_family}_{dataset_short}_n{num_samples}_{today}"
    return name


def _show_preset_info(preset_name: str):
    """Display information about a preset"""
    if preset_name not in PRESETS:
        console.print(f"❌ Preset '{preset_name}' not found", style="red")
        return
    
    preset = PRESETS[preset_name]
    table = Table(title=f"Preset: {preset_name}")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Description", preset["description"])
    for key, value in preset["defaults"].items():
        table.add_row(key, str(value))
    
    console.print(table)


@app.command("list-presets")
def list_presets():
    """📋 List all available dataset preparation presets"""
    table = Table(title="Available Dataset Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Model Type", style="yellow")
    
    for name, info in PRESETS.items():
        model_type = info["defaults"]["tok_name"].split("/")[-1].split("-")[0].title()
        table.add_row(name, info["description"], model_type)
    
    console.print(table)
    console.print("\n💡 Use [bold]--preset [preset_name][/bold] to apply a preset configuration")


@app.command("preset-info")
def preset_info(
    preset: Annotated[str, typer.Argument(help="Preset name to show information for")]
):
    """📖 Show detailed information about a specific preset"""
    _show_preset_info(preset)


@app.command("prepare")
def prepare_dataset(
    # Model and tokenizer settings
    model: Annotated[str, typer.Option("--model", "-m", help="🤖 HuggingFace model identifier or local path")] = "unsloth/Qwen2.5-0.5B-Instruct",
    
    # Dataset settings
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="📊 HuggingFace dataset or local file path")] = "mlabonne/FineTome-100k",
    split: Annotated[str, typer.Option(help="📂 Dataset split to use")] = "train",
    samples: Annotated[int, typer.Option("--samples", "-n", help="🔢 Number of samples (-1 for all)")] = 1000,
    
    # Processing settings
    workers: Annotated[int, typer.Option("--workers", "-w", help="⚡ Number of parallel workers")] = 4,
    train_on_target_only: Annotated[bool, typer.Option("--target-only/--full-conversation", help="🎯 Train only on assistant responses")] = True,
    
    # Chat template settings
    chat_template: Annotated[Optional[str], typer.Option(help="💬 Chat template to use")] = None,
    instruction_part: Annotated[Optional[str], typer.Option(help="👤 Text that starts user messages")] = None,
    response_part: Annotated[Optional[str], typer.Option(help="🤖 Text that starts assistant responses")] = None,
    
    # Output settings
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="📁 Output directory (auto-generated if not specified)")] = None,
    
    # Advanced settings
    debug: Annotated[int, typer.Option(help="🐛 Number of samples to preview (0=disabled)")] = 0,
    hf_token: Annotated[Optional[str], typer.Option(help="🔑 HuggingFace token for gated models/datasets")] = None,
    preset: Annotated[Optional[str], typer.Option(help="⚙️ Use a preset configuration")] = None,
    
    # Utility flags
    dry_run: Annotated[bool, typer.Option("--dry-run", help="👀 Show configuration without processing")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="💪 Overwrite existing output directory")] = False,
):
    """
    🚀 Prepare a dataset for fine-tuning
    
    This command processes conversational datasets into the format needed for OpenSloth training.
    
    Examples:
    
    • Quick start with defaults:
      [bold]opensloth-dataset prepare[/bold]
    
    • Use a preset for Llama models:
      [bold]opensloth-dataset prepare --preset llama_chat --dataset my/dataset[/bold]
    
    • Custom configuration:
      [bold]opensloth-dataset prepare --model unsloth/Qwen2.5-7B-Instruct --dataset mlabonne/FineTome-100k --samples 5000[/bold]
    
    • Process local dataset:
      [bold]opensloth-dataset prepare --dataset ./local_data.json --samples 1000[/bold]
    """
    
    # Apply preset if specified
    config = {}
    if preset:
        if preset not in PRESETS:
            console.print(f"❌ Unknown preset: {preset}", style="red")
            console.print("Available presets:", style="yellow")
            list_presets()
            raise typer.Exit(1)
        
        console.print(f"📋 Applying preset: [bold]{preset}[/bold]")
        config.update(PRESETS[preset]["defaults"])
        _show_preset_info(preset)
        console.print()
    
    # Override with command line arguments
    config.update({
        "tok_name": model,
        "dataset_name": dataset,
        "split": split,
        "num_samples": samples,
        "num_proc": workers,
        "train_on_target_only": train_on_target_only,
        "debug": debug,
        "hf_token": hf_token,
    })
    
    # Override chat template settings if provided
    if chat_template:
        config["chat_template"] = chat_template
    if instruction_part:
        config["instruction_part"] = instruction_part
    if response_part:
        config["response_part"] = response_part
    
    # Auto-generate output directory if not specified
    if not output:
        output = f"data/{_generate_dataset_name(model, dataset, samples)}"
    config["output_dir"] = output
    
    # Validate configuration
    if train_on_target_only and (not config.get("instruction_part") or not config.get("response_part")):
        console.print("❌ [red]Error:[/red] --target-only requires instruction_part and response_part to be set")
        console.print("💡 Use a preset with [bold]--preset[/bold] or specify [bold]--instruction-part[/bold] and [bold]--response-part[/bold]")
        raise typer.Exit(1)
    
    # Check if output directory exists
    if os.path.exists(output) and not force:
        console.print(f"❌ Output directory already exists: [bold]{output}[/bold]")
        console.print("💡 Use [bold]--force[/bold] to overwrite or choose a different output path")
        raise typer.Exit(1)
    
    # Show configuration summary
    console.print("\n🎯 [bold]Dataset Preparation Configuration[/bold]")
    console.print("=" * 50)
    
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column()
    
    info_table.add_row("🤖 Model:", config["tok_name"])
    info_table.add_row("📊 Dataset:", config["dataset_name"])
    info_table.add_row("📂 Split:", config["split"])
    info_table.add_row("🔢 Samples:", str(config["num_samples"]) if config["num_samples"] > 0 else "All")
    info_table.add_row("⚡ Workers:", str(config["num_proc"]))
    info_table.add_row("📁 Output:", config["output_dir"])
    info_table.add_row("🎯 Target Only:", "✅" if config["train_on_target_only"] else "❌")
    
    if config.get("chat_template"):
        info_table.add_row("💬 Chat Template:", config["chat_template"])
    if config.get("debug"):
        info_table.add_row("🐛 Debug Samples:", str(config["debug"]))
    
    console.print(info_table)
    console.print("=" * 50)
    
    if dry_run:
        console.print("\n👀 [bold yellow]Dry run - configuration shown above[/bold yellow]")
        console.print("\n📋 Full configuration:")
        console.print(json.dumps(config, indent=2))
        return
    
    # Confirm before processing
    if not typer.confirm(f"\n🚀 Start processing dataset?"):
        console.print("❌ Cancelled")
        raise typer.Exit()
    
    # Run the dataset preparation
    try:
        console.print(f"\n🔄 Starting dataset preparation...")
        
        # Lazy import dataset preparers
        DatasetPrepConfig, BaseDatasetPreparer, QwenDatasetPreparer, GemmaDatasetPreparer = _get_preparers()
        
        # Determine which preparer to use
        model_name = config["tok_name"]
        model_lower = model_name.lower()
        
        if "qwen" in model_lower:
            preparer = QwenDatasetPreparer()
        elif "gemma" in model_lower:
            preparer = GemmaDatasetPreparer()
        else:
            # Default to Qwen if we can't detect
            preparer = QwenDatasetPreparer()
            console.print("⚠️  Could not detect model family, defaulting to Qwen", style="yellow")
        
        # Call the preparation function directly
        console.print("📝 [bold]Processing Dataset[/bold]")
        console.print("-" * 50)
        
        try:
            output_dir = preparer.run_with_config(config)
            
            console.print(f"\n🎉 [bold green]Dataset preparation completed successfully![/bold green]")
            console.print(f"📁 Dataset saved to: [bold]{output_dir}[/bold]")
            
            # Show next steps
            console.print(f"\n📝 [bold]Next Steps:[/bold]")
            console.print(f"1. Train with: [bold]opensloth-train train --dataset {output_dir}[/bold]")
            console.print(f"2. Or use in config: [bold]data_cache_path: {output_dir}[/bold]")
                
        except Exception as e:
            console.print(f"\n❌ [bold red]Dataset preparation failed: {e}[/bold red]")
            raise typer.Exit(1)
            
    except KeyboardInterrupt:
        console.print(f"\n⏹️  [yellow]Processing interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n❌ [bold red]Error during processing: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("list-datasets")
def list_datasets():
    """📂 List available processed datasets"""
    data_dir = Path("data")
    
    if not data_dir.exists():
        console.print("📂 No data directory found")
        return
    
    datasets = []
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a processed dataset
            if any((item / f).exists() for f in ["dataset_info.json", "state.json"]) or \
               any(f.suffix == ".arrow" for f in item.glob("*.arrow")):
                
                # Get dataset info
                size = "Unknown"
                config_file = item / "dataset_config.json"
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                            size = f"{config.get('num_samples', 'Unknown')} samples"
                    except:
                        pass
                
                datasets.append((str(item), size, item.stat().st_mtime))
    
    if not datasets:
        console.print("📂 No processed datasets found in the data directory")
        return
    
    # Sort by modification time (newest first)
    datasets.sort(key=lambda x: x[2], reverse=True)
    
    table = Table(title="Available Processed Datasets")
    table.add_column("Dataset Path", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="yellow")
    
    for path, size, mtime in datasets:
        mod_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        table.add_row(path, size, mod_time)
    
    console.print(table)
    console.print(f"\n💡 Use any dataset path with: [bold]opensloth-train --dataset <path>[/bold]")


@app.command("info")  
def dataset_info(
    path: Annotated[str, typer.Argument(help="Path to processed dataset directory")]
):
    """📖 Show information about a processed dataset"""
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        console.print(f"❌ Dataset directory not found: {path}", style="red")
        raise typer.Exit(1)
    
    # Load dataset info
    info = {}
    
    config_file = dataset_path / "dataset_config.json"
    if config_file.exists():
        with open(config_file) as f:
            info.update(json.load(f))
    
    dataset_info_file = dataset_path / "dataset_info.json"
    if dataset_info_file.exists():
        with open(dataset_info_file) as f:
            dataset_info_data = json.load(f)
            info.update(dataset_info_data)
    
    if not info:
        console.print(f"❌ No dataset information found in {path}", style="red")
        raise typer.Exit(1)
    
    # Display information
    panel_content = f"""
[bold cyan]Dataset Path:[/bold cyan] {dataset_path}
[bold cyan]Source Dataset:[/bold cyan] {info.get('dataset_name', 'Unknown')}
[bold cyan]Model/Tokenizer:[/bold cyan] {info.get('tok_name', 'Unknown')}
[bold cyan]Samples:[/bold cyan] {info.get('num_samples', 'Unknown')}
[bold cyan]Split:[/bold cyan] {info.get('split', 'Unknown')}
[bold cyan]Target Only:[/bold cyan] {info.get('train_on_target_only', 'Unknown')}
"""
    
    if info.get('chat_template'):
        panel_content += f"[bold cyan]Chat Template:[/bold cyan] {info['chat_template']}\n"
    
    console.print(Panel(panel_content.strip(), title="📊 Dataset Information", border_style="blue"))
    
    # Show file structure
    files = list(dataset_path.glob("*"))
    if files:
        console.print("\n📁 [bold]Files:[/bold]")
        for file in sorted(files):
            if file.is_file():
                size = file.stat().st_size / (1024*1024)  # MB
                console.print(f"  📄 {file.name} ({size:.1f} MB)")
            else:
                console.print(f"  📁 {file.name}/")


if __name__ == "__main__":
    app()
