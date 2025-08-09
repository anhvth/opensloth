


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
import itertools

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


console = Console(force_terminal=True, color_system="auto", markup=True, soft_wrap=True, highlight=False)

# ---- Plain terminal colors (avoid Rich for conversation preview) ----
ANSI_RESET = "\033[0m"
ANSI_COLORS = {
    "dim": "\033[90m",           # bright black / gray
    "green_bold": "\033[1;32m",
    "yellow": "\033[33m",
    "cyan_bold": "\033[1;36m",
    "white": "\033[37m",
}

def _ansi_wrap(text: str, color_key: str) -> str:
    return f"{ANSI_COLORS.get(color_key, '')}{text}{ANSI_RESET}"


# ---- Split/print helpers for CLI token coloring ----
def _get_split_points_cli(parts_mask: list[bool]) -> list[int]:
    """Find split points where trainable/non-trainable sections change."""
    split_points = [0]
    for i in range(1, len(parts_mask)):
        if parts_mask[i] != parts_mask[i - 1]:
            split_points.append(i)
    split_points.append(len(parts_mask))
    return split_points

def _print_colored_tokens(tokenizer, input_ids: list[int], labels: list[int], max_tokens: int = 20000):
    """
    Decode directly from tokenizer and print colored segments:
    - green (bold) for trainable tokens (label >= 0)
    - default (no color) for context (label == -100)
    Uses ANSI `print` to avoid Rich side effects.
    """
    parts_mask = [lbl >= 0 for lbl in labels]
    split_points = _get_split_points_cli(parts_mask)
    segments = []
    for a, b in itertools.pairwise(split_points):
        # decode this contiguous segment (truncate very long)
        slice_ids = input_ids[a:b]
        if len(slice_ids) > max_tokens:
            slice_ids = slice_ids[:max_tokens]
        text = tokenizer.decode(slice_ids, skip_special_tokens=False)
        if parts_mask[a]:
            segments.append(_ansi_wrap(text, "green_bold"))
        else:
            segments.append(text)
    print("Text:")
    print("".join(segments))

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

def _extract_conversation(sample) -> List[dict]:
    """Extract a list of messages with roles and content from various common schemas.
    Returns a list of dicts like {"role": str, "content": str} or an empty list if not found.
    """
    # Common schema: {"messages": [{"role": ..., "content": ...}, ...]}
    if isinstance(sample, dict):
        if "messages" in sample and isinstance(sample["messages"], list):
            msgs = []
            for m in sample["messages"]:
                if isinstance(m, dict):
                    role = str(m.get("role", "unknown"))
                    content = m.get("content", "")
                    # content might be dict / list in some datasets; stringify safely
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    msgs.append({"role": role, "content": content})
            return msgs
        
        # Alternative schema (ShareGPT / HC3): {"conversations": [{"from": "human|gpt", "value": ...}, ...]
        if "conversations" in sample and isinstance(sample["conversations"], list):
            role_map = {"human": "user", "gpt": "assistant"}
            msgs = []
            for m in sample["conversations"]:
                if isinstance(m, dict):
                    raw_role = str(m.get("from", "unknown")).lower()
                    role = role_map.get(raw_role, raw_role)
                    content = m.get("value", "")
                    if not isinstance(content, str):
                        content = json.dumps(content, ensure_ascii=False)
                    msgs.append({"role": role, "content": content})
            return msgs
        
        # Alpaca-like: {"instruction": ..., "input": ..., "output": ...}
        if "instruction" in sample and "output" in sample:
            user_text = sample.get("instruction", "")
            if sample.get("input"):
                user_text = f"{user_text}\n{sample.get('input')}".strip()
            return [
                {"role": "user", "content": str(user_text)},
                {"role": "assistant", "content": str(sample.get("output", ""))},
            ]
    return []



def _print_conversation_fused(messages: List[dict]):
    """Print a fused chat transcript with <|im_start|>/<|im_end|> tags, color-coding roles using ANSI (no Rich)."""
    # Header
    print("Text:")

    # Role -> color mapping
    role_to_color = {
        "user": "dim",
        "system": "dim",
        "assistant": "green_bold",
        "tool": "yellow",
        "developer": "cyan_bold",
        "unknown": "white",
    }

    # Print each block in its role color; escape nothing, print literally
    for m in messages:
        role = str(m.get("role", "unknown")).lower()
        content = str(m.get("content", ""))
        block = f"<|im_start|>{role}\n{content}<|im_end|>"
        print(_ansi_wrap(block, role_to_color.get(role, "white")))
def _generate_dataset_name(model_name: str, dataset_name: str, num_samples: int, max_seq_length: int = 4096) -> str:
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
    
    # Format: model-dataset-samples-lNNNN-mmdd (where lNNNN indicates max length)
    name = f"{model_family}_{dataset_short}_n{num_samples}_l{max_seq_length}_{today}"
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
    # Positional argument for file or dataset
    input_file: Annotated[Optional[str], typer.Argument(help="📄 Local JSON/JSONL file path, or use --dataset for HuggingFace datasets")] = None,
    
    # Model and tokenizer settings
    model: Annotated[str, typer.Option("--model", "-m", help="🤖 HuggingFace model identifier or local path")] = "unsloth/Qwen2.5-0.5B-Instruct",
    
    # Dataset settings
    dataset: Annotated[Optional[str], typer.Option("--dataset", "-d", help="📊 HuggingFace dataset name")] = None,
    split: Annotated[str, typer.Option(help="📂 Dataset split to use")] = "train",
    samples: Annotated[int, typer.Option("--samples", "-n", help="🔢 Number of samples (-1 for all)")] = 1000,
    
    # Processing settings
    workers: Annotated[int, typer.Option("--workers", "-w", help="⚡ Number of parallel workers")] = 4,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length", help="📏 Maximum sequence length (tokens)")] = 4096,
    train_on_target_only: Annotated[bool, typer.Option("--target-only/--full-conversation", help="🎯 Train only on assistant responses")] = False,
    
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
    yes: Annotated[bool, typer.Option("--yes", "-y", help="✅ Skip confirmation prompt")] = False,
):
    """
    🚀 Prepare a dataset for fine-tuning
    
    This command processes conversational datasets into the format needed for OpenSloth training.
    It supports both HuggingFace datasets and local JSON/JSONL files.
    
    📊 **Supported Data Formats:**
    
    • **HuggingFace datasets** with 'conversations' or 'messages' columns
    • **Local JSON files** with list of conversations: [{"messages": [...]}, ...]
    • **Local JSONL files** with one conversation per line: {"messages": [...]}
    
    🔧 **Chat Format Requirements:**
    
    Each conversation should have messages with 'role' and 'content' fields:
    ```json
    {"messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]}
    ```
    
    Examples:
    
    • **Process local file:**
      [bold]opensloth-dataset prepare my_data.json --preset qwen_chat -y[/bold]
    
    • **Use HuggingFace dataset:**
      [bold]opensloth-dataset prepare --dataset mlabonne/FineTome-100k --preset qwen_chat[/bold]
    
    • **Custom configuration:**
      [bold]opensloth-dataset prepare --model unsloth/Qwen2.5-7B-Instruct --dataset mlabonne/FineTome-100k --samples 5000 --max-seq-length 8192[/bold]
    
    • **Response-only training:**
      [bold]opensloth-dataset prepare data.json --target-only --instruction-part "<|user|>" --response-part "<|assistant|>"[/bold]
    
    💡 **Tips:**
    • Use `--preset` for quick setup with popular models
    • Use `--target-only` to train only on assistant responses (requires instruction/response parts)
    • Use `--debug 5` to preview how your data will be processed
    • Use `-y` to skip confirmation prompts
    """
    
    # Determine data source
    if input_file and dataset:
        console.print("❌ [red]Error:[/red] Cannot specify both input file and --dataset. Use one or the other.")
        raise typer.Exit(1)
    
    if not input_file and not dataset:
        console.print("❌ [red]Error:[/red] Must specify either an input file or use --dataset for HuggingFace datasets")
        console.print("💡 Examples:")
        console.print("  • Local file: [bold]opensloth-dataset prepare my_data.json[/bold]")
        console.print("  • HuggingFace: [bold]opensloth-dataset prepare --dataset mlabonne/FineTome-100k[/bold]")
        raise typer.Exit(1)
    
    # Use input_file if provided, otherwise use dataset
    actual_dataset = input_file if input_file else dataset
    
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
    
    # Override with command line arguments (only if not using preset defaults)
    updates = {
        "tok_name": model,
        "dataset_name": actual_dataset,
        "split": split,
        "num_samples": samples,
        "num_proc": workers,
        "max_seq_length": max_seq_length,
        "debug": debug,
        "hf_token": hf_token,
    }
    
    # Only override train_on_target_only if not using a preset or explicitly set
    if not preset:
        updates["train_on_target_only"] = train_on_target_only
    
    config.update(updates)
    
    # If input_file is used, pass it to the preparer
    if input_file:
        config["input_file"] = input_file
    
    # Override chat template settings if provided
    if chat_template:
        config["chat_template"] = chat_template
    if instruction_part:
        config["instruction_part"] = instruction_part
    if response_part:
        config["response_part"] = response_part
    
    # Early validation for target-only training
    if config.get("train_on_target_only") and (not config.get("instruction_part") or not config.get("response_part")):
        console.print("❌ [red]Error:[/red] Target-only training requires instruction_part and response_part to be set")
        console.print("💡 Use a preset with [bold]--preset[/bold] or specify [bold]--instruction-part[/bold] and [bold]--response-part[/bold]")
        raise typer.Exit(1)
    
    # Validate instruction/response parts against chat template if target-only training is enabled
    if config.get("train_on_target_only"):
        _validate_chat_template_compatibility(config)
    
    # Auto-generate output directory if not specified
    if not output:
        output = f"data/{_generate_dataset_name(model, actual_dataset, samples, max_seq_length)}"
    config["output_dir"] = output
    
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
    info_table.add_row("📏 Max Length:", f"{config['max_seq_length']} tokens")
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
    if not yes and not typer.confirm(f"\n🚀 Start processing dataset?"):
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


@app.command("debug")
def debug_dataset(
    dataset: Annotated[Optional[str], typer.Option("--dataset", "-d", help="📁 Dataset directory to debug")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="🤖 Model to use for tokenization")] = None,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length", help="📏 Maximum sequence length")] = 4096,
    num_samples: Annotated[int, typer.Option("--samples", "-n", help="🔢 Number of samples to analyze")] = 5,
    show_tokens: Annotated[bool, typer.Option("--show-tokens", help="🔤 Show token details for each sample")] = False,
):
    """
    🔍 Debug and analyze your training dataset
    
    This command helps you understand how your data will be processed during training.
    It shows you:
    - Dataset statistics and configuration
    - Sample data with tokenization details  
    - Training mask visualization (what tokens will be trained on)
    - Sequence length distribution
    - Potential issues or warnings
    
    Examples:
    
    • Analyze a dataset:
      [bold]opensloth-dataset debug --dataset data/my_dataset[/bold]
    
    • Show detailed token information:
      [bold]opensloth-dataset debug --dataset data/my_dataset --show-tokens[/bold]
      
    • Analyze with specific model:
      [bold]opensloth-dataset debug --dataset data/my_dataset --model unsloth/Qwen2.5-7B-Instruct[/bold]
    """
    
    try:
        # Interactive dataset selection if not provided
        if not dataset:
            datasets = _list_cached_datasets()
            if not datasets:
                console.print("❌ No processed datasets found in 'data/' directory")
                console.print("💡 First prepare a dataset with: [bold]opensloth-dataset prepare[/bold]")
                raise typer.Exit(1)
            
            console.print("📂 [bold]Available datasets:[/bold]")
            for i, dataset_path in enumerate(datasets, 1):
                console.print(f"  {i}. {dataset_path}")
            
            while True:
                try:
                    choice = typer.prompt("\n🔢 Select dataset number")
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(datasets):
                        dataset = datasets[choice_idx]
                        console.print(f"✅ Selected: {dataset}")
                        break
                    else:
                        console.print(f"❌ Invalid choice. Please enter 1-{len(datasets)}")
                except ValueError:
                    console.print("❌ Please enter a valid number")
        
        # Validate dataset exists
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            console.print(f"❌ Dataset not found: {dataset}")
            raise typer.Exit(1)
        
        console.print(f"\n🔍 [bold]Analyzing Dataset: {dataset}[/bold]")
        console.print("=" * 60)
        
        # Load dataset configuration
        config_file = dataset_path / "dataset_config.json"
        dataset_config = {}
        if config_file.exists():
            with open(config_file) as f:
                dataset_config = json.load(f)
        
        # Show dataset configuration
        _show_dataset_config(dataset_config, dataset_path)
        
        # Load and analyze dataset
        _analyze_dataset_samples(dataset_path, model, max_seq_length, num_samples, show_tokens, dataset_config)
        
        console.print(f"\n🎯 [bold]Debug Summary[/bold]")
        console.print("=" * 60)
        console.print("✅ Dataset analysis complete! Use this information to:")
        console.print("  • Verify your data is correctly formatted")
        console.print("  • Check that training tokens are properly masked")
        console.print("  • Ensure sequence lengths are within limits")
        console.print("  • Identify any potential data quality issues")
        console.print(f"\n💡 Ready to train? Use: [bold]opensloth-train train --dataset {dataset}[/bold]")
        
    except KeyboardInterrupt:
        console.print(f"\n⏹️  [yellow]Debug interrupted by user[/yellow]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n❌ [bold red]Error during debug: {e}[/bold red]")
        raise typer.Exit(1)


def _list_cached_datasets() -> List[str]:
    """List available processed datasets."""
    datasets = []
    data_dir = Path("data")
    
    if not data_dir.exists():
        return []
    
    for item in data_dir.iterdir():
        if item.is_dir():
            datasets.append(str(item))
    
    # Sort by modification time (newest first)
    datasets.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return datasets


def _show_dataset_config(dataset_config: dict, dataset_path: Path):
    """Show dataset configuration information."""
    console.print("\n📋 [bold]Dataset Configuration[/bold]")
    
    config_table = Table.grid(padding=(0, 2))
    config_table.add_column(style="cyan")
    config_table.add_column()
    
    if dataset_config:
        config_table.add_row("🤖 Model/Tokenizer:", dataset_config.get("tok_name", "Unknown"))
        config_table.add_row("💬 Chat Template:", dataset_config.get("chat_template", "Unknown"))
        config_table.add_row("📏 Max Seq Length:", str(dataset_config.get("max_seq_length", "Unknown")))
        config_table.add_row("🎯 Target Only:", "✅" if dataset_config.get("train_on_target_only") else "❌")
        config_table.add_row("📊 Original Dataset:", dataset_config.get("dataset_name", "Unknown"))
        config_table.add_row("🔢 Num Samples:", str(dataset_config.get("num_samples", "Unknown")))
        
        if dataset_config.get("train_on_target_only"):
            config_table.add_row("📝 Instruction Part:", repr(dataset_config.get("instruction_part", "")))
            config_table.add_row("📝 Response Part:", repr(dataset_config.get("response_part", "")))
    else:
        config_table.add_row("⚠️  Configuration:", "No dataset_config.json found")
    
    console.print(config_table)


def _analyze_dataset_samples(dataset_path: Path, model: Optional[str], max_seq_length: int, 
                           num_samples: int, show_tokens: bool, dataset_config: dict):
    """Analyze dataset samples and show training details."""
    
    try:
        # Import here to avoid slow startup
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        
        console.print(f"\n📂 [bold]Loading dataset from {dataset_path}...[/bold]")
        dataset = load_from_disk(str(dataset_path))
        
        console.print(f"✅ Dataset loaded: {len(dataset)} samples")
        
        # Load tokenizer
        if not model:
            model = dataset_config.get("tok_name", "unsloth/Qwen2.5-0.5B-Instruct")
        
        console.print(f"🤖 Loading tokenizer: {model}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        
        # Dataset statistics
        _show_dataset_statistics(dataset, max_seq_length)
        
        # Sample analysis
        _show_sample_analysis(dataset, tokenizer, num_samples, show_tokens, dataset_config)
        
    except ImportError as e:
        console.print(f"❌ Missing dependencies: {e}")
        console.print("💡 Please install required packages: datasets, transformers")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error loading dataset: {e}")
        raise typer.Exit(1)


def _show_dataset_statistics(dataset, max_seq_length: int):
    """Show dataset statistics."""
    console.print(f"\n📊 [bold]Dataset Statistics[/bold]")
    
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="cyan")
    stats_table.add_column()
    
    stats_table.add_row("📦 Total Samples:", str(len(dataset)))
    stats_table.add_row("📋 Features:", ", ".join(dataset.column_names))
    
    # Sequence length analysis
    if "input_ids" in dataset.column_names and len(dataset) > 0:
        lengths = [len(sample["input_ids"]) for sample in dataset]
        avg_length = sum(lengths) / len(lengths)
        max_length = max(lengths)
        min_length = min(lengths)
        
        stats_table.add_row("📏 Avg Length:", f"{avg_length:.1f} tokens")
        stats_table.add_row("📏 Min Length:", f"{min_length} tokens")
        stats_table.add_row("📏 Max Length:", f"{max_length} tokens")
        
        # Check if any samples exceed max_seq_length
        over_length = sum(1 for length in lengths if length > max_seq_length)
        if over_length > 0:
            stats_table.add_row("⚠️  Over Max Length:", f"{over_length} samples ({over_length/len(lengths)*100:.1f}%)")
        else:
            stats_table.add_row("✅ Length Check:", f"All samples ≤ {max_seq_length}")
    elif len(dataset) == 0:
        stats_table.add_row("⚠️  Warning:", "Dataset is empty!")
    
    console.print(stats_table)


def _show_sample_analysis(dataset, tokenizer, num_samples: int, show_tokens: bool, dataset_config: dict):
    """Show detailed analysis of sample data."""
    if len(dataset) == 0:
        console.print(f"\n⚠️  [yellow]Dataset is empty - no samples to analyze[/yellow]")
        return
    
    console.print(f"\n🔍 [bold]Sample Analysis (showing {min(num_samples, len(dataset))} samples)[/bold]")
    
    train_on_target_only = dataset_config.get("train_on_target_only", False)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        
        console.print(f"\n📝 [bold cyan]Sample {i+1}:[/bold cyan]")
        console.print("-" * 40)

        # Prefer showing colored token preview if possible
        if "input_ids" in sample and "labels" in sample:
            _print_colored_tokens(tokenizer, sample["input_ids"], sample["labels"])
        else:
            # Fallback: try messages/text if tokenized fields are missing
            messages = _extract_conversation(sample if isinstance(sample, dict) else {})
            if messages:
                _print_conversation_fused(messages)
            elif "text" in sample:
                text = sample["text"]
                if isinstance(text, (list, dict)):
                    text = json.dumps(text, ensure_ascii=False)
                console.print(f"[bold]Text:[/bold] {text}")

        # Show tokenization info (if available)
        if "input_ids" in sample:
            input_ids = sample["input_ids"]
            labels = sample.get("labels", input_ids)
            
            console.print(f"[bold]Length:[/bold] {len(input_ids)} tokens")
            
            if train_on_target_only:
                # Count training tokens (not -100)
                training_tokens = sum(1 for label in labels if label != -100)
                masked_tokens = sum(1 for label in labels if label == -100)
                console.print(f"[bold]Training Tokens:[/bold] [green]{training_tokens}[/green] ([green]{training_tokens/len(input_ids)*100:.1f}%[/green])")
                console.print(f"[bold]Masked Tokens:[/bold] [red]{masked_tokens}[/red] ([red]{masked_tokens/len(input_ids)*100:.1f}%[/red])")
                
                if training_tokens == 0:
                    console.print("⚠️  [red]Warning: No training tokens found in this sample![/red]")


def _show_token_details(tokenizer, input_ids, labels, train_on_target_only: bool):
    """Show detailed token information with color coding."""
    console.print(f"\n[bold]Token Details:[/bold]")
    
    # Show first 50 tokens to avoid overwhelming output
    display_tokens = min(50, len(input_ids))
    
    token_table = Table(show_header=True, header_style="bold magenta")
    token_table.add_column("Pos", style="dim", width=4)
    token_table.add_column("Token", width=15)
    token_table.add_column("ID", style="dim", width=8)
    if train_on_target_only:
        token_table.add_column("Label", width=8)
        token_table.add_column("Train", width=5)
    
    for i in range(display_tokens):
        token_id = input_ids[i]
        token = tokenizer.decode([token_id])
        label = labels[i] if i < len(labels) else token_id
        
        # Clean up token display
        token_display = repr(token)
        if len(token_display) > 15:
            token_display = token_display[:12] + "..."
        
        if train_on_target_only:
            is_training = label != -100
            label_display = str(label) if label != -100 else "-100"
            train_display = "✅" if is_training else "❌"
            
            # Apply color styling to entire row based on training status
            if is_training:
                row_style = "green"
            else:
                row_style = "red"
            
            token_table.add_row(
                str(i), 
                token_display, 
                str(token_id), 
                label_display, 
                train_display,
                style=row_style
            )
        else:
            token_table.add_row(str(i), token_display, str(token_id))
    
    console.print(token_table)
    
    if len(input_ids) > display_tokens:
        console.print(f"... ([yellow]{len(input_ids) - display_tokens}[/yellow] more tokens)")


def _validate_chat_template_compatibility(config: dict):
    """Validate that instruction/response parts are compatible with the chat template."""
    try:
        from transformers import AutoTokenizer
        from unsloth.chat_templates import get_chat_template
        
        # Load tokenizer and apply chat template
        tokenizer = AutoTokenizer.from_pretrained(config["tok_name"])
        chat_template = config.get("chat_template")
        
        if chat_template:
            tokenizer = get_chat_template(tokenizer, chat_template=chat_template)
        
        # Get the actual chat template text
        template_text = tokenizer.chat_template or ""
        
        instruction_part = config.get("instruction_part", "")
        response_part = config.get("response_part", "")
        
        # Clean up the parts for comparison (remove escaping)
        instruction_clean = instruction_part.replace("\\n", "\n").replace("\\", "")
        response_clean = response_part.replace("\\n", "\n").replace("\\", "")
        
        # Check if instruction and response parts are in the template
        instruction_found = instruction_clean in template_text
        response_found = response_clean in template_text
        
        if not instruction_found or not response_found:
            console.print("⚠️  [yellow]Warning: Chat template validation[/yellow]")
            if not instruction_found:
                console.print(f"   • Instruction part '{instruction_part}' not found in chat template")
            if not response_found:
                console.print(f"   • Response part '{response_part}' not found in chat template")
            console.print(f"   • Chat template: {chat_template}")
            console.print("   • This may cause issues with response-only training")
            console.print("   • Consider using a preset or verify your instruction/response parts")
        
    except Exception as e:
        console.print(f"⚠️  [yellow]Warning: Could not validate chat template compatibility: {e}[/yellow]")


if __name__ == "__main__":
    app()
