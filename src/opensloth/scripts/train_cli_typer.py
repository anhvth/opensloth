#!/usr/bin/env python3
"""
OpenSloth Training CLI

A modern, intuitive command-line interface for fine-tuning large language models.
Built with Typer for excellent user experience, type safety, and rich output.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, List, Annotated
from datetime import datetime
from enum import Enum

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

# Add the parent directory to sys.path for imports
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

app = typer.Typer(
    name="opensloth-train",
    help="🦥 OpenSloth Training - Fine-tune large language models with ease",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

# Training presets for different scenarios
TRAINING_PRESETS = {
    "quick_test": {
        "description": "Quick test run with minimal steps for validation",
        "config": {
            "training_args": {
                "max_steps": 50,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "logging_steps": 1,
                "save_total_limit": 1,
                "report_to": "none",
            }
        }
    },
    "small_model": {
        "description": "Optimized for models < 3B parameters",
        "config": {
            "opensloth_config": {
                "fast_model_args": {
                    "max_seq_length": 2048,
                    "load_in_4bit": True,
                },
                "lora_args": {
                    "r": 16,
                    "lora_alpha": 32,
                }
            },
            "training_args": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "num_train_epochs": 3,
                "warmup_steps": 100,
            }
        }
    },
    "large_model": {
        "description": "Optimized for models > 7B parameters",
        "config": {
            "opensloth_config": {
                "fast_model_args": {
                    "max_seq_length": 4096,
                    "load_in_4bit": True,
                },
                "lora_args": {
                    "r": 8,
                    "lora_alpha": 16,
                }
            },
            "training_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "num_train_epochs": 2,
                "warmup_steps": 50,
            }
        }
    },
    "full_finetune": {
        "description": "Full parameter fine-tuning (requires lots of VRAM)",
        "config": {
            "opensloth_config": {
                "fast_model_args": {
                    "full_finetuning": True,
                    "load_in_4bit": False,
                    "max_seq_length": 2048,
                },
                "lora_args": None,
            },
            "training_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 32,
                "learning_rate": 5e-6,
                "num_train_epochs": 1,
                "warmup_steps": 10,
            }
        }
    },
    "memory_efficient": {
        "description": "Lowest memory usage configuration",
        "config": {
            "opensloth_config": {
                "fast_model_args": {
                    "max_seq_length": 1024,
                    "load_in_4bit": True,
                },
                "lora_args": {
                    "r": 4,
                    "lora_alpha": 8,
                }
            },
            "training_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 3e-4,
                "optim": "adamw_8bit",
            }
        }
    }
}

# Enums for better type safety and validation
class OptimizerType(str, Enum):
    adamw_torch = "adamw_torch"
    adamw_8bit = "adamw_8bit"  
    sgd = "sgd"

class SchedulerType(str, Enum):
    linear = "linear"
    cosine = "cosine"
    constant = "constant"
    constant_with_warmup = "constant_with_warmup"

class ReportType(str, Enum):
    none = "none"
    tensorboard = "tensorboard"
    wandb = "wandb"


def _get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return [0]  # Default to single GPU


def _list_cached_datasets() -> List[str]:
    """List available processed datasets."""
    datasets = []
    data_dir = Path("data")
    
    if not data_dir.exists():
        return datasets
    
    for item in data_dir.iterdir():
        if item.is_dir():
            # Check if it looks like a processed dataset
            if any((item / f).exists() for f in ["dataset_info.json", "state.json"]) or \
               any(f.suffix == ".arrow" for f in item.glob("*.arrow")):
                datasets.append(str(item))
    
    # Sort by modification time (newest first)
    datasets.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return datasets


def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate output directory from model and dataset."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name (remove path/org)
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    model_short = model_short.replace('-', '_')
    
    # Extract dataset name from path
    dataset_name = Path(dataset_path).name
    
    return f"outputs/{model_short}_{dataset_name}_{timestamp}"


def _validate_config(config: dict) -> None:
    """Validate the training configuration."""
    opensloth_config = config.get("opensloth_config", {})
    training_args = config.get("training_args", {})
    
    # Required fields
    if not opensloth_config.get("data_cache_path"):
        raise typer.BadParameter("Dataset path is required (--dataset)")
    
    fast_model_args = opensloth_config.get("fast_model_args", {})
    if not fast_model_args.get("model_name"):
        raise typer.BadParameter("Model name is required (--model)")
    
    # Check dataset exists
    dataset_path = opensloth_config["data_cache_path"]
    if not os.path.exists(dataset_path):
        raise typer.BadParameter(f"Dataset directory does not exist: {dataset_path}")
    
    # Validate GPU indices
    devices = opensloth_config.get("devices", [0])
    available_gpus = _get_available_gpus()
    for device in devices:
        if device not in available_gpus:
            console.print(f"⚠️  Warning: GPU {device} may not be available")
    
    # Check for conflicting options
    if training_args.get("max_steps") and training_args.get("num_train_epochs"):
        console.print("⚠️  Warning: Both max_steps and num_train_epochs specified. max_steps will take precedence.")


def _merge_configs(base: dict, override: dict) -> dict:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if value is None:
            continue
            
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def _apply_defaults(config: dict) -> dict:
    """Apply sensible defaults to configuration."""
    # Default OpenSloth config
    default_opensloth = {
        "devices": [0],
        "sequence_packing": True,
        "log_level": "info",
        "fast_model_args": {
            "max_seq_length": 4096,
            "load_in_4bit": True,
            "load_in_8bit": False,
            "full_finetuning": False,
        },
        "lora_args": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "use_rslora": False,
        }
    }
    
    # Default training args
    default_training = {
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "learning_rate": 2e-4,
        "num_train_epochs": 3,
        "lr_scheduler_type": "linear",
        "warmup_steps": 10,
        "logging_steps": 1,
        "save_total_limit": 2,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "report_to": "tensorboard",
    }
    
    result = {
        "opensloth_config": _merge_configs(default_opensloth, config.get("opensloth_config", {})),
        "training_args": _merge_configs(default_training, config.get("training_args", {}))
    }
    
    # Remove LoRA args if full finetuning
    if result["opensloth_config"]["fast_model_args"].get("full_finetuning"):
        result["opensloth_config"]["lora_args"] = None
    
    return result


def _print_config_summary(config: dict) -> None:
    """Print a summary of the training configuration."""
    opensloth_config = config["opensloth_config"]
    training_args = config["training_args"]
    fast_model_args = opensloth_config["fast_model_args"]
    lora_args = opensloth_config.get("lora_args")
    
    console.print("\n🎯 [bold]Training Configuration Summary[/bold]")
    console.print("=" * 50)
    
    # Create info table
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="cyan")
    info_table.add_column()
    
    # Model and dataset
    info_table.add_row("🤖 Model:", fast_model_args['model_name'])
    info_table.add_row("📊 Dataset:", opensloth_config['data_cache_path'])
    info_table.add_row("💾 Output:", training_args['output_dir'])
    
    # Hardware
    devices = opensloth_config['devices']
    info_table.add_row("🔧 GPUs:", f"{devices} ({len(devices)} GPU{'s' if len(devices) > 1 else ''})")
    
    # Model configuration
    max_length = fast_model_args['max_seq_length']
    quant = "4-bit" if fast_model_args.get('load_in_4bit') else "8-bit" if fast_model_args.get('load_in_8bit') else "none"
    info_table.add_row("📏 Max Length:", str(max_length))
    info_table.add_row("🔢 Quantization:", quant)
    
    # Training type
    if fast_model_args.get('full_finetuning'):
        info_table.add_row("🎯 Training Type:", "Full Fine-tuning")
    else:
        info_table.add_row("🎯 Training Type:", "LoRA")
        if lora_args:
            info_table.add_row("   - Rank:", str(lora_args['r']))
            info_table.add_row("   - Alpha:", str(lora_args['lora_alpha']))
    
    # Training parameters
    batch_size = training_args['per_device_train_batch_size']
    accumulation = training_args['gradient_accumulation_steps']
    effective_batch = batch_size * accumulation * len(devices)
    info_table.add_row("📦 Batch Size:", f"{batch_size} per GPU (effective: {effective_batch})")
    info_table.add_row("📚 Learning Rate:", str(training_args['learning_rate']))
    
    if training_args.get('max_steps'):
        info_table.add_row("⏰ Max Steps:", str(training_args['max_steps']))
    else:
        info_table.add_row("🔄 Epochs:", str(training_args.get('num_train_epochs', 'not set')))
    
    info_table.add_row("🔥 Optimizer:", training_args['optim'])
    info_table.add_row("📈 Scheduler:", training_args['lr_scheduler_type'])
    info_table.add_row("📊 Logging:", training_args['report_to'])
    
    console.print(info_table)
    console.print("=" * 50)


@app.command("list-presets")
def list_presets():
    """📋 List all available training presets"""
    table = Table(title="Available Training Presets")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    table.add_column("Use Case", style="yellow")
    
    for name, info in TRAINING_PRESETS.items():
        use_case = {
            "quick_test": "Testing & Validation",
            "small_model": "< 3B Parameters", 
            "large_model": "> 7B Parameters",
            "full_finetune": "Full Parameter Training",
            "memory_efficient": "Low VRAM"
        }.get(name, "General")
        
        table.add_row(name, info["description"], use_case)
    
    console.print(table)
    console.print("\n💡 Use [bold]--preset [preset_name][/bold] to apply a preset configuration")


@app.command("list-datasets")
def list_datasets():
    """📂 List available processed datasets"""
    datasets = _list_cached_datasets()
    
    if not datasets:
        console.print("📂 No processed datasets found in the data directory")
        console.print("💡 Use [bold]opensloth-dataset prepare[/bold] to prepare a dataset first")
        return
    
    table = Table(title="Available Processed Datasets")
    table.add_column("Dataset Path", style="cyan")
    table.add_column("Modified", style="yellow")
    
    for path in datasets:
        mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        table.add_row(path, mod_time)
    
    console.print(table)
    console.print(f"\n💡 Use any dataset path with: [bold]--dataset <path>[/bold]")


@app.command("train")
def train_model(
    # Required arguments
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="📊 Path to processed dataset directory")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="🤖 HuggingFace model identifier or local path")] = "unsloth/Qwen2.5-0.5B-Instruct",
    
    # Output settings
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="💾 Output directory for model checkpoints")] = None,
    
    # Hardware settings
    gpus: Annotated[Optional[str], typer.Option("--gpus", "-g", help="🔧 GPU indices (e.g. '0' or '0,1,2,3')")] = "0",
    
    # Training hyperparameters
    epochs: Annotated[Optional[int], typer.Option("--epochs", help="🔄 Number of training epochs")] = None,
    max_steps: Annotated[Optional[int], typer.Option("--max-steps", help="⏰ Maximum training steps (overrides epochs)")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="📦 Batch size per GPU")] = None,
    accumulation_steps: Annotated[Optional[int], typer.Option("--accumulation-steps", help="🔄 Gradient accumulation steps")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--learning-rate", "--lr", help="📚 Learning rate")] = None,
    warmup_steps: Annotated[Optional[int], typer.Option("--warmup-steps", help="🔥 Number of warmup steps")] = None,
    
    # Model configuration
    max_length: Annotated[Optional[int], typer.Option("--max-length", help="📏 Maximum sequence length")] = None,
    load_in_4bit: Annotated[bool, typer.Option("--4bit/--no-4bit", help="🔢 Use 4-bit quantization")] = True,
    load_in_8bit: Annotated[bool, typer.Option("--8bit", help="🔢 Use 8-bit quantization instead")] = False,
    full_finetune: Annotated[bool, typer.Option("--full-finetune", help="🎯 Full parameter fine-tuning instead of LoRA")] = False,
    
    # LoRA configuration
    lora_r: Annotated[Optional[int], typer.Option("--lora-r", help="🎯 LoRA rank (higher = more parameters)")] = None,
    lora_alpha: Annotated[Optional[int], typer.Option("--lora-alpha", help="🎯 LoRA alpha parameter")] = None,
    lora_dropout: Annotated[Optional[float], typer.Option("--lora-dropout", help="🎯 LoRA dropout rate")] = None,
    
    # Optimization settings
    optimizer: Annotated[OptimizerType, typer.Option("--optimizer", help="🔥 Optimizer to use")] = OptimizerType.adamw_8bit,
    scheduler: Annotated[SchedulerType, typer.Option("--scheduler", help="📈 Learning rate scheduler")] = SchedulerType.linear,
    weight_decay: Annotated[Optional[float], typer.Option("--weight-decay", help="⚖️ Weight decay")] = None,
    
    # Logging and checkpointing
    logging_steps: Annotated[Optional[int], typer.Option("--logging-steps", help="📝 Log every N steps")] = None,
    save_steps: Annotated[Optional[int], typer.Option("--save-steps", help="💾 Save checkpoint every N steps")] = None,
    save_total_limit: Annotated[Optional[int], typer.Option("--save-total-limit", help="💾 Max checkpoints to keep")] = None,
    report_to: Annotated[ReportType, typer.Option("--report-to", help="📊 Logging backend")] = ReportType.tensorboard,
    
    # Advanced settings
    sequence_packing: Annotated[bool, typer.Option("--packing/--no-packing", help="⚡ Enable sequence packing")] = True,
    resume: Annotated[Optional[str], typer.Option("--resume", help="🔄 Resume from checkpoint directory")] = None,
    
    # Preset and utility
    preset: Annotated[Optional[str], typer.Option("--preset", help="⚙️ Use a preset configuration")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="👀 Show configuration without training")] = False,
):
    """
    🚀 Train a large language model with OpenSloth
    
    This command fine-tunes language models using LoRA or full parameter training.
    
    Examples:
    
    • Quick training with defaults:
      [bold]opensloth-train train --dataset data/my_dataset[/bold]
    
    • Multi-GPU training:
      [bold]opensloth-train train --dataset data/my_dataset --gpus 0,1,2,3[/bold]
    
    • Use a preset for large models:
      [bold]opensloth-train train --dataset data/my_dataset --preset large_model[/bold]
    
    • Custom configuration:
      [bold]opensloth-train train --dataset data/my_dataset --model unsloth/Qwen2.5-7B-Instruct --epochs 5 --batch-size 2[/bold]
    
    • Full fine-tuning:
      [bold]opensloth-train train --dataset data/my_dataset --full-finetune --gpus 0,1,2,3[/bold]
    """
    
    # Interactive dataset selection if not provided
    if not dataset:
        datasets = _list_cached_datasets()
        if not datasets:
            console.print("❌ No processed datasets found!")
            console.print("💡 Use [bold]opensloth-dataset prepare[/bold] to prepare a dataset first")
            raise typer.Exit(1)
        
        console.print("📂 Available datasets:")
        for i, ds in enumerate(datasets):
            console.print(f"  {i+1}. {ds}")
        
        choice = typer.prompt("\n🔢 Select dataset number", type=int)
        if choice < 1 or choice > len(datasets):
            console.print("❌ Invalid selection")
            raise typer.Exit(1)
        dataset = datasets[choice-1]
        console.print(f"✅ Selected: [bold]{dataset}[/bold]")
    
    # Start with empty config
    config = {"opensloth_config": {}, "training_args": {}}
    
    # Apply preset if specified
    if preset:
        if preset not in TRAINING_PRESETS:
            console.print(f"❌ Unknown preset: {preset}", style="red")
            console.print("Available presets:", style="yellow")
            list_presets()
            raise typer.Exit(1)
        
        console.print(f"📋 Applying preset: [bold]{preset}[/bold]")
        preset_config = TRAINING_PRESETS[preset]["config"]
        config = _merge_configs(config, preset_config)
    
    # Build configuration from command line arguments
    cli_config = {"opensloth_config": {}, "training_args": {}}
    
    # OpenSloth config
    cli_config["opensloth_config"]["data_cache_path"] = dataset
    
    if gpus:
        try:
            devices = [int(d.strip()) for d in gpus.split(",")]
            cli_config["opensloth_config"]["devices"] = devices
        except ValueError:
            raise typer.BadParameter(f"Invalid GPU specification: {gpus}")
    
    cli_config["opensloth_config"]["sequence_packing"] = sequence_packing
    
    # Fast model args
    fast_model_args = {"model_name": model}
    if max_length:
        fast_model_args["max_seq_length"] = max_length
    if load_in_8bit:
        fast_model_args["load_in_8bit"] = True
        fast_model_args["load_in_4bit"] = False
    else:
        fast_model_args["load_in_4bit"] = load_in_4bit
    fast_model_args["full_finetuning"] = full_finetune
    cli_config["opensloth_config"]["fast_model_args"] = fast_model_args
    
    # LoRA args (if not full finetuning)
    if not full_finetune:
        lora_args = {}
        if lora_r:
            lora_args["r"] = lora_r
        if lora_alpha:
            lora_args["lora_alpha"] = lora_alpha
        if lora_dropout is not None:
            lora_args["lora_dropout"] = lora_dropout
        if lora_args:
            cli_config["opensloth_config"]["lora_args"] = lora_args
    
    # Training args
    training_args = {}
    if epochs:
        training_args["num_train_epochs"] = epochs
    if max_steps:
        training_args["max_steps"] = max_steps
    if batch_size:
        training_args["per_device_train_batch_size"] = batch_size
    if accumulation_steps:
        training_args["gradient_accumulation_steps"] = accumulation_steps
    if learning_rate:
        training_args["learning_rate"] = learning_rate
    if warmup_steps:
        training_args["warmup_steps"] = warmup_steps
    if weight_decay:
        training_args["weight_decay"] = weight_decay
    if logging_steps:
        training_args["logging_steps"] = logging_steps
    if save_steps:
        training_args["save_steps"] = save_steps
    if save_total_limit:
        training_args["save_total_limit"] = save_total_limit
    if resume:
        training_args["resume_from_checkpoint"] = resume
    
    training_args["optim"] = optimizer.value
    training_args["lr_scheduler_type"] = scheduler.value
    training_args["report_to"] = report_to.value
    
    cli_config["training_args"] = training_args
    
    # Merge with preset config
    config = _merge_configs(config, cli_config)
    
    # Apply defaults
    config = _apply_defaults(config)
    
    # Auto-generate output directory if not specified
    if not output:
        output = _generate_output_dir(model, dataset)
    config["training_args"]["output_dir"] = output
    
    # Validate configuration
    try:
        _validate_config(config)
    except typer.BadParameter as e:
        console.print(f"❌ Configuration error: {e}", style="red")
        raise typer.Exit(1)
    
    # Print configuration summary
    _print_config_summary(config)
    
    if dry_run:
        console.print("\n👀 [bold yellow]Dry run - configuration shown above[/bold yellow]")
        console.print("\n📋 Full configuration:")
        console.print(json.dumps(config, indent=2))
        return
    
    # Generate and save config file for user to review/edit
    config_file = Path(output) / "training_config.json"
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    console.print(f"\n📝 [bold]Configuration saved to:[/bold] {config_file}")
    console.print("💡 You can edit this file to customize training parameters")
    
    # Ask user if they want to edit the config
    edit_config = typer.confirm("📝 Would you like to edit the configuration before training?")
    
    if edit_config:
        console.print(f"\n📂 Opening config file for editing: {config_file}")
        console.print("📝 Please edit the file and save it, then press Enter to continue...")
        
        # Try to open the file in the user's default editor
        import subprocess as sp
        import shutil
        
        editor = os.environ.get('EDITOR', None)
        if not editor:
            # Try common editors
            for candidate in ['code', 'nano', 'vim', 'vi']:
                if shutil.which(candidate):
                    editor = candidate
                    break
        
        if editor:
            try:
                sp.run([editor, str(config_file)])
            except Exception:
                console.print(f"⚠️  Could not open editor. Please manually edit: {config_file}")
        else:
            console.print(f"⚠️  No editor found. Please manually edit: {config_file}")
        
        # Wait for user to confirm they've finished editing
        typer.confirm("✅ Have you finished editing the configuration?", abort=True)
        
        # Reload the config
        try:
            with open(config_file) as f:
                updated_config = json.load(f)
            config = updated_config
            console.print("✅ Configuration reloaded from file")
            
            # Re-validate
            _validate_config(config)
            
        except Exception as e:
            console.print(f"❌ Error loading updated config: {e}", style="red")
            raise typer.Exit(1)
    
    # Final confirmation before training
    console.print(f"\n🎯 [bold]Final Training Configuration:[/bold]")
    _print_config_summary(config)
    
    if not typer.confirm(f"\n🚀 Start training with this configuration?"):
        console.print("❌ Training cancelled")
        raise typer.Exit()
    
    # Run training
    try:
        console.print(f"\n🔄 Starting training...")
        
        # Call the existing training script
        proc = subprocess.Popen(
            ['python', 'prepare_dataset/run_train_job.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        
        proc.stdin.write(json.dumps(config))
        proc.stdin.close()
        
        # Stream output with rich formatting
        console.print("📝 [bold]Live Training Logs[/bold]")
        console.print("-" * 50)
        
        for line in proc.stdout:
            # Strip ANSI escape codes and enhanced log formatting
            import re
            line = line.rstrip()
            # Remove ANSI escape sequences
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_line = ansi_escape.sub('', line)
            
            if any(word in clean_line.lower() for word in ["error", "failed", "exception"]):
                console.print(f"❌ {clean_line}", style="red")
            elif any(word in clean_line.lower() for word in ["warning", "warn"]):
                console.print(f"⚠️  {clean_line}", style="yellow")
            elif any(word in clean_line.lower() for word in ["epoch", "step", "loss"]):
                console.print(f"📊 {clean_line}", style="blue")
            elif "starting" in clean_line.lower() or "initializing" in clean_line.lower():
                console.print(f"🚀 {clean_line}", style="green")
            elif "completed" in clean_line.lower() or "finished" in clean_line.lower():
                console.print(f"✅ {clean_line}", style="green bold")
            else:
                console.print(clean_line)
        
        code = proc.wait()
        
        if code == 0:
            console.print(f"\n🎉 [bold green]Training completed successfully![/bold green]")
            console.print(f"📁 Model saved to: [bold]{output}[/bold]")
            
            # Show next steps
            console.print(f"\n📝 [bold]Next Steps:[/bold]")
            console.print(f"1. Test model: [bold]python scripts/test_model.py {output}[/bold]")
            console.print(f"2. Merge LoRA: [bold]python scripts/merge_lora.py {output}[/bold]")
            console.print(f"3. Push to Hub: [bold]python scripts/push_to_hub.py {output}[/bold]")
            
        else:
            console.print(f"\n❌ [bold red]Training failed with exit code: {code}[/bold red]")
            raise typer.Exit(code)
            
    except KeyboardInterrupt:
        console.print(f"\n⏹️  [yellow]Training interrupted by user[/yellow]")
        if proc.poll() is None:
            proc.terminate()
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n❌ [bold red]Error during training: {e}[/bold red]")
        raise typer.Exit(1)


# Add the train command as the default
@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version and exit")] = False
):
    """🦥 OpenSloth Training CLI - Fine-tune large language models with ease"""
    if version:
        console.print("OpenSloth Training CLI v1.0.0")
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        # If no subcommand, show help
        console.print(ctx.get_help())
        console.print("\n💡 [bold]Quick start:[/bold] [cyan]opensloth-train train --dataset data/my_dataset[/cyan]")


if __name__ == "__main__":
    app()
