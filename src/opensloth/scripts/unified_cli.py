#!/usr/bin/env python3
"""
OpenSloth Unified CLI

A clean, unified command-line interface for OpenSloth.
- `os prepare-data` - Prepare datasets for fine-tuning
- `os train` - Train models with pre-processed datasets

No redundancy, no confusion, just clean workflows.
"""

import os
import sys
import json
import statistics
import subprocess
import re
from pathlib import Path
from typing import Optional, List, Annotated, Dict, Tuple
from datetime import datetime
import tempfile
import textwrap

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

from opensloth._debug_dataloader import print_dataloader_example_short

# Main app
app = typer.Typer(
    name="os",
    help="🦥 OpenSloth - Unified CLI for dataset preparation and model training",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console(force_terminal=True, color_system="auto", markup=True, soft_wrap=True, highlight=False)

# ============================================================================
# SHARED UTILITIES
# ============================================================================

def _fail(msg: str):
    """Print error message and exit."""
    console.print(f"❌ [red]Error:[/red] {msg}")
    raise typer.Exit(1)

def _print_header(title: str):
    """Print a formatted header."""
    console.print(f"\n{title}")
    console.print("=" * 50)

def _print_kv(items: List[tuple]):
    """Print key-value pairs in a table format."""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    for key, value in items:
        table.add_row(key, str(value))
    console.print(table)

def _get_model_family(model_name: str) -> str:
    """Extract model family from model name."""
    model_lower = (model_name or "").lower()
    if "qwen" in model_lower:
        return "qwen"
    elif "gemma" in model_lower:
        return "gemma" 
    elif "llama" in model_lower:
        return "llama"
    elif "mistral" in model_lower:
        return "mistral"
    else:
        return "unknown"

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

def _load_dataset_config(dataset_path: str) -> dict:
    """Load dataset configuration if available."""
    config_file = Path(dataset_path) / "dataset_config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                return json.load(f)
        except Exception as e:
            console.print(f"⚠️  Warning: Could not load dataset config: {e}", style="yellow")
    return {}

# ============================================================================
# DATASET PREPARATION COMMANDS
# ============================================================================

# Import dataset preparation functionality
def _lazy_import_dataset_prep():
    """Lazy import of dataset preparation to avoid slow startup."""
    try:
        from opensloth.dataset import (
            DatasetPrepConfig,
            BaseDatasetPreparer,
            QwenDatasetPreparer,
            GemmaDatasetPreparer
        )
        return DatasetPrepConfig, BaseDatasetPreparer, QwenDatasetPreparer, GemmaDatasetPreparer
    except ImportError as e:
        _fail(f"Dataset preparation modules not available: {e}")

def _get_chat_template():
    """Lazy import of Unsloth chat template utilities."""
    # NOTE: Avoiding global unsloth import to prevent GPU registry issues
    # This function is not currently used in CLI workflow
    return None

# Chat template configurations
CHAT_TEMPLATES = {
    "chatml": {
        "description": "ChatML format (Qwen, OpenHermes, etc.)",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "train_on_target_only": True,
    },
    "qwen-2.5": {
        "description": "Qwen 2.5 format",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "train_on_target_only": True,
    },
    "llama-3": {
        "description": "Llama 3 format",
        "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
        "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "train_on_target_only": True,
    },
    "gemma": {
        "description": "Gemma format",
        "instruction_part": "<start_of_turn>user\n",
        "response_part": "<start_of_turn>model\n",
        "train_on_target_only": True,
    },
    "mistral": {
        "description": "Mistral Instruct format",
        "instruction_part": "[INST] ",
        "response_part": " [/INST]",
        "train_on_target_only": True,
    },
}

# Model detection heuristics for auto-template
NAME_MAP: Dict[str, str] = {
    r"\bqwen\s*3\b": "chatml",
    r"\bqwen\s*2\.?5\b": "qwen-2.5", 
    r"\bqwen\b": "chatml",
    r"\bllama[-\s]*3\b": "llama-3",
    r"\bgemma\b": "gemma",
    r"\bmixtral\b": "mistral",
    r"\bmistral\b": "mistral",
}

def _detect_template_by_model_name(model_name: str) -> Optional[str]:
    """Auto-detect chat template from model name."""
    if not model_name:
        return None
    
    name_lower = model_name.lower()
    for pattern, template_key in NAME_MAP.items():
        if re.search(pattern, name_lower):
            return template_key
    return None

def _apply_fallback_template(config: dict, model_name: str) -> bool:
    """Apply auto-detected template for target-only training."""
    if not config.get("train_on_target_only"):
        return False
    
    if config.get("instruction_part") and config.get("response_part"):
        return False  # Already configured
    
    detected_template = _detect_template_by_model_name(model_name)
    if not detected_template or detected_template not in CHAT_TEMPLATES:
        return False
    
    template_config = CHAT_TEMPLATES[detected_template]
    config.update({
        "chat_template": detected_template,
        "instruction_part": template_config["instruction_part"],
        "response_part": template_config["response_part"],
        "train_on_target_only": template_config["train_on_target_only"],
    })
    
    console.print(f"🎯 Auto-detected template: [bold]{detected_template}[/bold] (from model: {model_name})")
    return True

def _generate_dataset_name(model_name: str, dataset_name: str, num_samples: int, max_seq_length: int = 4096) -> str:
    """Auto-generate dataset directory name."""
    today = datetime.now().strftime("%m%d")
    model_family = _get_model_family(model_name)
    dataset_short = dataset_name.split('/')[-1].replace('-', '_').lower()
    return f"{model_family}_{dataset_short}_n{num_samples}_l{max_seq_length}_{today}"

def _execute_dataset_preparation(config: dict):
    """Execute dataset preparation."""
    console.print(f"\n🔄 Starting dataset preparation...")
    
    # Lazy import and select preparer
    if config.get("training_type") == "grpo":
        # Import GRPO preparer
        try:
            import importlib.util
            import pathlib
            prep_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "prepare_dataset" / "prepare_grpo.py"
            spec = importlib.util.spec_from_file_location("_grpo_prep", prep_path)
            if spec is None or spec.loader is None:
                _fail("Could not locate prepare_grpo.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            preparer = getattr(module, "GRPODatasetPreparer")()
        except Exception as e:
            _fail(f"Failed to import GRPO dataset preparer: {e}")
    else:
        # Regular SFT preparers
        DatasetPrepConfig, BaseDatasetPreparer, QwenDatasetPreparer, GemmaDatasetPreparer = _lazy_import_dataset_prep()
        
        model_family = _get_model_family(config["tok_name"])
        if model_family == "qwen":
            preparer = QwenDatasetPreparer()
        elif model_family == "gemma":
            preparer = GemmaDatasetPreparer()
        else:
            console.print("⚠️  Could not detect model family, defaulting to Qwen", style="yellow")
            preparer = QwenDatasetPreparer()
    
    console.print("📝 [bold]Processing Dataset[/bold]")
    console.print("-" * 50)
    
    output_dir = preparer.run_with_config(config)
    
    console.print(f"\n🎉 [bold green]Dataset preparation completed![/bold green]")
    console.print(f"📁 Dataset saved to: [bold]{output_dir}[/bold]")
    
    # Show next steps
    training_method = config.get("training_type", "sft")
    console.print(f"\n📝 [bold]Next Steps:[/bold]")
    console.print(f"1. Train with dataset: [cyan]os train {output_dir} --method {training_method}[/cyan]")
    console.print(f"2. Inspect dataset: [cyan]os info {output_dir}[/cyan]")
    
    return output_dir

@app.command("prepare-data")
def prepare_data(
    # Model (required unless generating example DPO dataset without data sources)
    model: Annotated[str, typer.Option("--model", "-m", help="🤖 HuggingFace model identifier (REQUIRED for SFT and for DPO training; can be placeholder for example DPO generation)")],
    # Method
    method: Annotated[str, typer.Option("--method", "-M", help="🎯 Dataset type: sft | dpo | grpo", rich_help_panel="Mode")] = "sft",
    
    # Data source (one is required)
    input_file: Annotated[Optional[str], typer.Argument(help="📄 Local JSON/JSONL file, or use --dataset for HuggingFace")] = None,
    dataset: Annotated[Optional[str], typer.Option("--dataset", "-d", help="📊 HuggingFace dataset name")] = None,
    
    # Chat template settings
    chat_template: Annotated[Optional[str], typer.Option("--chat-template", "-t", help="💬 Chat template (see list-templates)")] = None,
    target_only: Annotated[bool, typer.Option("--target-only/--full-conversation", help="🎯 Train only on assistant responses")] = False,
    instruction_part: Annotated[Optional[str], typer.Option(help="👤 Manual instruction part (for target-only)")] = None,
    response_part: Annotated[Optional[str], typer.Option(help="🤖 Manual response part (for target-only)")] = None,
    
    # Dataset settings
    split: Annotated[str, typer.Option(help="📂 Dataset split")] = "train",
    samples: Annotated[int, typer.Option("--samples", "-n", help="🔢 Number of samples (-1 for all)")] = 1000,
    max_seq_length: Annotated[int, typer.Option("--max-seq-length", help="📏 Max sequence length (tokens)")] = 4096,
    
    # Processing settings
    workers: Annotated[int, typer.Option("--workers", "-w", help="⚡ Number of workers")] = 4,
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="📁 Output directory")] = None,
    
    # Utility
    debug: Annotated[int, typer.Option(help="🐛 Debug samples to preview")] = 0,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="👀 Show config without processing")] = False,
    force: Annotated[bool, typer.Option("--force", "-f", help="💪 Overwrite existing output")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="✅ Skip confirmation")] = False,
):
    """
    📝 Prepare dataset for fine-tuning
    
    Process conversational datasets into OpenSloth training format.
    
    **Examples:**
    
    • **Quick start with template:**
      [cyan]os prepare-data my_data.json --model unsloth/Qwen2.5-7B-Instruct --chat-template chatml[/cyan]
    
    • **HuggingFace dataset:**
      [cyan]os prepare-data --dataset mlabonne/FineTome-100k --model unsloth/Qwen2.5-7B-Instruct --chat-template chatml[/cyan]
    
    • **Target-only training:**
      [cyan]os prepare-data my_data.json --model MODEL --target-only --chat-template chatml[/cyan]
    
    **💡 Tips:**
    • Use `os list-templates` to see available chat templates
    • Use `--debug 5` to preview data processing
    • Output auto-named if not specified
    """
    
    try:
        method_l = method.lower()
        if method_l not in {"sft", "dpo", "grpo"}:
            _fail("--method must be one of: sft, dpo, grpo")

        if method_l == "dpo" and (chat_template or target_only or instruction_part or response_part):
            console.print("⚠️  Ignoring chat template / target-only flags for DPO datasets", style="yellow")

        if method_l == "grpo" and (chat_template or target_only or instruction_part or response_part):
            console.print("⚠️  Ignoring chat template / target-only flags for GRPO datasets", style="yellow")

        # Shortcut: generate example DPO dataset
        if method_l == "dpo" and dataset is None and input_file is None:
            try:
                from prepare_dataset.prepare_dpo_dataset import prepare_dpo_dataset_for_opensloth  # type: ignore
            except ImportError:
                # Fallback absolute path import if package not installed
                import importlib.util, pathlib
                prep_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "prepare_dataset" / "prepare_dpo_dataset.py"
                spec = importlib.util.spec_from_file_location("_dpo_prep", prep_path)
                if spec is None or spec.loader is None:
                    _fail("Could not locate prepare_dpo_dataset.py for DPO example generation")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                prepare_dpo_dataset_for_opensloth = getattr(module, "prepare_dpo_dataset_for_opensloth")
            out_dir = output or "data/dpo_example"
            if os.path.exists(out_dir) and not force:
                _fail(f"Output directory exists: {out_dir}. Use --force to overwrite.")
            prepare_dpo_dataset_for_opensloth(out_dir)
            console.print(f"\n🚀 Next: os train {out_dir} --method dpo --model {model}")
            return

        # SFT requires a data source
        if method_l in ["sft", "grpo"] and not (input_file or dataset):
            _fail("Must specify either input file or --dataset for SFT/GRPO")
        if input_file and dataset:
            _fail("Cannot specify both input file and --dataset")

        actual_dataset = input_file or dataset
        config = {
            "tok_name": model,
            "dataset_name": actual_dataset,
            "split": split,
            "num_samples": samples,
            "num_proc": workers,
            "max_seq_length": max_seq_length,
            "debug": debug,
            "train_on_target_only": bool(target_only) if method_l == "sft" else False,
            "training_type": method_l,  # Add training type for preparer selection
        }
        if input_file:
            config["input_file"] = input_file

        if method_l == "sft" and chat_template:
            if chat_template not in CHAT_TEMPLATES:
                _fail(f"Unknown chat template: {chat_template}. Available: {', '.join(CHAT_TEMPLATES)}")
            tpl = CHAT_TEMPLATES[chat_template]
            config.update(
                {
                    "chat_template": chat_template,
                    "train_on_target_only": tpl["train_on_target_only"],
                    "instruction_part": tpl["instruction_part"],
                    "response_part": tpl["response_part"],
                }
            )
            console.print(f"📋 Applied chat template: [bold]{chat_template}[/bold]")

        else:
            if instruction_part:
                config["instruction_part"] = instruction_part
            if response_part:
                config["response_part"] = response_part

        if method_l == "sft" and target_only:
            if not (config.get("instruction_part") and config.get("response_part")):
                if not _apply_fallback_template(config, model):
                    _fail("Target-only requires --chat-template or both --instruction-part & --response-part")

        if not output:
            if method_l == "dpo":
                base = (actual_dataset or "dpo").split("/")[-1]
                output = f"data/dpo_{base}_n{samples if samples>0 else 'all'}"
            elif method_l == "grpo":
                base = (actual_dataset or "grpo").split("/")[-1]
                output = f"data/grpo_{base}_n{samples if samples>0 else 'all'}"
            else:
                output = f"data/{_generate_dataset_name(model, actual_dataset, samples, max_seq_length)}"
        if os.path.exists(output) and not force:
            _fail(f"Output directory exists: {output}. Use --force to overwrite")
        config["output_dir"] = output

        _print_header("🎯 [bold]Dataset Preparation Configuration[/bold]")
        summary = [
            ("🧪 Method:", method_l.upper()),
            ("🤖 Model:", model),
            ("📊 Dataset:", actual_dataset),
            ("📂 Split:", split),
            ("🔢 Samples:", str(samples) if samples > 0 else "All"),
            ("📏 Max Length:", f"{max_seq_length} tokens"),
            ("⚡ Workers:", str(workers)),
            ("📁 Output:", output),
            ("🎯 Target Only:", "✅" if config.get("train_on_target_only") else "❌"),
        ]
        if method_l == "sft" and config.get("chat_template"):
            summary.append(("💬 Chat Template:", config["chat_template"]))
        _print_kv(summary)

        if dry_run:
            console.print("\n👀 [bold yellow]Dry run - configuration shown above[/bold yellow]")
            console.print("\n📋 Full configuration:")
            console.print(json.dumps(config, indent=2))
            return

        if not yes and not typer.confirm("\n🚀 Start processing dataset?"):
            console.print("❌ Cancelled")
            raise typer.Exit(0)

        if method_l == "dpo":
            try:
                from prepare_dataset.prepare_dpo_dataset import (
                    convert_existing_dataset_to_dpo,
                    prepare_dpo_dataset_for_opensloth,
                )  # type: ignore
            except ImportError:
                import importlib.util, pathlib
                prep_path = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "prepare_dataset" / "prepare_dpo_dataset.py"
                spec = importlib.util.spec_from_file_location("_dpo_prep", prep_path)
                if spec is None or spec.loader is None:
                    _fail("Could not import DPO dataset preparer")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore
                convert_existing_dataset_to_dpo = getattr(module, "convert_existing_dataset_to_dpo")
                prepare_dpo_dataset_for_opensloth = getattr(module, "prepare_dpo_dataset_for_opensloth")
            if input_file:
                convert_existing_dataset_to_dpo(input_file, output)
            else:
                if dataset:
                    _fail("Direct HF -> DPO conversion not supported yet; supply JSON preference file")
                prepare_dpo_dataset_for_opensloth(output)
            console.print(f"\n🚀 Next: os train {output} --method dpo --model {model}")
        else:
            _execute_dataset_preparation(config)
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n⏹️  Interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n❌ [bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)
@app.command("list-templates")
def list_templates():
    """📋 List available chat templates"""
    table = Table(title="Available Chat Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Instruction Part", style="yellow")
    table.add_column("Response Part", style="magenta")
    
    for name, info in CHAT_TEMPLATES.items():
        # Escape Rich markup characters
        instruction_preview = info["instruction_part"][:20].replace("[", r"\[").replace("]", r"\]") + "..."
        response_preview = info["response_part"][:20].replace("[", r"\[").replace("]", r"\]") + "..."
        
        table.add_row(
            name, 
            info["description"],
            instruction_preview,
            response_preview
        )
    
    console.print(table)
    console.print("\n💡 Use [bold]--chat-template [name][/bold] with prepare-data")

# ============================================================================
# TRAINING COMMANDS  
# ============================================================================

# Training presets
TRAINING_PRESETS = {
    "quick": {
        "description": "Quick test run (50 steps)",
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
    "small": {
        "description": "Small models < 3B params",
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
    "large": {
        "description": "Large models > 7B params",
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
    "memory-efficient": {
        "description": "Lowest memory usage",
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

def _get_available_gpus() -> List[int]:
    """Get available GPU indices."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return [0]

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    model_short = model_short.replace('-', '_')
    dataset_name = Path(dataset_path).name
    return f"outputs/{model_short}_{dataset_name}_{timestamp}"

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

def _apply_training_defaults(config: dict, dataset_path: str) -> dict:
    """Apply training defaults, inheriting from dataset config."""
    dataset_config = _load_dataset_config(dataset_path)
    
    # Default configuration
    default_config = {
        "opensloth_config": {
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
            }
        },
        "training_args": {
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
    }
    
    # Inherit from dataset configuration
    if dataset_config:
        console.print("📋 [bold]Inheriting from dataset configuration...[/bold]")
        
        # Inherit model
        if (not config.get("opensloth_config", {}).get("fast_model_args", {}).get("model_name") and 
            dataset_config.get("tok_name")):
            console.print(f"  🤖 Model: {dataset_config['tok_name']}")
            default_config["opensloth_config"]["fast_model_args"]["model_name"] = dataset_config["tok_name"]
        
        # Inherit max_seq_length
        if dataset_config.get("max_seq_length"):
            dataset_max_length = dataset_config["max_seq_length"]
            if not config.get("opensloth_config", {}).get("fast_model_args", {}).get("max_seq_length"):
                console.print(f"  📏 Max Length: {dataset_max_length}")
                default_config["opensloth_config"]["fast_model_args"]["max_seq_length"] = dataset_max_length
    
    # Merge defaults with user config
    result = _merge_configs(default_config, config)
    
    # Remove LoRA args if full finetuning
    if result["opensloth_config"]["fast_model_args"].get("full_finetuning"):
        result["opensloth_config"]["lora_args"] = None
    
    return result

def _validate_training_config(config: dict, dataset_path: str):
    """Validate training configuration."""
    opensloth_config = config.get("opensloth_config", {})
    fast_model_args = opensloth_config.get("fast_model_args", {})
    
    # Check required fields
    if not fast_model_args.get("model_name"):
        _fail("Model name is required")
    
    # Check dataset exists
    if not os.path.exists(dataset_path):
        _fail(f"Dataset directory does not exist: {dataset_path}")
    
    # Check model/dataset compatibility
    dataset_config = _load_dataset_config(dataset_path)
    if dataset_config:
        current_model = fast_model_args["model_name"]
        dataset_model = dataset_config.get("tok_name", "")
        
        current_family = _get_model_family(current_model)
        dataset_family = _get_model_family(dataset_model)
        
        if current_family != dataset_family and dataset_family != "unknown":
            console.print(f"⚠️  [yellow]Model family mismatch: dataset={dataset_family}, training={current_family}[/yellow]")
        
        # Check sequence length
        training_max_length = fast_model_args.get("max_seq_length", 4096)
        dataset_max_length = dataset_config.get("max_seq_length", 4096)
        
        if training_max_length < dataset_max_length:
            _fail(f"Training max_seq_length ({training_max_length}) < dataset max_seq_length ({dataset_max_length})")

def _print_training_config_summary(config: dict, dataset_path: str):
    """Print training configuration summary."""
    opensloth_config = config["opensloth_config"]
    training_args = config["training_args"]
    fast_model_args = opensloth_config["fast_model_args"]
    lora_args = opensloth_config.get("lora_args")
    
    _print_header("🎯 [bold]Training Configuration[/bold]")
    
    items = [
        ("🤖 Model:", fast_model_args['model_name']),
        ("📊 Dataset:", dataset_path),
        ("💾 Output:", training_args['output_dir']),
        ("🔧 GPUs:", f"{opensloth_config['devices']} ({len(opensloth_config['devices'])} GPU{'s' if len(opensloth_config['devices']) > 1 else ''})"),
        ("📏 Max Length:", str(fast_model_args['max_seq_length'])),
        ("🔢 Quantization:", "4-bit" if fast_model_args.get('load_in_4bit') else "8-bit" if fast_model_args.get('load_in_8bit') else "none"),
    ]
    
    # Training type
    if fast_model_args.get('full_finetuning'):
        items.append(("🎯 Training Type:", "Full Fine-tuning"))
    else:
        items.append(("🎯 Training Type:", f"LoRA (r={lora_args.get('r', 8)}, α={lora_args.get('lora_alpha', 16)})"))
    
    # Training parameters
    batch_size = training_args['per_device_train_batch_size']
    accumulation = training_args['gradient_accumulation_steps']
    effective_batch = batch_size * accumulation * len(opensloth_config['devices'])
    items.extend([
        ("📦 Batch Size:", f"{batch_size} per GPU (effective: {effective_batch})"),
        ("📚 Learning Rate:", str(training_args['learning_rate'])),
        ("🔄 Epochs:", str(training_args.get('num_train_epochs', 'not set'))),
        ("🔥 Optimizer:", training_args['optim']),
        ("📊 Logging:", training_args['report_to']),
    ])
    
    _print_kv(items)

@app.command("train")
def train_model(
    # Required
    dataset: Annotated[str, typer.Argument(help="📊 Path to processed dataset directory")],
    method: Annotated[str, typer.Option("--method", "-M", help="🎯 Training method: sft | dpo | grpo")] = "sft",
    
    # Model (auto-detect from dataset if not specified)
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="🤖 Model (auto-detect from dataset if not specified)")] = None,
    
    # Output
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="💾 Output directory")] = None,
    
    # Hardware
    gpus: Annotated[str, typer.Option("--gpus", "-g", help="🔧 GPU indices (e.g. '0' or '0,1,2,3')")] = "0",
    
    # Training parameters
    epochs: Annotated[Optional[int], typer.Option("--epochs", help="🔄 Number of epochs")] = None,
    max_steps: Annotated[Optional[int], typer.Option("--max-steps", help="⏰ Max training steps")] = None,
    batch_size: Annotated[Optional[int], typer.Option("--batch-size", help="📦 Batch size per GPU")] = None,
    learning_rate: Annotated[Optional[float], typer.Option("--lr", help="📚 Learning rate")] = None,
    
    # Model settings
    max_seq_length: Annotated[Optional[int], typer.Option("--max-seq-length", help="📏 Max sequence length")] = None,
    load_4bit: Annotated[bool, typer.Option("--4bit/--no-4bit", help="🔢 4-bit quantization")] = True,
    full_finetune: Annotated[bool, typer.Option("--full-finetune", help="🎯 Full parameter training")] = False,
    
    # LoRA settings
    lora_r: Annotated[Optional[int], typer.Option("--lora-r", help="🎯 LoRA rank")] = None,
    lora_alpha: Annotated[Optional[int], typer.Option("--lora-alpha", help="🎯 LoRA alpha")] = None,
    
    # GRPO-specific settings
    grpo_task_type: Annotated[Optional[str], typer.Option("--grpo-task-type", help="🎯 GRPO task type: math | code | general | reasoning")] = None,
    grpo_group_size: Annotated[Optional[int], typer.Option("--grpo-group-size", help="👥 Number of responses per prompt")] = None,
    grpo_temperature: Annotated[Optional[float], typer.Option("--grpo-temp", help="🌡️ GRPO sampling temperature")] = None,
    grpo_max_new_tokens: Annotated[Optional[int], typer.Option("--grpo-max-tokens", help="📝 Max new tokens for GRPO generation")] = None,
    grpo_reward_functions: Annotated[Optional[str], typer.Option("--grpo-rewards", help="🏆 Comma-separated reward function names")] = None,
    
    # Preset and utility
    preset: Annotated[Optional[str], typer.Option("--preset", help="⚙️ Use preset configuration")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="👀 Show config without training")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="✅ Skip confirmations")] = False,
    # Tmux options
    use_tmux: Annotated[bool, typer.Option("--use-tmux/--no-tmux", help="🖥️ Launch multi-GPU training in a tmux session (one window per GPU)")] = False,
    tmux_session: Annotated[Optional[str], typer.Option("--tmux-session", help="📛 tmux session name (default: os_train_<timestamp>)")] = None,
    tmux_auto_kill: Annotated[bool, typer.Option("--tmux-auto-kill/--no-tmux-auto-kill", help="☠️ Auto-kill existing session if present")] = False,
):
    """
    🚀 Train a model with pre-processed dataset
    
    **Examples:**
    
    • **Basic training:**
      [cyan]os train data/my_dataset[/cyan]
    
    • **Multi-GPU:**
      [cyan]os train data/my_dataset --gpus 0,1,2,3[/cyan]
    
    • **Quick test:**
      [cyan]os train data/my_dataset --preset quick[/cyan]
    
    • **Custom settings:**
      [cyan]os train data/my_dataset --model unsloth/Qwen2.5-7B-Instruct --epochs 5 --batch-size 2[/cyan]
    
    **💡 Tips:**
    • Model auto-detected from dataset if not specified
    • Use `os list-presets` to see available presets
    • Use `--dry-run` to preview configuration
    """
    
    try:
        method_l = method.lower()
        if method_l not in {"sft", "dpo", "grpo"}:
            _fail("--method must be one of: sft, dpo, grpo")

        # Dataset path resolution / interactive select
        if not os.path.exists(dataset):
            console.print(f"❌ Dataset not found: {dataset}")
            ds_list = _list_cached_datasets()
            if not ds_list:
                console.print("💡 No datasets found. Use 'os prepare-data' first")
                raise typer.Exit(1)
            console.print("\n📂 Available datasets:")
            for idx, ds in enumerate(ds_list, 1):
                console.print(f"  {idx}. {ds}")
            choice = typer.prompt("\n🔢 Select dataset number", type=int)
            if not (1 <= choice <= len(ds_list)):
                _fail("Invalid selection")
            dataset = ds_list[choice - 1]
            console.print(f"✅ Selected: [bold]{dataset}[/bold]")

        # Base config
        config: dict = {"opensloth_config": {"training_type": method_l}, "training_args": {}}

        # Preset merge
        if preset:
            if preset not in TRAINING_PRESETS:
                _fail(f"Unknown preset: {preset}. Available: {', '.join(TRAINING_PRESETS)}")
            console.print(f"📋 Applying preset: [bold]{preset}[/bold]")
            config = _merge_configs(config, TRAINING_PRESETS[preset]["config"])

        # CLI overrides
        cli_cfg = {"opensloth_config": {}, "training_args": {}}
        # GPUs
        try:
            cli_cfg["opensloth_config"]["devices"] = [int(d.strip()) for d in gpus.split(',')]
        except ValueError:
            _fail(f"Invalid GPU list: {gpus}")
        # Model args
        fm_args = {}
        if model:
            fm_args["model_name"] = model
        if max_seq_length:
            fm_args["max_seq_length"] = max_seq_length
        fm_args["load_in_4bit"] = load_4bit
        fm_args["full_finetuning"] = full_finetune
        cli_cfg["opensloth_config"]["fast_model_args"] = fm_args
        # LoRA
        if not full_finetune and (lora_r or lora_alpha):
            lora_cfg = {}
            if lora_r:
                lora_cfg["r"] = lora_r
            if lora_alpha:
                lora_cfg["lora_alpha"] = lora_alpha
            cli_cfg["opensloth_config"]["lora_args"] = lora_cfg
        # Training args
        tr_args = {}
        if epochs:
            tr_args["num_train_epochs"] = epochs
        if max_steps:
            tr_args["max_steps"] = max_steps
        if batch_size:
            tr_args["per_device_train_batch_size"] = batch_size
        if learning_rate:
            tr_args["learning_rate"] = learning_rate
        cli_cfg["training_args"] = tr_args

        config = _merge_configs(config, cli_cfg)
        config = _apply_training_defaults(config, dataset)
        config["opensloth_config"]["data_cache_path"] = dataset

        # Method specific tweaks
        if method_l == "dpo":
            config["opensloth_config"]["sequence_packing"] = False
            if learning_rate is None and "learning_rate" not in config["training_args"]:
                config["training_args"]["learning_rate"] = 5e-6
            config["opensloth_config"]["dpo_args"] = {"beta": 0.1, "max_length": 1024, "max_prompt_length": 512}
        elif method_l == "grpo":
            # Disable sequence packing; adjust default LR if not set
            config["opensloth_config"]["sequence_packing"] = False
            if learning_rate is None and "learning_rate" not in config["training_args"]:
                config["training_args"]["learning_rate"] = 5e-6
            
            # Configure GRPO args with enhanced options
            grpo_config = {
                "group_size": grpo_group_size if grpo_group_size is not None else 4,
                "max_new_tokens": grpo_max_new_tokens if grpo_max_new_tokens is not None else 256,
                "temperature": grpo_temperature if grpo_temperature is not None else 1.0,
                "top_p": 0.9,
                "top_k": None,
                "min_p": 0.1,
                "kl_coef": 0.05,
                
                # Task-specific configuration
                "task_type": grpo_task_type if grpo_task_type is not None else "general",
                "reward_functions": grpo_reward_functions.split(",") if grpo_reward_functions else [],
                "use_custom_chat_template": True,
                
                # Prompt processing
                "max_prompt_length": 512,
                "prompt_length_percentile": 0.9,
                
                # Training control
                "eval_interval": 50,
                "save_interval": 100,
                "print_sample_every": 10,
                
                # vLLM settings
                "stop_sequences": [],
                "include_stop_str_in_output": True,
            }
            
            config["opensloth_config"]["grpo_args"] = grpo_config

        final_model = config["opensloth_config"]["fast_model_args"]["model_name"]
        if not output:
            output = _generate_output_dir(final_model, dataset)
        config["training_args"]["output_dir"] = output

        _validate_training_config(config, dataset)
        _print_training_config_summary(config, dataset)
        console.print(f"🧪 Method: [bold]{method_l.upper()}[/bold]")
        if use_tmux and len(config["opensloth_config"]["devices"]) == 1:
            console.print("⚠️  --use-tmux requested but only one GPU detected; ignoring.", style="yellow")
            use_tmux = False

        if dry_run:
            console.print("\n👀 [bold yellow]Dry run - configuration shown above[/bold yellow]")
            console.print("\n📋 Full configuration:")
            console.print(json.dumps(config, indent=2))
            return

        if not yes and not typer.confirm("\n🚀 Start training?"):
            console.print("❌ Cancelled")
            raise typer.Exit(0)

        from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
        opensloth_config = OpenSlothConfig(**config["opensloth_config"])
        training_args = TrainingArguments(**config["training_args"])

        # Tmux launch path (multi-GPU only)
        if use_tmux and len(opensloth_config.devices) > 1:
            console.print("\n🖥️  Launching tmux multi-GPU session...")
            # Verify tmux installed
            if os.system("command -v tmux > /dev/null 2>&1") != 0:
                _fail("tmux not found in PATH. Install tmux or omit --use-tmux.")
            session = tmux_session or f"os_train_{datetime.now().strftime('%H%M%S')}"
            # Write ephemeral config file
            tmp_dir = tempfile.mkdtemp(prefix="opensloth_tmux_")
            cfg_path = Path(tmp_dir) / "tmux_config.py"
            cfg_code = textwrap.dedent(
                f"""# Auto-generated by OpenSloth CLI --use-tmux
import json
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

_opensloth_cfg_json = r'''{json.dumps(opensloth_config.model_dump())}'''
_training_args_json = r'''{json.dumps(training_args.model_dump())}'''

opensloth_config = OpenSlothConfig(**json.loads(_opensloth_cfg_json))
training_config = TrainingArguments(**json.loads(_training_args_json))
"""
            )
            cfg_path.write_text(cfg_code)
            # Construct command
            cmd = [
                sys.executable,
                "-m",
                "opensloth.scripts.opensloth_sft_trainer",
                str(cfg_path),
                "--tmux",
                session,
            ] + (["-y"] if tmux_auto_kill else [])
            console.print(f"🧾 Temp config: {cfg_path}")
            console.print(f"🧪 Session: {session}")
            console.print(f"▶️  Command: {' '.join(cmd)}")
            # Set env to signal tmux usage (script also checks args)
            env = os.environ.copy()
            env["USE_TMUX"] = "1"
            subprocess.run(cmd, check=True, env=env)
            console.print("\n✅ tmux session launched. Use: tmux attach -t " + session)
            return

        # Direct (non-tmux) path
        console.print("\n🔄 Starting training (direct multi-process)...")
        from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs
        setup_envs(opensloth_config, training_args)
        run_mp_training(opensloth_config.devices, opensloth_config, training_args)
        console.print(f"\n🎉 [bold green]Training completed![/bold green]")
        console.print(f"📁 Model saved to: [bold]{output}[/bold]")
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n⏹️  Interrupted by user")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n❌ [bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)

@app.command("list-presets")
def list_presets():
    """📋 List available training presets"""
    table = Table(title="Training Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")
    
    for name, info in TRAINING_PRESETS.items():
        table.add_row(name, info["description"])
    
    console.print(table)
    console.print("\n💡 Use [bold]--preset [name][/bold] with train command")

# ============================================================================
# DEBUG COMMANDS
# ============================================================================

@app.command("debug")
def debug_dataset(
    dataset: Annotated[str, typer.Argument(help="📊 Path to processed dataset directory")],
    samples_per_page: Annotated[int, typer.Option("--samples", "-n", help="🔢 Samples per page")] = 3,
    max_to_print_per_content: Annotated[int, typer.Option("--max-to-print-per-content", "-k", help="Max tokens per example (first 25 + last 25 if exceeded)")] = 50,
):
    """🐛 Simple dataset preview (terminal only)

    - Color-coded output (green = context, yellow = trainable) via `print_dataloader_example_short`
    - Press ENTER for next page, 'q' to quit
    """
    try:
        if not os.path.exists(dataset):
            _fail(f"Dataset not found: {dataset}")

        # Load dataset configuration to get tokenizer/model name
        ds_cfg = _load_dataset_config(dataset)
        tok_name = ds_cfg.get("tok_name")
        if not tok_name:
            _fail("Could not determine tokenizer name (tok_name) from dataset_config.json")

        console.print(f"\n🔧 Loading tokenizer: [bold]{tok_name}[/bold]")
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=True, trust_remote_code=True)
        except Exception as e:
            _fail(f"Failed to load tokenizer '{tok_name}': {e}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load dataset
        console.print(f"📂 Opening dataset at: [bold]{dataset}[/bold]")
        try:
            from datasets import load_from_disk
            ds_or_dict = load_from_disk(dataset)
        except Exception as e:
            _fail(f"Failed to load dataset from disk: {e}")

        # Pick split if DatasetDict
        if hasattr(ds_or_dict, "column_names"):
            ds = ds_or_dict
            split_name = "(single split)"
        else:
            available_splits = list(ds_or_dict.keys())
            if not available_splits:
                _fail("No splits found in dataset")
            split = "train" if "train" in ds_or_dict else available_splits[0]
            ds = ds_or_dict[split]
            split_name = split

        total = len(ds)
        console.print(f"\n🔎 Split: [bold]{split_name}[/bold] — Samples: [bold]{total}[/bold]")
        console.print("💡 Press [bold]ENTER[/bold] for next page, type [bold]q[/bold] then Enter to quit.\n")

        # Helper to normalize columns to 1-D Long tensors
        def _to_1d_long(x):
            import torch
            if torch.is_tensor(x):
                return x.to(dtype=torch.long).view(-1)
            try:
                import numpy as np
                if isinstance(x, np.ndarray):
                    return torch.as_tensor(x, dtype=torch.long).view(-1)
            except Exception:
                pass
            if isinstance(x, (list, tuple)):
                # flatten one level if nested
                if x and isinstance(x[0], (list, tuple)):
                    x = [t for sub in x for t in sub]
                return torch.tensor(x, dtype=torch.long).view(-1)
            return torch.tensor(x, dtype=torch.long).view(-1)

        idx = 0
        shown = 0
        while idx < total:
            end = min(idx + samples_per_page, total)
            for i in range(idx, end):
                ex = ds[i]
                input_ids = ex.get("input_ids")
                if input_ids is None:
                    _fail("Dataset does not contain 'input_ids' column")
                label_ids = ex.get("labels", ex.get("label_ids"))
                if label_ids is None:
                    _fail("Dataset does not contain 'labels' or 'label_ids' column")

                # Normalize to tensors to avoid list-vs-int ops
                input_ids = _to_1d_long(input_ids)
                label_ids = _to_1d_long(label_ids)

                console.print(f"\n📝 [bold cyan]Sample {i}/{total - 1}[/bold cyan]")
                try:
                    print_dataloader_example_short(
                        tokenizer=tokenizer,
                        input_ids=input_ids,
                        label_ids=label_ids,
                        max_to_print_per_content=max_to_print_per_content,
                    )
                except Exception as e:
                    console.print(f"⚠️  Skipping sample {i} due to error: {e}", style="yellow")
                    continue
                shown += 1

            idx = end
            try:
                user_in = input("[Enter]=next  |  q=quit > ").strip().lower()
            except EOFError:
                break
            if user_in == "q":
                break

        console.print(f"\n✅ Done. Shown samples: [bold]{shown}[/bold]")

    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print("\n⏹️  Interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n❌ [bold red]Debug error: {e}[/bold red]")
        raise typer.Exit(1)



# ============================================================================
# INFO COMMANDS
# ============================================================================

@app.command("list-datasets")
def list_datasets():
    """📂 List available processed datasets"""
    datasets = _list_cached_datasets()
    
    if not datasets:
        console.print("📂 No processed datasets found in data/ directory")
        console.print("💡 Use [bold]os prepare-data[/bold] to prepare a dataset first")
        return
    
    table = Table(title="Available Datasets")
    table.add_column("Dataset Path", style="cyan")
    table.add_column("Modified", style="yellow")
    table.add_column("Samples", style="green")
    
    for path in datasets:
        mod_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
        
        # Try to get sample count
        dataset_config = _load_dataset_config(path)
        samples = str(dataset_config.get("num_samples", "Unknown")) if dataset_config else "Unknown"
        
        table.add_row(path, mod_time, samples)
    
    console.print(table)
    console.print(f"\n💡 Use [bold]os train <dataset_path>[/bold] to train")

@app.command("info")
def dataset_info(
    dataset: Annotated[str, typer.Argument(help="Path to dataset directory")]
):
    """📖 Show dataset information"""
    if not os.path.exists(dataset):
        _fail(f"Dataset not found: {dataset}")
    
    dataset_config = _load_dataset_config(dataset)
    
    if not dataset_config:
        _fail(f"No configuration found in {dataset}")
    
    _print_header(f"📊 [bold]Dataset Information: {dataset}[/bold]")
    
    items = [
        ("🤖 Model/Tokenizer:", dataset_config.get("tok_name", "Unknown")),
        ("📊 Original Dataset:", dataset_config.get("dataset_name", "Unknown")),
        ("🔢 Samples:", str(dataset_config.get("num_samples", "Unknown"))),
        ("📏 Max Seq Length:", str(dataset_config.get("max_seq_length", "Unknown"))),
        ("💬 Chat Template:", dataset_config.get("chat_template", "None")),
        ("🎯 Target Only:", "✅" if dataset_config.get("train_on_target_only") else "❌"),
        ("📂 Split:", dataset_config.get("split", "Unknown")),
    ]
    
    _print_kv(items)
    
    # Show compatible training command
    console.print(f"\n🚀 [bold]Training Command:[/bold]")
    console.print(f"[cyan]os train {dataset}[/cyan]")
    
    # Model family compatibility
    model_family = _get_model_family(dataset_config.get("tok_name", ""))
    console.print(f"\n💡 [bold]Compatibility:[/bold]")
    console.print(f"• Dataset prepared for [cyan]{model_family}[/cyan] model family")
    console.print(f"• Training models must support max_seq_length >= {dataset_config.get('max_seq_length', 'unknown')}")


@app.command("list-rewards")
def list_reward_functions():
    """
    🏆 List available GRPO reward functions
    
    Shows all available reward functions for GRPO training with descriptions and task presets.
    """
    try:
        from opensloth.grpo_rewards import list_reward_functions, create_reward_preset
        
        _print_header("🏆 [bold]Available GRPO Reward Functions[/bold]")
        
        # List all reward functions
        reward_funcs = list_reward_functions()
        console.print("📋 [bold]Individual Reward Functions:[/bold]")
        
        # Import to get descriptions
        from opensloth.grpo_rewards import _REWARD_REGISTRY
        
        for name in reward_funcs:
            func = _REWARD_REGISTRY[name]
            console.print(f"  • [cyan]{name}[/cyan]: {func.description}")
        
        # Show task presets
        console.print("\n🎯 [bold]Task Presets:[/bold]")
        
        task_types = ["math", "code", "general", "reasoning"]
        for task_type in task_types:
            try:
                preset_funcs = create_reward_preset(task_type)
                console.print(f"  • [cyan]{task_type}[/cyan]: {', '.join(preset_funcs)}")
            except ValueError:
                pass
        
        console.print("\n💡 [bold]Usage Examples:[/bold]")
        console.print("• [cyan]os train data/math_dataset --method grpo --grpo-task-type math[/cyan]")
        console.print("• [cyan]os train data/code_dataset --method grpo --grpo-task-type code[/cyan]") 
        console.print("• [cyan]os train data/dataset --method grpo --grpo-rewards math_format,length_penalty[/cyan]")
        
    except ImportError as e:
        _fail(f"Could not import GRPO reward functions: {e}")


# ============================================================================
# MAIN
# ============================================================================

@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    version: Annotated[bool, typer.Option("--version", help="Show version")] = False
):
    """🦥 OpenSloth - Unified CLI for LLM fine-tuning"""
    if version:
        console.print("OpenSloth CLI v1.0.0")
        raise typer.Exit()
    
    if ctx.invoked_subcommand is None:
        console.print("🦥 [bold]OpenSloth - Unified CLI[/bold]")
        console.print("\n📝 [bold]Workflow:[/bold]")
        console.print("1. [cyan]os prepare-data[/cyan] - Prepare dataset for training")
        console.print("2. [cyan]os train[/cyan] - Train model with prepared dataset")
        console.print("\n💡 [bold]Quick Examples:[/bold]")
        console.print("• [cyan]os prepare-data my_data.json --model unsloth/Qwen2.5-7B-Instruct --chat-template chatml[/cyan]")
        console.print("• [cyan]os train data/my_dataset --gpus 0,1,2,3[/cyan]")
        console.print("\n📋 [bold]Commands:[/bold]")
        console.print("• [cyan]os prepare-data[/cyan] - Prepare dataset")
        console.print("• [cyan]os train[/cyan] - Train model") 
        console.print("• [cyan]os debug[/cyan] - Debug dataset with color-coded tokens")
        console.print("• [cyan]os list-datasets[/cyan] - List available datasets")
        console.print("• [cyan]os list-templates[/cyan] - List chat templates")
        console.print("• [cyan]os list-presets[/cyan] - List training presets")
        console.print("• [cyan]os list-rewards[/cyan] - List GRPO reward functions")
        console.print("• [cyan]os info[/cyan] - Show dataset information")
        console.print("\nUse [cyan]os [command] --help[/cyan] for detailed help")

def main():
    """Entry point for the unified CLI."""
    try:
        app()
    except typer.Exit as e:
        import sys
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        import sys
        sys.exit(130)

if __name__ == "__main__":
    main()
