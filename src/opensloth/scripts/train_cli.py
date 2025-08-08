#!/usr/bin/env python3
"""
OpenSloth Training CLI

A unified command-line interface for fine-tuning large language models.
Supports multi-GPU training, LoRA, full fine-tuning, and various optimizations.
"""

import argparse
import os
import sys
import json
import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None

# Import OpenSloth configuration classes
current_dir = Path(__file__).parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments, FastModelArgs, LoraArgs
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs, train_on_single_gpu


# Default configurations for different training scenarios
TRAINING_PRESETS = {
    "quick_test": {
        "description": "Quick test run with minimal steps",
        "training_args": {
            "max_steps": 50,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "save_total_limit": 1,
            "report_to": "none",
        }
    },
    "small_model": {
        "description": "Optimized for models < 3B parameters",
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
    },
    "large_model": {
        "description": "Optimized for models > 7B parameters", 
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
    },
    "full_finetune": {
        "description": "Full parameter fine-tuning (requires lots of VRAM)",
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
    },
    "memory_efficient": {
        "description": "Lowest memory usage configuration",
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

# Common output directory patterns
OUTPUT_DIR_TEMPLATES = {
    "timestamp": "outputs/train_{timestamp}",
    "model_based": "outputs/{model_name}_{timestamp}",
    "dataset_based": "outputs/{model_name}_{dataset_name}_{timestamp}",
}


def get_available_gpus() -> List[int]:
    """Get list of available GPU indices."""
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except ImportError:
        pass
    return [0]  # Default to single GPU


def generate_output_dir(template: str, model_name: str, dataset_path: str) -> str:
    """Generate output directory from template."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract model name (remove path/org)
    model_short = model_name.split('/')[-1] if '/' in model_name else model_name
    model_short = model_short.replace('-', '_')
    
    # Extract dataset name from path
    dataset_name = Path(dataset_path).name
    
    return template.format(
        timestamp=timestamp,
        model_name=model_short,
        dataset_name=dataset_name
    )


def list_presets() -> List[str]:
    """List available training presets."""
    presets = []
    for name, info in TRAINING_PRESETS.items():
        description = info.get("description", "")
        presets.append(f"{name}: {description}" if description else name)
    return presets


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    """Load a training preset configuration."""
    return TRAINING_PRESETS.get(name)


def save_preset(name: str, config: Dict[str, Any]) -> None:
    """Save training configuration as a preset."""
    preset_dir = Path(__file__).parent.parent.parent.parent / "prepare_dataset" / "presets" / "train"
    preset_dir.mkdir(parents=True, exist_ok=True)
    
    filename = name.lower().replace(" ", "_") + ".json"
    filepath = preset_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training preset saved as '{filename}'")


def load_custom_preset(name: str) -> Optional[Dict[str, Any]]:
    """Load a custom training preset from file."""
    preset_dir = Path(__file__).parent.parent.parent.parent / "prepare_dataset" / "presets" / "train"
    
    filename = name.lower().replace(" ", "_") + ".json"
    filepath = preset_dir / filename
    
    if not filepath.exists():
        # Try searching for partial matches
        for file in preset_dir.glob("*.json"):
            if name.lower() in file.stem.lower():
                filepath = file
                break
        else:
            return None
    
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading preset '{name}': {e}")
        return None


def list_cached_datasets() -> List[str]:
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


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenSloth Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick start with processed dataset
  opensloth-train --dataset data/qwen_finetome_1000_0808 --model unsloth/Qwen2.5-0.5B-Instruct

  # Multi-GPU training
  opensloth-train --dataset data/my_dataset --model unsloth/Qwen2.5-7B-Instruct --gpus 0,1,2,3

  # Use a preset configuration  
  opensloth-train --dataset data/my_dataset --preset large_model

  # Full fine-tuning instead of LoRA
  opensloth-train --dataset data/my_dataset --model qwen --full-finetune

  # Memory-efficient training
  opensloth-train --dataset data/my_dataset --preset memory_efficient --max-length 1024

  # Custom training parameters
  opensloth-train \\
    --dataset data/qwen_finetome_10k \\
    --model unsloth/Qwen2.5-7B-Instruct \\
    --output outputs/qwen_custom \\
    --epochs 3 \\
    --batch-size 2 \\
    --accumulation-steps 8 \\
    --learning-rate 2e-4 \\
    --lora-r 16 \\
    --max-length 4096

  # Continue training from checkpoint
  opensloth-train --dataset data/my_dataset --resume outputs/qwen_custom/checkpoint-500

  # List available datasets or presets
  opensloth-train --list-datasets
  opensloth-train --list-presets

Training Presets:
  - quick_test: Fast test run with minimal steps
  - small_model: Optimized for models < 3B parameters
  - large_model: Optimized for models > 7B parameters  
  - full_finetune: Full parameter fine-tuning
  - memory_efficient: Lowest memory usage

Model Types:
  - LoRA: Parameter-efficient fine-tuning (default)
  - Full: Full parameter fine-tuning (--full-finetune)
  - Quantization: 4-bit (default), 8-bit (--load-in-8bit), or none

For more information: https://github.com/anhvth/opensloth
        """
    )
    
    # Dataset and model configuration
    parser.add_argument(
        "--dataset", "--data-cache-path", dest="dataset_path",
        type=str, required=False,
        help="Path to processed dataset directory"
    )
    
    parser.add_argument(
        "--model", "--model-name", dest="model_name", 
        type=str,
        help="HuggingFace model identifier or local path"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--gpus", "--devices", dest="devices",
        type=str,
        help="GPU indices to use (e.g., '0' or '0,1,2,3')"
    )
    
    # Output configuration
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir",
        type=str,
        help="Output directory for model checkpoints"
    )
    
    parser.add_argument(
        "--output-template", dest="output_template",
        type=str, choices=list(OUTPUT_DIR_TEMPLATES.keys()),
        default="timestamp",
        help="Template for auto-generating output directory"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", "--num-train-epochs", dest="num_train_epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max-steps", dest="max_steps",
        type=int,
        help="Maximum number of training steps (overrides epochs)"
    )
    
    parser.add_argument(
        "--batch-size", "--per-device-train-batch-size", dest="per_device_train_batch_size",
        type=int,
        help="Batch size per GPU"
    )
    
    parser.add_argument(
        "--accumulation-steps", "--gradient-accumulation-steps", dest="gradient_accumulation_steps", 
        type=int,
        help="Gradient accumulation steps"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate",
        type=float,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--warmup-steps", dest="warmup_steps",
        type=int,
        help="Number of warmup steps"
    )
    
    parser.add_argument(
        "--weight-decay", dest="weight_decay",
        type=float,
        help="Weight decay"
    )
    
    parser.add_argument(
        "--optimizer", "--optim", dest="optim",
        type=str, choices=["adamw_torch", "adamw_8bit", "sgd"],
        help="Optimizer to use"
    )
    
    parser.add_argument(
        "--scheduler", "--lr-scheduler-type", dest="lr_scheduler_type",
        type=str, choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler"
    )
    
    # Model configuration
    parser.add_argument(
        "--max-length", "--max-seq-length", dest="max_seq_length",
        type=int,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--load-in-4bit", dest="load_in_4bit",
        action="store_true",
        help="Use 4-bit quantization (default)"
    )
    
    parser.add_argument(
        "--load-in-8bit", dest="load_in_8bit", 
        action="store_true",
        help="Use 8-bit quantization instead of 4-bit"
    )
    
    parser.add_argument(
        "--no-quantization", dest="no_quantization",
        action="store_true", 
        help="Disable quantization"
    )
    
    parser.add_argument(
        "--full-finetune", dest="full_finetuning",
        action="store_true",
        help="Full parameter fine-tuning instead of LoRA"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r", dest="lora_r",
        type=int,
        help="LoRA rank (higher = more parameters)"
    )
    
    parser.add_argument(
        "--lora-alpha", dest="lora_alpha",
        type=int,
        help="LoRA alpha parameter"
    )
    
    parser.add_argument(
        "--lora-dropout", dest="lora_dropout",
        type=float,
        help="LoRA dropout rate"
    )
    
    parser.add_argument(
        "--target-modules", dest="target_modules",
        type=str,
        help="Comma-separated list of target modules for LoRA"
    )
    
    parser.add_argument(
        "--use-rslora", dest="use_rslora",
        action="store_true",
        help="Use Rank-stabilized LoRA"
    )
    
    # Optimization features
    parser.add_argument(
        "--sequence-packing", dest="sequence_packing",
        action="store_true", default=True,
        help="Enable sequence packing (default)"
    )
    
    parser.add_argument(
        "--no-sequence-packing", dest="sequence_packing",
        action="store_false",
        help="Disable sequence packing"
    )
    
    # Logging and monitoring
    parser.add_argument(
        "--logging-steps", dest="logging_steps",
        type=int,
        help="Log every N steps"
    )
    
    parser.add_argument(
        "--save-steps", dest="save_steps",
        type=int,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--save-total-limit", dest="save_total_limit",
        type=int,
        help="Maximum number of checkpoints to keep"
    )
    
    parser.add_argument(
        "--report-to", dest="report_to",
        type=str, choices=["tensorboard", "wandb", "none"],
        help="Logging backend"
    )
    
    parser.add_argument(
        "--log-level", dest="log_level",
        type=str, choices=["info", "debug"],
        help="OpenSloth logging level"
    )
    
    # Checkpointing
    parser.add_argument(
        "--resume", "--resume-from-checkpoint", dest="resume_from_checkpoint",
        type=str,
        help="Resume training from checkpoint directory"
    )
    
    parser.add_argument(
        "--pretrained-lora", dest="pretrained_lora",
        type=str,
        help="Path to pretrained LoRA for continued training"
    )
    
    # Preset management
    parser.add_argument(
        "--preset", dest="preset_name",
        type=str,
        help="Load configuration from preset"
    )
    
    parser.add_argument(
        "--save-preset", dest="save_preset_name",
        type=str,
        help="Save current configuration as preset"
    )
    
    parser.add_argument(
        "--list-presets", dest="list_presets",
        action="store_true",
        help="List available training presets"
    )
    
    # Utility commands
    parser.add_argument(
        "--list-datasets", dest="list_datasets",
        action="store_true",
        help="List available processed datasets"
    )
    
    parser.add_argument(
        "--config", dest="config_file",
        type=str,
        help="Load configuration from JSON file"
    )
    
    parser.add_argument(
        "--save-config", dest="save_config_file",
        type=str,
        help="Save configuration to JSON file"
    )
    
    parser.add_argument(
        "--dry-run", dest="dry_run",
        action="store_true",
        help="Print configuration without starting training"
    )
    
    return parser


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if value is None:
            continue
            
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert command line arguments to configuration dictionary."""
    config = {
        "opensloth_config": {},
        "training_args": {}
    }
    
    # OpenSloth config mappings
    opensloth_mappings = {
        "dataset_path": "data_cache_path",
        "devices": "devices", 
        "sequence_packing": "sequence_packing",
        "pretrained_lora": "pretrained_lora",
        "log_level": "log_level",
    }
    
    # Fast model args mappings
    fast_model_mappings = {
        "model_name": "model_name",
        "max_seq_length": "max_seq_length",
        "load_in_4bit": "load_in_4bit",
        "load_in_8bit": "load_in_8bit", 
        "full_finetuning": "full_finetuning",
    }
    
    # LoRA args mappings
    lora_mappings = {
        "lora_r": "r",
        "lora_alpha": "lora_alpha",
        "lora_dropout": "lora_dropout",
        "target_modules": "target_modules",
        "use_rslora": "use_rslora",
    }
    
    # Training args mappings
    training_mappings = {
        "output_dir": "output_dir",
        "num_train_epochs": "num_train_epochs",
        "max_steps": "max_steps",
        "per_device_train_batch_size": "per_device_train_batch_size",
        "gradient_accumulation_steps": "gradient_accumulation_steps",
        "learning_rate": "learning_rate",
        "warmup_steps": "warmup_steps",
        "weight_decay": "weight_decay",
        "optim": "optim",
        "lr_scheduler_type": "lr_scheduler_type",
        "logging_steps": "logging_steps",
        "save_steps": "save_steps",
        "save_total_limit": "save_total_limit",
        "report_to": "report_to",
        "resume_from_checkpoint": "resume_from_checkpoint",
    }
    
    # Process OpenSloth config
    for arg_name, config_key in opensloth_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config["opensloth_config"][config_key] = value
    
    # Process fast model args
    fast_model_args = {}
    for arg_name, config_key in fast_model_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            fast_model_args[config_key] = value
    
    if fast_model_args:
        config["opensloth_config"]["fast_model_args"] = fast_model_args
    
    # Process LoRA args
    lora_args = {}
    for arg_name, config_key in lora_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            if config_key == "target_modules" and isinstance(value, str):
                lora_args[config_key] = [m.strip() for m in value.split(",")]
            else:
                lora_args[config_key] = value
    
    if lora_args and not args.full_finetuning:
        config["opensloth_config"]["lora_args"] = lora_args
    
    # Process training args
    for arg_name, config_key in training_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config["training_args"][config_key] = value
    
    # Handle special cases
    
    # GPU devices parsing
    if args.devices:
        try:
            devices = [int(d.strip()) for d in args.devices.split(",")]
            config["opensloth_config"]["devices"] = devices
        except ValueError:
            raise ValueError(f"Invalid GPU device specification: {args.devices}")
    
    # Quantization logic
    if args.no_quantization:
        config["opensloth_config"].setdefault("fast_model_args", {})
        config["opensloth_config"]["fast_model_args"]["load_in_4bit"] = False
        config["opensloth_config"]["fast_model_args"]["load_in_8bit"] = False
    elif args.load_in_8bit:
        config["opensloth_config"].setdefault("fast_model_args", {})
        config["opensloth_config"]["fast_model_args"]["load_in_8bit"] = True
        config["opensloth_config"]["fast_model_args"]["load_in_4bit"] = False
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the training configuration."""
    opensloth_config = config.get("opensloth_config", {})
    training_args = config.get("training_args", {})
    
    # Required fields
    if not opensloth_config.get("data_cache_path"):
        raise ValueError("Dataset path is required (--dataset)")
    
    fast_model_args = opensloth_config.get("fast_model_args", {})
    if not fast_model_args.get("model_name"):
        raise ValueError("Model name is required (--model)")
    
    # Check dataset exists
    dataset_path = opensloth_config["data_cache_path"]
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset directory does not exist: {dataset_path}")
    
    # Validate GPU indices
    devices = opensloth_config.get("devices", [0])
    available_gpus = get_available_gpus()
    for device in devices:
        if device not in available_gpus:
            print(f"⚠️  Warning: GPU {device} may not be available")
    
    # Check for conflicting options
    if training_args.get("max_steps") and training_args.get("num_train_epochs"):
        print("⚠️  Warning: Both max_steps and num_train_epochs specified. max_steps will take precedence.")
    
    # Validate LoRA settings
    if not fast_model_args.get("full_finetuning", False):
        lora_args = opensloth_config.get("lora_args")
        if lora_args:
            r = lora_args.get("r", 8)
            alpha = lora_args.get("lora_alpha", 8) 
            if alpha < r:
                print(f"⚠️  Warning: LoRA alpha ({alpha}) is less than rank ({r}). Consider alpha >= rank.")


def apply_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
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
        "opensloth_config": merge_configs(default_opensloth, config.get("opensloth_config", {})),
        "training_args": merge_configs(default_training, config.get("training_args", {}))
    }
    
    # Remove LoRA args if full finetuning
    if result["opensloth_config"]["fast_model_args"].get("full_finetuning"):
        result["opensloth_config"]["lora_args"] = None
    
    return result


def print_config_summary(config: Dict[str, Any]) -> None:
    """Print a summary of the training configuration."""
    opensloth_config = config["opensloth_config"]
    training_args = config["training_args"]
    fast_model_args = opensloth_config["fast_model_args"]
    lora_args = opensloth_config.get("lora_args")
    
    print("\n🎯 Training Configuration Summary")
    print("=" * 50)
    
    # Model and dataset
    print(f"📊 Model: {fast_model_args['model_name']}")
    print(f"📁 Dataset: {opensloth_config['data_cache_path']}")
    print(f"💾 Output: {training_args['output_dir']}")
    
    # Hardware
    devices = opensloth_config['devices']
    print(f"🔧 GPUs: {devices} ({len(devices)} GPU{'s' if len(devices) > 1 else ''})")
    
    # Model configuration
    max_length = fast_model_args['max_seq_length']
    quant = "4-bit" if fast_model_args.get('load_in_4bit') else "8-bit" if fast_model_args.get('load_in_8bit') else "none"
    print(f"📏 Max Length: {max_length}")
    print(f"🔢 Quantization: {quant}")
    
    # Training type
    if fast_model_args.get('full_finetuning'):
        print("🎯 Training Type: Full Fine-tuning")
    else:
        print("🎯 Training Type: LoRA")
        if lora_args:
            print(f"   - Rank: {lora_args['r']}")
            print(f"   - Alpha: {lora_args['lora_alpha']}")
            print(f"   - Dropout: {lora_args['lora_dropout']}")
    
    # Training parameters
    batch_size = training_args['per_device_train_batch_size']
    accumulation = training_args['gradient_accumulation_steps']
    effective_batch = batch_size * accumulation * len(devices)
    print(f"📦 Batch Size: {batch_size} per GPU (effective: {effective_batch})")
    print(f"📚 Learning Rate: {training_args['learning_rate']}")
    
    if training_args.get('max_steps'):
        print(f"⏰ Max Steps: {training_args['max_steps']}")
    else:
        print(f"🔄 Epochs: {training_args.get('num_train_epochs', 'not set')}")
    
    print(f"🔥 Optimizer: {training_args['optim']}")
    print(f"📈 Scheduler: {training_args['lr_scheduler_type']}")
    print(f"📊 Logging: {training_args['report_to']}")
    
    # Features
    features = []
    if opensloth_config.get('sequence_packing'):
        features.append("Sequence Packing")
    if features:
        print(f"⚡ Features: {', '.join(features)}")
    
    print("=" * 50)


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle utility commands
        if args.list_presets:
            print("Available training presets:")
            for preset in list_presets():
                print(f"  - {preset}")
            
            # Also list custom presets
            preset_dir = Path(__file__).parent.parent.parent.parent / "prepare_dataset" / "presets" / "train"
            if preset_dir.exists():
                custom_presets = [f.stem for f in preset_dir.glob("*.json")]
                if custom_presets:
                    print("\nCustom presets:")
                    for preset in custom_presets:
                        print(f"  - {preset}")
            return
        
        if args.list_datasets:
            datasets = list_cached_datasets()
            if datasets:
                print("Available processed datasets:")
                for dataset in datasets:
                    print(f"  - {dataset}")
            else:
                print("No processed datasets found in 'data/' directory.")
                print("Use 'opensloth-dataset' to prepare a dataset first.")
            return
        
        # Start with empty config
        config = {"opensloth_config": {}, "training_args": {}}
        
        # Load from config file
        if args.config_file:
            if not os.path.exists(args.config_file):
                raise FileNotFoundError(f"Config file not found: {args.config_file}")
            
            with open(args.config_file) as f:
                file_config = json.load(f)
            config = merge_configs(config, file_config)
            print(f"📁 Loaded config from {args.config_file}")
        
        # Load preset
        if args.preset_name:
            preset_config = load_preset(args.preset_name)
            if preset_config is None:
                # Try custom presets
                preset_config = load_custom_preset(args.preset_name)
            
            if preset_config is None:
                raise ValueError(f"Preset not found: {args.preset_name}")
            
            config = merge_configs(config, preset_config)
            print(f"🎯 Applied preset '{args.preset_name}'")
        
        # Apply command line arguments
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)
        
        # Apply defaults
        config = apply_defaults(config)
        
        # Auto-generate output directory if needed
        if not config["training_args"].get("output_dir"):
            template = OUTPUT_DIR_TEMPLATES[args.output_template]
            model_name = config["opensloth_config"]["fast_model_args"]["model_name"]
            dataset_path = config["opensloth_config"]["data_cache_path"]
            config["training_args"]["output_dir"] = generate_output_dir(template, model_name, dataset_path)
        
        # Validate configuration
        validate_config(config)
        
        # Save config if requested
        if args.save_config_file:
            with open(args.save_config_file, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved config to {args.save_config_file}")
        
        # Save preset if requested
        if args.save_preset_name:
            save_preset(args.save_preset_name, config)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Dry run - just print config
        if args.dry_run:
            print("\n📄 Full Configuration:")
            print(json.dumps(config, indent=2))
            return
        
        # Convert to OpenSloth objects
        opensloth_config = OpenSlothConfig(**config["opensloth_config"])
        training_args = TrainingArguments(**config["training_args"])
        
        # Setup and run training
        print(f"\n🚀 Starting training...")
        
        setup_envs(opensloth_config, training_args)
        
        if len(opensloth_config.devices) > 1:
            print(f"🔥 Multi-GPU training on {len(opensloth_config.devices)} GPUs")
            run_mp_training(opensloth_config.devices, opensloth_config, training_args)
        else:
            print(f"🔥 Single-GPU training on GPU {opensloth_config.devices[0]}")
            train_on_single_gpu(opensloth_config.devices[0], opensloth_config, training_args)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved to: {training_args.output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
