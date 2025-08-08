#!/usr/bin/env python3
"""
OpenSloth Training CLI

A unified command-line interface for fine-tuning models with OpenSloth.
Supports LoRA and full fine-tuning with multi-GPU training.
"""

import argparse
import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent.parent


def list_datasets(base_dir: str = "data") -> List[str]:
    """List available processed datasets."""
    datasets = []
    if not os.path.isdir(base_dir):
        return datasets
    
    for root, _, files in os.walk(base_dir):
        if 'dataset_info.json' in files or any(f.endswith('.arrow') for f in files):
            datasets.append(root)
    
    # Sort by modification time (newest first)
    datasets.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return datasets


def list_training_presets() -> List[str]:
    """List available training preset configurations."""
    preset_dir = get_project_root() / "prepare_dataset" / "presets" / "train"
    
    if not preset_dir.exists():
        return []
    
    presets = []
    for file in preset_dir.glob("*.json"):
        try:
            with open(file) as f:
                data = json.load(f)
            description = data.get("description", "")
            name = file.stem.replace("_", " ").title()
            presets.append(f"{name}: {description}" if description else name)
        except Exception:
            presets.append(file.stem)
    
    return presets


def save_training_preset(name: str, config: Dict[str, Any]) -> None:
    """Save configuration as a training preset."""
    preset_dir = get_project_root() / "prepare_dataset" / "presets" / "train"
    preset_dir.mkdir(parents=True, exist_ok=True)
    
    filename = name.lower().replace(" ", "_") + ".json"
    filepath = preset_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Training preset saved as '{filename}'")


def load_training_preset(name: str) -> Optional[Dict[str, Any]]:
    """Load a training preset configuration."""
    preset_dir = get_project_root() / "prepare_dataset" / "presets" / "train"
    
    # Try exact filename first
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


def detect_model_family(model_name: str) -> str:
    """Auto-detect model family for default configurations."""
    model_lower = model_name.lower()
    
    if any(x in model_lower for x in ["qwen", "qwen2", "qwen3"]):
        return "qwen"
    elif any(x in model_lower for x in ["gemma"]):
        return "gemma"
    elif any(x in model_lower for x in ["llama"]):
        return "llama"
    elif any(x in model_lower for x in ["mistral", "mixtral"]):
        return "mistral"
    else:
        return "qwen"  # Default


def get_default_lora_targets(model_family: str) -> List[str]:
    """Get default LoRA target modules for model family."""
    # Most modern models use this pattern
    return [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]


def run_training(config: Dict[str, Any]) -> None:
    """Run training using the OpenSloth training script."""
    project_root = get_project_root()
    
    # Write config to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f, indent=2)
        config_file = f.name
    
    try:
        # Use the existing training script runner
        runner_script = project_root / "prepare_dataset" / "run_train_job.py"
        
        cmd = [sys.executable, str(runner_script)]
        
        # Run with config from stdin
        result = subprocess.run(
            cmd,
            input=json.dumps(config),
            cwd=str(project_root),
            text=True,
            capture_output=False
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with exit code {result.returncode}")
    
    finally:
        # Clean up temp file
        if os.path.exists(config_file):
            os.unlink(config_file)


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="OpenSloth Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick LoRA fine-tuning
  opensloth-train --dataset data/qwen_finetome_n1000_0808 --model unsloth/Qwen2.5-0.5B-Instruct --steps 100

  # Multi-GPU training 
  opensloth-train --dataset data/my_dataset --model unsloth/Qwen2.5-7B-Instruct --devices 0,1 --batch-size 2

  # Full fine-tuning (requires more VRAM)
  opensloth-train --dataset data/my_dataset --model unsloth/Qwen2.5-0.5B-Instruct --full-finetuning

  # Use training preset
  opensloth-train --preset qwen_lora_fast --dataset data/my_dataset

  # List available datasets
  opensloth-train --list-datasets

  # Save configuration as preset
  opensloth-train --dataset data/my_dataset --model unsloth/Qwen2.5-0.5B-Instruct --save-preset my_config

GPU Memory Tips:
  - 0.5B models: 6-8GB VRAM (LoRA), 12-16GB (full)
  - 1B models: 8-12GB VRAM (LoRA), 20-24GB (full)  
  - 3B models: 12-16GB VRAM (LoRA), 40GB+ (full)
  - 7B models: 16-24GB VRAM (LoRA), 80GB+ (full)
  
  Use --load-in-4bit to reduce memory usage
  Reduce --batch-size if you get OOM errors
        """
    )
    
    # Dataset and model
    parser.add_argument(
        "--dataset", "--data-cache-path", dest="dataset_path",
        type=str, required=False,
        help="Path to processed dataset directory"
    )
    
    parser.add_argument(
        "--model", "--model-name", dest="model_name", 
        type=str, required=False,
        help="HuggingFace model identifier or local path"
    )
    
    # Hardware configuration
    parser.add_argument(
        "--devices", "--gpus", dest="devices",
        type=str, default="0",
        help="GPU indices to use (e.g., '0' or '0,1,2,3')"
    )
    
    parser.add_argument(
        "--max-seq-length", dest="max_seq_length",
        type=int, default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    
    parser.add_argument(
        "--load-in-4bit", dest="load_in_4bit",
        action="store_true", default=True,
        help="Use 4-bit quantization to save VRAM (default: enabled)"
    )
    
    parser.add_argument(
        "--no-4bit", dest="load_in_4bit",
        action="store_false",
        help="Disable 4-bit quantization"
    )
    
    parser.add_argument(
        "--load-in-8bit", dest="load_in_8bit",
        action="store_true", default=False,
        help="Use 8-bit quantization instead of 4-bit"
    )
    
    # Training type
    parser.add_argument(
        "--full-finetuning", dest="full_finetuning",
        action="store_true", default=False,
        help="Full parameter fine-tuning (requires much more VRAM)"
    )
    
    parser.add_argument(
        "--lora", dest="full_finetuning",
        action="store_false",
        help="Use LoRA fine-tuning (default, memory efficient)"
    )
    
    # LoRA configuration
    parser.add_argument(
        "--lora-r", dest="lora_r",
        type=int, default=16,
        help="LoRA rank (higher = more parameters, default: 16)"
    )
    
    parser.add_argument(
        "--lora-alpha", dest="lora_alpha",
        type=int, default=32,
        help="LoRA alpha parameter (default: 32)"
    )
    
    parser.add_argument(
        "--lora-dropout", dest="lora_dropout",
        type=float, default=0.0,
        help="LoRA dropout rate (default: 0.0)"
    )
    
    parser.add_argument(
        "--lora-targets", dest="lora_targets",
        type=str, 
        help="Comma-separated LoRA target modules (auto-detected if not specified)"
    )
    
    parser.add_argument(
        "--use-rslora", dest="use_rslora",
        action="store_true", default=False,
        help="Use Rank-stabilized LoRA (experimental)"
    )
    
    # Training parameters
    parser.add_argument(
        "--output", "--output-dir", dest="output_dir",
        type=str,
        help="Output directory for saved model (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--batch-size", "--per-device-train-batch-size", dest="batch_size",
        type=int, default=1,
        help="Batch size per GPU (default: 1)"
    )
    
    parser.add_argument(
        "--gradient-accumulation-steps", dest="gradient_accumulation_steps",
        type=int, default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr", dest="learning_rate",
        type=float, default=1e-4,
        help="Learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--steps", "--max-steps", dest="max_steps",
        type=int,
        help="Maximum training steps (overrides epochs)"
    )
    
    parser.add_argument(
        "--epochs", "--num-train-epochs", dest="num_train_epochs",
        type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    
    parser.add_argument(
        "--warmup-steps", dest="warmup_steps",
        type=int, default=10,
        help="Number of warmup steps (default: 10)"
    )
    
    parser.add_argument(
        "--lr-scheduler", dest="lr_scheduler_type",
        type=str, default="linear",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler (default: linear)"
    )
    
    parser.add_argument(
        "--weight-decay", dest="weight_decay",
        type=float, default=0.01,
        help="Weight decay (default: 0.01)"
    )
    
    parser.add_argument(
        "--optimizer", "--optim", dest="optim",
        type=str, default="adamw_8bit",
        choices=["adamw_8bit", "adamw_torch", "sgd"],
        help="Optimizer (default: adamw_8bit)"
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging-steps", dest="logging_steps",
        type=int, default=1,
        help="Log every N steps (default: 1)"
    )
    
    parser.add_argument(
        "--save-steps", dest="save_steps",
        type=int,
        help="Save checkpoint every N steps"
    )
    
    parser.add_argument(
        "--save-total-limit", dest="save_total_limit",
        type=int, default=1,
        help="Maximum number of checkpoints to keep (default: 1)"
    )
    
    parser.add_argument(
        "--seed", dest="seed",
        type=int, default=3407,
        help="Random seed (default: 3407)"
    )
    
    parser.add_argument(
        "--report-to", dest="report_to",
        type=str, default="none",
        choices=["none", "tensorboard", "wandb"],
        help="Logging backend (default: none)"
    )
    
    # Sequence packing
    parser.add_argument(
        "--sequence-packing", dest="sequence_packing",
        action="store_true", default=True,
        help="Enable sequence packing for efficiency (default: enabled)"
    )
    
    parser.add_argument(
        "--no-sequence-packing", dest="sequence_packing",
        action="store_false",
        help="Disable sequence packing"
    )
    
    # Preset management
    parser.add_argument(
        "--preset", dest="preset_name",
        type=str,
        help="Load configuration from a training preset"
    )
    
    parser.add_argument(
        "--save-preset", dest="save_preset_name", 
        type=str,
        help="Save current configuration as a training preset"
    )
    
    parser.add_argument(
        "--list-presets", dest="list_presets",
        action="store_true",
        help="List all available training presets"
    )
    
    # Dataset management
    parser.add_argument(
        "--list-datasets", dest="list_datasets",
        action="store_true",
        help="List all available processed datasets"
    )
    
    # Configuration file
    parser.add_argument(
        "--config", dest="config_file",
        type=str,
        help="Load configuration from JSON file"
    )
    
    parser.add_argument(
        "--save-config", dest="save_config_file",
        type=str,
        help="Save current configuration to JSON file"
    )
    
    return parser


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries recursively."""
    result = base.copy()
    for key, value in override.items():
        if value is not None:
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_configs(result[key], value)
            else:
                result[key] = value
    return result


def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Convert argparse namespace to OpenSloth configuration format."""
    
    # Parse devices list
    if args.devices:
        devices = [int(x.strip()) for x in args.devices.split(',')]
    else:
        devices = [0]
    
    # Parse LoRA targets
    lora_targets = None
    if args.lora_targets:
        lora_targets = [x.strip() for x in args.lora_targets.split(',')]
    elif args.model_name and not args.full_finetuning:
        # Auto-detect based on model family
        family = detect_model_family(args.model_name)
        lora_targets = get_default_lora_targets(family)
    
    # Generate output directory if not specified
    output_dir = args.output_dir
    if not output_dir and args.dataset_path and args.model_name:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model_name.split('/')[-1].lower()
        dataset_short = os.path.basename(args.dataset_path.rstrip('/'))
        output_dir = f"outputs/exps/{model_short}_{dataset_short}_{timestamp}"
    
    # Build configuration structure matching OpenSloth format
    config = {
        "opensloth_config": {
            "data_cache_path": args.dataset_path,
            "devices": devices,
            "sequence_packing": args.sequence_packing,
            "fast_model_args": {
                "model_name": args.model_name,
                "max_seq_length": args.max_seq_length,
                "load_in_4bit": args.load_in_4bit and not args.load_in_8bit,
                "load_in_8bit": args.load_in_8bit,
                "full_finetuning": args.full_finetuning,
            }
        },
        "training_args": {
            "output_dir": output_dir,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "logging_steps": args.logging_steps,
            "warmup_steps": args.warmup_steps,
            "lr_scheduler_type": args.lr_scheduler_type,
            "save_total_limit": args.save_total_limit,
            "weight_decay": args.weight_decay,
            "optim": args.optim,
            "seed": args.seed,
            "report_to": args.report_to,
        }
    }
    
    # Add LoRA configuration if not full fine-tuning
    if not args.full_finetuning:
        config["opensloth_config"]["lora_args"] = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "use_rslora": args.use_rslora,
        }
        
        if lora_targets:
            config["opensloth_config"]["lora_args"]["target_modules"] = lora_targets
    
    # Add training steps or epochs
    if args.max_steps:
        config["training_args"]["max_steps"] = args.max_steps
        # Remove num_train_epochs when max_steps is specified
        if "num_train_epochs" in config["training_args"]:
            del config["training_args"]["num_train_epochs"]
    else:
        config["training_args"]["num_train_epochs"] = args.num_train_epochs
    
    # Add save_steps if specified
    if args.save_steps:
        config["training_args"]["save_steps"] = args.save_steps
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the training configuration."""
    opensloth_config = config.get("opensloth_config", {})
    training_args = config.get("training_args", {})
    
    # Check required fields
    if not opensloth_config.get("data_cache_path"):
        raise ValueError("Dataset path (data_cache_path) is required")
    
    if not opensloth_config.get("fast_model_args", {}).get("model_name"):
        raise ValueError("Model name is required")
    
    # Check dataset exists
    dataset_path = opensloth_config["data_cache_path"]
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Check for required dataset files
    dataset_info_file = os.path.join(dataset_path, "dataset_info.json")
    arrow_files = list(Path(dataset_path).glob("*.arrow"))
    
    if not os.path.exists(dataset_info_file) and not arrow_files:
        raise ValueError(f"Invalid dataset directory (missing dataset_info.json or .arrow files): {dataset_path}")
    
    # Check GPU indices are valid
    devices = opensloth_config.get("devices", [0])
    if not isinstance(devices, list) or not all(isinstance(d, int) and d >= 0 for d in devices):
        raise ValueError("Invalid device list - must be list of non-negative integers")


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Handle utility commands first
        if args.list_presets:
            presets = list_training_presets()
            if presets:
                print("Available training presets:")
                for preset in presets:
                    print(f"  - {preset}")
            else:
                print("No training presets found.")
            return
        
        if args.list_datasets:
            datasets = list_datasets()
            if datasets:
                print("Available processed datasets:")
                for dataset in datasets[:10]:  # Show most recent 10
                    mtime = os.path.getmtime(dataset)
                    import time
                    time_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
                    print(f"  - {dataset} (modified: {time_str})")
                if len(datasets) > 10:
                    print(f"  ... and {len(datasets) - 10} more")
            else:
                print("No processed datasets found.")
            return
        
        # Start with empty config
        config = {}
        
        # Load from config file if specified
        if args.config_file:
            if not os.path.exists(args.config_file):
                raise FileNotFoundError(f"Config file not found: {args.config_file}")
            
            with open(args.config_file) as f:
                config = json.load(f)
            print(f"📁 Loaded config from {args.config_file}")
        
        # Load from preset if specified
        if args.preset_name:
            preset_config = load_training_preset(args.preset_name)
            if preset_config is None:
                raise ValueError(f"Training preset not found: {args.preset_name}")
            
            config = merge_configs(config, preset_config)
            print(f"🎯 Applied training preset '{args.preset_name}'")
        
        # Generate configuration from CLI arguments
        cli_config = args_to_config(args)
        config = merge_configs(config, cli_config)
        
        # Validate configuration
        validate_config(config)
        
        # Save config if requested
        if args.save_config_file:
            with open(args.save_config_file, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved config to {args.save_config_file}")
        
        # Save as preset if requested
        if args.save_preset_name:
            save_training_preset(args.save_preset_name, config)
        
        # Print training summary
        opensloth_config = config["opensloth_config"]
        training_args = config["training_args"]
        model_args = opensloth_config["fast_model_args"]
        
        print(f"\n🚀 Starting OpenSloth training...")
        print(f"🤖 Model: {model_args['model_name']}")
        print(f"📊 Dataset: {opensloth_config['data_cache_path']}")
        print(f"🔧 Type: {'Full Fine-tuning' if model_args.get('full_finetuning') else 'LoRA'}")
        print(f"⚡ GPUs: {opensloth_config['devices']}")
        print(f"📏 Max length: {model_args['max_seq_length']}")
        print(f"🎯 Batch size: {training_args['per_device_train_batch_size']} per GPU")
        print(f"🔄 Gradient accumulation: {training_args['gradient_accumulation_steps']}")
        print(f"📈 Learning rate: {training_args['learning_rate']}")
        
        if "max_steps" in training_args:
            print(f"⏱️  Steps: {training_args['max_steps']}")
        else:
            print(f"🔄 Epochs: {training_args['num_train_epochs']}")
        
        print(f"💾 Output: {training_args['output_dir']}")
        print()
        
        # Run training
        run_training(config)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved to: {training_args['output_dir']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
