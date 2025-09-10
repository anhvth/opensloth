#!/usr/bin/env python3
"""
Simple, direct training script for OpenSloth SFT.
A single file that handles everything needed for supervised fine-tuning.
"""
import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Add the src directory to the path so we can import opensloth
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from opensloth.api import run_training
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments


def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/sft_{model_short}_{dataset_short}_{timestamp}"


def create_config(args: argparse.Namespace) -> tuple[OpenSlothConfig, TrainingArguments]:
    """Create configuration objects from command line arguments."""
    
    # Build OpenSlothConfig
    opensloth_config = {
        'data_cache_path': args.data_cache_path,
        'devices': [0],  # Default to single GPU
        'fast_model_args': {
            'model_name': args.model,
            'max_seq_length': args.max_seq_length,
            'load_in_4bit': args.load_in_4bit,
            'full_finetuning': args.full_finetuning,
        },
        'training_type': 'sft',
        'sequence_packing': True,
        'filter_overlength_samples': True,
        'log_level': 'info',
    }
    
    # Add LoRA config if not full finetuning
    if not args.full_finetuning:
        opensloth_config['lora_args'] = {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
            'use_rslora': args.use_rslora,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        }
    
    # Build TrainingArguments
    training_args = {
        'output_dir': args.output,
        'per_device_train_batch_size': args.bs,
        'learning_rate': args.lr,
        'gradient_accumulation_steps': args.grad_accum,
        'num_train_epochs': args.epochs,
        'warmup_steps': args.warmup,
        'logging_steps': 1,
        'lr_scheduler_type': 'linear',
        'save_total_limit': 2,
        'optim': 'adamw_8bit',
        'weight_decay': 0.01,
        'save_only_model': False,
        'seed': 42,
        'report_to': 'tensorboard',
        'eval_strategy': 'no',
        'dataset_num_proc': 8,
    }
    
    # Create Pydantic objects
    opensloth_cfg = OpenSlothConfig(**opensloth_config)
    train_args = TrainingArguments(**training_args)
    
    return opensloth_cfg, train_args


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Simple OpenSloth SFT Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('dataset', help='Path to the processed and sharded dataset')
    parser.add_argument('--model', required=True, help='The model name or path to use')
    
    # Model options
    parser.add_argument('--max-seq-length', type=int, default=4096, 
                       help='Maximum sequence length for the model')
    parser.add_argument('--load-in-4bit', action='store_true', default=True, 
                       help='Load the model in 4-bit (QLoRA)')
    parser.add_argument('--no-load-in-4bit', dest='load_in_4bit', action='store_false', 
                       help='Disable 4-bit loading')
    parser.add_argument('--full-finetuning', action='store_true', 
                       help='Perform full fine-tuning instead of LoRA')
    
    # LoRA options
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--use-rslora', action='store_true', 
                       help='Use RSLoRA (Rank-Stabilized LoRA)')
    
    # Training options
    parser.add_argument('--output', default='', 
                       help='Output directory for checkpoints and logs (auto-generated if empty)')
    parser.add_argument('--bs', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-4, help='The initial learning rate')
    parser.add_argument('--grad-accum', type=int, default=4, 
                       help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup steps for LR scheduler')
    
    # System options
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print configuration and exit without running')
    parser.add_argument('--tmux', action='store_true', help='Use tmux for multi-GPU training')
    parser.add_argument('--data-cache-path', default='./cache', 
                       help='Path to cache directory for datasets')
    
    args = parser.parse_args()
    
    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"Error: Dataset path '{args.dataset}' does not exist.")
        sys.exit(1)
    
    # Create configurations
    try:
        opensloth_cfg, train_args = create_config(args)
    except Exception as e:
        print(f"Error creating configuration: {e}")
        sys.exit(1)
    
    # Auto-generate output directory if not specified
    if not args.output:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, args.dataset)
    
    # Handle dry run
    if args.dry_run:
        print("DRY RUN: SFT training configuration:")
        summary = {
            "dataset": args.dataset,
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        print(json.dumps(summary, indent=2))
        return
    
    # Run training
    print("🚀 Starting SFT training...")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {opensloth_cfg.fast_model_args.model_name}")
    print(f"Output: {train_args.output_dir}")
    
    try:
        run_training(opensloth_cfg, train_args, use_tmux=args.tmux)
        print(f"✅ SFT Training complete. Model saved to: {train_args.output_dir}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()