# opensloth/cli/autogen.py

import argparse
from typing import Any


def add_sft_args(parser: argparse.ArgumentParser) -> None:
    """Add SFT-specific arguments to an ArgumentParser."""
    
    # Core options
    parser.add_argument('dataset', help='Path to the processed and sharded dataset')
    
    # Model options
    parser.add_argument('--model', required=True, help='The model name or path to use')
    parser.add_argument('--max-seq-length', type=int, default=4096, help='Maximum sequence length for the model')
    parser.add_argument('--load-in-4bit', action='store_true', default=True, help='Load the model in 4-bit (QLoRA)')
    parser.add_argument('--no-load-in-4bit', dest='load_in_4bit', action='store_false', help='Disable 4-bit loading')
    parser.add_argument('--full-finetuning', action='store_true', help='Perform full fine-tuning instead of LoRA')
    
    # LoRA options
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--use-rslora', action='store_true', help='Use RSLoRA (Rank-Stabilized LoRA)')
    
    # Training options
    parser.add_argument('--output', default='saves/loras/', help='Output directory for checkpoints and logs')
    parser.add_argument('--bs', type=int, default=2, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=2e-4, help='The initial learning rate')
    parser.add_argument('--grad-accum', type=int, default=4, help='Number of gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=3, help='Total number of training epochs')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup steps for LR scheduler')
    
    # System options
    parser.add_argument('--dry-run', action='store_true', help='Print configuration and exit without running')
    parser.add_argument('--tmux', action='store_true', help='Use tmux for multi-GPU training')
    parser.add_argument('--data-cache-path', default='./cache', help='Path to cache directory for datasets')


def parse_sft_config(args: argparse.Namespace) -> dict[str, Any]:
    """Convert parsed args to config dictionaries."""
    
    # Build OpenSlothConfig
    opensloth_config = {
        'data_cache_path': args.data_cache_path,
        'devices': [0],  # Default to single GPU, can be extended later
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
    
    return {
        'opensloth_config': opensloth_config,
        'training_args': training_args,
    }


__all__ = ["add_sft_args", "parse_sft_config"]
