#!/usr/bin/env python3
"""Unified SFT pipeline: prepare dataset then launch training.

Usage (singl    # Model l    args = p.parse_args()
    
    # Set default quantization if neither is specified
    if not args.load_in_4bit and not args.load_in_8bit:
        args.load_in_4bit = True  # Default to 4-bit
    
    # Validation
    if args.load_in_4bit and args.load_in_8bit:
        p.error(\"Cannot use both --load-in-4bit and --load-in-8bit simultaneously\")
    
    if args.full_finetuning and any([args.lora_r != 8, args.lora_alpha is not None, 
                                    args.lora_dropout != 0.0, args.use_rslora]):
        print(\"Warning: LoRA arguments will be ignored when using --full-finetuning\")
    
    return argsiguration
    p.add_argument(\"--full-finetuning\", action=\"store_true\", help=\"Perform full fine-tuning instead of LoRA\")
    p.add_argument(\"--load-in-4bit\", action=\"store_true\", help=\"Load model in 4-bit (QLoRA)\")
    p.add_argument(\"--load-in-8bit\", action=\"store_true\", help=\"Load model in 8-bit\")ONL file with LoRA):
  opensloth-sft-run \
    --model /path/to/model \
    --input data/x1.jsonl \
    --output-dir outputs/demo_run \
    --devices 0,1 \
    --samples 1000 --max-seq-length 4096

Usage (full fine-tuning with 8-bit quantization):
  opensloth-sft-run \
    --model /path/to/model \
    --input data/x1.jsonl \
    --output-dir outputs/demo_run \
    --devices 0,1 \
    --full-finetuning \
    --load-in-8bit \
    --samples 1000 --max-seq-length 4096

Usage (LoRA with no quantization):
  opensloth-sft-run \
    --model /path/to/model \
    --input data/x1.jsonl \
    --output-dir outputs/demo_run \
    --devices 0,1 \
    --lora-r 16 --lora-alpha 32 \
    --samples 1000 --max-seq-length 4096
    
Note: Defaults to 4-bit quantization unless --load-in-8bit is specified or both are disabled.

This will create:
  outputs/demo_run/dataset/  (processed shards + metadata)
  outputs/demo_run/train/    (training outputs: checkpoints, logs)

Dataset prep arguments largely mirror DatasetPrepConfig fields.
Training arguments can be minimally overridden via CLI; otherwise a template is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime

# Import unsloth FIRST (critical for OpenSloth) when training stage begins; we delay until needed.
# Heavy imports are lazily loaded inside functions to make --help fast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified dataset preparation + SFT training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core model/tokenizer
    p.add_argument("--model", required=True, help="Model / tokenizer path or HF id (FastModelArgs.model_name & tokenizer_name)")

    # Data source
    p.add_argument("--input", required=True, help="HF dataset repo or local JSON/JSONL file path")
    p.add_argument("--split", default="train", help="Dataset split (HF datasets)")
    p.add_argument("--samples", type=int, default=-1, help="Limit number of samples (-1 = all)")
    p.add_argument("--workers", type=int, default=8, help="Number of processes for map/tokenization")

    # Output root
    p.add_argument("--output-dir", required=True, help="Root output directory (will contain dataset/ & train/)")

    # GPU / sharding
    p.add_argument("--devices", default="0", help="Comma separated GPU device indices")

    # Formatting / labeling
    p.add_argument("--chat-template", default="chatml", help="Chat template name")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--train-on-target-only", action="store_true", help="Mask non-assistant tokens")
    p.add_argument("--instruction-part", default="<|im_start|>user\n")
    p.add_argument("--response-part", default="<|im_start|>assistant\n")

    # Training overrides (lightweight)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--bs", type=int, default=2, help="Per device batch size")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--lr-scheduler-type", default="linear")

    # LoRA configuration
    p.add_argument("--lora-r", type=int, default=8, help="LoRA rank (r)")
    p.add_argument("--lora-alpha", type=int, help="LoRA alpha. If not provided, defaults to 2x LoRA rank")
    p.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    p.add_argument("--lora-bias", default="none", help="LoRA bias type")
    p.add_argument("--lora-random-state", type=int, default=3407, help="LoRA random state")
    p.add_argument("--lora-target-modules", nargs="*", default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                   help="LoRA target modules (space-separated list)")
    p.add_argument("--use-rslora", action="store_true", help="Use RSLoRA (Rank-Stabilized LoRA)")
    p.add_argument("--finetune-vision-layers", action="store_true", help="Finetune vision layers")
    p.add_argument("--finetune-language-layers", action="store_true", default=True, help="Finetune language layers")
    p.add_argument("--finetune-attention-modules", action="store_true", default=True, help="Finetune attention modules")
    p.add_argument("--finetune-mlp-modules", action="store_true", default=True, help="Finetune MLP modules")

    # Model loading configuration
    p.add_argument("--full-finetuning", action="store_true", help="Perform full fine-tuning instead of LoRA")
    p.add_argument("--load-in-4bit", action="store_true", default=True, help="Load model in 4-bit (QLoRA)")
    p.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit")

    # Misc
    p.add_argument("--hf-token", help="HF token for gated resources")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tmux", action="store_true", help="Use tmux for multi-GPU training")

    args = p.parse_args()
    
    # Validation
    if args.load_in_4bit and args.load_in_8bit:
        p.error("Cannot use both --load-in-4bit and --load-in-8bit simultaneously")
    
    if args.full_finetuning and any([args.lora_r != 8, args.lora_alpha is not None, 
                                    args.lora_dropout != 0.0, args.use_rslora]):
        print("Warning: LoRA arguments will be ignored when using --full-finetuning")
    
    return args


def create_dataset_hash(args: argparse.Namespace, num_gpus: int) -> str:
    """Create a hash from dataset preparation arguments to use as dataset identifier."""
    # Include all arguments that affect dataset preparation
    hash_data = {
        'model': args.model,
        'input': args.input,
        'split': args.split,
        'samples': args.samples,
        'chat_template': args.chat_template,
        'max_seq_length': args.max_seq_length,
        'train_on_target_only': args.train_on_target_only,
        'instruction_part': args.instruction_part,
        'response_part': args.response_part,
        'num_gpus': num_gpus,
    }
    
    # Create a stable string representation
    hash_string = json.dumps(hash_data, sort_keys=True)
    
    # Create SHA256 hash and take first 12 characters for readability
    return hashlib.sha256(hash_string.encode()).hexdigest()[:12]


def build_prep_config(args: argparse.Namespace, dataset_dir: Path, num_gpus: int):
    # Lazy import to make --help fast
    from opensloth.opensloth_config import DatasetPrepConfig
    
    return DatasetPrepConfig(
        tokenizer_name=args.model,
        chat_template=args.chat_template,
        dataset_name=args.input,
        input_file=args.input if os.path.exists(args.input) else None,
        split=args.split,
        num_samples=args.samples,
        num_proc=args.workers,
        gpus=num_gpus,
        output_dir=str(dataset_dir),
        train_on_target_only=args.train_on_target_only,
        instruction_part=args.instruction_part,
        response_part=args.response_part,
        max_seq_length=args.max_seq_length,
        hf_token=args.hf_token,
    )



def build_training_configs(model_name: str, max_seq_length: int, num_gpus: int, train_output_dir: Path, args: argparse.Namespace):
    # Lazy import to make --help fast
    from opensloth.scripts.prepare_dataset import get_training_config_template
    
    template = get_training_config_template(
        model_name, 
        num_gpus=num_gpus, 
        max_seq_length=max_seq_length, 
        lora_r=args.lora_r, 
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_bias=args.lora_bias,
        lora_random_state=args.lora_random_state,
        lora_target_modules=args.lora_target_modules,
        use_rslora=args.use_rslora,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        full_finetuning=args.full_finetuning,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )
    # Apply overrides
    targs = template["training_args"]
    targs["num_train_epochs"] = args.epochs
    targs["per_device_train_batch_size"] = args.bs
    targs["gradient_accumulation_steps"] = args.grad_accum
    targs["learning_rate"] = args.lr
    targs["warmup_steps"] = args.warmup
    targs["lr_scheduler_type"] = args.lr_scheduler_type
    targs["seed"] = args.seed
    targs["output_dir"] = str(train_output_dir)
    
    # Handle full finetuning - disable LoRA
    if args.full_finetuning:
        template["opensloth_config"]["lora_args"] = None
    
    return template


def main():  # noqa: C901
    args = parse_args()

    # Lazy imports after argument parsing to make --help fast
    from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
    from opensloth.scripts.prepare_dataset import prepare_dataset
    from opensloth.api import run_training

    devices = [int(x) for x in args.devices.split(',') if x.strip()]
    num_gpus = len(devices)
    
    # Create hash-based dataset identifier
    dataset_hash = create_dataset_hash(args, num_gpus)
    
    root_out = Path(args.output_dir).absolute()
    dataset_dir = root_out / "dataset" / f"dataset_{dataset_hash}"
    train_dir = root_out / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset preparation - check if hash-based dataset exists
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        print(f"[Dataset] Found existing dataset with hash {dataset_hash}, reusing: {dataset_dir}")
    else:
        print(f"[Dataset] Creating new dataset with hash {dataset_hash}: {dataset_dir}")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        prep_cfg = build_prep_config(args, dataset_dir, num_gpus)
        prepare_dataset(prep_cfg)

    # 2. Build training configuration
    training_cfg_dict = build_training_configs(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        num_gpus=num_gpus,
        train_output_dir=train_dir,
        args=args,
    )

    # Inject dataset path
    training_cfg_dict['opensloth_config']['data_cache_path'] = str(dataset_dir)
    training_cfg_dict['opensloth_config']['devices'] = devices

    # Create Pydantic objects
    opensloth_cfg = OpenSlothConfig(**training_cfg_dict['opensloth_config'])
    train_args = TrainingArguments(**training_cfg_dict['training_args'])

    summary = {
        "dataset_dir": str(dataset_dir),
        "train_output_dir": str(train_dir),
        "devices": devices,
        "global_batch_size": len(devices) * train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps,
    }
    print("[Config] Unified run summary:\n" + json.dumps(summary, indent=2))

    # 3. Training
    print("[Train] Starting SFT training ...")
    run_training(opensloth_cfg, train_args, use_tmux=args.tmux)
    print(f"[Done] Training complete. Outputs in {train_dir}")


if __name__ == "__main__":
    main()
