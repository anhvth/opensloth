#!/usr/bin/env python3
"""
Minimal, read-only training script for OpenSloth SFT.
Simply loads configuration from dataset directory and starts training.
"""
import argparse
import json
import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import opensloth
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from opensloth.api import run_training
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Minimal OpenSloth SFT Training - loads all config from dataset directory",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core arguments
    parser.add_argument('dataset', help='Path to the processed and sharded dataset')
    parser.add_argument('--devices', type=str, default='0', help='Comma-separated GPU device IDs')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print configuration and exit without running')
    parser.add_argument('--tmux', action='store_true', help='Use tmux for multi-GPU training')
    
    args = parser.parse_args()
    
    # Validate dataset path
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path '{args.dataset}' does not exist.")
        sys.exit(1)
    
    # Check for required config files
    training_config_path = dataset_path / "training_config.json"
    dataset_config_path = dataset_path / "dataset_config.json"
    
    if not training_config_path.exists():
        print(f"❌ Error: training_config.json not found in {args.dataset}")
        print("💡 Make sure to run prepare_qwen_dataset.py first to generate the training config.")
        sys.exit(1)
    
    if not dataset_config_path.exists():
        print(f"❌ Error: dataset_config.json not found in {args.dataset}")
        print("💡 Make sure to run prepare_qwen_dataset.py first to generate the dataset config.")
        sys.exit(1)
    
    # Load configurations
    try:
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        
        with open(dataset_config_path, 'r') as f:
            dataset_config = json.load(f)
            
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)
    
    # Validate device count matches dataset shards
    # devices = args.devices.split(',')
    # num_shards = dataset_config.get('num_shards', 1)
    
    # if len(devices) != num_shards:
    #     print(f"❌ Error: Number of devices ({len(devices)}) must match number of dataset shards ({num_shards})")
    #     print(f"💡 Use --devices with {num_shards} GPU(s), e.g., --devices 0,1")
    #     sys.exit(1)
    
    # Override device configuration with user input
    # training_config['opensloth_config']['devices'] = [int(d.strip()) for d in devices]
    # training_config['opensloth_config']['data_cache_path'] = str(dataset_path.absolute())
    
    # Generate output directory based on dataset name and timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = training_config['opensloth_config']['fast_model_args']['model_name'].split("/")[-1].replace("-", "_").lower()
    dataset_short = dataset_path.name.replace("-", "_").lower()
    output_dir = f"outputs/sft_{model_short}_{dataset_short}_{timestamp}"
    print(f"ℹ️ Output directory set to: {output_dir}")
    training_config['training_args']['output_dir'] = output_dir
    
    # Create Pydantic objects
    try:
        opensloth_cfg = OpenSlothConfig(**training_config['opensloth_config'])
        train_args = TrainingArguments(**training_config['training_args'])
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"❌ Error creating configuration objects: {e}")
        sys.exit(1)
    
    # Handle dry run
    if args.dry_run:
        print("🔍 DRY RUN: SFT training configuration:")
        summary = {
            "dataset": str(dataset_path),
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        print(json.dumps(summary, indent=2))
        return
    
    # Print training summary
    print("🚀 Starting SFT training...")
    print(f"📁 Dataset: {dataset_path}")
    print(f"🤖 Model: {opensloth_cfg.fast_model_args.model_name}")
    print(f"💾 Output: {train_args.output_dir}")
    print(f"🖥️  Devices: {opensloth_cfg.devices}")
    
    global_batch_size = len(opensloth_cfg.devices) * train_args.per_device_train_batch_size * train_args.gradient_accumulation_steps
    print(f"📊 Global batch size: {global_batch_size}")
    
    # Run training
    try:
        run_training(opensloth_cfg, train_args, use_tmux=args.tmux)
        print(f"✅ SFT Training complete. Model saved to: {train_args.output_dir}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()