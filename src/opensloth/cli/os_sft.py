# opensloth/cli/os_sft.py
"""
CLI for running Supervised Fine-Tuning (SFT) with OpenSloth.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from opensloth.api import run_training
from opensloth.cli.autogen import add_sft_args, parse_sft_config
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments


def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    """Generate an automatic output directory name."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = Path(dataset_path).name.replace("-", "_").lower()
    return f"outputs/sft_{model_short}_{dataset_short}_{timestamp}"


def train(args: argparse.Namespace) -> None:
    """
    Trains a model using Supervised Fine-Tuning (SFT) on a prepared dataset.
    """
    # Parse configuration from args
    config = parse_sft_config(args)
    
    # Create Pydantic objects
    opensloth_cfg = OpenSlothConfig(**config['opensloth_config'])
    train_args = TrainingArguments(**config['training_args'])
    
    # Auto-generate output directory if not specified
    if not train_args.output_dir or train_args.output_dir == 'saves/loras/':
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, args.dataset)

    if args.dry_run:
        print("DRY RUN: SFT training configuration:")
        summary = {
            "opensloth_config": opensloth_cfg.model_dump(),
            "training_args": train_args.model_dump(),
        }
        print(json.dumps(summary, indent=2))
        return

    print("🚀 Starting SFT training...")
    run_training(opensloth_cfg, train_args, use_tmux=args.tmux)
    print(f"✅ SFT Training complete. Model saved to: {train_args.output_dir}")


def main():
    """Entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Run Supervised Fine-Tuning (SFT) with OpenSloth",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_sft_args(parser)
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
