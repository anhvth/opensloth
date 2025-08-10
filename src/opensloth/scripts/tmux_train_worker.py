#!/usr/bin/env python3
"""
TMux training worker entry point for OpenSloth.
This script is called by the tmux training orchestrator for each GPU worker.
"""

import argparse
import os
import sys


def main():
    # Setup comprehensive logging interception
    from opensloth.logging_config import setup_huggingface_logging_interception, setup_stdout_interception_for_training
    setup_huggingface_logging_interception()
    
    # Set training active flag for stdout interception
    os.environ["OPENSLOTH_TRAINING_ACTIVE"] = "1"
    setup_stdout_interception_for_training()
    

    from opensloth.scripts.opensloth_trainer import load_config_from_path, train_on_single_gpu

    parser = argparse.ArgumentParser(description="Train worker for tmux mode")
    parser.add_argument("config_file", help="Path to config file")
    parser.add_argument("--rank", type=int, required=True, help="GPU rank to use")
    args = parser.parse_args()

    opensloth_config, training_config = load_config_from_path(args.config_file)
    
    # Set environment variables for rank
    os.environ["OPENSLOTH_LOCAL_RANK"] = str(args.rank)
    os.environ["OPENSLOTH_WORLD_SIZE"] = str(len(opensloth_config.devices))
    
    gpu = opensloth_config.devices[args.rank]
    
    # Run training on the assigned GPU
    train_on_single_gpu(gpu, opensloth_config, training_config)

if __name__ == "__main__":
    main()
