#!/usr/bin/env python3
"""
Tmux worker stipt for running distributed training on specific GPU ranks.
This script is called by the tmux session manager to run training on individual GPUs.
"""

import argparse
import os
import sys
from pathlib import Path


def main():
    """Main entry point for tmux worker."""
    parser = argparse.ArgumentParser(description="OpenSloth tmux worker for distributed training")
    parser.add_argument("config_file", help="Path to the config file")
    parser.add_argument("--rank", type=int, required=True, help="Local rank for this worker")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of workers")
    
    args = parser.parse_args()
    
    # Import the config file
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist")
        sys.exit(1)
        
    # Set up environment variables for this worker
    os.environ["OPENSLOTH_LOCAL_RANK"] = str(args.rank)
    os.environ["OPENSLOTH_WORLD_SIZE"] = str(args.world_size)
    os.environ["USE_TMUX"] = "1"
    
    # Execute the config file to get the configurations
    config_globals = {}
    with open(config_path, 'r') as f:
        exec(f.read(), config_globals)
    
    opensloth_config = config_globals["opensloth_config"]
    training_config = config_globals["training_config"]
    
    # Get the GPU ID for this rank
    gpu_id = opensloth_config.devices[args.rank]
    
    # Import the training function
    from opensloth.trainer_factory.opensloth_trainer import train_on_single_gpu
    
    # Run training for this specific GPU
    train_on_single_gpu(
        gpu=gpu_id,
        opensloth_config=opensloth_config,
        hf_train_args=training_config
    )


if __name__ == "__main__":
    main()