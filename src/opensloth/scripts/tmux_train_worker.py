#!/usr/bin/env python3
"""
TMux training worker entry point for OpenSloth.
This script is called by the tmux training orchestrator for each GPU worker.
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="TMux training worker for OpenSloth")
    parser.add_argument("config_file", help="Path to the configuration file")
    parser.add_argument("--rank", type=int, required=True, help="Local rank of this worker")
    parser.add_argument("--world_size", type=int, required=True, help="Total number of workers")
    
    args = parser.parse_args()
    
    # Set environment variables for tmux mode
    os.environ["USE_TMUX"] = "1"
    os.environ["OPENSLOTH_LOCAL_RANK"] = str(args.rank)
    
    # Import and run the training function
    from opensloth.scripts.opensloth_trainer import initialize_training_config, train_on_single_gpu
    
    # Load configuration
    opensloth_config, training_config = initialize_training_config(args.config_file)
    
    # Get the GPU for this rank
    gpu = opensloth_config.devices[args.rank]
    
    # Run training on the assigned GPU
    train_on_single_gpu(gpu, opensloth_config, training_config)

if __name__ == "__main__":
    main()
