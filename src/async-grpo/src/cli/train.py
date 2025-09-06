#!/usr/bin/env python3
"""
GRPO Training CLI

Usage:
    python src/cli/train.py [TRAINER_CONFIG] [OPTIONS]

Examples:
    # Train with default config
    python src/cli/train.py

    # Train with specific trainer setup
    python src/cli/train.py src/app/trainer_setup_gsmk.py

    # Train with custom device
    python src/cli/train.py src/app/trainer_setup_gsmk.py --device 1

    # Distributed training with device control
    python src/cli/train.py --master_device_id 0 --worker_device_ids 1,2,3
"""

import argparse
import importlib.util
import multiprocessing
import os
import shutil
import subprocess
import sys
from typing import Optional


def load_trainer_from_config(config_path: str, device_id: int = 0):
    """
    Load get_trainer function from a configuration file.

    Args:
        config_path: Path to Python file containing get_trainer function
        device_id: CUDA device ID to use

    Returns:
        Configured trainer instance
    """
    # Convert relative path to absolute
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.getcwd(), config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Trainer config file not found: {config_path}")

    # Load module from file
    spec = importlib.util.spec_from_file_location("trainer_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load trainer config from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Get the get_trainer function
    if not hasattr(module, "get_trainer"):
        raise AttributeError(f"No 'get_trainer' function found in {config_path}")

    get_trainer = getattr(module, "get_trainer")
    

    # Call get_trainer with device_id
    print(f"Loading trainer from {config_path} on device {device_id}...")
    return get_trainer()


def train_single(config_path: str, device_id: int = 0, max_steps: Optional[int] = None):
    """
    Train a single GRPO model using the specified configuration.

    Args:
        config_path: Path to trainer configuration file
        device_id: CUDA device ID to use
        max_steps: Maximum training steps (overrides config)
    """
    print("Starting GRPO training...")
    print(f"Config: {config_path}")
    print(f"Device: {device_id}")

    try:
        # Load trainer
        trainer = load_trainer_from_config(config_path, device_id)

        # Override max_steps if provided
        if max_steps is not None:
            trainer.args.max_steps = max_steps
            print(f"Overriding max_steps to: {max_steps}")

        print(f"Training for {trainer.args.max_steps} steps...")
        print(f"Batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Learning rate: {trainer.args.learning_rate}")
        print(f"Output directory: {trainer.args.output_dir}")

        # Start training
        trainer.train()

        print("Training completed successfully!")

    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


# ==================== Distributed Training ====================


def run_in_process(name: str, cmd: str, env: dict = {}):
    """Run a shell command in a subprocess, log to file, and stream to terminal with prefix."""
    os.makedirs("worker", exist_ok=True)
    log_path = os.path.join("worker", f"log_{name}.txt")

    # Prepare environment - start with current env and update with custom vars
    process_env = os.environ.copy()
    if env:
        process_env.update(env)

    with open(log_path, "w") as logfile:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )
        # Stream output line by line
        for line in process.stdout:
            # Write to logfile
            logfile.write(line)
            logfile.flush()
            # Write to terminal with prefix
            sys.stdout.write(f"[{name}] {line}")
            sys.stdout.flush()

        process.wait()


def start_distributed_training(config_path: str, master_device_id: int = 0, worker_device_ids: Optional[list] = None):
    """Start the parameter server and workers in separate processes with logs + terminal prefix."""
    if worker_device_ids is None:
        worker_device_ids = [1]  # Default to GPU 1
    
    server_cmd = (
        "parameter_server",
        f"python src/cli/parameter_server.py --config {config_path} --device {master_device_id}",
    )
    worker_cmds = [
        (f"worker_{d}", f"python src/cli/worker.py -d {d} --config {config_path}")
        for d in worker_device_ids
    ]

    processes = []

    # Start server
    server_env = {"FAST_INFERENCE": "0"}
    server_proc = multiprocessing.Process(
        target=run_in_process, args=(server_cmd[0], server_cmd[1], server_env)
    )
    server_proc.start()
    processes.append(server_proc)

    # Start workers
    for name, cmd in worker_cmds:
        proc = multiprocessing.Process(target=run_in_process, args=(name, cmd, None))
        proc.start()
        processes.append(proc)

    # Join all processes (wait for them to complete)
    for proc in processes:
        proc.join()


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GRPO Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # Train with default config
  %(prog)s src/app/trainer_setup_gsmk.py           # Train with GSM8K config
  %(prog)s src/app/trainer_setup.py --device 1     # Train on GPU 1
  %(prog)s src/app/trainer_setup_gsmk.py --steps 50  # Override training steps
  %(prog)s --master_device_id 0 --worker_device_ids 1,2,3  # Distributed training with device control
        """,
    )

    parser.add_argument(
        "config",
        nargs="?",
        default="src/app/trainer_setup_gsmk.py",
        help="Path to trainer configuration file (default: src/app/trainer_setup_gsmk.py)",
    )

    parser.add_argument(
        "--device", "-d", type=int, default=0, help="CUDA device ID to use (default: 0)"
    )

    parser.add_argument(
        "--steps", "-s", type=int, help="Maximum training steps (overrides config)"
    )

    parser.add_argument(
        "--num-workers", "-n", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--master_device_id",
        type=int,
        help="CUDA device ID for the parameter server (master)",
    )

    parser.add_argument(
        "--worker_device_ids",
        type=str,
        help="Comma-separated list of CUDA device IDs for workers (e.g., '1,2,3')",
    )

    args = parser.parse_args()

    # clean worker folder if exists
    if os.path.exists("worker"):
        shutil.rmtree("worker")

    # Check for distributed training mode
    use_distributed = args.master_device_id is not None or args.worker_device_ids is not None
    
    if use_distributed:
        # Parse worker device IDs
        worker_device_ids = None
        if args.worker_device_ids:
            try:
                worker_device_ids = [int(x.strip()) for x in args.worker_device_ids.split(',')]
            except ValueError:
                print("Error: worker_device_ids must be a comma-separated list of integers")
                sys.exit(1)
        
        # Default master device ID
        master_device_id = args.master_device_id if args.master_device_id is not None else 0
        
        print("Starting distributed training...")
        print(f"Master device ID: {master_device_id}")
        print(f"Worker device IDs: {worker_device_ids}")
        
        start_distributed_training(args.config, master_device_id, worker_device_ids)
    else:
        train_single(args.config, args.device, args.steps)


if __name__ == "__main__":
    main()
