# worker.py
import argparse
import os
import signal
import time

# ============================================================================
# Device & Environment Bootstrap (CRITICAL: before importing Unsloth/torch)
# ============================================================================


def bootstrap_env():
    """Bootstrap environment settings. MUST be called before importing Unsloth/torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--device", "-d", type=int, default=int(os.environ.get("WORKER_DEVICE", "1"))
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/app/trainer_setup_gsmk.py",
        help="Path to trainer configuration file"
    )
    args, _ = parser.parse_known_args()
    # Disable tqdm globally and its monitor thread to avoid rare shutdown crashes
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Also avoid torch.compile/Dynamo to reduce background compile threads
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        from tqdm import tqdm as _tqdm
        _tqdm.monitor_interval = 0
        try:
            from tqdm.auto import tqdm as _tqdm_auto
            _tqdm_auto.monitor_interval = 0
        except Exception:
            pass
    except Exception:
        pass
    d = args.device
    if d == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{d},0"
    os.makedirs(".cache/", exist_ok=True)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = ".cache/UNSLOTH_CACHE_DIR_" + str(d)
    import unsloth
    _ = unsloth

    print(
        f"[Worker] Bootstrap: device={d}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}"
    )
    return d, args.config


worker_device, trainer_config_path = bootstrap_env()

# Now safe to import torch and other libraries (after device bootstrap)

import importlib.util

from async_grpo_ipc import DEFAULT_HANDLE_PATH, create_stream
from utils.filesystem import read_global_step
from utils.job_queue import claim_job
from utils.logger import logger
from utils.performance import print_performance_summary
from utils.worker_utils import process_job, update_model_weights, wait_for_ps_ready

HANDLE_PATH = os.environ.get("CUDA_IPC_HANDLE_PATH", DEFAULT_HANDLE_PATH)


def load_trainer_from_config(config_path: str, device_id: int = 0):
    """
    Load get_trainer function from a configuration file.

    Args:
        config_path: Path to Python file containing get_trainer function
        device_id: CUDA device ID to use

    Returns:
        get_trainer function from the config
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

    return getattr(module, "get_trainer")


def main():
    """Main worker loop that processes jobs and updates model weights."""
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
        shutdown_requested = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info(f"Starting worker on GPU {worker_device}")

    # Setup trainer and stream
    get_trainer = load_trainer_from_config(trainer_config_path, device_id=worker_device)
    trainer = get_trainer()
    stream = create_stream(non_blocking=True, device=0)

    last_seen_mtime, last_global_step = None, -1

    remote = wait_for_ps_ready(HANDLE_PATH, worker_device=0 if worker_device == 0 else 1)

    # Log connection details
    if hasattr(remote, "buffers"):
        total_bytes = sum(rm.nbytes for rm in remote.buffers.values())
        logger.info(
            f"Opened remote bundle -> {len(remote.buffers)} buffers, {total_bytes}B total"
        )
    else:
        logger.info(
            f"Opened remote handle -> ptr=0x{remote.ptr:x}, nbytes={remote.nbytes}"
        )

    last_seen_mtime, last_global_step = update_model_weights(
        trainer, remote, stream, 0, last_seen_mtime, last_global_step, HANDLE_PATH
    )

    idle_count, base_sleep, max_sleep = 0, 0.01, 0.05
    logger.info("Starting job processing loop...")

    while not shutdown_requested:
        try:
            job_data, job_id, claimed_job_path = claim_job(worker_device)

            if job_data and job_id:
                idle_count = 0
                logger.info(
                    f"Claimed job {job_id} from {os.path.basename(claimed_job_path)}"
                )

                current_global_step = read_global_step()
                last_seen_mtime, last_global_step = update_model_weights(
                    trainer,
                    remote,
                    stream,
                    0,
                    last_seen_mtime,
                    last_global_step,
                    HANDLE_PATH,
                )

                # Process the job and consume the input file upon success
                if process_job(
                    trainer, job_data, job_id, current_global_step, worker_device
                ):
                    from utils.filesystem import archive_file

                    archive_file(claimed_job_path, "processed")
                else:
                    # Handle processing failure (e.g., move to an error directory)
                    from utils.filesystem import archive_file

                    logger.error(
                        f"Processing failed for job {job_id}, archiving input file."
                    )
                    archive_file(claimed_job_path, "failed_processing")
            else:
                idle_count += 1
                sleep_time = (
                    min(max_sleep, base_sleep * 2) if idle_count > 20 else base_sleep
                )
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception(f"Error in main loop: {e}. Continuing...")
            time.sleep(1.0)

    logger.info("Cleaning up...")

    # Print final performance summary
    try:
        print_performance_summary()
    except Exception as e:
        logger.warning(f"Error printing performance summary: {e}")

    if "remote" in locals():
        remote.close()
    logger.success("Graceful shutdown complete")


if __name__ == "__main__":
    main()
