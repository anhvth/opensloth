# ps.py
import argparse
import importlib.util
import os


def bootstrap_env():
    """Bootstrap environment settings. MUST be called before importing Unsloth/torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", "-d", type=int, default=0)
    args, _ = parser.parse_known_args()
    
    # Set CUDA_VISIBLE_DEVICES based on device argument
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    
    # Disable tqdm progress bars and monitor threads early to avoid crashes
    os.environ.setdefault("TQDM_DISABLE", "1")
    # Disable torch.compile/TorchDynamo/Inductor to improve stability with background threads
    os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    
    print(f"[ParameterServer] Bootstrap: device={args.device}, CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    return args.device


ps_device = bootstrap_env()

import threading
import time
import uuid
from collections import deque

import unsloth as _unsloth  # Must be imported before torch; keep for side-effects

_ = _unsloth  # keep a reference to satisfy linters
import torch
from transformers import TrainerCallback
try:
    # Disable the global tqdm monitor thread which can cause shutdown crashes
    from tqdm import tqdm as _tqdm
    _tqdm.monitor_interval = 0  # 0 disables the monitor thread
    try:
        from tqdm.auto import tqdm as _tqdm_auto
        _tqdm_auto.monitor_interval = 0
    except Exception:
        pass
except Exception:
    pass


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


from async_grpo_ipc import (
    DEFAULT_HANDLE_PATH,
    export_serialized_tensors_handle,
    get_trainable_params,
    serialize_tensors,
)
from utils.filesystem import (
    ps_cleanup_on_startup,
    write_global_step,
    write_version_info,
)
from utils.logger import logger
from utils.ps_utils import (
    collect_results,
    get_next_job_result,
    throw_job_data,
    update_shared_weights,
)

HANDLE_PATH = os.environ.get("CUDA_IPC_HANDLE_PATH", DEFAULT_HANDLE_PATH)

# Clean startup and ensure directories exist
ps_cleanup_on_startup(HANDLE_PATH)

# Global job management state
job_queue = deque()
result_cache = {}
result_cache_lock = threading.Lock()
shutdown_event = threading.Event()
PS_STATE = {}


if __name__ == "__main__":
    # Parse command line arguments (device already parsed in bootstrap)
    parser = argparse.ArgumentParser(description="GRPO Parameter Server")
    parser.add_argument(
        "--config", 
        type=str, 
        default="src/app/trainer_setup_gsmk.py",
        help="Path to trainer configuration file"
    )
    parser.add_argument(
        "--device", "-d",
        type=int, 
        default=0,
        help="CUDA device ID for parameter server (default: 0)"
    )
    args = parser.parse_args()
    
    # Use the device from bootstrap
    device_id = ps_device
    
    # Load trainer from config
    get_trainer = load_trainer_from_config(args.config, device_id=device_id)
    trainer = get_trainer()

    # Initial setup: Serialize and export weights
    trainable_params = get_trainable_params(trainer.model)
    buckets, meta = serialize_tensors(trainable_params)
    logger.info(
        f"Serialized {len(trainable_params)} params into {len(buckets)} dtype buckets: {list(buckets.keys())}"
    )

    export_serialized_tensors_handle(buckets, meta, HANDLE_PATH)
    logger.info(f"Exported CUDA IPC bundle to {HANDLE_PATH}")

    PS_STATE.update({"buckets": buckets, "meta": meta})
    logger.info("Stored persistent state for in-place updates")

    # Create initial version and global step files
    initial_version_info = {
        "global_step": 0,
        "timestamp": time.time(),
        "version_id": str(uuid.uuid4())[:8],
    }
    version_file = HANDLE_PATH.replace(".json", "_version.json")
    write_version_info(initial_version_info, version_file)
    write_global_step(0)

    # Signal workers that PS is ready
    ready_file = "./worker/ps_ready.signal"
    with open(ready_file, "w") as f:
        f.write("ready")
    logger.info(f"Created ready signal at {ready_file}")
    logger.success("Parameter server ready, starting training...")

    def move_tensors_to_device(obj, device):
        """Recursively move all tensors in a nested structure to the specified device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: move_tensors_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [move_tensors_to_device(item, device) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(move_tensors_to_device(item, device) for item in obj)
        else:
            return obj

    def _prepare_inputs(batch: dict) -> dict:
        """Prepare inputs by getting results from the async job system."""
        worker_results = get_next_job_result(
            shutdown_event, job_queue, result_cache, result_cache_lock
        )
        if worker_results is None:
            raise StopIteration("Training interrupted by shutdown")

        # Move all tensors to PS device to avoid device mismatch
        result_on_device = move_tensors_to_device(worker_results["result"], f"cuda:{device_id}")
        assert isinstance(result_on_device, dict)
        return result_on_device 

    class AfterOptimizerStepCallback(TrainerCallback):
        def on_optimizer_step(self, args, state, control, **kwargs):
            trainer_obj = kwargs.get("trainer", trainer)
            update_shared_weights(trainer_obj, HANDLE_PATH, shutdown_event, PS_STATE)
            return control

    trainer.add_callback(AfterOptimizerStepCallback())
    trainer._prepare_inputs = _prepare_inputs

    # Start async job management threads
    logger.info("Starting async job management threads...")
    job_thread = throw_job_data(trainer, shutdown_event, job_queue, result_cache_lock, 8, 8)
    results_thread = collect_results(shutdown_event, result_cache, result_cache_lock)
    time.sleep(1.0)

    logger.info("Starting training with async job system...")
    try:
        trainer.train()
    except (KeyboardInterrupt, StopIteration):
        logger.warning("Training stopped, shutting down...")
    except Exception as e:
        logger.exception(f"Training error: {e}")
    finally:
        logger.info("Setting shutdown event...")
        shutdown_event.set()
        try:
            if 'job_thread' in locals() and job_thread is not None:
                job_thread.join(timeout=5.0)
        except Exception:
            pass
        try:
            if 'results_thread' in locals() and results_thread is not None:
                results_thread.join(timeout=5.0)
        except Exception:
            pass
        time.sleep(0.2)
        logger.success("Shutdown complete")
