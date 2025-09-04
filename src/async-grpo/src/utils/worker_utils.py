import os
import time
import torch

from utils.logger import logger
from utils.atomic_file_ops import AtomicFileOps
from async_grpo_ipc import (
    open_remote_memory,
    stream_set_trainable_params,
)
from utils.filesystem import (
    get_version_file_mtime,
    read_version_info,
)
from utils.lock import WeightUpdateLock
from utils.performance import record_job_metrics, record_sync_metrics


def wait_for_ps_ready(
    handle_path: str,
    max_retries: int = 300,
    retry_interval: float = 1.0,
    worker_device: int = 0,
):
    """
    Wait for Parameter Server to be ready by checking for ready signal and IPC handle.
    """
    ready_file = "./worker/ps_ready.signal"
    logger.info("Waiting for Parameter Server to be ready...")
    logger.info(f"Will check every {retry_interval}s for up to {max_retries}s")
    logger.info(f"Looking for: {ready_file} and {handle_path}")

    for attempt in range(max_retries):
        try:
            if not os.path.exists(ready_file):
                if attempt % 10 == 0:
                    logger.info(
                        f"Waiting for PS ready signal at {ready_file}... ({attempt}/{max_retries})"
                    )
                time.sleep(retry_interval)
                continue

            if not os.path.exists(handle_path):
                if attempt % 10 == 0:
                    logger.info(
                        f"Waiting for IPC handle at {handle_path}... ({attempt}/{max_retries})"
                    )
                time.sleep(retry_interval)
                continue

            logger.info("Both files found, attempting to connect to PS...")
            remote = open_remote_memory(handle_path, target_device=worker_device)
            logger.success(f"Successfully connected to PS on attempt {attempt + 1}")
            return remote

        except Exception as e:
            if attempt % 10 == 0 or attempt == max_retries - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e}"
                )
            time.sleep(retry_interval)

    logger.error(f"Could not connect to PS after {max_retries} attempts")
    raise RuntimeError(f"Could not connect to PS after {max_retries} attempts")


def update_model_weights(
    trainer, remote, stream, worker_device, last_seen_mtime, last_step, handle_path
):
    """
    Update model weights from remote memory with caching and throttling.
    """
    try:
        version_file = handle_path.replace(".json", "_version.json")
        current_mtime = get_version_file_mtime(version_file)

        if current_mtime == last_seen_mtime:
            return last_seen_mtime, last_step

        version_info = read_version_info(version_file)
        current_global_step = version_info.get("global_step", -1)

        if current_global_step == last_step:
            return current_mtime, last_step

        t1 = time.time()
        with WeightUpdateLock(is_writer=False):
            with torch.cuda.device(worker_device):
                with stream:
                    if remote.serialization_meta or getattr(remote, "buffers", None):
                        num_updated = stream_set_trainable_params(
                            trainer.model, remote, stream
                        )
                        stream.synchronize()
                        torch.cuda.synchronize(worker_device)

                        update_time = time.time() - t1
                        version_str = f"v{version_info.get('version_id', 'uk')}, step {current_global_step}"
                        logger.info(
                            f"Updated {num_updated} params ({version_str}) in {update_time:.3f}s"
                        )

                        # Record weight sync metrics
                        record_sync_metrics(update_time, worker_device)
                    else:
                        logger.warning("No serialization metadata found")

        return current_mtime, current_global_step

    except Exception as e:
        logger.warning(f"Error updating model weights: {e}")
        return last_seen_mtime, last_step


def move_tensors_to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_tensors_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_tensors_to_cpu(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_tensors_to_cpu(item) for item in obj)
    else:
        return obj


def process_job(
    trainer, job_data: dict, job_id: str, current_global_step: int, worker_device: int
):
    """Process a single job and save the results."""
    generation_batch = job_data["generation_batch"]
    job_global_step = job_data.get("global_step", -1)
    created_at = job_data.get("created_at", 0)
    claimed_at = job_data.get("claimed_at", 0)

    # Calculate job age for monitoring
    now = time.time()
    job_age = now - created_at if created_at > 0 else 0
    queue_time = claimed_at - created_at if claimed_at > created_at > 0 else 0

    if hasattr(trainer, "state"):
        trainer.state.global_step = (
            current_global_step  # Use current step, not job step
        )

    logger.info(
        f"Processing job {job_id} at global step {job_global_step} (current: {current_global_step}, age: {job_age:.2f}s, queue: {queue_time:.2f}s)"
    )

    assert hasattr(trainer, "_prepare_inputs"), (
        "Trainer must have _prepare_inputs method"
    )
    result = trainer._prepare_inputs(generation_batch)

    metrics = trainer._metrics

    # Move all tensors to CPU before saving to avoid device mismatch issues
    result_cpu = move_tensors_to_cpu(result)
    metrics_cpu = move_tensors_to_cpu(metrics)

    result_data = {
        "result": result_cpu,
        "job_id": job_id,
        "processed_at_step": current_global_step,
        "worker_device": worker_device,
        "job_created_step": job_global_step,
        "job_age": job_age,
        "queue_time": queue_time,
        "processing_time": time.time() - now,
        "metrics": metrics_cpu,
    }

    # Atomic write using the new stateful naming system
    result_file = f"./worker/queue/{job_id}_complete_from_dev{worker_device}.pt"

    if not AtomicFileOps.write_torch(result_file, result_data):
        logger.error(f"Failed to save result for job {job_id}")
        return False

    logger.success(
        f"Completed job {job_id} (proc: {result_data['processing_time']:.2f}s)"
    )

    # Record performance metrics
    record_job_metrics(
        job_id, result_data["processing_time"], job_age, queue_time, worker_device
    )

    return True
