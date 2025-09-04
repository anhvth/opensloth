import os
import time
import uuid
import torch
import threading
from fastcore.all import threaded

from utils.logger import logger
from utils.atomic_file_ops import AtomicFileOps
from async_grpo_ipc import get_trainable_params, serialize_tensors
from utils.filesystem import write_version_info, write_global_step, archive_file
from utils.lock import WeightUpdateLock
from utils.job_queue import create_job


def update_shared_weights(trainer, handle_path: str, shutdown_event: threading.Event, ps_state: dict):
    """Update the shared weights in IPC memory with in-place copy (no re-export)."""
    if shutdown_event.is_set():
        return
        
    with WeightUpdateLock(is_writer=True):
        # Get current trainable parameters and serialize them
        new_buckets, _ = serialize_tensors(get_trainable_params(trainer.model))

        # Single in-place copy pass under torch.no_grad()
        with torch.no_grad():
            for k in new_buckets:
                ps_state["buckets"][k].copy_(new_buckets[k])

        # Atomically write version and global step together
        version_info = {
            "global_step": trainer.state.global_step,
            "timestamp": time.time(),
            "version_id": str(uuid.uuid4())[:8],
        }

        version_file = handle_path.replace(".json", "_version.json")
        write_version_info(version_info, version_file)
        write_global_step(trainer.state.global_step)

        logger.info(f"Bucket update step {trainer.state.global_step} (v{version_info['version_id']})")


@threaded
def throw_job_data(trainer, shutdown_event, job_queue, result_cache_lock, max_input_in_folder, max_pending_output):
    """
    Continuously populate the input folder with jobs to keep workers busy.
    """
    dataloader = trainer.get_train_dataloader()
    logger.info(f"Starting job creation thread (max_input={max_input_in_folder}, max_pending={max_pending_output})")
    data_iter = iter(dataloader)
    
    # State for per-step job indexing
    last_step_for_job_idx = -1
    job_index_this_step = 0
    total_jobs_created = 0

    while not shutdown_event.is_set():
        try:
            # Use atomic file ops to count files
            current_inputs = len(AtomicFileOps.list_files("./worker/queue/", "*_pending.pt"))
            current_pending = len(AtomicFileOps.list_files("./worker/queue/", "*_complete_from_dev*.pt"))

            if current_inputs < max_input_in_folder and current_pending < max_pending_output:
                try:
                    generation_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(trainer.get_train_dataloader())
                    generation_batch = next(data_iter)

                # Reset job index if global step has advanced
                current_global_step = trainer.state.global_step
                if current_global_step != last_step_for_job_idx:
                    logger.info(f"Global step changed from {last_step_for_job_idx} to {current_global_step}. Resetting job index.")
                    last_step_for_job_idx = current_global_step
                    job_index_this_step = 0

                # Create composite job ID: {step}_{index}
                job_id = f"{current_global_step:06d}_{job_index_this_step:03d}"
                if create_job(generation_batch, job_id):
                    with result_cache_lock:
                        job_queue.append({"job_id": job_id})
                    
                    total_jobs_created += 1
                    job_index_this_step += 1 # Increment index for next job in this step
                    
                    if total_jobs_created % 10 == 1:
                        logger.info(f"Created job {job_id}. Total created: {total_jobs_created}, queue size: {len(job_queue)}")
                else:
                    logger.warning(f"Failed to create job {job_id}, retrying...")
            
            time.sleep(0.1)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.exception(f"Error in throw_job_data: {e}")
            time.sleep(1.0)
    
    logger.info("Job creation thread shutting down")


@threaded
def collect_results(shutdown_event, result_cache, result_cache_lock):
    """
    Continuously collect completed job results and cache them.
    """
    logger.info("Starting result collection thread")
    result_count = 0
    while not shutdown_event.is_set():
        try:
            # Clean up any leftover temporary files first
            temp_files = AtomicFileOps.list_files("./worker/queue/", "*.tmp")
            for temp_file in temp_files:
                archive_file(temp_file, "orphaned_temp")
            
            # Use atomic claiming to get result files safely
            result_files = AtomicFileOps.list_files("./worker/queue/", "*_complete_from_dev*.pt", sort_by_mtime=True)
            
            for result_file in result_files:
                try:
                    # Extract job_id from filename: {step}_{index}_complete_from_dev{d}.pt
                    basename = os.path.basename(result_file)
                    parts = basename.replace('.pt', '').split('_')
                    if len(parts) < 5:
                        continue
                    job_id = f"{parts[0]}_{parts[1]}"
                    
                    # Atomically claim the result file by renaming to a 'collected' state
                    claimed_path = f"./worker/queue/{job_id}_collected.pt"
                    claimed_file = AtomicFileOps.claim_file(result_file, claimed_path)
                    if not claimed_file:
                        continue  # File was already claimed
                    
                    # Safely load the claimed file
                    worker_results = AtomicFileOps.read_torch(claimed_file, weights_only=False)
                    if not worker_results:
                        logger.warning(f"Failed to load claimed file {claimed_file}")
                        archive_file(claimed_file, "failed_load")
                        continue
                        
                    # Validate job_id matches filename
                    file_job_id = worker_results.get("job_id")
                    if not file_job_id or file_job_id != job_id:
                        logger.warning(f"Job ID mismatch: file={job_id}, content={file_job_id}")
                        archive_file(claimed_file, "job_id_mismatch")
                        continue
                    
                    # Cache the result atomically
                    with result_cache_lock:
                        if job_id not in result_cache:
                            result_cache[job_id] = {
                                "result_data": worker_results,
                                "result_file_path": claimed_file,
                            }
                            result_count += 1
                            if result_count % 10 == 1:
                                logger.info(f"Cached {result_count} results, cache_size={len(result_cache)}")
                        else:
                            # Duplicate result - clean up the claimed file
                            logger.debug(f"Duplicate result for job {job_id}, discarding")
                            archive_file(claimed_file, "duplicate")
                            
                except Exception as e:
                    if not shutdown_event.is_set():
                        logger.warning(f"Error processing result file {result_file}: {e}")
                            
            time.sleep(0.05)
        except Exception as e:
            if not shutdown_event.is_set():
                logger.exception(f"Error in collect_results: {e}")
            time.sleep(1.0)
    
    # Clean up any leftover claimed files during shutdown
    logger.info("Cleaning up leftover claimed files...")
    claimed_files = AtomicFileOps.list_files("./worker/queue/", "*_collected.pt")
    for claimed_file in claimed_files:
        archive_file(claimed_file, "shutdown")
    
    logger.info("Result collection thread shutting down")


def get_next_job_result(shutdown_event, job_queue, result_cache, result_cache_lock):
    """
    Get the next available job result, blocking until one is available.
    """
    while not shutdown_event.is_set():
        with result_cache_lock:
            # First, try to find any available result for jobs in the queue
            if job_queue and result_cache:
                # Look for any job in the queue that has a result ready
                for i, job_item in enumerate(job_queue):
                    job_id = job_item["job_id"]
                    if job_id in result_cache:
                        # Remove this job from the queue (it may not be the first)
                        del job_queue[i]
                        result = result_cache.pop(job_id)
                        # Clean up the claimed file using archive_file
                        archive_file(result["result_file_path"], f"consumed_{job_id}")
                        logger.debug(f"Consumed result for job {job_id}")
                        return result["result_data"]
            
            # If no job-result pairs are ready but we have results, 
            # log a warning about orphaned results
            if result_cache and not job_queue:
                orphaned_jobs = list(result_cache.keys())
                logger.warning(f"Found {len(orphaned_jobs)} orphaned results (no matching jobs in queue): {orphaned_jobs[:5]}")
                # Clear orphaned results to prevent memory leak
                for job_id in orphaned_jobs:
                    result = result_cache.pop(job_id)
                    archive_file(result["result_file_path"], f"orphaned_{job_id}")
                
        time.sleep(0.01)
    return None