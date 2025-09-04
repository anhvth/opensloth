import os
import time
from typing import Tuple, Optional, Dict
from utils.logger import logger
from utils.atomic_file_ops import AtomicFileOps


def create_job(generation_batch: Dict, job_id: str) -> str:
    """Create a job file for workers to process.
    Return create status
    """
    # Use a descriptive filename: {job_id}_pending.pt
    input_file = f"./worker/queue/{job_id}_pending.pt"

    job_data = {
        "generation_batch": generation_batch,
        "job_id": job_id,
        "created_at": time.time(),  # Track when job was created
    }

    # Use atomic write to prevent corruption
    if not AtomicFileOps.write_torch(input_file, job_data):
        logger.error(f"Failed to create job {job_id}")
        return False
    
    return True


def claim_job(worker_device: int) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Atomically claim an available job by renaming the input file.
    Returns (job_data, job_id, claimed_path) or (None, None, None) if no job was claimed.
    """
    # Get available job files (sorted by creation time for FIFO)
    input_files = AtomicFileOps.list_files("./worker/queue/", "*_pending.pt", sort_by_mtime=True)
    
    for input_file in input_files:
        try:
            # Extract job_id from filename: {job_id}_pending.pt
            basename = os.path.basename(input_file)
            job_id = basename.replace('_pending.pt', '')
            
            claimed_path = f"./worker/queue/{job_id}_processing_by_dev{worker_device}.pt"
            
            # Atomically claim the job by renaming
            claimed_file = AtomicFileOps.claim_file(input_file, claimed_path)
            if not claimed_file:
                continue  # Job was claimed by another worker
            
            # Load the job data safely
            job_data = AtomicFileOps.read_torch(claimed_file, weights_only=False)
            if not job_data:
                logger.warning(f"Failed to load claimed job {claimed_file}")
                from utils.filesystem import archive_file
                archive_file(claimed_file, "failed_load")
                continue
            
            # CRITICAL: Do NOT consume the file here. The worker will do it after processing.
            
            # Add processing metadata
            job_data["claimed_at"] = time.time()
            job_data["worker_device"] = worker_device
            
            logger.debug(f"Successfully claimed job {job_id}")
            return job_data, job_id, claimed_file
            
        except Exception as e:
            logger.warning(f"Error claiming job {input_file}: {e}")
            continue
    
    return None, None, None