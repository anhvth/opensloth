"""
Centralized filesystem utilities for async-grpo.
Provides common helpers for PS and worker to manage job directories, 
trash cleanup, and atomic file I/O.
"""

import os
from typing import Optional, Dict
from utils.logger import logger
from utils.atomic_file_ops import AtomicFileOps


def ensure_directories():
    """Ensure required directories exist."""
    directories = [
        "./worker/queue",
        "./worker/done"
    ]
    for directory in directories:
        AtomicFileOps.ensure_directory(directory)


def archive_file(file_path: str, reason: str) -> bool:
    """Atomically move a file to the 'done' directory with a reason."""
    if not os.path.exists(file_path):
        return False
    
    try:
        os.makedirs("./worker/done", exist_ok=True)
        
        filename = os.path.basename(file_path)
        # Get base name without extension to append new status
        base, ext = os.path.splitext(filename)
        archive_path = os.path.join("./worker/done", f"{base}_{reason}{ext}")
        
        os.rename(file_path, archive_path)
        logger.debug(f"Archived file: {file_path} -> {archive_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Error archiving file {file_path}: {e}")
        return False


def ps_cleanup_on_startup(handle_path: str):
    """Clean up old files during PS startup to ensure a fresh state."""
    logger.info("Performing startup cleanup...")
    
    # Ensure directories exist before cleaning
    ensure_directories()
    
    # Clean specific files
    if os.path.exists(handle_path):
        logger.info(f"Removing existing IPC handle file: {handle_path}")
        os.remove(handle_path)
    
    version_file = handle_path.replace(".json", "_version.json")
    if os.path.exists(version_file):
        os.remove(version_file)

    ready_file = "./worker/ps_ready.signal"
    if os.path.exists(ready_file):
        os.remove(ready_file)

    # Clean up old job files from queue and done directories
    old_queue_files = AtomicFileOps.list_files("./worker/queue/", "*")
    for f in old_queue_files:
        os.remove(f)
    
    old_done_files = AtomicFileOps.list_files("./worker/done/", "*")
    if len(old_done_files) > 100: # Keep some history
        files_to_remove = sorted(old_done_files, key=os.path.getmtime)[:-100]
        for f in files_to_remove:
            os.remove(f)
            
    logger.info("Startup cleanup complete.")




def write_global_step(step: int, global_step_file: str = "./worker/global_step.txt"):
    """Write the current global step to a file for workers to check."""
    success = AtomicFileOps.write_text(global_step_file, str(step))
    if not success:
        logger.error(f"Error writing global step {step}")


def read_global_step(global_step_file: str = "./worker/global_step.txt") -> int:
    """Read the current global step from file."""
    content = AtomicFileOps.read_text(global_step_file)
    try:
        return int(content.strip()) if content else -1
    except ValueError:
        return -1


def get_version_file_mtime(version_file: str) -> Optional[float]:
    """Get modification time of version file, return None if file doesn't exist."""
    try:
        return os.path.getmtime(version_file)
    except (OSError, FileNotFoundError):
        return None


def write_version_info(version_info: dict, version_file: str):
    """Atomically write version info to file."""
    success = AtomicFileOps.write_json(version_file, version_info)
    if not success:
        logger.error(f"Error writing version info to {version_file}")


def read_version_info(version_file: str) -> Dict:
    """Read version info from file, return empty dict if file doesn't exist or is invalid."""
    return AtomicFileOps.read_json(version_file, default={})