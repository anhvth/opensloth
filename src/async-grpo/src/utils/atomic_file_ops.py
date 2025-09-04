"""
Atomic File Operations System for async-grpo

Provides thread-safe and process-safe file operations with automatic locking,
atomic writes, protected reads, and consistent error handling.

Usage Examples:
    # Simple atomic write/read
    AtomicFileOps.write_json("config.json", {"key": "value"})
    data = AtomicFileOps.read_json("config.json")
    
    # Atomic PyTorch operations
    AtomicFileOps.write_torch("model.pt", tensor_dict)
    tensors = AtomicFileOps.read_torch("model.pt")
    
    # File claiming pattern
    claimed_path = AtomicFileOps.claim_file("job.input.pt", "job.claimed.pt")
    if claimed_path:
        data = AtomicFileOps.read_torch(claimed_path)
        AtomicFileOps.consume_file(claimed_path)  # atomic delete
        
    # Directory operations
    files = AtomicFileOps.list_files("./worker/inputs/", "*.input.pt")
    AtomicFileOps.cleanup_files("./worker/trash/", max_files=100)
"""

import os
import json
import time
import fcntl
import torch
import tempfile
import contextlib
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils.logger import logger


class FileLock:
    """
    File-based lock using fcntl for cross-process coordination.
    Supports both exclusive (write) and shared (read) locks.
    """
    def __init__(self, file_path: str, exclusive: bool = True, timeout: float = 10.0):
        self.file_path = file_path
        self.lock_path = f"{file_path}.lock"
        self.exclusive = exclusive
        self.timeout = timeout
        self._lock_fd = None
        
    def __enter__(self):
        return self.acquire()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        
    def acquire(self) -> bool:
        """Acquire the lock, return True if successful."""
        try:
            # Ensure lock file exists
            os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
            if not os.path.exists(self.lock_path):
                with open(self.lock_path, 'w') as f:
                    f.write("")
            
            # Open lock file
            self._lock_fd = open(self.lock_path, 'r+')
            
            # Choose lock type
            lock_type = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
            
            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < self.timeout:
                try:
                    fcntl.flock(self._lock_fd.fileno(), lock_type | fcntl.LOCK_NB)
                    return True
                except BlockingIOError:
                    time.sleep(0.01)
                    
            # Timeout reached
            self._lock_fd.close()
            self._lock_fd = None
            return False
            
        except Exception as e:
            if self._lock_fd:
                self._lock_fd.close()
                self._lock_fd = None
            logger.warning(f"Error acquiring lock for {self.file_path}: {e}")
            return False
    
    def release(self):
        """Release the lock."""
        if self._lock_fd:
            try:
                fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
                self._lock_fd.close()
            except Exception as e:
                logger.warning(f"Error releasing lock for {self.file_path}: {e}")
            finally:
                self._lock_fd = None


class AtomicFileOps:
    """
    Unified atomic file operations with automatic locking and error handling.
    All operations are thread-safe and process-safe.
    """
    
    DEFAULT_TIMEOUT = 10.0
    DEFAULT_RETRY_COUNT = 3
    DEFAULT_RETRY_DELAY = 0.1
    
    @classmethod
    @contextlib.contextmanager
    def _atomic_write_context(cls, file_path: str, timeout: float = DEFAULT_TIMEOUT):
        """
        Context manager for atomic write operations.
        Creates temp file, yields it, then atomically renames to final path.
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create temp file in same directory to ensure atomic rename
        temp_fd, temp_path = tempfile.mkstemp(
            suffix='.tmp',
            prefix=f"{file_path.name}.",
            dir=file_path.parent
        )
        
        try:
            with FileLock(str(file_path), exclusive=True, timeout=timeout):
                os.close(temp_fd)  # Close fd, we'll use the path
                yield temp_path
                
                # Atomic rename to final location
                os.rename(temp_path, file_path)
                logger.debug(f"Atomically wrote {file_path}")
                
        except Exception:
            # Cleanup temp file on error
            try:
                os.unlink(temp_path)
            except Exception:
                pass
            raise
    
    @classmethod
    def _protected_read(cls, file_path: str, reader_func, timeout: float = DEFAULT_TIMEOUT, 
                       retry_count: int = DEFAULT_RETRY_COUNT, retry_delay: float = DEFAULT_RETRY_DELAY):
        """
        Protected read with shared lock and retry logic.
        """
        for attempt in range(retry_count):
            try:
                with FileLock(file_path, exclusive=False, timeout=timeout):
                    if not os.path.exists(file_path):
                        return None
                    return reader_func(file_path)
                    
            except (FileNotFoundError, EOFError):
                return None
            except Exception as e:
                if attempt == retry_count - 1:
                    logger.warning(f"Failed to read {file_path} after {retry_count} attempts: {e}")
                    raise
                time.sleep(retry_delay * (attempt + 1))
        
        return None
    
    # ============================================================================
    # JSON Operations
    # ============================================================================
    
    @classmethod
    def write_json(cls, file_path: str, data: Dict, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """Atomically write JSON data to file."""
        try:
            with cls._atomic_write_context(file_path, timeout) as temp_path:
                with open(temp_path, 'w') as f:
                    json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error writing JSON to {file_path}: {e}")
            return False
    
    @classmethod
    def read_json(cls, file_path: str, default: Optional[Dict] = None, 
                  timeout: float = DEFAULT_TIMEOUT) -> Optional[Dict]:
        """Read JSON data with protection against concurrent writes."""
        def _read_json(path):
            with open(path, 'r') as f:
                return json.load(f)
                
        result = cls._protected_read(file_path, _read_json, timeout)
        return result if result is not None else default
    
    # ============================================================================
    # PyTorch Operations
    # ============================================================================
    
    @classmethod
    def write_torch(cls, file_path: str, data: Any, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """Atomically write PyTorch data to file."""
        try:
            with cls._atomic_write_context(file_path, timeout) as temp_path:
                torch.save(data, temp_path)
            return True
        except Exception as e:
            logger.error(f"Error writing PyTorch data to {file_path}: {e}")
            return False
    
    @classmethod
    def read_torch(cls, file_path: str, map_location: str = 'cpu', weights_only: bool = False,
                   timeout: float = DEFAULT_TIMEOUT) -> Any:
        """Read PyTorch data with protection against concurrent writes."""
        def _read_torch(path):
            return torch.load(path, map_location=map_location, weights_only=weights_only)
            
        return cls._protected_read(file_path, _read_torch, timeout)
    
    # ============================================================================
    # Text Operations
    # ============================================================================
    
    @classmethod
    def write_text(cls, file_path: str, content: str, timeout: float = DEFAULT_TIMEOUT) -> bool:
        """Atomically write text content to file."""
        try:
            with cls._atomic_write_context(file_path, timeout) as temp_path:
                with open(temp_path, 'w') as f:
                    f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing text to {file_path}: {e}")
            return False
    
    @classmethod
    def read_text(cls, file_path: str, default: str = "", timeout: float = DEFAULT_TIMEOUT) -> str:
        """Read text content with protection against concurrent writes."""
        def _read_text(path):
            with open(path, 'r') as f:
                return f.read()
                
        result = cls._protected_read(file_path, _read_text, timeout)
        return result if result is not None else default
    
    # ============================================================================
    # File Claiming Operations
    # ============================================================================
    
    @classmethod
    def claim_file(cls, source_path: str, claimed_path: Optional[str] = None, 
                   timeout: float = DEFAULT_TIMEOUT) -> Optional[str]:
        """
        Atomically claim a file by renaming it.
        Returns the claimed file path if successful, None if already claimed.
        """
        if not os.path.exists(source_path):
            return None
            
        if claimed_path is None:
            # Generate claimed path by adding .claimed suffix
            path = Path(source_path)
            claimed_path = str(path.with_suffix(f'.claimed{path.suffix}'))
        
        try:
            with FileLock(source_path, exclusive=True, timeout=timeout):
                if not os.path.exists(source_path):
                    return None
                os.rename(source_path, claimed_path)
                logger.debug(f"Claimed file: {source_path} → {claimed_path}")
                return claimed_path
                
        except (OSError, FileNotFoundError):
            # File was already claimed by another process
            return None
        except Exception as e:
            logger.warning(f"Error claiming file {source_path}: {e}")
            return None
    
    @classmethod
    def consume_file(cls, file_path: str, trash_dir: str = "./worker/trash", 
                     reason: str = "consumed") -> bool:
        """
        Atomically consume (move to trash) a file.
        """
        if not os.path.exists(file_path):
            return False
            
        try:
            os.makedirs(trash_dir, exist_ok=True)
            
            filename = os.path.basename(file_path)
            timestamp = int(time.time())
            trash_path = os.path.join(trash_dir, f"{timestamp}_{filename}_{reason}")
            
            os.rename(file_path, trash_path)
            logger.debug(f"Consumed file: {file_path} → {trash_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Error consuming file {file_path}: {e}")
            return False
    
    # ============================================================================
    # Batch File Operations  
    # ============================================================================
    
    @classmethod
    def list_files(cls, directory: str, pattern: str = "*", sort_by_mtime: bool = True) -> List[str]:
        """
        List files matching pattern with optional sorting by modification time.
        """
        try:
            search_pattern = os.path.join(directory, pattern)
            files = glob(search_pattern)
            
            if sort_by_mtime:
                # Filter out files that no longer exist and cache mtime to avoid double calls
                valid_files_with_mtime = []
                for f in files:
                    try:
                        mtime = os.path.getmtime(f)  # Get mtime once
                        valid_files_with_mtime.append((f, mtime))
                    except (OSError, FileNotFoundError):
                        # File was deleted between glob and mtime check
                        continue
                
                # Sort by cached mtime (no second mtime call)
                valid_files_with_mtime.sort(key=lambda x: x[1])
                return [f for f, _ in valid_files_with_mtime]
            else:
                # Just filter for existence
                return [f for f in files if os.path.exists(f)]
                
        except Exception as e:
            logger.warning(f"Error listing files in {directory}: {e}")
            return []
    
    @classmethod
    def cleanup_files(cls, directory: str, pattern: str = "*", max_files: int = 100, 
                     keep_newest: bool = True) -> int:
        """
        Clean up old files in directory, keeping only max_files newest/oldest.
        Returns number of files removed.
        """
        try:
            files = cls.list_files(directory, pattern, sort_by_mtime=True)
            
            if len(files) <= max_files:
                return 0
                
            # Decide which files to remove
            if keep_newest:
                files_to_remove = files[:-max_files]  # Remove oldest
            else:
                files_to_remove = files[max_files:]   # Remove newest
            
            removed_count = 0
            for file_path in files_to_remove:
                try:
                    os.remove(file_path)
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Error removing file {file_path}: {e}")
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} files from {directory}")
                
            return removed_count
            
        except Exception as e:
            logger.error(f"Error during cleanup in {directory}: {e}")
            return 0
    
    @classmethod
    def batch_claim_files(cls, directory: str, pattern: str, max_files: int = 1, 
                         claimed_suffix: str = ".claimed") -> List[str]:
        """
        Claim multiple files atomically, up to max_files.
        Returns list of claimed file paths.
        """
        claimed_files = []
        files = cls.list_files(directory, pattern, sort_by_mtime=True)
        
        for file_path in files[:max_files]:
            path = Path(file_path)
            claimed_path = str(path.with_suffix(f'{claimed_suffix}{path.suffix}'))
            
            claimed = cls.claim_file(file_path, claimed_path)
            if claimed:
                claimed_files.append(claimed)
            
        return claimed_files
    
    # ============================================================================
    # High-level Convenience Methods
    # ============================================================================
    
    @classmethod
    def safe_update_json(cls, file_path: str, update_func, default: Optional[Dict] = None) -> bool:
        """
        Safely update JSON file by reading, modifying, and writing atomically.
        update_func receives current data and should return updated data.
        """
        try:
            current_data = cls.read_json(file_path, default or {})
            updated_data = update_func(current_data)
            return cls.write_json(file_path, updated_data)
        except Exception as e:
            logger.error(f"Error updating JSON file {file_path}: {e}")
            return False
    
    @classmethod
    def ensure_directory(cls, directory: str) -> bool:
        """Ensure directory exists."""
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False


# Backward-compat convenience wrappers removed to reduce API surface.