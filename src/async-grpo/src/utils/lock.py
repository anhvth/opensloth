import os
import fcntl

WEIGHT_LOCK_FILE = "./worker/weight_update.lock"

class WeightUpdateLock:
    """
    File-based lock to coordinate weight updates between PS and workers.
    - PS uses an exclusive lock (LOCK_EX) for writing.
    - Workers use a shared lock (LOCK_SH) for reading.
    """
    def __init__(self, lock_file: str = WEIGHT_LOCK_FILE, is_writer: bool = False):
        self.lock_file = lock_file
        self._lock_fd = None
        self._lock_type = fcntl.LOCK_EX if is_writer else fcntl.LOCK_SH

    def __enter__(self):
        # Create lock file if it doesn't exist
        if not os.path.exists(self.lock_file):
            open(self.lock_file, 'w').close()
        
        # Open in read mode for both lock types to avoid truncating the file
        self._lock_fd = open(self.lock_file, 'r')
        fcntl.flock(self._lock_fd.fileno(), self._lock_type)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._lock_fd:
            fcntl.flock(self._lock_fd.fileno(), fcntl.LOCK_UN)
            self._lock_fd.close()
            self._lock_fd = None