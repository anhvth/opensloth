"""Memory monitoring utilities for OpenSloth to detect and prevent memory issues."""

import gc
import psutil
import torch
from typing import Dict, Optional
import warnings
from opensloth.logging_config import get_opensloth_logger


class MemoryMonitor:
    """Monitor system and GPU memory usage to prevent fatal memory errors."""
    
    def __init__(self, warning_threshold: float = 0.85, critical_threshold: float = 0.95):
        """
        Initialize memory monitor.
        
        Args:
            warning_threshold: Fraction of memory usage to trigger warnings (0.85 = 85%)
            critical_threshold: Fraction of memory usage to trigger critical cleanup (0.95 = 95%)
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = get_opensloth_logger()
        self.last_warning_step = -1
        self.last_critical_step = -1
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        stats = {}
        
        # System memory
        memory = psutil.virtual_memory()
        stats['system_used_gb'] = memory.used / (1024**3)
        stats['system_total_gb'] = memory.total / (1024**3)
        stats['system_percent'] = memory.percent
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_stats()
            allocated = gpu_memory.get('allocated_bytes.all.current', 0)
            reserved = gpu_memory.get('reserved_bytes.all.current', 0)
            
            stats['gpu_allocated_gb'] = allocated / (1024**3)
            stats['gpu_reserved_gb'] = reserved / (1024**3)
            
            # Get total GPU memory
            total_memory = torch.cuda.get_device_properties(0).total_memory
            stats['gpu_total_gb'] = total_memory / (1024**3)
            stats['gpu_percent'] = (reserved / total_memory) * 100
        
        return stats
    
    def check_memory_health(self, current_step: int) -> bool:
        """
        Check memory health and trigger cleanup if needed.
        
        Args:
            current_step: Current training step
            
        Returns:
            True if memory is healthy, False if critical intervention was needed
        """
        stats = self.get_memory_stats()
        
        # Check system memory
        if stats['system_percent'] / 100 > self.critical_threshold:
            if current_step != self.last_critical_step:
                self.logger.error(
                    f"CRITICAL: System memory at {stats['system_percent']:.1f}% "
                    f"({stats['system_used_gb']:.1f}GB/{stats['system_total_gb']:.1f}GB)"
                )
                self._force_cleanup()
                self.last_critical_step = current_step
                return False
        
        elif stats['system_percent'] / 100 > self.warning_threshold:
            if current_step != self.last_warning_step:
                self.logger.warning(
                    f"System memory high: {stats['system_percent']:.1f}% "
                    f"({stats['system_used_gb']:.1f}GB/{stats['system_total_gb']:.1f}GB)"
                )
                self.last_warning_step = current_step
        
        # Check GPU memory
        if torch.cuda.is_available() and 'gpu_percent' in stats:
            if stats['gpu_percent'] / 100 > self.critical_threshold:
                if current_step != self.last_critical_step:
                    self.logger.error(
                        f"CRITICAL: GPU memory at {stats['gpu_percent']:.1f}% "
                        f"({stats['gpu_reserved_gb']:.1f}GB/{stats['gpu_total_gb']:.1f}GB)"
                    )
                    self._force_cleanup()
                    self.last_critical_step = current_step
                    return False
            
            elif stats['gpu_percent'] / 100 > self.warning_threshold:
                if current_step != self.last_warning_step:
                    self.logger.warning(
                        f"GPU memory high: {stats['gpu_percent']:.1f}% "
                        f"({stats['gpu_reserved_gb']:.1f}GB/{stats['gpu_total_gb']:.1f}GB)"
                    )
                    self.last_warning_step = current_step
        
        return True
    
    def _force_cleanup(self):
        """Force aggressive memory cleanup."""
        self.logger.info("Forcing aggressive memory cleanup...")
        
        # Python garbage collection
        collected = gc.collect()
        self.logger.info(f"Garbage collection freed {collected} objects")
        
        # GPU cache cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self.logger.info("GPU cache cleared")
        
        # Log memory stats after cleanup
        stats = self.get_memory_stats()
        self.logger.info(
            f"After cleanup - System: {stats['system_percent']:.1f}%, "
            f"GPU: {stats.get('gpu_percent', 0):.1f}%"
        )
    
    def log_memory_summary(self, step: int, prefix: str = ""):
        """Log a summary of current memory usage."""
        stats = self.get_memory_stats()
        
        msg = f"{prefix}Step {step} memory: "
        msg += f"System {stats['system_percent']:.1f}% ({stats['system_used_gb']:.1f}GB), "
        
        if torch.cuda.is_available() and 'gpu_percent' in stats:
            msg += f"GPU {stats['gpu_percent']:.1f}% ({stats['gpu_reserved_gb']:.1f}GB)"
        
        self.logger.info(msg)


def check_for_memory_leaks(model: torch.nn.Module, step: int) -> bool:
    """
    Check for potential memory leaks in the model.
    
    Args:
        model: The model to check
        step: Current training step
        
    Returns:
        True if no leaks detected, False if potential leaks found
    """
    logger = get_opensloth_logger()
    
    # Check for None parameters (potential sign of reference counting bugs)
    none_params = []
    for name, param in model.named_parameters():
        if param is None:
            none_params.append(name)
    
    if none_params:
        logger.error(f"Step {step}: Found None parameters: {none_params}")
        logger.error("This may indicate a reference counting bug!")
        return False
    
    # Check for parameters with invalid gradients
    invalid_grads = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                invalid_grads.append(name)
    
    if invalid_grads:
        logger.warning(f"Step {step}: Found invalid gradients in: {invalid_grads}")
        return False
    
    return True


def safe_tensor_operation(operation_name: str, operation_func, *args, **kwargs):
    """
    Safely execute tensor operations with proper error handling.
    
    Args:
        operation_name: Name of the operation for logging
        operation_func: Function to execute
        *args, **kwargs: Arguments for the function
        
    Returns:
        Result of the operation
        
    Raises:
        RuntimeError: If operation fails with memory-related error
    """
    logger = get_opensloth_logger()
    
    try:
        return operation_func(*args, **kwargs)
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for memory-related errors
        if any(keyword in error_msg for keyword in ['none_dealloc', 'reference', 'memory', 'cuda']):
            logger.error(f"Memory-related error in {operation_name}: {e}")
            
            # Attempt cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            raise RuntimeError(f"Critical memory error in {operation_name}: {e}")
        else:
            # Re-raise non-memory errors as-is
            raise