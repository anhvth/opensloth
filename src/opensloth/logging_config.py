# Duration threshold constants
DURATION_SHORT_SKIP = 3
DURATION_SECONDS_IN_MINUTE = 60
DURATION_SECONDS_IN_HOUR = 3600
DURATION_MS_THRESHOLD = 0.1
"""
Enhanced logging configuration for opensloth with improved formatting and organization.
"""

import os
import sys
import time
import logging

from rich.console import Console
from rich.table import Table

# from speedy_utils import setup_logger
COUNT = 0


def setup_comprehensive_logging_interception():
    """
    Comprehensive interception of logging output only (not stdout/stderr).
    Redirects logging to loguru for unified logging with GPU context.
    """
    # Check if already intercepted to avoid duplicate setup
    if hasattr(setup_comprehensive_logging_interception, '_intercepted'):
        return
    
    class LoguruHandler(logging.Handler):
        """Custom handler to redirect standard logging to loguru files"""
        
        def __init__(self):
            super().__init__()
            
            # File handles for writing intercepted output
            self.hf_log_file = None
            
            # Get GPU context
            self.gpu_id = os.environ.get("OPENSLOTH_LOCAL_RANK", "0")
            
            # Setup log files
            self._setup_log_files()
        
        def _setup_log_files(self):
            """Setup log files for intercepted output"""
            try:
                output_dir = os.environ.get("OPENSLOTH_OUTPUT_DIR")
                if output_dir:
                    log_dir = os.path.join(output_dir, "logs")
                    os.makedirs(log_dir, exist_ok=True)
                    
                    # HF library logs (standard logging)
                    hf_log_path = os.path.join(log_dir, f"huggingface_gpu_{self.gpu_id}.log")
                    self.hf_log_file = open(hf_log_path, 'a', encoding='utf-8')
            except Exception:
                pass
        
        def emit(self, record):
            try:
                # Format the message including logger name for context
                message = record.getMessage()
                if record.name and record.name != "root":
                    message = f"[{record.name}] {message}"
                
                # Format timestamp
                import datetime
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                # Create log line with GPU context
                log_line = f"{timestamp} | {record.levelname:<8} | GPU{self.gpu_id} | [HF] | {message}\n"
                
                # Write to file if available
                if self.hf_log_file:
                    try:
                        self.hf_log_file.write(log_line)
                        self.hf_log_file.flush()
                    except Exception:
                        pass
                        
            except Exception:
                self.handleError(record)
        
        def close(self):
            if self.hf_log_file:
                try:
                    self.hf_log_file.close()
                except Exception:
                    pass
            super().close()
    
    # Remove existing handlers and add our interceptor
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    loguru_handler = LoguruHandler()
    loguru_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(loguru_handler)
    root_logger.setLevel(logging.DEBUG)
    
    # Set reasonable levels for common libraries
    logging.getLogger("transformers").setLevel(logging.INFO)
    logging.getLogger("datasets").setLevel(logging.INFO)
    logging.getLogger("accelerate").setLevel(logging.INFO)
    logging.getLogger("trl").setLevel(logging.INFO)
    logging.getLogger("unsloth").setLevel(logging.INFO)
    # Suppress very verbose network logs
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    
    # Mark as intercepted
    setup_comprehensive_logging_interception._intercepted = True


def setup_stdout_interception_for_training():
    """
    Setup stdout/stderr interception specifically for training phases.
    This should be called when training starts.
    """
    # Check if already intercepted
    if hasattr(setup_stdout_interception_for_training, '_intercepted'):
        return
    
    class TrainingOutputInterceptor:
        """Intercepts stdout/stderr during training only"""
        
        def __init__(self):
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            self.gpu_id = os.environ.get("OPENSLOTH_LOCAL_RANK", "0")
            
            # Setup training log file
            self.training_log_file = None
            try:
                output_dir = os.environ.get("OPENSLOTH_OUTPUT_DIR")
                if output_dir:
                    log_dir = os.path.join(output_dir, "logs")
                    os.makedirs(log_dir, exist_ok=True)
                    
                    training_log_path = os.path.join(log_dir, f"training_stdout_gpu_{self.gpu_id}.log")
                    self.training_log_file = open(training_log_path, 'a', encoding='utf-8')
            except Exception:
                pass
        
        def write(self, text):
            if not text or text.isspace():
                return len(text) if text else 0
            
            # Format timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            text = text.strip()
            if not text:
                return 0
            
            # Create log line with GPU context
            log_line = f"{timestamp} | INFO     | GPU{self.gpu_id} | [STDOUT] | {text}\n"
            
            # Write to file
            if self.training_log_file:
                try:
                    self.training_log_file.write(log_line)
                    self.training_log_file.flush()
                except Exception:
                    pass
            
            return len(text)
        
        def flush(self):
            try:
                if self.training_log_file:
                    self.training_log_file.flush()
            except Exception:
                pass
        
        def close(self):
            try:
                if self.training_log_file:
                    self.training_log_file.close()
            except Exception:
                pass
    
    # Only intercept if we're actually in training
    if os.environ.get("OPENSLOTH_TRAINING_ACTIVE") == "1":
        interceptor = TrainingOutputInterceptor()
        sys.stdout = interceptor
        sys.stderr = interceptor
    
    setup_stdout_interception_for_training._intercepted = True


def disable_huggingface_console_callbacks():
    """
    Disable Hugging Face's default console callbacks that print training progress.
    This should be called when setting up the trainer.
    """
    try:
        # Import HF callback classes
        from transformers.trainer_callback import (
            ProgressCallback, 
            PrinterCallback,
            DefaultFlowCallback
        )
        
        # Store original print methods to disable them
        if hasattr(ProgressCallback, 'on_log'):
            ProgressCallback._original_on_log = ProgressCallback.on_log
            ProgressCallback.on_log = lambda self, args, state, control, logs=None, **kwargs: None
        
        if hasattr(PrinterCallback, 'on_log'):
            PrinterCallback._original_on_log = PrinterCallback.on_log  
            PrinterCallback.on_log = lambda self, args, state, control, logs=None, **kwargs: None
            
        # Also disable progress bar updates that print to stdout
        if hasattr(ProgressCallback, 'on_step_end'):
            ProgressCallback._original_on_step_end = ProgressCallback.on_step_end
            ProgressCallback.on_step_end = lambda self, args, state, control, **kwargs: None
        
        if hasattr(ProgressCallback, 'on_epoch_end'):
            ProgressCallback._original_on_epoch_end = ProgressCallback.on_epoch_end
            ProgressCallback.on_epoch_end = lambda self, args, state, control, **kwargs: None
            
    except ImportError:
        # HF not available or different version, skip
        pass


class OpenSlothTrainingLogCallback:
    """
    Custom callback to handle training progress logs through loguru with GPU context.
    Replaces Hugging Face's default console logging.
    """
    
    def __init__(self):
        self.gpu_id = os.environ.get("OPENSLOTH_LOCAL_RANK", "0")
        
        # Setup log file for training progress
        self.training_progress_file = None
        try:
            output_dir = os.environ.get("OPENSLOTH_OUTPUT_DIR")
            if output_dir:
                log_dir = os.path.join(output_dir, "logs")
                os.makedirs(log_dir, exist_ok=True)
                
                # Training progress logs
                progress_log_path = os.path.join(log_dir, f"training_progress_gpu_{self.gpu_id}.log")
                self.training_progress_file = open(progress_log_path, 'a', encoding='utf-8')
        except Exception:
            pass
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Handle training progress logs through loguru"""
        if logs is None:
            return
        
        # Format timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Format the logs nicely with GPU context
        log_parts = []
        for key, value in logs.items():
            if isinstance(value, float):
                if key in ['loss', 'train_loss', 'eval_loss']:
                    log_parts.append(f"{key}={value:.4f}")
                elif key in ['learning_rate']:
                    log_parts.append(f"{key}={value:.2e}")
                elif key in ['epoch']:
                    log_parts.append(f"{key}={value:.2f}")
                else:
                    log_parts.append(f"{key}={value:.4f}")
            else:
                log_parts.append(f"{key}={value}")
        
        log_message = " | ".join(log_parts)
        
        # Create formatted log line
        step = state.global_step if hasattr(state, 'global_step') else 0
        log_line = f"{timestamp} | INFO     | GPU{self.gpu_id} | [STEP {step:>4}] | {log_message}\n"
        
        # Write to file
        if self.training_progress_file:
            try:
                self.training_progress_file.write(log_line)
                self.training_progress_file.flush()
            except Exception:
                pass
    
    def close(self):
        """Close file handles"""
        if self.training_progress_file:
            try:
                self.training_progress_file.close()
            except Exception:
                pass


def add_opensloth_logging_callback(trainer):
    """
    Add OpenSloth's custom logging callback to the trainer.
    This should be called after trainer creation.
    """
    # Remove default HF callbacks that print to console
    callbacks_to_remove = []
    
    for callback in trainer.callback_handler.callbacks:
        callback_name = callback.__class__.__name__
        if callback_name in ['ProgressCallback', 'PrinterCallback']:
            callbacks_to_remove.append(callback)
    
    for callback in callbacks_to_remove:
        trainer.callback_handler.callbacks.remove(callback)
    
    # Add our custom callback
    opensloth_callback = OpenSlothTrainingLogCallback()
    trainer.add_callback(opensloth_callback)
    
    return trainer


def setup_huggingface_logging_interception():
    """
    Legacy function for backward compatibility.
    Now calls the comprehensive logging interception.
    """
    setup_comprehensive_logging_interception()
    disable_huggingface_console_callbacks()


class StepTimer:
    """Helper class to track timing for individual steps."""

    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = time.time()
        self.end_time: float | None = None

    def finish(self) -> float:
        """Finish timing and return duration."""
        self.end_time = time.time()
        return self.duration

    @property
    def duration(self) -> float:
        """Get duration in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


class OpenslothLogger:
    """Enhanced logger for opensloth with better formatting and GPU-aware logging."""

    def __init__(self, allow_unknown_gpu: bool = False):
        """Initialize the OpenslothLogger with specified log level and GPU awareness."""
        self.allow_unknown_gpu = (
            allow_unknown_gpu  # allow to run without setting OPENSLOTH_LOCAL_RANK
        )
        self.log_level = os.environ.get("OPENSLOTH_LOG_LEVEL", "INFO").upper()
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(
                f"Invalid log level: {self.log_level}. Must be one of DEBUG, INFO, WARNING, ERROR, CRITICAL."
            )
        self.console = Console()

        # Timing tracking
        self.step_timers: dict[str, StepTimer] = {}
        self.step_durations: dict[str, list] = {}  # Store history of durations
        self.total_training_start: float | None = None

        self._setup_logger()

    @property
    def gpu_id(self) -> str:
        id = os.environ.get("OPENSLOTH_LOCAL_RANK", "UNSET")
        if id == "UNSET" and not self.allow_unknown_gpu:
            raise ValueError(
                'Both "OPENSLOTH_LOCAL_RANK" is not set and "allow_unknown_gpu" is False. '
                "Please set the environment variable or allow unknown GPU."
            )
        return id

    def _should_log_to_stderr(self) -> bool:
        """Determine if stderr logging should be enabled based on tmux mode and rank."""
        # Check if we're in tmux mode
        use_tmux = os.environ.get("USE_TMUX")
        is_tmux_mode = use_tmux == "1"

        # Get local rank for distributed training
        local_rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))

        # In tmux mode, all ranks log to stderr
        if is_tmux_mode:
            return True

        # In non-tmux mode, only rank 0 logs to stderr
        return local_rank == 0

    def _setup_logger(self) -> None:
        """Setup loguru logger with enhanced formatting and mode-aware configuration."""
        from loguru import logger as base_logger

        self.logger = base_logger.bind(gpu_id=self.gpu_id)
        self.logger.remove()
        del base_logger  # Avoid confusion with loguru's logger
        log_format = (
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>GPU{extra[gpu_id]}</cyan> | "
            "<cyan>{file}:{line}</cyan> | "
            "<level>{message}</level>"
        )

        # Check if we should add stderr handler based on tmux mode and rank
        should_log_to_stderr = self._should_log_to_stderr()

        if should_log_to_stderr:
            self.logger.add(
                sys.stderr,
                format=log_format,
                level=self.log_level,
                colorize=True,
                enqueue=True,
            )

        # File handlers for rank-specific logs (always add these)
        try:
            output_dir = os.environ["OPENSLOTH_OUTPUT_DIR"]
            
            # Create logs directory
            log_dir = os.path.join(output_dir, "logs")
            os.makedirs(log_dir, exist_ok=True)
            
            # Rank-specific log file
            rank_log_file = os.path.join(log_dir, f"gpu_{self.gpu_id}.log")
            self.logger.add(
                rank_log_file,
                format=log_format,
                level="DEBUG",
                rotation="10 MB",
                retention="1 week",
                enqueue=True,
            )
            
            # Legacy training.log for backward compatibility (only on main rank)
            if self.gpu_id == "0" or self.gpu_id == 0:
                legacy_log_file = os.path.join(output_dir, "training.log")
                self.logger.add(
                    legacy_log_file,
                    format=log_format,
                    level="DEBUG",
                    rotation="10 MB",
                    retention="1 week",
                    enqueue=True,
                )
                
        except KeyError:
            # OPENSLOTH_OUTPUT_DIR not set, skip file logging
            pass

    def _log_with_depth(self, level: str, message: str, depth: int = 2) -> None:
        """Log message with loguru's built-in caller information."""
        # Convert level to uppercase since loguru levels are case-sensitive
        level_upper = level.upper()
        self.logger.opt(depth=depth).log(level_upper, message)

    # === TIMING METHODS ===
    def start_timing(self, step_name: str) -> None:
        """Start timing a major step."""
        self.step_timers[step_name] = StepTimer(step_name)
        if step_name not in self.step_durations:
            self.step_durations[step_name] = []

        self._log_with_depth("debug", f"⏱️  Started timing: {step_name}", depth=2)

    def finish_timing(self, step_name: str, log_result: bool = True) -> float:
        """Finish timing a step and optionally log the result."""
        if step_name not in self.step_timers:
            self._log_with_depth(
                "warning", f"⚠️  Timer '{step_name}' was not started", depth=2
            )
            return 0.0

        timer = self.step_timers[step_name]
        duration = timer.finish()
        self.step_durations[step_name].append(duration)

        if log_result:
            self._log_step_duration(step_name, duration)

        # Clean up the timer
        del self.step_timers[step_name]
        return duration

    def _log_step_duration(self, step_name: str, duration: float) -> None:
        """Log the duration of a completed step."""
        # Skip logging very short durations (less than 0.5 seconds) to reduce noise
        if duration < DURATION_SHORT_SKIP:
            return
        if duration < DURATION_SECONDS_IN_MINUTE:
            duration_str = f"{duration:.2f}s"
        elif duration < DURATION_SECONDS_IN_HOUR:
            duration_str = f"{duration/60:.1f}m"
        else:
            duration_str = f"{duration/3600:.1f}h"

        self._log_with_depth("info", f"⏱️  {step_name}: {duration_str}", depth=2)

    def start_total_training_timer(self) -> None:
        """Start the total training timer."""
        self.total_training_start = time.time()
        self._log_with_depth("info", "🚀 Starting total training timer", depth=2)

    def log_training_summary(self) -> None:
        """Log a summary of all timing information."""
        if not self.step_durations:
            return

        if self.gpu_id == "0":  # Only master GPU logs summary
            table = Table(
                title="[bold green]⏱️  Training Step Timing Summary[/bold green]",
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Step", style="cyan", width=25)
            table.add_column("Count", style="yellow", width=8)
            table.add_column("Avg Duration", style="green", width=12)
            table.add_column("Total Duration", style="blue", width=12)
            table.add_column("Min/Max", style="magenta", width=15)

            total_time = 0.0
            for step_name, durations in self.step_durations.items():
                if not durations:
                    continue

                count = len(durations)
                avg_duration = sum(durations) / count
                total_duration = sum(durations)
                min_duration = min(durations)
                max_duration = max(durations)

                total_time += total_duration

                # Format durations
                def format_duration(dur: float) -> str:
                    if dur < DURATION_MS_THRESHOLD:
                        return f"{dur*1000:.1f}ms"
                    elif dur < DURATION_SECONDS_IN_MINUTE:
                        return f"{dur:.2f}s"
                    elif dur < DURATION_SECONDS_IN_HOUR:
                        return f"{dur/60:.1f}m"
                    else:
                        return f"{dur/3600:.1f}h"

                table.add_row(
                    step_name,
                    str(count),
                    format_duration(avg_duration),
                    format_duration(total_duration),
                    f"{format_duration(min_duration)}/{format_duration(max_duration)}",
                )

            # Add total training time if available
            if self.total_training_start:
                total_training_time = time.time() - self.total_training_start
                table.add_row(
                    "[bold]TOTAL TRAINING[/bold]",
                    "-",
                    "-",
                    f"[bold]{self._format_duration(total_training_time)}[/bold]",
                    "-",
                )

            self.console.print(table)

    def log_step_timing_progress(
        self, step_name: str, current_step: int, total_steps: int
    ) -> None:
        """Log timing progress for steps showing average and estimated remaining time."""
        if step_name not in self.step_durations or not self.step_durations[step_name]:
            return

        durations = self.step_durations[step_name]
        avg_duration = sum(durations) / len(durations)
        remaining_steps = total_steps - current_step
        estimated_remaining = avg_duration * remaining_steps

        progress_msg = (
            f"📊 {step_name} Progress: {current_step}/{total_steps} "
            f"(Avg: {self._format_duration(avg_duration)}, "
            f"ETA: {self._format_duration(estimated_remaining)})"
        )

        if current_step % 10 == 0 or current_step == total_steps:  # Log every 10 steps
            self._log_with_depth("info", progress_msg, depth=2)

    def _format_duration(self, duration: float) -> str:
        """Format duration consistently."""
        if duration < DURATION_MS_THRESHOLD:
            return f"{duration*1000:.1f}ms"
        elif duration < DURATION_SECONDS_IN_MINUTE:
            return f"{duration:.2f}s"
        elif duration < DURATION_SECONDS_IN_HOUR:
            return f"{duration/60:.1f}m"
        else:
            return f"{duration/3600:.1f}h"

    def info(self, message: str) -> None:
        """Log info message with GPU context."""
        self._log_with_depth("info", message, depth=2)

    def debug(self, message: str) -> None:
        """Log debug message with GPU context."""
        self._log_with_depth("debug", message, depth=2)

    def warning(self, message: str) -> None:
        """Log warning message with GPU context."""
        self._log_with_depth("warning", message, depth=2)

    def error(self, message: str) -> None:
        """Log error message with GPU context."""
        self._log_with_depth("error", message, depth=2)


VALID_LOGGER = None


def get_opensloth_logger(allow_unknown_gpu=False) -> OpenslothLogger:
    """Setup and return enhanced logger instance."""
    if get_opensloth_logger.VALID_LOGGER is not None:
        return get_opensloth_logger.VALID_LOGGER

    logger = OpenslothLogger(allow_unknown_gpu=allow_unknown_gpu)
    if not allow_unknown_gpu:
        get_opensloth_logger.VALID_LOGGER = logger
    return logger

get_opensloth_logger.VALID_LOGGER = None
