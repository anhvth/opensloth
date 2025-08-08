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

from rich.console import Console
from rich.table import Table

# from speedy_utils import setup_logger
COUNT = 0


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
        """Setup loguru logger with enhanced formatting."""
        from loguru import logger as base_logger

        # global COUNT
        # COUNT += 1

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

        # File handler for individual GPU logs (always add these)
        try:
            log_file = os.path.join(os.environ["OPENSLOTH_OUTPUT_DIR"], "training.log")

            self.logger.add(
                log_file,
                format=log_format,
                level="DEBUG",
                rotation="10 MB",
                retention="1 week",
                enqueue=True,
            )
        except KeyError:
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
