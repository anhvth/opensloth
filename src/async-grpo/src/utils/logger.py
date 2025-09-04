import sys
import os
from loguru import logger

# Remove the default handler to avoid duplicate logs
logger.remove()

# Get log level from environment variable, default to INFO
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()

# Add a new handler with a custom format
logger.add(
    sys.stderr,
    level=log_level,
    format="<green>{time:HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{file.name}:{line:<4}</cyan> - <level>{message}</level>",
    colorize=True,
)

# Export the configured logger
__all__ = ["logger"]