#!/usr/bin/env python3
"""
Test script to verify that Hugging Face logging is properly intercepted by loguru.
"""

import os
import tempfile
import sys

# Setup test environment
test_output_dir = tempfile.mkdtemp()
os.environ["OPENSLOTH_OUTPUT_DIR"] = test_output_dir
os.environ["OPENSLOTH_LOCAL_RANK"] = "0"
os.environ["OPENSLOTH_LOG_LEVEL"] = "INFO"

print(f"Test output directory: {test_output_dir}")

# Import and setup logging interception
from src.opensloth.logging_config import setup_huggingface_logging_interception, get_opensloth_logger

print("Setting up Hugging Face logging interception...")
setup_huggingface_logging_interception()

# Get OpenSloth logger
logger = get_opensloth_logger(allow_unknown_gpu=True)
logger.info("OpenSloth logger initialized successfully")

print("Testing standard logging interception...")

# Test standard logging (what Hugging Face libraries use)
import logging

# Create some test loggers similar to what Hugging Face libraries use
transformers_logger = logging.getLogger("transformers")
datasets_logger = logging.getLogger("datasets")
trl_logger = logging.getLogger("trl")
accelerate_logger = logging.getLogger("accelerate")

print("Sending test messages through standard logging (should appear in loguru format):")

# Test different log levels
transformers_logger.info("This is a test INFO message from transformers")
datasets_logger.warning("This is a test WARNING message from datasets")
trl_logger.error("This is a test ERROR message from trl")
accelerate_logger.debug("This is a test DEBUG message from accelerate")

# Test generic logging
root_logger = logging.getLogger()
root_logger.info("This is a test message from root logger")

print("Test messages sent. Check log files and console output above.")
print(f"Log files should be in: {test_output_dir}")

# List log files created
import glob
log_files = glob.glob(os.path.join(test_output_dir, "**", "*.log"), recursive=True)
if log_files:
    print(f"Log files created: {log_files}")
    for log_file in log_files:
        print(f"\nContents of {log_file}:")
        try:
            with open(log_file, 'r') as f:
                print(f.read())
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
else:
    print("No log files found")

print("Test completed!")
