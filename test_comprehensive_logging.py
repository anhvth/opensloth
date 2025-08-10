#!/usr/bin/env python3

import os
import tempfile

def test_comprehensive_logging_interception():
    """Test the comprehensive logging interception"""
    
    # Setup test environment
    temp_dir = tempfile.mkdtemp(prefix='opensloth_test_')
    os.environ['OPENSLOTH_OUTPUT_DIR'] = temp_dir
    os.environ['OPENSLOTH_LOCAL_RANK'] = '0'
    
    print("Test output directory:", temp_dir)
    
    print("Setting up comprehensive logging interception...")
    
    # Import and setup the comprehensive logging
    from opensloth.logging_config import setup_comprehensive_logging_interception
    setup_comprehensive_logging_interception()
    
    print("Testing comprehensive logging interception...")
    
    # Test 1: Print statements should be intercepted
    print("This is a test print statement - should be intercepted")
    
    # Test 2: Standard logging should be intercepted  
    import logging
    logging.info("This is a test logging.info message")
    logging.warning("This is a test logging.warning message")
    
    # Test 3: Various library logging
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.info("This is a test transformers message")
    
    datasets_logger = logging.getLogger("datasets")
    datasets_logger.warning("This is a test datasets message")
    
    # Test 4: Direct stdout write
    import sys
    sys.stdout.write("Direct stdout write test\n")
    
    print("Test messages sent. Check log files...")
    
    # List log files created
    import glob
    log_files = glob.glob(os.path.join(temp_dir, "logs", "*.log"))
    print(f"Log files created: {log_files}")
    
    # Show contents of log files
    for log_file in log_files:
        print(f"\nContents of {log_file}:")
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                print(content)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    print("Test completed!")

if __name__ == "__main__":
    test_comprehensive_logging_interception()
