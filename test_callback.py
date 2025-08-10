#!/usr/bin/env python3

import os
import tempfile

def test_training_callback():
    """Test the OpenSloth training callback"""
    
    # Setup test environment
    temp_dir = tempfile.mkdtemp(prefix='opensloth_test_')
    os.environ['OPENSLOTH_OUTPUT_DIR'] = temp_dir
    os.environ['OPENSLOTH_LOCAL_RANK'] = '0'
    
    print("Test output directory:", temp_dir)
    
    # Import and test the callback
    from opensloth.logging_config import OpenSlothTrainingLogCallback
    
    # Create mock trainer state
    class MockState:
        global_step = 10
    
    # Create mock args
    class MockArgs:
        pass
    
    # Create callback
    callback = OpenSlothTrainingLogCallback()
    
    # Test logging
    test_logs = {
        'loss': 0.7577,
        'grad_norm': 0.40557193756103516,
        'learning_rate': 2e-05,
        'epoch': 0.05
    }
    
    print("Testing training callback with sample logs...")
    callback.on_log(MockArgs(), MockState(), None, test_logs)
    callback.close()
    
    # Check log files
    import glob
    log_files = glob.glob(os.path.join(temp_dir, "logs", "*.log"))
    print(f"Log files created: {log_files}")
    
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
    test_training_callback()
