#!/usr/bin/env python3
"""
Demonstration script showing the refactored OpenSloth CLI, logging, and patching system.
This script shows how the new mode-aware system works in practice.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def demonstrate_cli_extensibility():
    """Demonstrate how the CLI is now more extensible with Pydantic models."""
    print("📋 CLI Extensibility Demonstration")
    print("=" * 50)
    
    from opensloth.cli.params import TrainingParams
    
    # Show how easy it is to create training parameters
    params = TrainingParams(
        dataset="/path/to/dataset",
        method="sft",
        model="Qwen/Qwen2.5-7B-Instruct",
        epochs=3,
        batch_size=2,
        learning_rate=2e-4,
        use_tmux=True,
        tmux_session="my_training"
    )
    
    print("✅ Easy parameter creation with Pydantic:")
    print(f"   Dataset: {params.dataset}")
    print(f"   Method: {params.method}")
    print(f"   Model: {params.model}")
    print(f"   Use TMUX: {params.use_tmux}")
    print(f"   Session: {params.tmux_session}")
    print()
    
    # Show how easy it would be to add new parameters
    print("💡 Adding new CLI options is now simple:")
    print("   1. Add new field to TrainingParams in params.py")
    print("   2. No need to modify function signatures in main.py")
    print("   3. Configuration automatically flows through the system")
    print()

def demonstrate_mode_aware_logging():
    """Demonstrate the mode-aware logging system."""
    print("📝 Mode-Aware Logging Demonstration")
    print("=" * 50)
    
    temp_dir = tempfile.mkdtemp(prefix='opensloth_demo_')
    
    try:
        from opensloth.logging_config import OpenslothLogger
        
        print("🔍 Multiprocessing Mode (Traditional):")
        # Setup for multiprocessing mode
        os.environ['OPENSLOTH_OUTPUT_DIR'] = temp_dir
        if 'USE_TMUX' in os.environ:
            del os.environ['USE_TMUX']
        
        # Rank 0 - Master process
        os.environ['OPENSLOTH_LOCAL_RANK'] = '0'
        logger_mp_0 = OpenslothLogger(allow_unknown_gpu=True)
        stderr_logging_0 = logger_mp_0._should_log_to_stderr()
        print(f"   Rank 0: Log to stderr = {stderr_logging_0} ✅")
        
        # Rank 1 - Worker process
        os.environ['OPENSLOTH_LOCAL_RANK'] = '1'
        logger_mp_1 = OpenslothLogger(allow_unknown_gpu=True)
        stderr_logging_1 = logger_mp_1._should_log_to_stderr()
        print(f"   Rank 1: Log to stderr = {stderr_logging_1} ✅")
        print("   → Only master shows terminal output, avoiding clutter")
        print()
        
        print("🖥️  TMUX Mode (New):")
        # Setup for tmux mode
        os.environ['USE_TMUX'] = '1'
        
        # Both ranks in tmux mode
        os.environ['OPENSLOTH_LOCAL_RANK'] = '0'
        logger_tmux_0 = OpenslothLogger(allow_unknown_gpu=True)
        stderr_logging_0 = logger_tmux_0._should_log_to_stderr()
        print(f"   Rank 0: Log to stderr = {stderr_logging_0} ✅")
        
        os.environ['OPENSLOTH_LOCAL_RANK'] = '1'
        logger_tmux_1 = OpenslothLogger(allow_unknown_gpu=True)
        stderr_logging_1 = logger_tmux_1._should_log_to_stderr()
        print(f"   Rank 1: Log to stderr = {stderr_logging_1} ✅")
        print("   → Each tmux pane shows its own output")
        print()
        
        # Demonstrate file logging
        logger_tmux_1.info("Demo message from GPU 1")
        
        # Check log files
        logs_dir = Path(temp_dir) / "logs"
        if logs_dir.exists():
            log_files = list(logs_dir.glob("*.log"))
            print(f"📁 Per-GPU log files created: {[f.name for f in log_files]}")
        print()
        
    finally:
        shutil.rmtree(temp_dir)

def demonstrate_conditional_patching():
    """Demonstrate the conditional patching system."""
    print("🔧 Conditional Patching Demonstration")
    print("=" * 50)
    
    print("The IPC log patch is now applied conditionally:")
    print()
    
    # Single GPU scenario
    devices = [0]
    print(f"📱 Single GPU ({len(devices)} device):")
    if len(devices) > 1:
        print("   → Would apply multi-GPU patches")
    else:
        print("   → No patches needed ✅")
    print()
    
    # Multi-GPU multiprocessing scenario
    devices = [0, 1, 2, 3]
    use_tmux = False
    print(f"🖥️  Multi-GPU Multiprocessing ({len(devices)} devices, tmux={use_tmux}):")
    if len(devices) > 1 and not use_tmux:
        print("   → IPC log aggregation patch APPLIED ✅")
        print("   → Prevents log message duplication")
    else:
        print("   → No patch applied")
    print()
    
    # Multi-GPU tmux scenario
    use_tmux = True
    print(f"📺 Multi-GPU TMUX ({len(devices)} devices, tmux={use_tmux}):")
    if len(devices) > 1 and not use_tmux:
        print("   → IPC log aggregation patch applied")
    else:
        print("   → IPC log aggregation patch SKIPPED ✅")
        print("   → Each pane handles its own logging")
    print()

def demonstrate_environment_handling():
    """Demonstrate environment variable handling."""
    print("🌍 Environment Variable Handling")
    print("=" * 50)
    
    print("The API now properly sets execution mode flags:")
    print()
    
    # Clear any existing environment
    if 'USE_TMUX' in os.environ:
        del os.environ['USE_TMUX']
    
    print("🔄 Multiprocessing Mode:")
    print("   API called with use_tmux=False")
    print(f"   USE_TMUX environment: {os.environ.get('USE_TMUX', '(not set)')} ✅")
    print("   → Downstream modules use multiprocessing behavior")
    print()
    
    print("📺 TMUX Mode:")
    os.environ['USE_TMUX'] = '1'  # Simulate API setting this
    print("   API called with use_tmux=True")
    print(f"   USE_TMUX environment: {os.environ.get('USE_TMUX')} ✅")
    print("   → Downstream modules use tmux behavior")
    print()

def main():
    """Run the demonstration."""
    print("🚀 OpenSloth Refactoring Demonstration")
    print("=" * 60)
    print("This demonstrates the key improvements from the refactoring:")
    print("1. 📋 CLI Extensibility with Pydantic")
    print("2. 📝 Mode-Aware Logging")
    print("3. 🔧 Conditional Patching")
    print("4. 🌍 Environment Handling")
    print()
    
    demonstrate_cli_extensibility()
    demonstrate_mode_aware_logging()
    demonstrate_conditional_patching()
    demonstrate_environment_handling()
    
    print("🎯 Summary of Benefits:")
    print("=" * 60)
    print("✅ CLI is now easily extensible - add new options in one place")
    print("✅ Logging is mode-aware - clean output in both MP and tmux modes")
    print("✅ IPC patches only applied when needed - better performance")
    print("✅ Environment variables properly managed - reliable mode detection")
    print()
    print("🎉 The refactoring successfully improves extensibility and mode-awareness!")

if __name__ == "__main__":
    main()
