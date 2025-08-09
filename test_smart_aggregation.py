#!/usr/bin/env python3
"""
Test script to verify the smart aggregation functionality works correctly.
"""

import sys
import os
import numpy as np
import tempfile
from pathlib import Path

# Add the src directory to the path so we can import opensloth modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from opensloth.patching.patch_log_for_multi_gpu import (
    AGGREGATION_STRATEGIES, 
    DEFAULT_AGGREGATION,
    _aggregate_logs,
    _initialize_mmaps_dynamically,
    FileLock
)

def test_aggregation_strategies():
    """Test that different aggregation strategies work correctly."""
    print("🧪 Testing aggregation strategies...")
    
    # Create mock data for 2 GPUs
    world_size = 2
    
    # Test metrics with different strategies
    test_cases = [
        # (metric_name, [gpu0_value, gpu1_value], expected_result, expected_strategy)
        ("loss", [0.5, 0.3], 0.4, "mean"),  # (0.5 + 0.3) / 2 = 0.4
        ("rewards/chosen", [1.2, 0.8], 1.0, "mean"),  # (1.2 + 0.8) / 2 = 1.0
        ("num_input_tokens_seen", [1000, 2000], 3000, "sum"),  # 1000 + 2000 = 3000
        ("completions/min_length", [10, 5], 5, "min"),  # min(10, 5) = 5
        ("completions/max_length", [20, 30], 30, "max"),  # max(20, 30) = 30
        ("unknown_metric", [2.0, 4.0], 3.0, "mean"),  # Default to mean: (2.0 + 4.0) / 2 = 3.0
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mmap = {}
        log_locks = {}
        
        # Initialize mmaps for all test metrics
        test_keys = [case[0] for case in test_cases]
        _initialize_mmaps_dynamically(
            test_keys, tmpdir, world_size, 0, True, log_mmap, log_locks
        )
        
        # Populate the mmaps with test data
        for metric_name, values, expected, expected_strategy in test_cases:
            if metric_name in log_mmap:
                with log_locks[metric_name]:
                    for gpu_id, value in enumerate(values):
                        log_mmap[metric_name][gpu_id] = value
                    log_mmap[metric_name].flush()
        
        # Test aggregation
        dummy_logs = {case[0]: case[1][0] for case in test_cases}  # Use GPU 0 values
        aggregated = _aggregate_logs(dummy_logs, test_keys, log_mmap, log_locks)
        
        # Verify results
        all_passed = True
        for metric_name, values, expected, expected_strategy in test_cases:
            actual = aggregated[metric_name]
            strategy = AGGREGATION_STRATEGIES.get(metric_name, DEFAULT_AGGREGATION)
            
            print(f"  📊 {metric_name}: {values} -> {actual} (expected: {expected}, strategy: {strategy})")
            
            if abs(actual - expected) > 1e-6:
                print(f"    ❌ FAILED: Expected {expected}, got {actual}")
                all_passed = False
            else:
                print(f"    ✅ PASSED")
        
        return all_passed

def test_dynamic_reward_function_handling():
    """Test that dynamic reward function keys are handled correctly."""
    print("\n🧪 Testing dynamic reward function handling...")
    
    test_cases = [
        ("rewards/custom_math_reward/mean", [0.8, 0.6], 0.7, "mean"),
        ("rewards/custom_math_reward/std", [0.1, 0.2], 0.15, "mean"),
        ("rewards/unknown_reward/mean", [1.0, 2.0], 1.5, "mean"),
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mmap = {}
        log_locks = {}
        world_size = 2
        
        test_keys = [case[0] for case in test_cases]
        _initialize_mmaps_dynamically(
            test_keys, tmpdir, world_size, 0, True, log_mmap, log_locks
        )
        
        # Populate test data
        for metric_name, values, expected, expected_strategy in test_cases:
            if metric_name in log_mmap:
                with log_locks[metric_name]:
                    for gpu_id, value in enumerate(values):
                        log_mmap[metric_name][gpu_id] = value
                    log_mmap[metric_name].flush()
        
        # Test aggregation
        dummy_logs = {case[0]: case[1][0] for case in test_cases}
        aggregated = _aggregate_logs(dummy_logs, test_keys, log_mmap, log_locks)
        
        all_passed = True
        for metric_name, values, expected, expected_strategy in test_cases:
            actual = aggregated[metric_name]
            print(f"  📊 {metric_name}: {values} -> {actual} (expected: {expected})")
            
            if abs(actual - expected) > 1e-6:
                print(f"    ❌ FAILED: Expected {expected}, got {actual}")
                all_passed = False
            else:
                print(f"    ✅ PASSED")
        
        return all_passed

def test_file_naming_with_slashes():
    """Test that metrics with slashes in names get proper file names."""
    print("\n🧪 Testing file naming with slashes...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mmap = {}
        log_locks = {}
        world_size = 2
        
        # Test metrics with slashes
        slash_keys = ["rewards/chosen", "completions/mean_length", "clip_ratio/low_mean"]
        
        _initialize_mmaps_dynamically(
            slash_keys, tmpdir, world_size, 0, True, log_mmap, log_locks
        )
        
        # Check that files were created with proper names
        all_passed = True
        for key in slash_keys:
            expected_filename = f"log_{key.replace('/', '_')}.mmap"
            expected_path = Path(tmpdir) / expected_filename
            
            if expected_path.exists():
                print(f"  ✅ {key} -> {expected_filename}")
            else:
                print(f"  ❌ FAILED: {expected_filename} not found")
                all_passed = False
        
        return all_passed

def main():
    """Run all tests."""
    print("🚀 Testing Smart Aggregation Patch Implementation\n")
    
    tests = [
        test_aggregation_strategies,
        test_dynamic_reward_function_handling,
        test_file_naming_with_slashes,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Smart aggregation is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
