#!/usr/bin/env python3
"""
Integration test to verify that multi-GPU GRPO training would use smart aggregation.
This simulates what would happen during actual training.
"""

import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

# Add the src directory to the path so we can import opensloth modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_grpo_training_scenario():
    """
    Simulate a realistic GRPO training scenario with multiple GPUs 
    and verify that metrics would be aggregated correctly.
    """
    print("🧪 Testing GRPO Multi-GPU Training Scenario")
    
    # Mock GRPO metrics that would be logged during actual training
    grpo_metrics_from_training = {
        # Standard training metrics
        "loss": [0.432, 0.456, 0.411, 0.423],  # 4 GPUs
        "learning_rate": [0.0002, 0.0002, 0.0002, 0.0002],  # Should be same
        "grad_norm": [1.23, 1.45, 1.12, 1.34],
        
        # GRPO-specific reward metrics  
        "reward": [0.724, 0.689, 0.712, 0.695],  # Group-wise normalized rewards
        "reward_std": [0.123, 0.145, 0.134, 0.129],
        
        # KL divergence
        "kl": [0.051, 0.049, 0.053, 0.048],
        
        # Policy metrics
        "entropy": [2.45, 2.52, 2.41, 2.49],
        "clip_ratio": [0.12, 0.15, 0.11, 0.14],
        
        # Completion statistics
        "completions/mean_length": [45.2, 43.8, 46.1, 44.5],
        "completions/min_length": [12, 15, 10, 14],  # Should use min
        "completions/max_length": [128, 125, 127, 126],  # Should use max
        "completions/clipped_ratio": [0.023, 0.019, 0.025, 0.021],
        
        # Custom reward functions (dynamic)
        "rewards/math_accuracy/mean": [0.834, 0.821, 0.845, 0.829],
        "rewards/math_accuracy/std": [0.156, 0.167, 0.149, 0.162],
        "rewards/code_quality/mean": [0.712, 0.698, 0.725, 0.706],
        
        # Token counting (should be summed)
        "num_input_tokens_seen": [15420, 15380, 15440, 15400],
        "num_tokens": [892, 876, 905, 883],
    }
    
    world_size = 4
    expected_aggregations = {}
    
    # Calculate expected results
    for metric, values in grpo_metrics_from_training.items():
        from opensloth.patching.patch_log_for_multi_gpu import AGGREGATION_STRATEGIES, DEFAULT_AGGREGATION
        
        strategy = AGGREGATION_STRATEGIES.get(metric, DEFAULT_AGGREGATION)
        
        # Handle dynamic reward function keys
        if strategy == DEFAULT_AGGREGATION and metric.startswith("rewards/") and "/mean" in metric:
            strategy = "mean"
        elif strategy == DEFAULT_AGGREGATION and metric.startswith("rewards/") and "/std" in metric:
            strategy = "mean"
            
        if strategy == "mean":
            expected_aggregations[metric] = np.mean(values)
        elif strategy == "sum":
            expected_aggregations[metric] = np.sum(values)
        elif strategy == "min":
            expected_aggregations[metric] = np.min(values)
        elif strategy == "max":
            expected_aggregations[metric] = np.max(values)
        else:
            expected_aggregations[metric] = np.mean(values)  # fallback
    
    # Test the aggregation
    from opensloth.patching.patch_log_for_multi_gpu import _aggregate_logs, _initialize_mmaps_dynamically, FileLock
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_mmap = {}
        log_locks = {}
        
        # Initialize mmaps for all metrics
        test_keys = list(grpo_metrics_from_training.keys())
        _initialize_mmaps_dynamically(
            test_keys, tmpdir, world_size, 0, True, log_mmap, log_locks
        )
        
        # Populate with test data
        for metric, values in grpo_metrics_from_training.items():
            if metric in log_mmap:
                with log_locks[metric]:
                    for gpu_id, value in enumerate(values):
                        log_mmap[metric][gpu_id] = value
                    log_mmap[metric].flush()
        
        # Aggregate (using GPU 0's values as base logs)
        base_logs = {metric: values[0] for metric, values in grpo_metrics_from_training.items()}
        aggregated = _aggregate_logs(base_logs, test_keys, log_mmap, log_locks)
        
        # Verify results
        all_passed = True
        print("\n📊 Metric Aggregation Results:")
        print("=" * 80)
        
        for metric in sorted(grpo_metrics_from_training.keys()):
            strategy = AGGREGATION_STRATEGIES.get(metric, DEFAULT_AGGREGATION)
            if strategy == DEFAULT_AGGREGATION and metric.startswith("rewards/"):
                strategy = "mean"  # Dynamic handling
                
            values = grpo_metrics_from_training[metric]
            expected = expected_aggregations[metric]
            actual = aggregated[metric]
            
            # Format output
            values_str = f"[{', '.join(f'{v:.3f}' for v in values)}]"
            
            print(f"{metric:35} | {strategy:>4} | {values_str:>30} -> {actual:8.3f}")
            
            if abs(actual - expected) > 1e-4:  # More lenient for floating point
                print(f"    ❌ FAILED: Expected {expected:.6f}, got {actual:.6f}")
                all_passed = False
            
        print("=" * 80)
        
        if all_passed:
            print("✅ All GRPO metrics would be aggregated correctly!")
            print(f"📈 Verified {len(grpo_metrics_from_training)} metrics with smart aggregation")
            
            # Show some key insights
            print("\n🔍 Key Insights:")
            print(f"  • Loss averaging: {grpo_metrics_from_training['loss']} -> {aggregated['loss']:.3f}")
            print(f"  • Reward averaging: {grpo_metrics_from_training['reward']} -> {aggregated['reward']:.3f}")
            print(f"  • Token summing: {grpo_metrics_from_training['num_input_tokens_seen']} -> {int(aggregated['num_input_tokens_seen'])}")
            print(f"  • Min length: {grpo_metrics_from_training['completions/min_length']} -> {int(aggregated['completions/min_length'])}")
            print(f"  • Max length: {grpo_metrics_from_training['completions/max_length']} -> {int(aggregated['completions/max_length'])}")
            
        return all_passed

def main():
    """Run the integration test."""
    print("🚀 GRPO Multi-GPU Smart Aggregation Integration Test\n")
    
    try:
        success = test_grpo_training_scenario()
        
        if success:
            print("\n🎉 Integration test PASSED!")
            print("✨ Multi-GPU GRPO training will use correct metric aggregation")
            return 0
        else:
            print("\n❌ Integration test FAILED!")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
