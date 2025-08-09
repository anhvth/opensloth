#!/usr/bin/env python3
"""
Test script for GRPO improvements.
Tests the new demonstration output and improved configuration.
"""

import sys
import os
sys.path.insert(0, 'src')

def test_demo_reward():
    """Test the demo reward function."""
    print("🧪 Testing DemoReward function...")
    
    from opensloth.grpo_rewards import get_reward_function
    
    # Get the demo reward function
    demo_reward = get_reward_function("demo_reward")
    
    # Mock some sample data
    prompts = [
        [{"role": "user", "content": "What is 2 + 2?"}]
    ]
    completions = [
        [{"content": "The answer is 4. Let me explain: 2 + 2 = 4 because when you have two items and add two more items, you get four items total."}]
    ]
    kwargs = {"answer": ["4"]}
    
    # Test the reward function (should print demo output)
    scores = demo_reward(prompts, completions, **kwargs)
    
    print(f"✅ Demo reward returned scores: {scores}")
    return True

def test_reward_presets():
    """Test that reward presets include demo_reward."""
    print("🧪 Testing reward presets...")
    
    from opensloth.grpo_rewards import create_reward_preset
    
    task_types = ["math", "code", "general", "reasoning"]
    
    for task_type in task_types:
        preset = create_reward_preset(task_type)
        if "demo_reward" in preset:
            print(f"✅ {task_type} preset includes demo_reward: {preset}")
        else:
            print(f"❌ {task_type} preset missing demo_reward: {preset}")
            return False
    
    return True

def test_list_rewards():
    """Test that demo_reward is in the available rewards."""
    print("🧪 Testing reward function listing...")
    
    from opensloth.grpo_rewards import list_reward_functions
    
    rewards = list_reward_functions()
    if "demo_reward" in rewards:
        print(f"✅ demo_reward found in available rewards: {rewards}")
        return True
    else:
        print(f"❌ demo_reward missing from available rewards: {rewards}")
        return False

def main():
    print("🚀 Testing GRPO improvements...")
    print("=" * 60)
    
    tests = [
        test_list_rewards,
        test_reward_presets,
        test_demo_reward,
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with error: {e}")
            print()
    
    print("=" * 60)
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! GRPO improvements are working.")
        return 0
    else:
        print("💥 Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
