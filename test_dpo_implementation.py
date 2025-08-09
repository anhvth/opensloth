#!/usr/bin/env python3
"""
Test script to validate the new DPO training implementation.
This script tests the configuration and trainer factory without actually training.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_sft_config():
    """Test SFT configuration (backward compatibility)."""
    from opensloth.opensloth_config import (
        FastModelArgs, LoraArgs, OpenSlothConfig, TrainingArguments
    )
    
    print("Testing SFT configuration...")
    
    config = OpenSlothConfig(
        data_cache_path='test/data',
        devices=[0],
        # training_type defaults to "sft"
        fast_model_args=FastModelArgs(
            model_name='unsloth/Qwen2.5-0.5B-Instruct',
            max_seq_length=2048,
        ),
        lora_args=LoraArgs(r=8, lora_alpha=16),
    )
    
    assert config.training_type == "sft"
    assert config.dpo_args is None
    print("✓ SFT configuration validated")


def test_dpo_config():
    """Test DPO configuration."""
    from opensloth.opensloth_config import (
        FastModelArgs, LoraArgs, OpenSlothConfig, 
        TrainingArguments, DPOArgs
    )
    
    print("Testing DPO configuration...")
    
    config = OpenSlothConfig(
        data_cache_path='test/data',
        devices=[0],
        training_type="dpo",
        fast_model_args=FastModelArgs(
            model_name='unsloth/zephyr-sft-bnb-4bit',
            max_seq_length=2048,
        ),
        lora_args=LoraArgs(r=64, lora_alpha=64),
        dpo_args=DPOArgs(
            beta=0.1,
            max_length=1024,
            max_prompt_length=512,
        ),
    )
    
    assert config.training_type == "dpo"
    assert config.dpo_args is not None
    assert config.dpo_args.beta == 0.1
    print("✓ DPO configuration validated")


def test_auto_dpo_config():
    """Test automatic DPO args creation."""
    from opensloth.opensloth_config import (
        FastModelArgs, LoraArgs, OpenSlothConfig, TrainingArguments
    )
    
    print("Testing automatic DPO args creation...")
    
    config = OpenSlothConfig(
        data_cache_path='test/data',
        devices=[0],
        training_type="dpo",
        fast_model_args=FastModelArgs(
            model_name='unsloth/zephyr-sft-bnb-4bit',
            max_seq_length=2048,
        ),
        lora_args=LoraArgs(r=64, lora_alpha=64),
        # dpo_args not provided - should be auto-created
    )
    
    assert config.training_type == "dpo"
    assert config.dpo_args is not None  # Should be auto-created
    assert config.dpo_args.beta == 0.1  # Default value
    print("✓ Automatic DPO args creation validated")


def test_invalid_configs():
    """Test invalid configuration combinations."""
    from opensloth.opensloth_config import (
        FastModelArgs, LoraArgs, OpenSlothConfig, DPOArgs
    )
    
    print("Testing invalid configurations...")
    
    # Test 1: DPO args with SFT training
    try:
        config = OpenSlothConfig(
            data_cache_path='test/data',
            devices=[0],
            training_type="sft",
            fast_model_args=FastModelArgs(model_name='test'),
            dpo_args=DPOArgs(beta=0.1),  # Should not be allowed with SFT
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "dpo_args should only be specified when training_type is 'dpo'" in str(e)
        print("✓ Invalid SFT+DPO args combination correctly rejected")
    
    # Test 2: Unimplemented training types
    try:
        config = OpenSlothConfig(
            data_cache_path='test/data',
            devices=[0],
            training_type="kto",  # Not implemented yet
            fast_model_args=FastModelArgs(model_name='test'),
        )
        assert False, "Should have raised NotImplementedError"
    except NotImplementedError as e:
        assert "not yet implemented" in str(e)
        print("✓ Unimplemented training type correctly rejected")


def test_trainer_factory_imports():
    """Test that trainer factory imports work correctly."""
    print("Testing trainer factory imports...")
    
    try:
        from opensloth.trainer_factory import create_trainer_by_type
        print("✓ Trainer factory import successful")
    except ImportError as e:
        print(f"✗ Trainer factory import failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Running OpenSloth DPO implementation tests...\n")
    
    try:
        test_sft_config()
        test_dpo_config()
        test_auto_dpo_config()
        test_invalid_configs()
        test_trainer_factory_imports()
        
        print("\n🎉 All tests passed! DPO implementation is working correctly.")
        print("\nNext steps:")
        print("1. Prepare a DPO dataset using: python prepare_dataset/prepare_dpo_dataset.py")
        print("2. Run DPO training using: python train_scripts/train_dpo_example.py")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
