#!/usr/bin/env python3
"""Test script to verify the ShuffleData callback fix."""

import os
import sys
sys.path.insert(0, 'src')

# Set dummy environment variables for testing
os.environ["OPENSLOTH_LOCAL_RANK"] = "0" 
os.environ["OPENSLOTH_WORLD_SIZE"] = "1"

from opensloth.patching.patch_sampler import ShuffleData
from transformers.trainer_callback import TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

def test_shuffle_data_callback():
    """Test that ShuffleData callback accepts the correct arguments."""
    print("Testing ShuffleData callback...")
    
    # Create a dummy callback instance
    callback = ShuffleData()
    
    # Create dummy arguments that the transformers library would pass
    args = TrainingArguments(output_dir="/tmp/test", per_device_train_batch_size=1)
    state = TrainerState()
    state.epoch = 0
    control = TrainerControl()
    
    # Test that the callback method accepts the correct arguments
    try:
        callback.on_epoch_begin(args, state, control, model=None)
        print("✅ SUCCESS: ShuffleData.on_epoch_begin() accepts correct arguments")
        return True
    except Exception as e:
        print(f"❌ FAILED: ShuffleData.on_epoch_begin() error: {e}")
        return False

if __name__ == "__main__":
    success = test_shuffle_data_callback()
    sys.exit(0 if success else 1)
