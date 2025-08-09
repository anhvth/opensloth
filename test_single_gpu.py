#!/usr/bin/env python3

from opensloth.api import run_training
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments, FastModelArgs

# Simple test config
opensloth_config = OpenSlothConfig(
    data_cache_path='data/test_grpo_sharded',
    devices=[0],  # Single GPU
    training_type='sft',
    fast_model_args=FastModelArgs(
        model_name='./hf-models/unsloth/Qwen3-4B-Base-bnb-4bit'
    )
)

training_args = TrainingArguments(
    output_dir='outputs/test_single_gpu_mode',
    per_device_train_batch_size=1,
    max_steps=1,  # Just one step to test
    logging_steps=1
)

# Test single GPU mode (should use generate-then-execute)
print('Testing single GPU mode path...')
try:
    run_training(opensloth_config, training_args, use_tmux=False)
    print('Single GPU mode: SUCCESS')
except Exception as e:
    print(f'Single GPU mode: {type(e).__name__}: {e}')
