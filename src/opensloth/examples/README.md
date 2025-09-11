"""
Examples module for OpenSloth configurations.

This module provides pre-configured examples for dataset preparation and training
with different setups (single GPU, multi-GPU, debug, etc.).

All configuration functions now return Pydantic models for type safety and validation.

Usage:
from opensloth.examples import qwen_config_2gpus
config = qwen_config_2gpus(dataset_name='your-dataset')
prepare_qwen_dataset(config)

Available configurations:

- qwen_config_1gpu: Single GPU setup
- qwen_config_2gpus: 2-GPU setup
- qwen_config_4gpus: 4-GPU setup
- qwen_config_debug: Debug configuration with small dataset
- qwen_config_full_finetuning: Full fine-tuning (no LoRA)

You can also create custom configurations:
from opensloth.examples.qwen_configs import qwen_config_1gpu
config = qwen_config_1gpu(
dataset_name='mlabonne/FineTome-100k',
num_samples=1000,
max_seq_length=8192
)

The configurations are now type-safe Pydantic models with automatic validation.
"""
