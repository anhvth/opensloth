"""
Example configurations for OpenSloth dataset preparation and training.
"""

from .qwen_configs import (
    qwen_config_1gpu,
    qwen_config_2gpus,
    qwen_config_4gpus,
    qwen_config_debug,
    qwen_config_full_finetuning,
)

__all__ = [
    "qwen_config_1gpu",
    "qwen_config_2gpus", 
    "qwen_config_4gpus",
    "qwen_config_debug",
    "qwen_config_full_finetuning",
]