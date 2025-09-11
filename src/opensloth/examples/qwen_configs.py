"""
Qwen model configuration examples for dataset preparation and training.
"""

from typing import Dict, Any


# Base Qwen configuration
QWEN_BASE_CONFIG = {
    "tokenizer_name": 'unsloth/Qwen2.5-0.5B-Instruct',
    "chat_template": "qwen-2.5",
    "dataset_name": 'mlabonne/FineTome-100k',
    "input_file": None,
    "split": 'train',
    "num_samples": -1,
    "num_proc": 16,
    "output_dir": None,
    "gpus": 1,
    "max_seq_length": 16096,
    "train_on_target_only": False,
    "instruction_part": '<|im_start|>user\n',
    "response_part": '<|im_start|>assistant\n',
    "debug": 0,
}


def qwen_config_1gpu(**overrides) -> Dict[str, Any]:
    """Configuration for single GPU training."""
    config = QWEN_BASE_CONFIG.copy()
    config.update({
        "gpus": 1,
        "num_proc": 8,
    })
    config.update(overrides)
    return config


def qwen_config_2gpus(**overrides) -> Dict[str, Any]:
    """Configuration for 2-GPU training."""
    config = QWEN_BASE_CONFIG.copy()
    config.update({
        "gpus": 2,
        "num_proc": 16,
    })
    config.update(overrides)
    return config


def qwen_config_4gpus(**overrides) -> Dict[str, Any]:
    """Configuration for 4-GPU training."""
    config = QWEN_BASE_CONFIG.copy()
    config.update({
        "gpus": 4,
        "num_proc": 32,
    })
    config.update(overrides)
    return config


def qwen_config_debug(**overrides) -> Dict[str, Any]:
    """Debug configuration with small dataset."""
    config = QWEN_BASE_CONFIG.copy()
    config.update({
        "num_samples": 10,
        "debug": 3,
        "gpus": 1,
        "num_proc": 1,
    })
    config.update(overrides)
    return config


def qwen_config_full_finetuning(**overrides) -> Dict[str, Any]:
    """Configuration for full fine-tuning (no LoRA)."""
    config = QWEN_BASE_CONFIG.copy()
    config.update({
        "gpus": 1,
        "num_proc": 8,
        "train_on_target_only": True,
    })
    config.update(overrides)
    return config


# Training configuration templates
def get_training_config_template(num_gpus: int = 1, max_seq_length: int = 16096) -> Dict[str, Any]:
    """Get a training configuration template."""
    return {
        "opensloth_config": {
            "data_cache_path": None,  # Will be set by dataset preparation
            "devices": list(range(num_gpus)),
            "fast_model_args": {
                "model_name": 'unsloth/Qwen2.5-0.5B-Instruct',
                "max_seq_length": max_seq_length,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "full_finetuning": False,
                "use_gradient_checkpointing": "unsloth",
                "fast_inference": False,
                "max_lora_rank": None,
                "gpu_memory_utilization": 0.7
            },
            "lora_args": {
                "finetune_vision_layers": False,
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "bias": "none",
                "random_state": 3407,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "use_rslora": False
            },
            "pretrained_lora": None,
            "sequence_packing": True,
            "log_level": "info",
            "filter_overlength_samples": True
        },
        "training_args": {
            "output_dir": "saves/loras/",
            "per_device_train_batch_size": 2,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
            "logging_steps": 1,
            "num_train_epochs": 3,
            "lr_scheduler_type": "linear",
            "warmup_steps": 10,
            "save_total_limit": 2,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "save_only_model": False,
            "resume_from_checkpoint": None,
            "seed": 42,
            "report_to": "tensorboard",
            "eval_strategy": "no",
            "dataset_num_proc": 8
        }
    }


def get_dataset_config_template(**kwargs) -> Dict[str, Any]:
    """Get a dataset configuration template."""
    base_config = QWEN_BASE_CONFIG.copy()
    base_config.update(kwargs)
    return base_config