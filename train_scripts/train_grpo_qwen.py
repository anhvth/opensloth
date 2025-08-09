#!/usr/bin/env python3
"""
GRPO Training Script using OpenSloth and Unsloth
Supports configurable reward functions for different task types.
"""

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
from opensloth.grpo_rewards import list_reward_functions, create_reward_preset

# Model configuration
opensloth_config = OpenSlothConfig(
    devices=[0],  # Single GPU for now, can use [0, 1] for multi-GPU
    training_type="grpo",
    
    # Model settings
    fast_model_args={
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": True,
        "fast_inference": True,
        "gpu_memory_utilization": 0.7,
    },
    
    # LoRA settings
    lora_args={
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.0,
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "use_rslora": False,
        "random_state": 3407,
    },
    
    # GRPO-specific settings
    grpo_args={
        "task_type": "math",  # "math", "code", "general", or "reasoning"
        "group_size": 4,
        "max_new_tokens": 256,
        "temperature": 1.0,
        "top_p": 0.9,
        "top_k": None,
        "min_p": 0.1,
        
        # Reward functions (auto-selected if empty based on task_type)
        "reward_functions": [],  # or explicitly: ["math_format", "math_answer", "math_number"]
        
        # Chat template
        "use_custom_chat_template": True,
        
        # Prompt filtering
        "max_prompt_length": 512,
        "prompt_length_percentile": 0.9,
        
        # Training control
        "eval_interval": 50,
        "save_interval": 100,
        "print_sample_every": 10,
        
        # vLLM settings
        "stop_sequences": [],
        "include_stop_str_in_output": True,
    },
    
    # Data path
    data_cache_path="./data/grpo_math",  # Path to your prepared dataset
    
    # Logging
    log_level="INFO",
    sequence_packing=False,  # Not used for GRPO
)

# Training configuration
training_config = TrainingArguments(
    output_dir="./outputs/grpo_qwen_0.5b_math",
    
    # Training schedule
    max_steps=200,  # Adjust based on your dataset size
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    
    # Optimizer
    learning_rate=5e-6,
    weight_decay=0.01,
    optim="adamw_8bit",
    
    # Learning rate schedule  
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    
    # Logging and saving
    logging_steps=5,
    save_steps=100,
    save_only_model=True,
    
    # Evaluation
    eval_strategy="no",  # Required for multi-GPU
    
    # Reporting
    report_to="none",  # Avoid issues with vLLM serialization
    
    # Other
    seed=42,
    dataloader_num_workers=4,
)

if __name__ == "__main__":
    print("Available reward functions:", list_reward_functions())
    print("Math task reward preset:", create_reward_preset("math"))
    print("Code task reward preset:", create_reward_preset("code"))
    print("General task reward preset:", create_reward_preset("general"))
    
    print(f"\\nGRPO Config:")
    print(f"  Task type: {opensloth_config.grpo_args.task_type}")
    print(f"  Reward functions: {opensloth_config.grpo_args.reward_functions or 'auto-selected'}")
    print(f"  Group size: {opensloth_config.grpo_args.group_size}")
    print(f"  Max new tokens: {opensloth_config.grpo_args.max_new_tokens}")
    print(f"  Data path: {opensloth_config.data_cache_path}")
    print(f"  Output dir: {training_config.output_dir}")
    print(f"  Max steps: {training_config.max_steps}")
    print(f"  Devices: {opensloth_config.devices}")
