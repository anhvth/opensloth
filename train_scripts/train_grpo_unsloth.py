"""
GRPO training script using Unsloth native implementation (not TRL).
This script demonstrates how to configure and run GRPO training with OpenSloth.
"""

from opensloth.opensloth_config import (
    FastModelArgs,
    GRPOArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# GRPO configuration for 0.5B model with fast experiment settings
opensloth_config = OpenSlothConfig(
    data_cache_path="data/grpo_math_test/",  # Update this to your dataset path
    devices=[0],  # Single GPU for testing
    training_type="grpo",
    fast_model_args=FastModelArgs(
        model_name="unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    ),
    lora_args=LoraArgs(
        r=32,
        lora_alpha=64,  # 2 * r for better performance
        lora_dropout=0.0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ),
    grpo_args=GRPOArgs(
        group_size=4,  # Number of generated responses per prompt
        max_new_tokens=256,  # Max tokens for reasoning + solution
        temperature=1.0,
        top_p=0.9,
        top_k=None,  # Disabled by default
        kl_coef=0.05,  # KL divergence coefficient
        max_prompt_length=512,
        eval_interval=50,
        save_interval=100,
    ),
    sequence_packing=False,  # GRPO doesn't use sequence packing
    log_level="info",
)

# Training arguments optimized for GRPO
training_config = TrainingArguments(
    output_dir="outputs/grpo_unsloth_test",
    per_device_train_batch_size=1,  # Keep small for GRPO
    gradient_accumulation_steps=4,  # Accumulate for larger effective batch
    learning_rate=5e-6,  # Lower LR for GRPO stability
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    optim="adamw_8bit",
    logging_steps=1,
    max_steps=100,  # Small number for testing
    save_steps=50,
    report_to="none",  # GRPO doesn't support tensorboard due to vLLM serialization issues
    save_total_limit=2,
    save_only_model=False,
    seed=3407,
)

if __name__ == "__main__":
    print("Starting GRPO training with Unsloth native implementation...")
    print(f"Model: {opensloth_config.fast_model_args.model_name}")
    print(f"Dataset: {opensloth_config.data_cache_path}")
    print(f"Output: {training_config.output_dir}")
    
    # Setup environment
    setup_envs(opensloth_config, training_config)
    
    # Run training
    if len(opensloth_config.devices) > 1:
        print(f"Multi-GPU GRPO training on GPUs: {opensloth_config.devices}")
        run_mp_training(
            gpus=opensloth_config.devices,
            opensloth_config=opensloth_config,
            training_config=training_config,
        )
    else:
        print(f"Single GPU GRPO training on GPU: {opensloth_config.devices[0]}")
        from opensloth.scripts.opensloth_sft_trainer import train_on_single_gpu
        train_on_single_gpu(
            gpu=opensloth_config.devices[0],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
    
    print("GRPO training completed!")
    print(f"Model saved to: {training_config.output_dir}")
    print("LoRA adapter saved as: grpo_lora_adapter/")
