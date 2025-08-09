"""
Example training script for DPO (Direct Preference Optimization) training using OpenSloth.
This script demonstrates how to configure and run DPO training.
"""

from opensloth.opensloth_config import (
    FastModelArgs, 
    LoraArgs, 
    OpenSlothConfig, 
    TrainingArguments,
    DPOArgs
)
from opensloth.scripts.opensloth_trainer import run_mp_training, setup_envs

# Example DPO configuration
opensloth_config = OpenSlothConfig(
    data_cache_path='data/dpo_dataset_cache',  # Path to DPO dataset cache
    devices=[0],  # GPU devices to use
    training_type="dpo",  # Specify DPO training
    
    # Model configuration
    fast_model_args=FastModelArgs(
        model_name='unsloth/zephyr-sft-bnb-4bit',  # Pre-trained SFT model for DPO
        max_seq_length=2048,
        load_in_8bit=False,
        load_in_4bit=True,
        full_finetuning=False,
    ),
    
    # LoRA configuration
    lora_args=LoraArgs(
        r=64,
        lora_alpha=64,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
    ),
    
    # DPO-specific configuration
    dpo_args=DPOArgs(
        beta=0.1,  # DPO beta parameter
        max_length=1024,  # Maximum sequence length for DPO
        max_prompt_length=512,  # Maximum prompt length
    ),
    
    # Disable sequence packing for DPO (recommended)
    sequence_packing=False,
)

# Training configuration
training_config = TrainingArguments(
    output_dir='outputs/dpo_training',
    per_device_train_batch_size=2,  # Smaller batch size for DPO
    gradient_accumulation_steps=8,
    learning_rate=5e-6,  # Lower learning rate for DPO
    num_train_epochs=3,
    warmup_ratio=0.1,
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.0,
    lr_scheduler_type="linear",
    seed=42,
    report_to="tensorboard",
    save_total_limit=2,
    save_only_model=False,
)

if __name__ == "__main__":
    # Setup environment
    setup_envs(opensloth_config, training_config)
    
    # Run training
    if len(opensloth_config.devices) > 1:
        # Multi-GPU DPO training
        run_mp_training(
            gpus=opensloth_config.devices,
            opensloth_config=opensloth_config,
            training_config=training_config,
        )
    else:
        # Single GPU DPO training
        from opensloth.scripts.opensloth_trainer import train_on_single_gpu
        train_on_single_gpu(
            gpu=opensloth_config.devices[0],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
        
    print("DPO training completed!")
