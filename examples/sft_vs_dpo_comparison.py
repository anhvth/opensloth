"""
Comprehensive example showing both SFT and DPO training configurations.
This script demonstrates the differences and how to use both training types.
"""

from opensloth.opensloth_config import DPOArgs, FastModelArgs, LoraArgs, OpenSlothConfig, TrainingArguments


def create_sft_config():
    """Create configuration for SFT (Supervised Fine-Tuning) training."""
    
    opensloth_config = OpenSlothConfig(
        data_cache_path='data/sft_dataset_cache',  # Path to SFT dataset
        devices=[0],
        training_type="sft",  # Traditional SFT training
        
        # Model configuration
        fast_model_args=FastModelArgs(
            model_name='unsloth/Qwen2.5-0.5B-Instruct',  # Base model
            max_seq_length=2048,
            load_in_4bit=True,
            full_finetuning=False,
        ),
        
        # LoRA configuration for SFT
        lora_args=LoraArgs(
            r=16,  # Smaller rank for SFT
            lora_alpha=16,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                           'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.1,
            bias="none",
        ),
        
        # SFT works well with sequence packing
        sequence_packing=True,
    )
    
    # SFT Training configuration
    training_config = TrainingArguments(
        output_dir='outputs/sft_training',
        per_device_train_batch_size=8,  # Larger batch size for SFT
        learning_rate=2e-4,  # Standard learning rate for SFT
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        warmup_steps=100,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        save_total_limit=3,
    )
    
    return opensloth_config, training_config


def create_dpo_config():
    """Create configuration for DPO (Direct Preference Optimization) training."""
    
    opensloth_config = OpenSlothConfig(
        data_cache_path='data/dpo_test_cache',  # Path to DPO dataset
        devices=[0],
        training_type="dpo",  # DPO training
        
        # Model configuration - typically starts from SFT model
        fast_model_args=FastModelArgs(
            model_name='unsloth/zephyr-sft-bnb-4bit',  # Pre-trained SFT model
            max_seq_length=1024,  # Often shorter for DPO
            load_in_4bit=True,
            full_finetuning=False,
        ),
        
        # LoRA configuration for DPO
        lora_args=LoraArgs(
            r=64,  # Higher rank often used for DPO
            lora_alpha=64,
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                           'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout=0.0,  # No dropout for DPO stability
            bias="none",
        ),
        
        # DPO-specific configuration
        dpo_args=DPOArgs(
            beta=0.1,  # DPO beta parameter - controls preference strength
            max_length=1024,  # Maximum total sequence length
            max_prompt_length=512,  # Maximum prompt length
        ),
        
        # Disable packing for DPO (recommended)
        sequence_packing=False,
    )
    
    # DPO Training configuration
    training_config = TrainingArguments(
        output_dir='outputs/dpo_training',
        per_device_train_batch_size=2,  # Smaller batch size due to memory
        learning_rate=5e-6,  # Much lower learning rate for DPO
        gradient_accumulation_steps=8,  # Higher to maintain effective batch size
        num_train_epochs=3,
        warmup_ratio=0.1,  # Use ratio instead of steps
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.0,  # No weight decay for DPO
        lr_scheduler_type="linear",
        save_total_limit=2,
        seed=42,
    )
    
    return opensloth_config, training_config


def compare_configurations():
    """Compare SFT and DPO configurations side by side."""
    
    print("=" * 80)
    print("COMPARISON: SFT vs DPO Training Configurations")
    print("=" * 80)
    
    sft_config, sft_training = create_sft_config()
    dpo_config, dpo_training = create_dpo_config()
    
    print("\n📊 KEY DIFFERENCES:\n")
    
    # Dataset requirements
    print("🗃️  DATASET REQUIREMENTS:")
    print("   SFT: input_ids, labels (input-output pairs)")
    print("   DPO: prompt, chosen, rejected (preference pairs)")
    
    # Model starting point
    print("\n🤖 MODEL STARTING POINT:")
    print(f"   SFT: {sft_config.fast_model_args.model_name} (base model)")
    print(f"   DPO: {dpo_config.fast_model_args.model_name} (SFT model)")
    
    # Learning rates
    print("\n📈 LEARNING RATES:")
    print(f"   SFT: {sft_training.learning_rate} (standard)")
    print(f"   DPO: {dpo_training.learning_rate} (much lower)")
    
    # Batch sizes
    print("\n📦 BATCH SIZES:")
    print(f"   SFT: {sft_training.per_device_train_batch_size} (larger)")
    print(f"   DPO: {dpo_training.per_device_train_batch_size} (smaller)")
    
    # LoRA settings
    print("\n🔧 LORA SETTINGS:")
    print(f"   SFT: r={sft_config.lora_args.r}, dropout={sft_config.lora_args.lora_dropout}")
    print(f"   DPO: r={dpo_config.lora_args.r}, dropout={dpo_config.lora_args.lora_dropout}")
    
    # Sequence packing
    print("\n📋 SEQUENCE PACKING:")
    print(f"   SFT: {sft_config.sequence_packing} (enabled for efficiency)")
    print(f"   DPO: {dpo_config.sequence_packing} (disabled for stability)")
    
    # DPO-specific parameters
    print("\n🎯 DPO-SPECIFIC PARAMETERS:")
    print(f"   Beta: {dpo_config.dpo_args.beta} (preference strength)")
    print(f"   Max Length: {dpo_config.dpo_args.max_length}")
    print(f"   Max Prompt Length: {dpo_config.dpo_args.max_prompt_length}")
    
    print("\n" + "=" * 80)


def run_sft_training_example():
    """Example of running SFT training."""
    
    print("\n🚀 SFT TRAINING EXAMPLE:\n")
    
    sft_config, sft_training = create_sft_config()
    
    print("To run SFT training:")
    print("1. Prepare SFT dataset with input-output pairs")
    print("2. Configure training_type='sft'")
    print("3. Run training:")
    print()
    print("```python")
    print("from opensloth.scripts.opensloth_trainer import train_on_single_gpu")
    print()
    print("train_on_single_gpu(")
    print("    gpu=0,")
    print("    opensloth_config=sft_config,")
    print("    hf_train_args=sft_training,")
    print(")")
    print("```")


def run_dpo_training_example():
    """Example of running DPO training."""
    
    print("\n🎯 DPO TRAINING EXAMPLE:\n")
    
    dpo_config, dpo_training = create_dpo_config()
    
    print("To run DPO training:")
    print("1. Prepare DPO dataset with preference pairs")
    print("2. Configure training_type='dpo'")
    print("3. Run training:")
    print()
    print("```python")
    print("from opensloth.scripts.opensloth_trainer import train_on_single_gpu")
    print()
    print("train_on_single_gpu(")
    print("    gpu=0,")
    print("    opensloth_config=dpo_config,")
    print("    hf_train_args=dpo_training,")
    print(")")
    print("```")


def main():
    """Main function demonstrating both training types."""
    
    print("OpenSloth Multi-Training Type Example")
    print("Supports: SFT, DPO, and future RL methods (KTO, ORPO, GRPO)")
    
    # Compare configurations
    compare_configurations()
    
    # Show training examples
    run_sft_training_example()
    run_dpo_training_example()
    
    print("\n📚 ADDITIONAL RESOURCES:")
    print("- DPO Documentation: docs/DPO_TRAINING.md")
    print("- Example Scripts: train_scripts/train_dpo_example.py")
    print("- Dataset Preparation: prepare_dataset/prepare_dpo_dataset.py")
    
    print("\n✨ FUTURE TRAINING TYPES:")
    print("- KTO (Kahneman-Tversky Optimization) - Coming soon")
    print("- ORPO (Odds Ratio Preference Optimization) - Coming soon")
    print("- GRPO (Group Relative Policy Optimization) - Coming soon")


if __name__ == "__main__":
    main()
