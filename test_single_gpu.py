from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# Single GPU test configuration
DEVICES = [0]  # Just use GPU 0

opensloth_config = OpenSlothConfig(
    data_cache_path="qwen_dataset/0808-dataset/",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name='/data/hf-models/unsloth/Qwen3-0.6B-bnb-4bit/',
        max_seq_length=2048,  # Smaller for testing
        load_in_8bit = False,
        load_in_4bit=True,
        full_finetuning=False
    ),
    lora_args=LoraArgs(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0,
        bias="none",
        use_rslora=False,
    ),
    sequence_packing=True,
)

training_config = TrainingArguments(
    output_dir="outputs/test_callback_fix",
    save_only_model=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=1,
    max_steps=2,  # Just test for 2 steps
    lr_scheduler_type="linear",
    warmup_steps=1,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="none",  # No logging for test
)

if __name__ == "__main__":
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
