"""
tested with:

pip install unsloth==2025.5.7 unsloth-zoo==2025.5.8
"""

from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_trainer import run_mp_training, setup_envs

GLOBAL_BZ = 32

DEVICES = [0, 2]

BZ = 2


opensloth_config = OpenSlothConfig(
    data_cache_path="data/cache_gemma_responses_only",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name="unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
        max_seq_length=16000,
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        r=8,
        lora_alpha=16,
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
    output_dir="outputs/exps/qwen3-0.6b-FineTome-2gpu-packing",
    per_device_train_batch_size=BZ,
    gradient_accumulation_steps=GLOBAL_BZ // (len(DEVICES) * BZ),
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=5,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="none",  # or wandb/tensorboard
    max_steps=100,
)


if __name__ == "__main__":
    print(
        "→ Global batch size:",
        len(DEVICES) * BZ * training_config.gradient_accumulation_steps,
    )
    print("→ Grad accumulation:", training_config.gradient_accumulation_steps)
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
    print("Training completed.")
