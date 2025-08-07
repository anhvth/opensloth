from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

# 2 GPUs with packing configuration
GLOBAL_BZ = 64

DEVICES = [0, 1,2,3]

BZ = 1  # if sequence packing, then should be 1, larger does not contribute to speed
opensloth_config = OpenSlothConfig(
    data_cache_path="data/cache_qwen3_dataset_250807/",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        model_name="./unsloth/Qwen3-32B-bnb-4bit",
        # model_name='./unsloth/qwen3-0.6b-bnb-4bit',
        max_seq_length=8000,
        load_in_4bit=True,
    ),
    lora_args=LoraArgs(
        r=256,
        lora_alpha=512,
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
    output_dir="outputs/exps/qwen3-30ba3-cache_qwen3_dataset_250807",
    save_only_model=True,
    per_device_train_batch_size=BZ,
    gradient_accumulation_steps=GLOBAL_BZ // (len(DEVICES) * BZ),
    learning_rate=1e-5,
    logging_steps=1,
    num_train_epochs=1,
    lr_scheduler_type="linear",
    warmup_steps=20,
    save_total_limit=1,
    weight_decay=0.01,
    optim="adamw_8bit",
    seed=3407,
    report_to="tensorboard",  # or wandb/tensorboard
)


if __name__ == "__main__":
    import os
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
