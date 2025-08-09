from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_trainer import run_mp_training, setup_envs

# 2 GPUs with packing configuration
GLOBAL_BZ = 32

DEVICES = [0, 1,2,3]

BZ = 1  # if sequence packing, then should be 1, larger does not contribute to speed
opensloth_config = OpenSlothConfig(
    data_cache_path="qwen_dataset/0808-dataset/",
    devices=DEVICES,
    fast_model_args=FastModelArgs(
        # model_name="/data/hf-models/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8/",
        model_name='/data/hf-models/unsloth/Qwen3-0.6B-bnb-4bit/',
        max_seq_length=16000,
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
    # output_dir="outputs/exps/qwen3-30ba3-cache_qwen3_dataset_250808",
    output_dir="outputs/exps/qwen3-30b-cache_qwen3_dataset_250808",
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
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
