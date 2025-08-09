"""Minimal single-GPU DPO training simulation to validate pipeline."""
from opensloth.opensloth_config import (
    OpenSlothConfig, FastModelArgs, LoraArgs, TrainingArguments, DPOArgs
)
from opensloth.scripts.opensloth_sft_trainer import train_on_single_gpu, setup_envs

opensloth_config = OpenSlothConfig(
    data_cache_path="data/dpo_sim_run",
    devices=[0],
    training_type="dpo",
    fast_model_args=FastModelArgs(
        model_name="unsloth/zephyr-sft-bnb-4bit",
        max_seq_length=1024,
        load_in_4bit=True,
        full_finetuning=False,
    ),
    lora_args=LoraArgs(r=32, lora_alpha=32),
    dpo_args=DPOArgs(beta=0.1, max_length=512, max_prompt_length=256),
    sequence_packing=False,
)

train_args = TrainingArguments(
    output_dir="outputs/dpo_sim_run",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=1,
    warmup_steps=0,
    optim="adamw_8bit",
    save_total_limit=1,
    report_to="none",
)

if __name__ == "__main__":
    setup_envs(opensloth_config, train_args)
    train_on_single_gpu(0, opensloth_config, train_args)
    print("Single-GPU DPO simulation complete.")
