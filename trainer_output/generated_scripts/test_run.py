from opensloth.opensloth_config import (FastModelArgs, LoraArgs, OpenSlothConfig, TrainingArguments)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

opensloth_config = OpenSlothConfig(
    data_cache_path='data/cache_qwen3_dataset',
    devices=[0],
    fast_model_args=FastModelArgs(
        model_name='unsloth/Qwen2.5-0.5B-Instruct',
        max_seq_length=2048,
        load_in_8bit=False,
        load_in_4bit=True,
        full_finetuning=False,
    ),
    lora_args=LoraArgs(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0,
        bias="none",
        use_rslora=False,
    ),
    sequence_packing=True,
)

training_config = TrainingArguments(
    output_dir='outputs/exps/test_run',
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=0.0001,
    logging_steps=1,
    max_steps=50,
    num_train_epochs=1,
    lr_scheduler_type='linear',
    warmup_steps=10,
    save_total_limit=1,
    weight_decay=0.01,
    optim='adamw_8bit',
    seed=3407,
    report_to='none',
)

if __name__ == "__main__":
    setup_envs(opensloth_config, training_config)
    run_mp_training(opensloth_config.devices, opensloth_config, training_config)
