from opensloth.opensloth_config import (
    FastModelArgs,
    HFDatasetConfig,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs

if __name__ == "__main__":
    import os

    experiment_setups = [
        # 1 gpu, no packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0],
            "sequence_packing": False,
            "num_samples": 10000,
            "exp_name": "1gpu_no_packing",
        },
        # 1 gpu, packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0],
            "sequence_packing": True,
            "num_samples": 10000,
            "exp_name": "1gpu_packing",
        },
        # 2 gpus, no packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0, 1],
            "sequence_packing": False,
            "num_samples": 10000,
            "exp_name": "2gpu_no_packing",
        },
        # 2 gpus, packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0, 1],
            "sequence_packing": True,
            "num_samples": 10000,
            "exp_name": "2gpu_packing",
        },
        # 4 gpus, no packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0, 1, 2, 3],
            "sequence_packing": False,
            "num_samples": 10000,
            "exp_name": "4gpu_no_packing",
        },
        # 4 gpus, packing, global bz 128, total 10000 samples, bz 8
        {
            "devices": [0, 1, 2, 3],
            "sequence_packing": True,
            "num_samples": 10000,
            "exp_name": "4gpu_packing",
        },
    ]

    for exp in experiment_setups:
        # Calculate gradient accumulation steps for global batch size 128
        num_devices = len(exp["devices"])
        per_device_batch_size = 8
        target_global_batch_size = 128
        gradient_accumulation_steps = target_global_batch_size // (
            num_devices * per_device_batch_size
        )

        opensloth_config = OpenSlothConfig(
            data=HFDatasetConfig(
                tokenizer_name="Qwen/Qwen3-8B",
                chat_template="qwen3",
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
                num_samples=exp["num_samples"],
                nproc=52,
                max_seq_length=16000,
                source_type="hf",
                dataset_name="mlabonne/FineTome-100k",
                split="train",
            ),
            devices=exp["devices"],
            fast_model_args=FastModelArgs(
                model_name="model_store/unsloth/Qwen3-14B-bnb-4bit",
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
            sequence_packing=exp["sequence_packing"],
        )

        # Training arguments with calculated gradient accumulation
        training_config = TrainingArguments(
            output_dir=f"outputs/exps/qwen3-0.6b-FineTome-{exp['exp_name']}",
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=1e-5,
            logging_steps=1,
            num_train_epochs=1,
            lr_scheduler_type="linear",
            warmup_steps=5,
            save_total_limit=1,
            weight_decay=0.01,
            optim="adamw_8bit",
            seed=3407,
            report_to="wandb",
        )

        # Setup wandb with proper naming
        os.environ["WANDB_PROJECT"] = "opensloth"
        packing_str = "packing" if exp["sequence_packing"] else "no_packing"
        wandb_name = (
            f"qwen3-0.6b_{num_devices}gpu_{packing_str}_"
            f"globalbz{target_global_batch_size}_samples{exp['num_samples']}"
        )
        os.environ["WANDB_NAME"] = wandb_name

        print(f"Running experiment: {exp['exp_name']}")
        print(
            f"Global batch size: {num_devices * per_device_batch_size * gradient_accumulation_steps}"
        )
        print(f"Gradient accumulation steps: {gradient_accumulation_steps}")

        setup_envs(opensloth_config, training_config)
        run_mp_training(opensloth_config.devices, opensloth_config, training_config)
