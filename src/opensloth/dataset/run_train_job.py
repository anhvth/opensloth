import sys
import json

from opensloth.opensloth_config import (
    FastModelArgs,
    LoraArgs,
    OpenSlothConfig,
    TrainingArguments,
)
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, setup_envs


def main() -> None:
    cfg = json.load(sys.stdin)
    print("[JOB] Building configs...")

    # New path: accept JSON objects mirroring Pydantic models
    if "opensloth_config" in cfg and "training_args" in cfg:
        oc = cfg["opensloth_config"]
        ta = cfg["training_args"]
        opensloth_config = OpenSlothConfig(**oc)
        training_args = TrainingArguments(**ta)
    else:
        # Backward compatibility with flat fields
        target_modules = cfg.get("target_modules") or [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]

        opensloth_config = OpenSlothConfig(
            data_cache_path=cfg["data_cache_path"],
            devices=cfg["devices"],
            fast_model_args=FastModelArgs(
                model_name=cfg["model_name"],
                max_seq_length=cfg["max_seq_length"],
                load_in_4bit=cfg["load_in_4bit"],
                load_in_8bit=cfg["load_in_8bit"],
                full_finetuning=cfg["full_finetuning"],
            ),
            lora_args=LoraArgs(
                r=cfg["r"],
                lora_alpha=cfg["lora_alpha"],
                lora_dropout=cfg["lora_dropout"],
                target_modules=target_modules,
                use_rslora=cfg["use_rslora"],
            ),
            sequence_packing=cfg["sequence_packing"],
        )

        training_args = TrainingArguments(
            output_dir=cfg["output_dir"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            logging_steps=cfg["logging_steps"],
            num_train_epochs=cfg["num_train_epochs"],
            lr_scheduler_type=cfg["lr_scheduler_type"],
            warmup_steps=cfg["warmup_steps"],
            save_total_limit=cfg["save_total_limit"],
            weight_decay=cfg["weight_decay"],
            optim=cfg["optim"],
            seed=cfg["seed"],
            report_to=cfg["report_to"],
        )

    print("[JOB] Starting training...")
    setup_envs(opensloth_config, training_args)
    run_mp_training(opensloth_config.devices, opensloth_config, training_args)
    print("[JOB] Training finished.")


if __name__ == "__main__":
    main()
