from __future__ import annotations

# ruff: noqa: I001

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
from .constructors import (
    create_sft_trainer,
    create_dpo_trainer,
    create_kto_trainer,
    create_orpo_trainer,
    create_grpo_trainer,
)

TRAINER_FACTORY_MAP = {
    "sft": create_sft_trainer,
    "dpo": create_dpo_trainer,
    "kto": create_kto_trainer,
    "orpo": create_orpo_trainer,
    "grpo": create_grpo_trainer,
}


def create_trainer_by_type(model, tokenizer, train_dataset, cfg: OpenSlothConfig, hf_args: TrainingArguments):
    factory = TRAINER_FACTORY_MAP.get(cfg.training_type)
    if factory is None:
        raise ValueError(f"Unsupported training_type {cfg.training_type}")
    return factory(model, tokenizer, train_dataset, cfg, hf_args)
