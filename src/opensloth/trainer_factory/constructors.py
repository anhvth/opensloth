from __future__ import annotations

# ruff: noqa: I001  (local/dynamic imports kept unsorted intentionally for clarity & conditional loading)

from typing import Any

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
from .base import validate_dataset_for_type


def create_sft_trainer(model, tokenizer, train_dataset, _cfg: OpenSlothConfig, hf_args: TrainingArguments):
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer

    validate_dataset_for_type(train_dataset, "sft")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    hf_args.skip_prepare_dataset = True  # type: ignore[attr-defined]
    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_args,          # type: ignore[arg-type]
        tokenizer=tokenizer,   # type: ignore[arg-type]
        data_collator=data_collator,
    )

