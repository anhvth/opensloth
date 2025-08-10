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


def create_dpo_trainer(model, tokenizer, train_dataset, cfg: OpenSlothConfig, hf_args: TrainingArguments):
    from trl import DPOTrainer, DPOConfig

    validate_dataset_for_type(train_dataset, "dpo")
    dpo_args = cfg.dpo_args
    if dpo_args is None:
        raise ValueError("dpo_args must be provided for DPO training")
    d = hf_args.to_dict()
    d.update(
        {
            "beta": dpo_args.beta,
            "max_length": dpo_args.max_length,
            "max_prompt_length": dpo_args.max_prompt_length,
        }
    )
    dpo_config = DPOConfig(**d)
    return DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )


def create_kto_trainer(model, tokenizer, train_dataset, _cfg: OpenSlothConfig, _args: TrainingArguments):
    raise NotImplementedError("KTO training not implemented yet")


def create_orpo_trainer(model, tokenizer, train_dataset, _cfg: OpenSlothConfig, _args: TrainingArguments):
    raise NotImplementedError("ORPO training not implemented yet")


# ---- GRPO ----

def _validate_grpo_dataset(ds) -> list[str]:
    feats = getattr(ds, "features", None)
    if feats is None:
        raise ValueError("GRPO dataset must have features")
    keys = list(feats.keys())
    if "prompt" not in keys and "input_ids" not in keys:
        raise ValueError(f"GRPO dataset must have 'prompt' or 'input_ids'; has {keys}")
    return keys


def create_grpo_trainer(model, tokenizer, train_dataset, cfg: OpenSlothConfig, hf_args: TrainingArguments):
    from trl import GRPOTrainer, GRPOConfig

    grpo_args = cfg.grpo_args
    if grpo_args is None:
        raise ValueError("grpo_args must be provided for GRPO training")

    try:
        from opensloth.grpo_rewards import (
            get_reward_functions,
            create_reward_preset,
        )
    except Exception:
        def _dummy_reward(*_a: Any, **_k: Any):
            return 0.0
        def get_reward_functions(_names):  # type: ignore
            return [_dummy_reward]
        def create_reward_preset(_task_type):  # type: ignore
            return ["dummy_reward"]

    _validate_grpo_dataset(train_dataset)

    cfg_dict = hf_args.to_dict()
    cfg_dict.pop("dataset_num_proc", None)
    cfg_dict.update(
        {
            "num_generations": grpo_args.group_size,
            "max_completion_length": grpo_args.max_new_tokens,
            "temperature": grpo_args.temperature,
            "top_p": grpo_args.top_p,
            "beta": grpo_args.kl_coef,
            "max_prompt_length": grpo_args.max_prompt_length,
        }
    )
    if grpo_args.top_k is not None:
        cfg_dict["top_k"] = grpo_args.top_k
    if grpo_args.min_p is not None:
        cfg_dict["min_p"] = grpo_args.min_p

    stop_sequences = list(grpo_args.stop_sequences)
    eos = getattr(tokenizer, "eos_token", None)
    if eos and eos not in stop_sequences:
        stop_sequences.append(eos)
    # "pad_token_id": eos
    # if stop_sequences:
        # generation_kwargs["stop_strings"] = stop_sequences
    # cfg_dict["generation_kwargs"] = {}

    grpo_config = GRPOConfig(**cfg_dict)

    reward_names = grpo_args.reward_functions or create_reward_preset(grpo_args.task_type)
    if not grpo_args.reward_functions:
        grpo_args.reward_functions = list(reward_names)  # type: ignore
    reward_fns = get_reward_functions(reward_names)

    def _wrap(rf):
        def inner(prompts, completions, **kw):
            return rf(prompts, completions, **kw)
        return inner

    reward_funcs = [_wrap(r) for r in reward_fns]
    try:
        return GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
        )
    except TypeError as exc:  # Older TRL versions
        if "reward_funcs" in str(exc):
            return GRPOTrainer(
                model=model,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
        raise
