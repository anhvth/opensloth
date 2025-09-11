from __future__ import annotations

import os
from typing import Any

from opensloth.logging_config import get_opensloth_logger
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

from .base import (
    maybe_hot_fix_gemma,
    setup_comm_backend,
    validate_dataset_compatibility,
    ensure_dataset_features,
)
from opensloth.patching.patch_log import patch_log_for_multi_gpu

def _init_model_and_tokenizer(cfg: OpenSlothConfig, unsloth_modules: dict[str, Any] | None = None):
    logger = get_opensloth_logger()
    
    # Lazy import for project rule compliance
    if unsloth_modules is not None:
        FastLanguageModel = unsloth_modules["FastLanguageModel"]  # noqa: N806
        FastModel = unsloth_modules.get("FastModel")  # noqa: N806
    else:
        from unsloth import FastLanguageModel, FastModel

    # If using a pretrained LoRA, we must identify the true base model
    # to correctly initialize vLLM (`fast_inference=True`).
    if cfg.pretrained_lora:
        logger.info(f"Using pretrained LoRA from: {cfg.pretrained_lora}")
        try:
            import json
            from pathlib import Path
            adapter_path = Path(cfg.pretrained_lora)
            config_file = adapter_path / "adapter_config.json"
            if config_file.is_file():
                with config_file.open() as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
                if base_model_name:
                    logger.info(f"Inferred base model for vLLM: {base_model_name}")
                    cfg.fast_model_args.model_name = base_model_name
                else:
                    logger.warning("`base_model_name_or_path` not found in adapter config. vLLM might fail.")
            else:
                logger.warning(f"adapter_config.json not found in {cfg.pretrained_lora}. Using LoRA path as model name.")
        except Exception as e:
            logger.error(f"Error reading adapter config: {e}. vLLM might fail.")

    model_args = cfg.fast_model_args.model_dump()

    # Load base model (potentially with vLLM enabled)
    model, tokenizer = FastLanguageModel.from_pretrained(**model_args)
    maybe_hot_fix_gemma(cfg, logger, tokenizer)
    setup_comm_backend(cfg)

    # Apply LoRA configuration
    if cfg.fast_model_args.full_finetuning:
        logger.info("Full fine-tuning enabled. Skipping PEFT model setup.")
    elif cfg.pretrained_lora:
        # Load the pretrained adapter onto the base model
        logger.info("Loading pretrained LoRA adapter onto the base model...")
        model = FastModel.from_pretrained(model=model, model_name=cfg.pretrained_lora)
    elif cfg.lora_args:
        # Initialize a new LoRA adapter
        logger.info("Initializing new LoRA adapter...")
        model = FastModel.get_peft_model(model, **cfg.lora_args.model_dump())
    
    return model, tokenizer


def _load_dataset(cfg: OpenSlothConfig, hf_train_args: TrainingArguments):
    from datasets import load_from_disk

    logger = get_opensloth_logger()
    base = cfg.data_cache_path
    rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    shard_path = os.path.join(base, f"shard_{rank}")
    path = shard_path if os.path.isdir(shard_path) else base
    train_ds = load_from_disk(path)
    # best-effort length compatibility check
    validate_dataset_compatibility(base, cfg.fast_model_args.max_seq_length)

    # Optional SFT over-length filtering
    if cfg.filter_overlength_samples:
        max_len = int(cfg.fast_model_args.max_seq_length)
        def _ok(ex):
            return len(ex.get("input_ids", [])) <= max_len
        before = len(train_ds)
        train_ds = train_ds.filter(_ok, num_proc=getattr(hf_train_args, "dataset_num_proc", 1))
        drop = before - len(train_ds)
        if drop:
            logger.info("Filtered %s over-length samples (>%s)", drop, max_len)

    ensure_dataset_features(train_ds, ["input_ids", "labels"])
    return train_ds


def create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    _cfg: OpenSlothConfig,  # kept for uniform signature
    hf_train_args: TrainingArguments,
):
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer
    import os

    # Only enable TensorBoard logging on main rank (LOCAL_RANK==0) to prevent thread hanging
    local_rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    if local_rank != 0:
        # Disable all reporting for non-main ranks to prevent hanging threads
        hf_train_args.report_to = "none"
        # Also disable progress bars for non-main ranks
        hf_train_args.disable_tqdm = True
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    
    
    hf_train_args.skip_prepare_dataset = True  # type: ignore[attr-defined]
    trainer =  SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_train_args,          # type: ignore[arg-type]
        tokenizer=tokenizer,         # type: ignore[arg-type]
        data_collator=data_collator,
    )
    trainer = patch_log_for_multi_gpu(trainer)
    return trainer


def setup_model_and_training(opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments, unsloth_modules: dict[str, Any] | None = None):
    """High-level API expected by external training scripts."""
    logger = get_opensloth_logger()
    logger.start_timing("total_setup")
    logger.start_timing("model_init")
    model, tokenizer = _init_model_and_tokenizer(opensloth_config, unsloth_modules)
    logger.finish_timing("model_init")
    logger.start_timing("trainer_creation")
    dataset = _load_dataset(opensloth_config, hf_train_args)
    trainer = create_sft_trainer(model, tokenizer, dataset, opensloth_config, hf_train_args)
    from opensloth.patching.inner_training_loop import (
        patch_inner_training_loop_for_sft,
    )
    patch_inner_training_loop_for_sft(trainer, opensloth_config.sequence_packing)
    logger.finish_timing("trainer_creation")
    logger.finish_timing("total_setup")
    return trainer, model, tokenizer
