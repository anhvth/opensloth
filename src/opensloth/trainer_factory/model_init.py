from __future__ import annotations

import os
from typing import Any

from opensloth.logging_config import get_opensloth_logger
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

from .base import (
    maybe_hot_fix_gemma,
    setup_comm_backend,
    validate_dataset_compatibility,
    validate_dataset_for_type,
)
from .dispatcher import create_trainer_by_type


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
    if cfg.training_type == "sft" and cfg.filter_overlength_samples:
        max_len = int(cfg.fast_model_args.max_seq_length)
        def _ok(ex):
            return len(ex.get("input_ids", [])) <= max_len
        before = len(train_ds)
        train_ds = train_ds.filter(_ok, num_proc=getattr(hf_train_args, "dataset_num_proc", 1))
        drop = before - len(train_ds)
        if drop:
            logger.info("Filtered %s over-length samples (>%s)", drop, max_len)

    validate_dataset_for_type(train_ds, cfg.training_type)
    return train_ds


def _configure_reporting(hf_args: TrainingArguments) -> None:
    rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    if rank != 0:
        hf_args.report_to = "none"


def _create_trainer(model, tokenizer, cfg: OpenSlothConfig, hf_args: TrainingArguments):
    trainer = create_trainer_by_type(model, tokenizer, _load_dataset(cfg, hf_args), cfg, hf_args)
    _configure_reporting(hf_args)
    if cfg.training_type == "sft":
        from opensloth.patching.inner_training_loop import (
            patch_inner_training_loop_for_sft,
        )
        patch_inner_training_loop_for_sft(trainer, cfg.sequence_packing)
    if len(cfg.devices) > 1 and os.getenv("OPENSLOTH_LOCAL_RANK", "0") != "0":
        def _no_op(*_a, **_k):
            return None
        trainer._save = _no_op  # type: ignore[attr-defined]
    return trainer


def setup_model_and_training(opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments, unsloth_modules: dict[str, Any] | None = None):
    """High-level API expected by external training scripts."""
    logger = get_opensloth_logger()
    logger.start_timing("total_setup")
    logger.start_timing("model_init")
    model, tokenizer = _init_model_and_tokenizer(opensloth_config, unsloth_modules)
    logger.finish_timing("model_init")
    logger.start_timing("trainer_creation")
    trainer = _create_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_creation")
    logger.finish_timing("total_setup")
    return trainer, model, tokenizer
