"""Module for initializing model/tokenizer and creating trainers (SFT, DPO, etc.)."""

import os
from typing import Tuple

from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing

from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


def _validate_dataset_compatibility(dataset_path: str, model_max_seq_length: int):
    """Validate that the dataset is compatible with the training configuration."""
    import json
    import os
    from pathlib import Path
    
    dataset_config_path = Path(dataset_path) / "dataset_config.json"
    
    if dataset_config_path.exists():
        with open(dataset_config_path) as f:
            dataset_config = json.load(f)
        
        dataset_max_seq_length = dataset_config.get("max_seq_length")
        
        if dataset_max_seq_length is not None:
            if model_max_seq_length < dataset_max_seq_length:
                raise ValueError(
                    f"Training configuration mismatch!\n"
                    f"Dataset was prepared with max_seq_length={dataset_max_seq_length}, "
                    f"but training config specifies max_seq_length={model_max_seq_length}.\n"
                    f"The training max_seq_length must be >= dataset max_seq_length.\n"
                    f"Either:\n"
                    f"  1. Increase training max_seq_length to {dataset_max_seq_length} or higher\n"
                    f"  2. Re-prepare the dataset with --max-seq-length {model_max_seq_length}"
                )
            elif model_max_seq_length > dataset_max_seq_length:
                logger = get_opensloth_logger()
                logger.warning(
                    f"Training max_seq_length ({model_max_seq_length}) is larger than "
                    f"dataset max_seq_length ({dataset_max_seq_length}). This is OK but may be inefficient. "
                    f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length} for optimal performance."
                )
        else:
            # Old dataset without max_seq_length - show warning
            logger = get_opensloth_logger()
            logger.warning(
                f"Dataset at {dataset_path} was prepared without max_seq_length specification. "
                f"This may cause issues if sequences exceed training max_seq_length ({model_max_seq_length}). "
                f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length}."
            )


def init_model_and_tokenizer(opensloth_config: OpenSlothConfig) -> Tuple[object, object]:
    """Initialize base/LoRA model and tokenizer; set up NCCL if multi-GPU."""
    logger = get_opensloth_logger()

    from unsloth import FastLanguageModel

    logger.start_timing("model_loading")
    if opensloth_config.pretrained_lora:
        logger.info(
            f"Loading model from {opensloth_config.pretrained_lora} with LoRA weights"
        )
        opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora

    model, tokenizer = FastLanguageModel.from_pretrained(
        **opensloth_config.fast_model_args.model_dump()
    )
    _maybe_hot_fix_gemma(opensloth_config, logger, tokenizer)
    logger.finish_timing("model_loading")

    # NCCL setup only if >1 GPU
    if len(opensloth_config.devices) > 1:
        logger.start_timing("nccl_setup")
        from opensloth.nccl_grad_sync import get_callback_and_setup_method

        setup_nccl_for_opensloth = get_callback_and_setup_method()[1]
        setup_nccl_for_opensloth(
            rank=int(os.environ["OPENSLOTH_LOCAL_RANK"]),
            gpus=opensloth_config.devices,
        )
        logger.finish_timing("nccl_setup")
    else:
        logger.info("Single GPU detected; skipping NCCL setup")

    logger.info(
        f"Model device: {model.device} | Tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
        and opensloth_config.lora_args is not None
    ):
        logger.start_timing("lora_setup")
        from unsloth import FastModel

        model = FastModel.get_peft_model(
            model, **opensloth_config.lora_args.model_dump()  # type: ignore
        )
        logger.finish_timing("lora_setup")
    return model, tokenizer

def _maybe_hot_fix_gemma(opensloth_config, logger, tokenizer):
    if (
        "gemma-3" in opensloth_config.fast_model_args.model_name
        and opensloth_config.sequence_packing
    ):
        logger.info(
            "Detected Gemma3 model, applying Unsloth patch for sequence packing."
        )
        patch_gemma3_unsloth_for_sequence_packing()

    if not hasattr(tokenizer, "pad") and opensloth_config.sequence_packing:
        logger.info(
            "Tokenizer missing 'pad' method; attempting to patch using "
            "transformers.AutoTokenizer. This may indicate an Unsloth issue. "
            "See: https://github.com/unslothai/unsloth/issues/2056#event-17007147800"
        )
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            opensloth_config.fast_model_args.model_name,
        )
        tokenizer.pad = hf_tokenizer.pad


def create_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create trainer (SFT/DPO/etc.) and apply SFT-only performance patches."""
    logger = get_opensloth_logger()
    logger.start_timing("trainer_setup")
    trainer = _get_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_setup")

    # Apply SFT-specific patches (inner loop, sampler, packing optimizations)
    if opensloth_config.training_type == "sft":
        logger.start_timing("training_loop_patch")
        from opensloth.patching.inner_training_loop import patch_inner_training_loop
        from opensloth.patching.patch_log import patch_log
        from opensloth.patching.patch_sampler import patch_sampler
        from .patching.get_batch_samples import patch_get_batch_samples

        patch_log(trainer)
        patch_inner_training_loop(trainer, opensloth_config.sequence_packing)
        patch_get_batch_samples(opensloth_config)
        trainer = patch_sampler(trainer)  # type: ignore
        logger.finish_timing("training_loop_patch")
    else:
        logger.info(
            f"Skipping SFT-specific patches for training_type='{opensloth_config.training_type}'."
        )

    # Disable save operations on non-master ranks (multi-GPU only)
    if len(opensloth_config.devices) > 1 and os.getenv("OPENSLOTH_LOCAL_RANK") != "0":
        def _no_op(*args, **kwargs):
            pass
        trainer._save = _no_op  # type: ignore
        logger.info("Patched _save to no-op on non-master rank")

    # Always add epoch shuffle callback for visibility (safe for all trainers)
    from .patching.patch_sampler import ShuffleData
    trainer.add_callback(ShuffleData())
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    """Adjust per-device batch size for multi-GPU and silence logging on non-zero ranks."""
    if num_gpus != 1:
        # Trainer will internally shard per device; here we keep user semantics consistent.
        hf_train_args.per_device_train_batch_size *= 1  # no-op placeholder for future logic
    if gpu_ith != 0:
        hf_train_args.report_to = "none"


def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an appropriate trainer instance (SFT, DPO, etc.) with a dataset loaded from disk.
    """
    from datasets import load_from_disk
    from .trainer_factory import create_trainer_by_type

    logger = get_opensloth_logger()
    logger.info(f"Loading dataset from {opensloth_config.data_cache_path}")
    try:
        train_dataset = load_from_disk(opensloth_config.data_cache_path)

        # Dataset compatibility warning/validation (best-effort)
        _validate_dataset_compatibility(
            opensloth_config.data_cache_path,
            opensloth_config.fast_model_args.max_seq_length,
        )

        # Optional filtering for SFT only (DPO datasets lack input_ids)
        if (
            opensloth_config.training_type == "sft"
            and opensloth_config.filter_overlength_samples
        ):
            max_len = int(opensloth_config.fast_model_args.max_seq_length)
            before_n = len(train_dataset)

            def _len_ok(ex):
                ids = ex.get("input_ids")
                if ids is None:
                    return True
                try:
                    return len(ids) <= max_len
                except Exception:
                    return True

            train_dataset = train_dataset.filter(
                _len_ok, num_proc=getattr(hf_train_args, "dataset_num_proc", 1)
            )
            dropped = before_n - len(train_dataset)
            if dropped > 0:
                logger.warning(
                    f"Filtered {dropped}/{before_n} samples ({dropped/max(1,before_n):.2%}) > max_len={max_len}."
                )
        else:
            if opensloth_config.training_type != "sft":
                logger.info(
                    "Skipping over-length filtering (not applicable to non-SFT datasets)."
                )
    except Exception as e:
        logger.error(
            "Failed to load dataset. Ensure it was prepared correctly. Error: %s",
            e,
        )
        raise

    trainer = create_trainer_by_type(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        opensloth_config=opensloth_config,
        hf_train_args=hf_train_args,
    )
    logger.info(
        f"Trainer created for training_type='{opensloth_config.training_type.upper()}'"
    )
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    if num_gpus != 1:
        hf_train_args.per_device_train_batch_size *= num_gpus  # This is the total batch size loaded by dataloader, the trainer later will chose the correct batch size for each GPU

    if gpu_ith != 0:
        hf_train_args.report_to = "none"


__all__ = ["configure_batch_size", "create_trainer", "init_model_and_tokenizer"]
