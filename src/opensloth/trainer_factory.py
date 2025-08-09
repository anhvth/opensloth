"""
Trainer factory and setup utilities for creating trainers (SFT, DPO, GRPO, etc.)
based on `OpenSlothConfig.training_type`.
"""

from typing import Dict, Tuple, Optional
import os

from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


# --------------------------- Dataset validation helpers ---------------------------

def _ensure_data_correct_for_training_type(train_dataset, training_type: str) -> None:
    """
    Ensure the dataset is correctly formatted for the specified training type.
    Raises an error if the dataset is not in the expected format.
    """
    logger = get_opensloth_logger()

    if training_type == "sft":
        # SFT requires input_ids and optionally labels
        if (
            not hasattr(train_dataset, "features")
            or "input_ids" not in train_dataset.features
        ):
            raise ValueError(
                "Dataset must have 'input_ids' feature for SFT training. "
                "Please check your dataset preparation."
            )
        if "labels" not in getattr(train_dataset, "features", {}):
            logger.warning(
                "Dataset does not have 'labels' feature. "
                "This may affect SFT training. Please check your dataset preparation."
            )

    elif training_type == "dpo":
        # DPO requires specific columns: prompt, chosen, rejected
        required_features = ["prompt", "chosen", "rejected"]
        if not hasattr(train_dataset, "features"):
            raise ValueError(
                "Dataset must have features for DPO training. "
                f"Required features: {required_features}"
            )

        missing = [f for f in required_features if f not in train_dataset.features]
        if missing:
            available = list(getattr(train_dataset, "features", {}).keys())
            raise ValueError(
                "Dataset missing required features for DPO training: "
                f"{missing}. Please ensure your dataset has columns: {required_features}. "
                f"Available features: {available}"
            )

        logger.info("DPO dataset validation passed.")

    else:
        raise NotImplementedError(
            f"Dataset validation for training type '{training_type}' is not implemented."
        )


# --------------------------- Trainer constructors ---------------------------

def _create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create an SFTTrainer instance."""
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer

    logger = get_opensloth_logger()
    _ensure_data_correct_for_training_type(train_dataset, "sft")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    # Some stacks add this flag; harmless if unused.
    setattr(hf_train_args, "skip_prepare_dataset", True)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_train_args,          # type: ignore
        tokenizer=tokenizer,         # type: ignore
        data_collator=data_collator,
    )

    logger.info("SFTTrainer setup completed successfully.")
    return trainer


def _create_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create a DPOTrainer instance."""
    logger = get_opensloth_logger()

    # Ensure PatchDPOTrainer is called (Unsloth compatibility)
    try:
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
        logger.info("Applied PatchDPOTrainer for Unsloth compatibility.")
    except ImportError as e:
        logger.error(f"Failed to import PatchDPOTrainer from unsloth: {e}")
        raise ImportError(
            "PatchDPOTrainer is required for DPO training with Unsloth. "
            "Please ensure you have the latest version of 'unsloth' installed."
        )
    except Exception as e:
        logger.warning(f"PatchDPOTrainer call failed: {e}. Continuing...")

    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        raise ImportError(
            f"Failed to import DPO components from 'trl': {e}. "
            "Please ensure you have TRL installed with DPO support."
        )

    _ensure_data_correct_for_training_type(train_dataset, "dpo")

    dpo_args = opensloth_config.dpo_args
    if dpo_args is None:
        raise ValueError("`dpo_args` must be configured for DPO training.")

    # Merge HF TrainingArguments with DPO-specific settings
    dpo_config_dict = hf_train_args.to_dict()
    dpo_config_dict.update(
        {
            "beta": dpo_args.beta,
            "max_length": dpo_args.max_length,
            "max_prompt_length": dpo_args.max_prompt_length,
        }
    )
    dpo_config = DPOConfig(**dpo_config_dict)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Let DPO create the reference model automatically
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=dpo_args.beta,
        max_length=dpo_args.max_length,
        max_prompt_length=dpo_args.max_prompt_length,
    )

    logger.info("DPOTrainer setup completed successfully.")
    return trainer


def _create_kto_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create a KTOTrainer instance."""
    raise NotImplementedError("KTO training is not yet implemented.")


def _create_orpo_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create an ORPOTrainer instance."""
    raise NotImplementedError("ORPO training is not yet implemented.")


def _create_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create a GRPO trainer instance using TRL's native GRPOTrainer."""
    logger = get_opensloth_logger()
    grpo_args = opensloth_config.grpo_args
    if grpo_args is None:
        raise ValueError("`grpo_args` must be set for GRPO training.")

    # Import TRL's GRPO components
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        raise ImportError(
            f"Failed to import GRPO components from 'trl': {e}. "
            "Please ensure you have TRL installed with GRPO support."
        )

    # Import reward helpers (best effort)
    try:
        from .grpo_rewards import (
            get_reward_functions,
            create_reward_preset,
            get_chat_template_for_task,  # imported for side-effects / parity
        )
        logger.info("GRPO reward functions loaded.")
    except Exception as e:
        logger.warning(f"Failed to import GRPO reward functions: {e}")

        def _dummy_reward(*args, **kwargs):
            # Return a list of zeros per completion
            if len(args) > 1 and isinstance(args[1], (list, tuple)):
                return [0.0] * len(args[1])
            return [0.0]

        def get_reward_functions(_names):
            class _RF:
                name = "dummy"

                def __call__(self, prompts, completions, **_kw):
                    return _dummy_reward(prompts, completions)

            return [_RF()]

        def create_reward_preset(_task_type):
            return ["dummy"]

    # Build GRPOConfig from HF TrainingArguments + GRPO-specific args
    grpo_config_dict = hf_train_args.to_dict()

    # Remove args not supported by GRPOConfig to avoid constructor errors
    unsupported_args = {
        "dataset_num_proc",
        "save_only_model",
        "push_to_hub_model_id",
        "push_to_hub_organization",
        "push_to_hub_token",
        "hub_model_id",
        "hub_token",
        "hub_private_repo",
        "hub_strategy",
        "hub_always_push",
        "gradient_checkpointing_kwargs",
        "include_inputs_for_metrics",
        "eval_do_concat_batches",
        "fp16_full_eval",
        "tf32",
        "jit_mode_eval",
        "use_ipex",
        "bf16_full_eval",
        "eval_on_start",
        "ignore_data_skip",
        "fsdp_config",
        "deepspeed_plugin",
        "label_smoothing_factor",
        "debug",
        "sharded_ddp",
        "accelerator_config",
        "dispatch_batches",
        "split_batches",
        "include_tokens_per_second",
        "neftune_noise_alpha",
        "optim_target_modules",
        "batch_eval_metrics",
        "eval_use_gather_object",
    }
    filtered_config = {k: v for k, v in grpo_config_dict.items() if k not in unsupported_args}

    # Map our args -> GRPO parameter names
    filtered_config.update(
        {
            "num_generations": grpo_args.group_size,
            "max_completion_length": grpo_args.max_new_tokens,
            "temperature": grpo_args.temperature,
            "top_p": grpo_args.top_p,
            "beta": grpo_args.kl_coef,  # GRPO uses 'beta' for KL weight
            "max_prompt_length": grpo_args.max_prompt_length,
        }
    )
    if grpo_args.top_k is not None:
        filtered_config["top_k"] = grpo_args.top_k

    try:
        grpo_config = GRPOConfig(**filtered_config)
    except Exception as e:
        logger.error(f"Failed to create GRPOConfig with keys: {list(filtered_config.keys())}")
        raise ValueError(f"GRPOConfig creation failed: {e}")

    # Validate dataset format
    if not hasattr(train_dataset, "features"):
        raise ValueError("Dataset must have features for GRPO training.")

    available_features = list(train_dataset.features.keys())
    logger.info(f"GRPO dataset features: {available_features}")

    if "prompt" not in available_features and "input_ids" not in available_features:
        raise ValueError(
            "GRPO dataset must have either 'prompt' or 'input_ids' column. "
            f"Available features: {available_features}"
        )

    # Setup reward functions
    reward_names = grpo_args.reward_functions or create_reward_preset(grpo_args.task_type)
    reward_functions = get_reward_functions(reward_names)
    logger.info(f"Using reward functions: {[getattr(rf, 'name', repr(rf)) for rf in reward_functions]}")

    def _wrap_reward(rf):
        def trl_reward(prompts, completions, **kwargs):
            return rf(prompts, completions, **kwargs)
        return trl_reward

    reward_funcs = [_wrap_reward(rf) for rf in reward_functions]

    # Create trainer; handle TRL/Unsloth signature differences
    try:
        trainer = GRPOTrainer(
            model=model,
            ref_model=None,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
        )
        logger.info("UnslothGRPOTrainer setup completed successfully with reward_funcs.")
    except TypeError as e:
        if "reward_funcs" in str(e):
            # Fallback to standard TRL GRPOTrainer without reward_funcs
            trainer = GRPOTrainer(
                model=model,
                ref_model=None,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
            logger.info("Standard TRL GRPOTrainer setup completed successfully.")
        else:
            raise

    return trainer


def _create_trainer_by_type(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Factory function to create the appropriate trainer based on training_type.
    """
    logger = get_opensloth_logger()
    training_type = opensloth_config.training_type
    logger.info(f"Creating trainer for training type: {training_type}")

    factory = {
        "sft": _create_sft_trainer,
        "dpo": _create_dpo_trainer,
        "kto": _create_kto_trainer,
        "orpo": _create_orpo_trainer,
        "grpo": _create_grpo_trainer,
    }.get(training_type)

    if factory is None:
        supported = ["sft", "dpo", "kto", "orpo", "grpo"]
        raise ValueError(f"Unsupported training type: {training_type}. Supported types: {supported}")

    try:
        return factory(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            opensloth_config=opensloth_config,
            hf_train_args=hf_train_args,
        )
    except Exception as e:
        logger.error(f"Failed to create {training_type.upper()} trainer: {e}")
        raise


# --------------------------- Model/tokenizer init & patches ---------------------------

def _validate_dataset_compatibility(dataset_path: str, model_max_seq_length: int) -> None:
    """Validate that the dataset is compatible with the training configuration."""
    import json
    from pathlib import Path

    dataset_config_path = Path(dataset_path) / "dataset_config.json"
    if not dataset_config_path.exists():
        logger = get_opensloth_logger()
        logger.warning(
            f"Dataset at {dataset_path} was prepared without max_seq_length specification. "
            f"This may cause issues if sequences exceed training max_seq_length ({model_max_seq_length}). "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length}."
        )
        return

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    dataset_max_seq_length = dataset_config.get("max_seq_length")
    if dataset_max_seq_length is None:
        logger = get_opensloth_logger()
        logger.warning(
            f"Dataset at {dataset_path} lacks 'max_seq_length'. "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length}."
        )
        return

    if model_max_seq_length < dataset_max_seq_length:
        raise ValueError(
            "Training configuration mismatch!\n"
            f"Dataset was prepared with max_seq_length={dataset_max_seq_length}, "
            f"but training config specifies max_seq_length={model_max_seq_length}.\n"
            "The training max_seq_length must be >= dataset max_seq_length.\n"
            "Either:\n"
            f"  1. Increase training max_seq_length to {dataset_max_seq_length} or higher\n"
            f"  2. Re-prepare the dataset with --max-seq-length {model_max_seq_length}"
        )

    if model_max_seq_length > dataset_max_seq_length:
        logger = get_opensloth_logger()
        logger.warning(
            f"Training max_seq_length ({model_max_seq_length}) is larger than "
            f"dataset max_seq_length ({dataset_max_seq_length}). This is OK but may be inefficient. "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length} for optimal performance."
        )


def _maybe_hot_fix_gemma(opensloth_config: OpenSlothConfig, logger, tokenizer) -> None:
    if "gemma-3" in opensloth_config.fast_model_args.model_name and opensloth_config.sequence_packing:
        from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing

        logger.info("Detected Gemma-3 model, applying Unsloth patch for sequence packing.")
        patch_gemma3_unsloth_for_sequence_packing()

    if not hasattr(tokenizer, "pad") and opensloth_config.sequence_packing:
        logger.info(
            "Tokenizer missing 'pad' method; attempting to patch using transformers.AutoTokenizer. "
            "This may indicate an Unsloth issue."
        )
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            opensloth_config.fast_model_args.model_name,
        )
        tokenizer.pad = hf_tokenizer.pad  # type: ignore[attr-defined]


def _init_model_and_tokenizer(
    opensloth_config: OpenSlothConfig,
    unsloth_modules: Optional[Dict[str, object]] = None,
) -> Tuple[object, object]:
    """Initialize base/LoRA model and tokenizer; set up NCCL if multi-GPU."""
    logger = get_opensloth_logger()

    # Use modules from the dictionary if provided, otherwise import (backward compatibility)
    if unsloth_modules is not None:
        FastLanguageModel = unsloth_modules["FastLanguageModel"]
    else:
        from unsloth import FastLanguageModel  # type: ignore

    logger.start_timing("model_loading")
    if opensloth_config.pretrained_lora:
        logger.info(f"Loading model from {opensloth_config.pretrained_lora} with LoRA weights")
        opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora

    model, tokenizer = FastLanguageModel.from_pretrained(**opensloth_config.fast_model_args.model_dump())
    _maybe_hot_fix_gemma(opensloth_config, logger, tokenizer)
    logger.finish_timing("model_loading")

    # NCCL setup only if >1 GPU
    if len(opensloth_config.devices) > 1:
        logger.start_timing("nccl_setup")
        from opensloth.nccl_grad_sync import get_callback_and_setup_method

        setup_nccl_for_opensloth = get_callback_and_setup_method()[1]
        setup_nccl_for_opensloth(
            rank=int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0")),
            gpus=opensloth_config.devices,
        )
        logger.finish_timing("nccl_setup")
    else:
        logger.info("Single GPU detected; skipping NCCL setup.")

    logger.info(f"Model device: {model.device} | Tokenizer: {tokenizer.__class__.__name__}")

    # Setup LoRA if requested and not loading a pretrained LoRA
    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
        and opensloth_config.lora_args is not None
    ):
        logger.start_timing("lora_setup")
        if unsloth_modules is not None:
            FastModel = unsloth_modules["FastModel"]
        else:
            from unsloth import FastModel  # type: ignore

        model = FastModel.get_peft_model(model, **opensloth_config.lora_args.model_dump())  # type: ignore
        logger.finish_timing("lora_setup")

    return model, tokenizer


# --------------------------- Trainer creation & global patches ---------------------------

def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an appropriate trainer instance (SFT, DPO, GRPO, etc.) with a dataset loaded from disk.
    """
    from datasets import load_from_disk

    logger = get_opensloth_logger()
    base_path = opensloth_config.data_cache_path
    local_rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    shard_path = os.path.join(base_path, f"shard_{local_rank}")
    if os.path.isdir(shard_path):
        logger.info(f"Loading rank-specific shard dataset from {shard_path}")
        dataset_load_path = shard_path
    else:
        logger.info(f"Shard path {shard_path} not found. Falling back to base dataset path {base_path}")
        dataset_load_path = base_path

    try:
        train_dataset = load_from_disk(dataset_load_path)

        # Dataset compatibility check (best-effort)
        _validate_dataset_compatibility(
            base_path,
            opensloth_config.fast_model_args.max_seq_length,
        )

        # Optional filtering for SFT only (DPO datasets lack input_ids)
        if opensloth_config.training_type == "sft" and opensloth_config.filter_overlength_samples:
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
                _len_ok,
                num_proc=getattr(hf_train_args, "dataset_num_proc", 1),
            )
            dropped = before_n - len(train_dataset)
            if dropped > 0:
                logger.warning(
                    f"Filtered {dropped}/{before_n} samples "
                    f"({dropped / max(1, before_n):.2%}) > max_len={max_len}."
                )
        else:
            if opensloth_config.training_type != "sft":
                logger.info("Skipping over-length filtering (not applicable to non-SFT datasets).")
    except Exception as e:
        logger.error(f"Failed to load dataset. Ensure it was prepared correctly. Error: {e}")
        raise




    trainer = _create_trainer_by_type(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        opensloth_config=opensloth_config,
        hf_train_args=hf_train_args,
    )
    logger.info(f"Trainer created for training_type='{opensloth_config.training_type.upper()}'")
    return trainer

def _configure_batch_size(hf_train_args: TrainingArguments) -> None:
    """
    Adjust per-device batch size settings for multi-GPU and silence logging on non-zero ranks.

    Note: We keep per-device batch size as specified by the user to avoid surprising scaling.
    """
    gpu_ith = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    num_gpus = int(os.environ.get("OPENSLOTH_WORLD_SIZE", "1"))
    if num_gpus != 1:
        # Intentionally a no-op for now; place-holder for future logic.
        hf_train_args.per_device_train_batch_size *= 1
    if gpu_ith != 0:
        hf_train_args.report_to = "none"


def _create_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create trainer (SFT/DPO/GRPO/etc.) and apply universal performance patches."""
    logger = get_opensloth_logger()
    logger.start_timing("trainer_setup")
    trainer = _get_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_setup")

    # Apply UNIVERSAL patches that all trainers need for multi-GPU
    logger.start_timing("universal_patch")
    _configure_batch_size(hf_train_args)
    # Removed custom sampler & batch patching: pre-sharded datasets handle per-rank isolation.
    logger.finish_timing("universal_patch")

    # Apply SFT-specific patches (inner loop, etc.)
    if opensloth_config.training_type == "sft":
        logger.start_timing("sft_specific_patch")
        from opensloth.patching.inner_training_loop import patch_inner_training_loop_for_sft
        from opensloth.patching.patch_log import patch_log_for_sft

        patch_log_for_sft(trainer)
        patch_inner_training_loop_for_sft(trainer, opensloth_config.sequence_packing)
        logger.finish_timing("sft_specific_patch")
    else:
        logger.info(f"Skipping SFT-specific patches for training_type='{opensloth_config.training_type}'.")

    # Disable save operations on non-master ranks (multi-GPU only)
    if len(opensloth_config.devices) > 1 and os.getenv("OPENSLOTH_LOCAL_RANK", "0") != "0":
        def _no_op(*_args, **_kwargs):
            pass

        trainer._save = _no_op  # type: ignore[attr-defined]
        logger.info("Patched _save to no-op on non-master rank.")

    # Always add epoch shuffle callback for visibility (safe for all trainers)
    # ShuffleData callback removed; per-rank sharded datasets rely on standard sampler behavior.

    return trainer


def setup_model_and_training(
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
    unsloth_modules: Optional[Dict[str, object]] = None,
):
    """
    Setup the model, tokenizer, dataset, and trainer for (potentially) multi-GPU training.
    """

    logger = get_opensloth_logger()

    # Start total setup timing
    logger.start_timing("total_setup")

    # Configure batch size
    

    # Model initialization
    logger.start_timing("model_init")
    model, tokenizer = _init_model_and_tokenizer(opensloth_config, unsloth_modules)
    logger.finish_timing("model_init")

    # Trainer creation
    logger.start_timing("trainer_creation")
    trainer = _create_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_creation")

    # Finish total setup timing
    logger.finish_timing("total_setup")

    return trainer, model, tokenizer
