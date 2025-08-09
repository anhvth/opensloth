"""
Trainer factory module for creating different types of trainers (SFT, DPO, KTO, etc.)
based on the OpenSlothConfig training_type.
"""

import os
from typing import Any

from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


def _ensure_data_correct_for_training_type(train_dataset, training_type: str):
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
        if not hasattr(train_dataset, "features") or "labels" not in train_dataset.features:
            logger.warning(
                "Dataset does not have 'labels' feature. "
                "This may affect SFT training. Please check your dataset preparation."
            )
    
    elif training_type == "dpo":
        # DPO requires specific columns: prompt, chosen, rejected
        required_features = ["prompt", "chosen", "rejected"]
        if not hasattr(train_dataset, "features"):
            raise ValueError(
                f"Dataset must have features for DPO training. "
                f"Required features: {required_features}"
            )
        
        missing_features = [f for f in required_features if f not in train_dataset.features]
        if missing_features:
            raise ValueError(
                f"Dataset missing required features for DPO training: {missing_features}. "
                f"Please ensure your dataset has columns: {required_features}. "
                f"Available features: {list(train_dataset.features.keys())}"
            )
        
        logger.info(f"DPO dataset validation passed. Features: {list(train_dataset.features.keys())}")
    
    else:
        raise NotImplementedError(f"Dataset validation for training type '{training_type}' is not implemented yet")


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
    
    # Validate dataset for SFT
    _ensure_data_correct_for_training_type(train_dataset, "sft")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    hf_train_args.skip_prepare_dataset = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_train_args,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        data_collator=data_collator,
    )
    
    logger.info("SFTTrainer setup completed successfully")
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
    
    # Ensure PatchDPOTrainer is called
    try:
        from unsloth import PatchDPOTrainer
        PatchDPOTrainer()
        logger.info("Successfully applied PatchDPOTrainer for Unsloth compatibility")
    except ImportError as e:
        logger.error(f"Failed to import PatchDPOTrainer from unsloth: {e}")
        raise ImportError(
            "PatchDPOTrainer is required for DPO training with Unsloth. "
            "Please ensure you have the latest version of unsloth installed."
        )
    except Exception as e:
        logger.warning(f"PatchDPOTrainer call failed: {e}. Continuing anyway...")
    
    # Import DPO components
    try:
        from trl import DPOTrainer, DPOConfig
    except ImportError as e:
        raise ImportError(
            f"Failed to import DPO components from trl: {e}. "
            "Please ensure you have TRL installed with DPO support."
        )
    
    # Validate dataset for DPO
    _ensure_data_correct_for_training_type(train_dataset, "dpo")
    
    # Get DPO configuration
    dpo_args = opensloth_config.dpo_args
    if dpo_args is None:
        raise ValueError("dpo_args must be configured for DPO training")
    
    # Create DPO configuration by combining TrainingArguments with DPO-specific args
    dpo_config_dict = hf_train_args.to_dict()
    dpo_config_dict.update({
        "beta": dpo_args.beta,
        "max_length": dpo_args.max_length,
        "max_prompt_length": dpo_args.max_prompt_length,
    })
    
    dpo_config = DPOConfig(**dpo_config_dict)
    
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # Let DPO create reference model automatically
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=dpo_args.beta,
        max_length=dpo_args.max_length,
        max_prompt_length=dpo_args.max_prompt_length,
    )
    
    logger.info("DPOTrainer setup completed successfully")
    return trainer


def _create_kto_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create a KTOTrainer instance."""
    # TODO: Implement KTO trainer creation
    raise NotImplementedError("KTO training is not yet implemented")


def _create_orpo_trainer(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create an ORPOTrainer instance."""
    # TODO: Implement ORPO trainer creation
    raise NotImplementedError("ORPO training is not yet implemented")


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
        raise ValueError("grpo_args must be set for GRPO training")

    # Import TRL's GRPO components
    try:
        from trl import GRPOTrainer, GRPOConfig
    except ImportError as e:
        raise ImportError(
            f"Failed to import GRPO components from trl: {e}. "
            "Please ensure you have TRL installed with GRPO support."
        )

    # Import math rewards
    try:
        from .training.grpo_rewards_math import (
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
            prepare_math_rewards,
        )
        logger.info("Math reasoning reward functions loaded")
    except ImportError as e:
        logger.warning(f"Failed to import math reward functions: {e}")
        # Fallback to basic rewards
        match_format_exactly = lambda *args, **kwargs: [0.0]
        match_format_approximately = lambda *args, **kwargs: [0.0] 
        check_answer = lambda *args, **kwargs: [0.0]
        check_numbers = lambda *args, **kwargs: [0.0]

    # Create GRPOConfig by combining TrainingArguments with GRPO-specific args
    grpo_config_dict = hf_train_args.to_dict()
    
    # Filter out arguments not supported by GRPOConfig
    unsupported_args = {
        'dataset_num_proc', 'save_only_model', 'push_to_hub_model_id', 
        'push_to_hub_organization', 'push_to_hub_token', 'hub_model_id',
        'hub_token', 'hub_private_repo', 'hub_strategy', 'hub_always_push',
        'gradient_checkpointing_kwargs', 'include_inputs_for_metrics',
        'eval_do_concat_batches', 'fp16_full_eval', 'tf32', 'jit_mode_eval',
        'use_ipex', 'bf16_full_eval', 'eval_on_start', 'ignore_data_skip',
        'fsdp_config', 'deepspeed_plugin', 'label_smoothing_factor',
        'debug', 'sharded_ddp', 'accelerator_config', 'dispatch_batches',
        'split_batches', 'include_tokens_per_second', 'neftune_noise_alpha',
        'optim_target_modules', 'batch_eval_metrics', 'eval_use_gather_object'
    }
    
    # Remove unsupported arguments
    filtered_config_dict = {k: v for k, v in grpo_config_dict.items() if k not in unsupported_args}
    
    # Add GRPO-specific parameters using correct parameter names
    filtered_config_dict.update({
        "num_generations": grpo_args.group_size,
        "max_completion_length": grpo_args.max_new_tokens,  # Correct parameter name
        "temperature": grpo_args.temperature,
        "top_p": grpo_args.top_p,
        "beta": grpo_args.kl_coef,  # GRPO uses beta instead of kl_coef
        "max_prompt_length": grpo_args.max_prompt_length,
    })
    
    # Add top_k if specified
    if grpo_args.top_k is not None:
        filtered_config_dict["top_k"] = grpo_args.top_k

    try:
        grpo_config = GRPOConfig(**filtered_config_dict)
    except Exception as e:
        logger.error(f"Failed to create GRPOConfig with args: {list(filtered_config_dict.keys())}")
        raise ValueError(f"GRPOConfig creation failed: {e}")
    
    # Validate dataset for GRPO
    # GRPO typically expects prompts/completions rather than input_ids/labels
    if not hasattr(train_dataset, "features"):
        raise ValueError("Dataset must have features for GRPO training")
    
    # GRPO can work with various dataset formats - check what we have
    available_features = list(train_dataset.features.keys())
    logger.info(f"GRPO dataset features: {available_features}")
    
    # Expected dataset formats for GRPO:
    # - 'prompt' column for prompts
    # - or 'input_ids' that can be decoded to prompts
    if "prompt" not in available_features and "input_ids" not in available_features:
        raise ValueError(
            "GRPO dataset must have either 'prompt' or 'input_ids' column. "
            f"Available features: {available_features}"
        )

    # Prepare reward functions list
    reward_funcs = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]

    # Create trainer - check if it's Unsloth's version that needs reward_funcs
    try:
        trainer = GRPOTrainer(
            model=model,
            ref_model=None,  # Let GRPO create reference model automatically
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,  # Try with reward_funcs first
        )
        logger.info("UnslothGRPOTrainer setup completed successfully with reward_funcs")
    except TypeError as e:
        if "reward_funcs" in str(e):
            # Fallback to standard TRL GRPOTrainer without reward_funcs
            try:
                trainer = GRPOTrainer(
                    model=model,
                    ref_model=None,
                    args=grpo_config,
                    train_dataset=train_dataset,
                    processing_class=tokenizer,
                )
                logger.info("Standard TRL GRPOTrainer setup completed successfully")
            except Exception as e2:
                raise ValueError(f"Both UnslothGRPOTrainer and TRL GRPOTrainer failed: {e}, {e2}")
        else:
            raise e
    
    return trainer


def create_trainer_by_type(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Factory function to create the appropriate trainer based on training_type.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        train_dataset: The training dataset
        opensloth_config: OpenSloth configuration
        hf_train_args: Training arguments
        
    Returns:
        The appropriate trainer instance (SFTTrainer, DPOTrainer, etc.)
    """
    logger = get_opensloth_logger()
    training_type = opensloth_config.training_type
    
    logger.info(f"Creating trainer for training type: {training_type}")
    
    # Factory mapping
    trainer_factory_map = {
        "sft": _create_sft_trainer,
        "dpo": _create_dpo_trainer,
        "kto": _create_kto_trainer,
        "orpo": _create_orpo_trainer,
        "grpo": _create_grpo_trainer,
    }
    
    if training_type not in trainer_factory_map:
        raise ValueError(
            f"Unsupported training type: {training_type}. "
            f"Supported types: {list(trainer_factory_map.keys())}"
        )
    
    factory_func = trainer_factory_map[training_type]
    
    try:
        trainer = factory_func(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            opensloth_config=opensloth_config,
            hf_train_args=hf_train_args,
        )
        return trainer
    except Exception as e:
        logger.error(f"Failed to create {training_type.upper()} trainer: {e}")
        raise
