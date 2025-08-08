"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing

from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


def init_model_and_tokenizer(opensloth_config: OpenSlothConfig):
    """Initialize and optionally set up LoRA for the model."""

    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert len(cuda_devices) == 1

    from unsloth import FastLanguageModel

    logger = get_opensloth_logger()

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

    logger.start_timing("nccl_setup")
    from opensloth.nccl_grad_sync import get_callback_and_setup_method

    setup_nccl_for_opensloth = get_callback_and_setup_method()[1]
    setup_nccl_for_opensloth(
        rank=int(os.environ["OPENSLOTH_LOCAL_RANK"]),
        gpus=opensloth_config.devices,
    )
    logger.finish_timing("nccl_setup")

    model_device = model.device
    logger.info(
        f"Model loaded on device {model_device}, tokenizer: {tokenizer.__class__.__name__}"
    )

    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
    ):
        logger.start_timing("lora_setup")
        from unsloth import FastModel
        model = FastModel.get_peft_model(
            model,
            **opensloth_config.lora_args.model_dump(),  # type: ignore
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
    """Load or prepare the dataset and create the SFTTrainer."""

    # Get enhanced logger for timing

    logger = get_opensloth_logger()

    logger.start_timing("trainer_setup")

    trainer = _get_trainer(
        model,
        tokenizer,
        opensloth_config,
        hf_train_args,
    )

    logger.finish_timing("trainer_setup")

    logger.start_timing("training_loop_patch")
    from opensloth.patching.inner_training_loop import patch_inner_training_loop
    from opensloth.patching.patch_log import patch_log
    from opensloth.patching.patch_sampler import patch_sampler
    # from trl import SFTConfig, SFTTrainer

    patch_log(trainer)
    patch_inner_training_loop(trainer, opensloth_config.sequence_packing)

    from .patching.get_batch_samples import patch_get_batch_samples

    patch_get_batch_samples(opensloth_config)

    # ====
    trainer = patch_sampler(trainer)  # type: ignore
    logger.finish_timing("training_loop_patch")

    # Patch save_model, _save, and _maybe_log_save_evaluate to no-op on non-master ranks

    if os.getenv("OPENSLOTH_LOCAL_RANK") != "0":
        print(
            f"[RANK={os.getenv('OPENSLOTH_LOCAL_RANK')}] Patching trainer.save_model, trainer._save, and trainer._maybe_log_save_evaluate to no-op on non-master rank."
        )

        def no_op(*args, **kwargs):
            pass

        # trainer.save_model = no_op
        trainer._save = no_op

        # @patch
        # def _maybe_log_save_evaluate(self: type(trainer), *args, **kwargs):
        #     logger.info(
        #         "Skipping _maybe_log_save_evaluate on non-master rank to avoid unnecessary operations."
        #     )

        # trainer._maybe_log_save_evaluate = no_op

    # ===
    from .patching.patch_sampler import ShuffleData

    logger.info(f"Add callback ShuffleData to Trainer {trainer.__class__.__name__}")
    trainer.add_callback(ShuffleData())

    return trainer


def _ensure_data_correct(train_dataset):
    """
    Ensure the dataset is correctly formatted for training.
    Raises an error if the dataset is not in the expected format.
    """
    if (
        not hasattr(train_dataset, "features")
        or "input_ids" not in train_dataset.features
    ):
        raise ValueError(
            "Dataset must have 'input_ids' feature for training. "
            "Please check your dataset preparation."
        )
    if not hasattr(train_dataset, "features") or "labels" not in train_dataset.features:
        logger = get_opensloth_logger()
        logger.warning(
            "Dataset does not have 'labels' feature. "
            "This may affect training. Please check your dataset preparation."
        )


def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an SFTTrainer instance with a dataset loaded from disk.
    """
    from datasets import load_from_disk
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer

    from .logging_config import get_opensloth_logger

    logger = get_opensloth_logger()

    logger.info(f"Loading dataset from {opensloth_config.data_cache_path}")
    try:
        train_dataset = load_from_disk(opensloth_config.data_cache_path)
        _ensure_data_correct(train_dataset)
        
        # Validate dataset configuration compatibility
        _validate_dataset_compatibility(opensloth_config.data_cache_path, opensloth_config.fast_model_args.max_seq_length)
        
        # Proactive filtering: drop samples longer than model max length if enabled
        max_len = int(opensloth_config.fast_model_args.max_seq_length)
        if opensloth_config.filter_overlength_samples:
            before_n = len(train_dataset)
            def _len_ok(ex):
                ids = ex.get("input_ids")
                # Accept None (let collator handle) but prefer to keep simple robust check
                if ids is None:
                    return True
                # Some datasets store arrays; use generic len when possible
                try:
                    return len(ids) <= max_len
                except Exception:
                    return True
            train_dataset = train_dataset.filter(
                _len_ok,
                num_proc=getattr(hf_train_args, "dataset_num_proc", 1),
            )
            after_n = len(train_dataset)
            dropped = before_n - after_n
            if dropped > 0:
                logger.warning(
                    f"Filtered {dropped}/{before_n} samples ({dropped/max(1,before_n):.2%}) exceeding max_len={max_len}."
                )
            else:
                logger.info("No over-length samples found during dataset validation.")
    except Exception:
        logger.error(
            f"Failed to load dataset from {opensloth_config.data_cache_path}. "
            "Please verify that the path exists and the dataset is correctly prepared. "
            "Refer to the documentation at "
            "https://github.com/anhvth/opensloth/blob/main/cache_unsloth_dataset/README.md for guidance."
        )
        raise

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    hf_train_args.skip_prepare_dataset = True

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_train_args,  # type: ignore
        tokenizer=tokenizer,  # type: ignore
        data_collator=data_collator,
    )

    logger.info("Trainer setup completed successfully")
    return trainer


def configure_batch_size(hf_train_args, gpu_ith, num_gpus):
    if num_gpus != 1:
        hf_train_args.per_device_train_batch_size *= num_gpus  # This is the total batch size loaded by dataloader, the trainer later will chose the correct batch size for each GPU

    if gpu_ith != 0:
        hf_train_args.report_to = "none"


__all__ = [
    "configure_batch_size",
    "create_trainer",
    "init_model_and_tokenizer",
]
