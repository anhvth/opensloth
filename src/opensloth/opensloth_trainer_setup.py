"""
Utility functions for multi-GPU training with Unsloth models.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import os

from opensloth.init_modules import (
    configure_batch_size,
    create_trainer,
    init_model_and_tokenizer,
)

from .opensloth_config import OpenSlothConfig, TrainingArguments

# from loguru import logger


# TODO: remove this part
def _change_compiler_location() -> None:
    import unsloth  # type: ignore  # noqa: F401
    from unsloth_zoo import compiler

    # ====== Patching the compiler location to avoid race conditions as it is shared between GPUs
    gpu_ith = int(os.environ["OPENSLOTH_LOCAL_RANK"])

    compiler.UNSLOTH_COMPILE_LOCATION = f".cache/{compiler.UNSLOTH_COMPILE_LOCATION}_{gpu_ith}"
    print(f"Using compiler location: {compiler.UNSLOTH_COMPILE_LOCATION}")


def setup_model_and_training(
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Setup the model, tokenizer, dataset, and trainer for multi-GPU training.
    """

    gpu_ith = int(os.environ["OPENSLOTH_LOCAL_RANK"])
    num_gpus = int(os.environ["OPENSLOTH_WORLD_SIZE"])

    # Get enhanced logger for timing
    from .logging_config import get_opensloth_logger

    logger = get_opensloth_logger()

    # Start total setup timing
    logger.start_timing("total_setup")

    _change_compiler_location()

    # Time batch size configuration
    configure_batch_size(hf_train_args, gpu_ith, num_gpus)

    # Time model initialization
    logger.start_timing("model_init")
    model, tokenizer = init_model_and_tokenizer(opensloth_config)
    logger.finish_timing("model_init")

    # Time trainer creation
    logger.start_timing("trainer_creation")
    trainer = create_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_creation")

    # Finish total setup timing
    logger.finish_timing("total_setup")

    return trainer, model, tokenizer
