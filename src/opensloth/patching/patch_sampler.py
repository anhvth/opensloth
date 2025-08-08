import os
import random
from typing import Iterator

from datasets import Dataset, IterableDataset
from fastcore.all import patch
from torch.utils.data.sampler import SequentialSampler
from transformers import Trainer, TrainerCallback

from opensloth.logging_config import get_opensloth_logger


class ShuffleData(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        local_rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))

        if local_rank != 0:
            return

        logger = get_opensloth_logger(allow_unknown_gpu=True)
        logger.info(f"🔄 Starting epoch {state.epoch + 1}")

        # try:
        #     from .._debug_dataloader import debug_chat_dataloader_for_training

        #     tok = kwargs["processing_class"]
        #     debug_chat_dataloader_for_training(train_dataloader, tokenizer=tok)
        #     logger.info(
        #         "📋 Dataloader examples logged to " ".log/dataloader_examples.html"
        #     )
        # except Exception as e:
        #     logger.debug(f"Dataloader debugging failed (non-critical): {e}")


class RandomSamplerSeededByEpoch(SequentialSampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.epoch = 0
        self.logger = get_opensloth_logger(allow_unknown_gpu=True)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        dataset_size = len(self.data_source)
        ids = list(range(dataset_size))

        # Shuffle with epoch-specific seed
        R = random.Random(42 + self.epoch)
        R.shuffle(ids)

        self.logger.info(
            f"🎲 Sampler epoch {self.epoch}: emitting {dataset_size} indices\nFirst ids dataset samples: {ids[:10]}\n...Last ids: {ids[-10:]}"
        )
        yield_ids = []
        for idx in ids:
            # self.logger.info(f"📤 Emitting index: {idx}")
            yield_ids.append(idx)
            yield idx
        # write to log for debugging
        self.logger.info(
            f"🎲 Sampler epoch {self.epoch}: dataset_size={dataset_size}\n"
            f"   📋 First 10 indices: {yield_ids[:10]}\n"
            f"   📋 Last 10 indices: {yield_ids[-10:]}"
        )


from ..opensloth_config import OpenSlothConfig


def patch_sampler(trainer: Trainer) -> None:
    """Patch trainer to use RandomSamplerSeededByEpoch."""
    print("🔧 Patching Trainer to use RandomSamplerSeededByEpoch")

    def _get_train_sampler(train_dataset=None) -> RandomSamplerSeededByEpoch:
        if train_dataset is None:
            train_dataset = trainer.train_dataset
        return RandomSamplerSeededByEpoch(train_dataset)

    trainer._get_train_sampler = _get_train_sampler
    return trainer
