from __future__ import annotations

import json
import os
from pathlib import Path

from opensloth.logging_config import get_opensloth_logger
from opensloth.opensloth_config import OpenSlothConfig


def ensure_dataset_features(dataset, required: list[str]) -> None:
    feats = getattr(dataset, "features", None)
    if feats is None:
        raise ValueError("Dataset missing features for SFT training")
    missing = [c for c in required if c not in feats]
    if missing:
        raise ValueError(f"Dataset missing required columns {missing} for SFT training")


def validate_dataset_compatibility(dataset_path: str, model_max_seq_length: int) -> None:
    logger = get_opensloth_logger()
    cfg_path = Path(dataset_path) / "dataset_config.json"
    if not cfg_path.exists():
        logger.warning("Dataset missing dataset_config.json; length mismatch checks skipped.")
        return
    try:
        with open(cfg_path) as f:
            data = json.load(f)
    except Exception as exc:
        logger.warning("Failed reading dataset_config.json: %s", exc)
        return
    ds_len = data.get("max_seq_length")
    if ds_len is None:
        logger.warning("dataset_config.json missing max_seq_length key")
        return
    if model_max_seq_length < ds_len:
        raise ValueError(
            "Training max_seq_length is smaller than dataset max_seq_length: "
            f"{model_max_seq_length} < {ds_len}. Rebuild dataset or increase training setting."
        )
    if model_max_seq_length > ds_len:
        logger.warning(
            "Training max_seq_length (%s) > dataset max_seq_length (%s); may waste padding.",
            model_max_seq_length,
            ds_len,
        )





def maybe_hot_fix_gemma(cfg: OpenSlothConfig, logger, tokenizer) -> None:
    if "gemma-3" in cfg.fast_model_args.model_name and cfg.sequence_packing:
        from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing
        logger.info("Applying Gemma-3 sequence packing patch.")
        patch_gemma3_unsloth_for_sequence_packing()
    if not hasattr(tokenizer, "pad") and cfg.sequence_packing:
        from transformers import AutoTokenizer
        logger.info("Tokenizer lacks pad(); patching from AutoTokenizer.")
        tk2 = AutoTokenizer.from_pretrained(cfg.fast_model_args.model_name)
        tokenizer.pad = tk2.pad  # type: ignore[attr-defined]


def setup_comm_backend(cfg: OpenSlothConfig) -> None:
    if len(cfg.devices) <= 1:
        return

    from opensloth.nccl_grad_sync import get_callback_and_setup_method
    _cb, setup_nccl, _destroy = get_callback_and_setup_method()
    setup_nccl(rank=int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0")), gpus=cfg.devices)
