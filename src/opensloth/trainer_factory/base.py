from __future__ import annotations

import json
import os
from pathlib import Path

from opensloth.logging_config import get_opensloth_logger
from opensloth.opensloth_config import OpenSlothConfig


def ensure_dataset_features(dataset, required: list[str], training_type: str) -> None:
    feats = getattr(dataset, "features", None)
    if feats is None:
        raise ValueError(f"Dataset missing features for training_type={training_type}")
    missing = [c for c in required if c not in feats]
    if missing:
        raise ValueError(f"Dataset missing required columns {missing} for {training_type}")


def validate_dataset_for_type(dataset, training_type: str) -> None:
    if training_type == "sft":
        ensure_dataset_features(dataset, ["input_ids", "labels"], training_type)
    elif training_type == "dpo":
        ensure_dataset_features(dataset, ["prompt", "chosen", "rejected"], training_type)
    elif training_type == "grpo":
        # GRPO can use prompt or pre-tokenized input_ids
        feats = getattr(dataset, "features", None)
        if feats is None or ("prompt" not in feats and "input_ids" not in feats):
            raise ValueError("GRPO dataset must have either 'prompt' or 'input_ids'")
    else:
        raise ValueError(f"Unsupported training_type for validation: {training_type}")


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


def apply_grpo_model_args(cfg: OpenSlothConfig, model_args: dict) -> None:
    if cfg.training_type != "grpo":
        return
    
    # Always enable fast_inference (vLLM) for GRPO
    model_args["fast_inference"] = True
    
    # Set max_lora_rank for vLLM memory calculation
    if model_args.get("max_lora_rank") is None:
        if cfg.lora_args:
            # New LoRA training
            model_args["max_lora_rank"] = cfg.lora_args.r
        elif cfg.pretrained_lora:
            # Pretrained LoRA - extract rank from adapter config
            try:
                import json
                from pathlib import Path
                adapter_path = Path(cfg.pretrained_lora) / "adapter_config.json"
                if adapter_path.exists():
                    with open(adapter_path) as f:
                        adapter_config = json.load(f)
                    lora_rank = adapter_config.get("r", 8)  # Default to 8 if not found
                    model_args["max_lora_rank"] = lora_rank
                else:
                    # Fallback default
                    model_args["max_lora_rank"] = 8
            except Exception:
                # Fallback default
                model_args["max_lora_rank"] = 8
    
    model_args.setdefault("gpu_memory_utilization", 0.6)


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
