"""Configuration builders for OpenSloth.

Centralises logic for constructing validated configuration objects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merges two dictionaries. Override values take precedence."""
    out = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_dataset_config(dataset_path: str) -> dict[str, Any]:
    """Loads the dataset_config.json file from the processed dataset directory."""
    cfg_file = Path(dataset_path) / "dataset_config.json"
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text())
    except Exception:
        return {}

class TrainingConfigBuilder:
    """
    Builds validated OpenSloth configuration objects by orchestrating defaults,
    dataset-inferred settings, and user-provided overrides.
    """

    def __init__(self, dataset_path: str, method: str = "sft") -> None:
        self.dataset_path = dataset_path
        self.method = method.lower()
        self._cli_overrides: dict[str, Any] = {}

    def with_cli_args(self, overrides: dict[str, Any]):
        """Applies configuration overrides from the command line."""
        self._cli_overrides = overrides
        return self

    def build(self) -> tuple[OpenSlothConfig, TrainingArguments]:
        """
        Constructs and validates the final configuration objects.

        The merge order of precedence is:
        1. Pydantic Model Defaults (lowest priority)
        2. Inferred settings from `dataset_config.json`
        3. CLI argument overrides (highest priority)
        """
        # 1. Infer settings from the dataset's metadata.
        dataset_cfg = _load_dataset_config(self.dataset_path)
        inferred_config: dict[str, Any] = {"opensloth_config": {}}
        
        # Infer model name and max sequence length from the dataset if available
        if dataset_cfg.get("tokenizer_name"):
            inferred_config["opensloth_config"]["fast_model_args"] = {"model_name": dataset_cfg["tokenizer_name"]}
        if dataset_cfg.get("max_seq_length"):
            inferred_config["opensloth_config"].setdefault("fast_model_args", {})["max_seq_length"] = dataset_cfg["max_seq_length"]
        
        # Infer device count from dataset sharding (with proper fallback logic)
        gpus = 1
        for key in ("gpus", "num_shards", "num_gpus"):
            if key in dataset_cfg:
                try:
                    gpus = int(dataset_cfg[key])
                    break
                except (ValueError, TypeError):
                    pass
        
        # Clamp to actual available CUDA device count to prevent oversubscription
        try:
            import torch
            if torch.cuda.is_available():
                available = torch.cuda.device_count()
                if available > 0 and gpus > available:
                    gpus = available
        except Exception:
            pass
            
        inferred_config["opensloth_config"]["devices"] = list(range(gpus))

        # 2. Merge inferred config with CLI overrides.
        # CLI overrides take precedence over inferred settings.
        final_config = _deep_merge(inferred_config, self._cli_overrides)

        # 3. Add mandatory/computed fields that are not part of the standard models.
        final_os_config = final_config.get("opensloth_config", {})
        final_ta_config = final_config.get("training_args", {})
        
        final_os_config["data_cache_path"] = self.dataset_path
        final_os_config["training_type"] = self.method
        
        # Special handling for GRPO task type presets
        if self.method == "grpo":
            grpo_args = final_os_config.get("grpo_args", {})
            if not grpo_args.get("task_type"):
                ds_lower = self.dataset_path.lower()
                if any(k in ds_lower for k in ["dapo", "math", "gsm", "openmath"]):
                    grpo_args["task_type"] = "math"
                else:
                    grpo_args["task_type"] = "general"
            final_os_config["grpo_args"] = grpo_args

        # 4. Pre-check for LoRA adapters to avoid conflicts
        # This must happen BEFORE creating the OpenSlothConfig object
        try:
            model_path = final_os_config.get("fast_model_args", {}).get("model_name")
            if model_path and Path(model_path).is_dir():
                adapter_file = Path(model_path) / "adapter_config.json"
                if adapter_file.exists() and self.method in {"dpo", "grpo"}:
                    # This is a pretrained LoRA model - clear any default LoRA config
                    final_os_config["pretrained_lora"] = model_path
                    # Explicitly set lora_args to None to prevent Pydantic defaults
                    final_os_config["lora_args"] = None
        except Exception:
            pass # Ignore errors in this best-effort check

        # 5. Instantiate Pydantic models. Pydantic will apply its own defaults
        # for any fields not provided in the final merged configuration.
        opensloth_config = OpenSlothConfig(**final_os_config)
        training_args = TrainingArguments(**final_ta_config)

        return opensloth_config, training_args


class PrepConfigBuilder:
    """Builder for dataset preparation configuration."""
    def __init__(self, method: str = "sft") -> None:
        self.method = method.lower()
        self._cfg: dict[str, Any] = {"training_type": self.method}
    def with_base(self, **kwargs):
        self._cfg.update({k: v for k, v in kwargs.items() if v is not None})
        return self
    def build(self) -> dict[str, Any]:
        return dict(self._cfg)

__all__ = ["PrepConfigBuilder", "TrainingConfigBuilder"]
