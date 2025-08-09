"""Configuration builders for OpenSloth.

Centralises logic for constructing validated configuration objects.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import json

from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

TRAINING_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "description": "Quick test", 
        "config": {
            "training_args": {
                "max_steps": 50, 
                "per_device_train_batch_size": 1, 
                "gradient_accumulation_steps": 4, 
                "learning_rate": 2e-4, 
                "logging_steps": 1, 
                "save_total_limit": 1, 
                "report_to": "none"
            }
        }
    },
    "small": {
        "description": "Small GPU setup (8GB VRAM)",
        "config": {
            "training_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "warmup_steps": 50,
                "num_train_epochs": 3,
                "logging_steps": 10,
                "save_total_limit": 2,
                "optim": "adamw_8bit"
            },
            "opensloth_config": {
                "fast_model_args": {
                    "load_in_4bit": True,
                    "max_seq_length": 2048
                }
            }
        }
    },
    "large": {
        "description": "Large GPU setup (24GB+ VRAM)",
        "config": {
            "training_args": {
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 4,
                "learning_rate": 2e-4,
                "warmup_steps": 100,
                "num_train_epochs": 3,
                "logging_steps": 10,
                "save_total_limit": 3,
                "optim": "adamw_torch"
            },
            "opensloth_config": {
                "fast_model_args": {
                    "load_in_4bit": False,
                    "max_seq_length": 4096
                }
            }
        }
    },
    "memory-efficient": {
        "description": "Memory-efficient setup for limited VRAM",
        "config": {
            "training_args": {
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 16,
                "learning_rate": 1e-4,
                "warmup_steps": 30,
                "num_train_epochs": 1,
                "logging_steps": 5,
                "save_total_limit": 1,
                "optim": "adamw_8bit",
                "dataloader_pin_memory": False
            },
            "opensloth_config": {
                "fast_model_args": {
                    "load_in_4bit": True,
                    "max_seq_length": 1024
                },
                "sequence_packing": True
            }
        }
    },
}

def _deep_merge(base: Dict, override: Dict) -> Dict:
    out = dict(base)
    for k, v in override.items():
        if v is None:
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _load_dataset_config(dataset_path: str) -> Dict[str, Any]:
    cfg_file = Path(dataset_path) / "dataset_config.json"
    if not cfg_file.exists():
        return {}
    try:
        return json.loads(cfg_file.read_text())
    except Exception:
        return {}

class TrainingConfigBuilder:
    def __init__(self, dataset_path: str, method: str = "sft") -> None:
        self.dataset_path = dataset_path
        self.method = method.lower()
        self._preset_name: Optional[str] = None
        self._cli_overrides: Dict[str, Any] = {}
        self._dataset_cfg: Dict[str, Any] = {}
        self._finalised = False
        self._merged: Dict[str, Any] = {}

    def with_preset(self, name: Optional[str]):
        if name:
            if name not in TRAINING_PRESETS:
                raise ValueError(f"Unknown preset '{name}'")
            self._preset_name = name
        return self

    def with_cli_args(self, **kwargs):
        self._cli_overrides.update(kwargs)
        return self

    def infer_from_dataset(self):
        self._dataset_cfg = _load_dataset_config(self.dataset_path)
        return self

    def finalise(self):
        if self._finalised:
            return self
        defaults: Dict[str, Any] = {
            "opensloth_config": {"training_type": self.method, "devices": [0], "sequence_packing": True, "log_level": "info", "fast_model_args": {"max_seq_length": 4096, "load_in_4bit": True, "load_in_8bit": False, "full_finetuning": False}, "lora_args": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]}},
            "training_args": {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "learning_rate": 2e-4, "num_train_epochs": 3, "lr_scheduler_type": "linear", "warmup_steps": 10, "logging_steps": 1, "save_total_limit": 2, "optim": "adamw_8bit", "weight_decay": 0.01, "report_to": "tensorboard"},
        }
        if self._preset_name:
            defaults = _deep_merge(defaults, TRAINING_PRESETS[self._preset_name]["config"])
        if self._dataset_cfg:
            fm = defaults["opensloth_config"]["fast_model_args"]
            if self._dataset_cfg.get("tok_name") and not fm.get("model_name"):
                fm["model_name"] = self._dataset_cfg["tok_name"]
            if self._dataset_cfg.get("max_seq_length") and not self._cli_overrides.get("max_seq_length"):
                fm["max_seq_length"] = self._dataset_cfg["max_seq_length"]
        cli_struct: Dict[str, Any] = {"opensloth_config": {}, "training_args": {}}
        fm_cli: Dict[str, Any] = {}
        if self._cli_overrides.get("model"):
            fm_cli["model_name"] = self._cli_overrides["model"]
        if fm_cli:
            cli_struct["opensloth_config"]["fast_model_args"] = fm_cli
        defaults = _deep_merge(defaults, cli_struct)
        defaults["opensloth_config"]["data_cache_path"] = self.dataset_path
        gpus = 1
        for key in ("gpus", "num_shards", "num_gpus"):
            if key in self._dataset_cfg:
                try:
                    gpus = int(self._dataset_cfg[key])
                    break
                except Exception:
                    pass
        defaults["opensloth_config"]["devices"] = list(range(gpus))
        # Set sequence_packing default based on training method
        if self.method == "sft":
            defaults["opensloth_config"]["sequence_packing"] = True
        else:
            defaults["opensloth_config"]["sequence_packing"] = False
        self._merged = defaults
        self._finalised = True
        return self

    def build(self) -> Tuple[OpenSlothConfig, TrainingArguments]:
        if not self._finalised:
            self.finalise()
        os_cfg = OpenSlothConfig(**self._merged["opensloth_config"])
        tr_args = TrainingArguments(**self._merged["training_args"])
        if not os_cfg.fast_model_args.model_name:
            raise ValueError("Model name required")
        return os_cfg, tr_args

class PrepConfigBuilder:
    def __init__(self, method: str = "sft") -> None:
        self.method = method.lower()
        self._cfg: Dict[str, Any] = {"training_type": self.method}
    def with_base(self, **kwargs):
        self._cfg.update({k: v for k, v in kwargs.items() if v is not None})
        return self
    def build(self) -> Dict[str, Any]:
        return dict(self._cfg)

__all__ = ["TrainingConfigBuilder", "PrepConfigBuilder", "TRAINING_PRESETS"]
