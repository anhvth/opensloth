"""Trainer factory package: creates and configures trainer instances (SFT, DPO, GRPO, etc.).

Public entrypoints:
- setup_model_and_training: main high-level function used by scripts.

Internal structure:
- base.py: shared validation & utilities
- constructors.py: individual _create_*_trainer functions
- model_init.py: model/tokenizer initialization and comm backend
- dispatcher.py: mapping training_type -> constructor
"""
from .model_init import setup_model_and_training  # re-export for external use

__all__ = ["setup_model_and_training"]
