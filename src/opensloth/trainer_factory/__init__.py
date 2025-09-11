"""Trainer factory package: creates and configures trainer instances (SFT only).

Public entrypoints:
- setup_model_and_training: main high-level function used by scripts.

Internal structure:
- base.py: shared validation & utilities
- model_init.py: model/tokenizer initialization and comm backend
"""
from .model_init import setup_model_and_training  # re-export for external use

__all__ = ["setup_model_and_training"]
