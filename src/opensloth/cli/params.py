"""
Pydantic models for CLI parameters to improve extensibility and maintainability.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class PrepareDataParams(BaseModel):
    """Parameters for the prepare-data command."""
    model: str = Field(..., description="Model name or path")
    method: str = Field(default="sft", description="Training method")
    input_file: Optional[str] = Field(default=None, description="Input file path")
    dataset: Optional[str] = Field(default=None, description="Dataset name")
    chat_template: Optional[str] = Field(default=None, description="Chat template name")
    target_only: bool = Field(default=False, description="Train on target only")
    split: str = Field(default="train", description="Dataset split")
    samples: int = Field(default=1000, description="Number of samples")
    max_seq_length: int = Field(default=4096, description="Maximum sequence length")
    workers: int = Field(default=4, description="Number of workers")
    gpus: int = Field(default=1, description="Number of GPUs")
    output: Optional[str] = Field(default=None, description="Output directory")
    debug: int = Field(default=0, description="Debug level")
    dry_run: bool = Field(default=False, description="Dry run mode")
    force: bool = Field(default=False, description="Force overwrite")


class TrainingParams(BaseModel):
    """Parameters for the train command."""
    # Required parameters
    dataset: str = Field(..., description="Path to processed dataset directory")
    
    # Training configuration
    method: str = Field(default="sft", description="Training method")
    model: Optional[str] = Field(default=None, description="Model name or path")
    output: Optional[str] = Field(default=None, description="Output directory")
    
    # Training hyperparameters
    epochs: Optional[int] = Field(default=None, description="Number of epochs")
    max_steps: Optional[int] = Field(default=None, description="Maximum steps")
    batch_size: Optional[int] = Field(default=None, description="Batch size per device")
    grad_accum: Optional[int] = Field(default=None, description="Gradient accumulation steps")
    learning_rate: Optional[float] = Field(default=None, description="Learning rate")
    max_seq_length: Optional[int] = Field(default=None, description="Maximum sequence length")
    
    # Model configuration
    load_4bit: bool = Field(default=True, description="Load model in 4-bit mode")
    full_finetune: bool = Field(default=False, description="Full fine-tuning mode")
    
    # LoRA configuration
    lora_r: Optional[int] = Field(default=None, description="LoRA rank")
    lora_alpha: Optional[int] = Field(default=None, description="LoRA alpha")
    
    # Presets and execution
    preset: Optional[str] = Field(default=None, description="Training preset")
    dry_run: bool = Field(default=False, description="Dry run mode")
    
    # Multi-GPU and tmux configuration
    use_tmux: bool = Field(default=False, description="Use tmux for multi-GPU training")
    tmux_session: Optional[str] = Field(default=None, description="Tmux session name")
    tmux_auto_kill: bool = Field(default=False, description="Auto-kill existing tmux session")


class DebugParams(BaseModel):
    """Parameters for the debug command."""
    dataset: str = Field(..., description="Dataset directory to debug")
    samples: int = Field(default=5, description="Number of samples to analyze")


class InfoParams(BaseModel):
    """Parameters for the info command."""
    path: str = Field(..., description="Path to processed dataset directory")
