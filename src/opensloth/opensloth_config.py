from multiprocessing import cpu_count
from typing import Any, Literal

from pydantic import BaseModel, Field

WORKERS = max(1, cpu_count() // 2)


class FastModelArgs(BaseModel):
    """Configuration for Unsloth's FastModel initialization.

    Derived from unsloth/models/loader.py: FastModel.from_pretrained
    """

    model_name: str
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    use_gradient_checkpointing: str = "unsloth"

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


def _default_target_modules() -> list[str]:
    """Default target modules for LoRA application."""
    return [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


class DPOArgs(BaseModel):
    """Configuration for DPO training parameters."""
    
    beta: float = Field(default=0.1, description="DPO beta parameter for preference strength")
    max_length: int = Field(default=1024, description="Maximum sequence length for DPO")
    max_prompt_length: int = Field(default=512, description="Maximum prompt length for DPO")
    
    class Config:
        """Pydantic configuration for DPOArgs."""
        extra = "allow"


class LoraArgs(BaseModel):
    """Configuration for LoRA parameters in PEFT."""

    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    bias: str = "none"
    random_state: int = 3407
    target_modules: list[str] = Field(
        default_factory=_default_target_modules,
        description="List of target modules for LoRA application",
    )
    use_rslora: bool = False

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"


class OpenSlothConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    data_cache_path: str = Field(
        description="Path to cache directory for datasets",
    )
    devices: list[int] = Field(default=[0], description="List of GPU indices to use")
    fast_model_args: FastModelArgs = Field(default_factory=FastModelArgs)
    lora_args: LoraArgs | None = Field(default_factory=LoraArgs)
    pretrained_lora: str | None = Field(
        default=None,
        description="Path to pretrained LoRA model for continous lora training",
    )
    sequence_packing: bool = Field(
        default=True,
        description="Disable packing of sequences for training",
    )
    
    # Training type and related configurations
    training_type: Literal["sft", "dpo", "kto", "orpo", "grpo"] = Field(
        default="sft",
        description="Type of training to perform: SFT, DPO, KTO, ORPO, or GRPO",
    )
    dpo_args: DPOArgs | None = Field(
        default=None,
        description="DPO-specific configuration parameters",
    )

    log_level: Literal["info", "debug"] = Field(
        default="info",
        description="Logging level for the training process",
    )

    # Dataset validation behavior
    filter_overlength_samples: bool = Field(
        default=True,
        description=(
            "If True, filter out dataset samples whose input_ids length exceeds "
            "fast_model_args.max_seq_length before training."
        ),
    )

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    # post assert ensure data_cache_path exists
    def model_post_init(self, __context: Any) -> None:
        """Post-initialization validation for OpenSlothConfig."""
        if self.data_cache_path is None:
            raise ValueError("data_cache_path must be specified")
        if not isinstance(self.devices, list) or not all(
            isinstance(d, int) for d in self.devices
        ):
            raise ValueError("devices must be a list of integers")
        if self.lora_args is not None and not isinstance(self.lora_args, LoraArgs):
            raise ValueError("lora_args must be an instance of LoraArgs")
        
        # Validate training type specific configurations
        if self.training_type == "dpo" and self.dpo_args is None:
            # Auto-create DPO args with defaults if not provided
            self.dpo_args = DPOArgs()
        elif self.training_type != "dpo" and self.dpo_args is not None:
            raise ValueError("dpo_args should only be specified when training_type is 'dpo'")
        
        # For future extensibility - validate other training types
        if self.training_type in ["kto", "orpo", "grpo"]:
            raise NotImplementedError(f"Training type '{self.training_type}' is not yet implemented. Currently only 'sft' and 'dpo' are supported.")


class TrainingArguments(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""

    output_dir: str = "saves/loras/"
    per_device_train_batch_size: int = 8
    learning_rate: float = 2e-4
    gradient_accumulation_steps: int = 16
    logging_steps: int = 1
    num_train_epochs: int = 1
    lr_scheduler_type: str = "linear"
    warmup_steps: int = 5
    save_total_limit: int = 2
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    save_only_model: bool = False
    resume_from_checkpoint: str | None = None

    seed: int = 42
    report_to: Literal["tensorboard", "wandb", "none"] = "tensorboard"
    eval_strategy: str = "no"  # must be no, when using multigpus
    dataset_num_proc: int = WORKERS

    class Config:
        """Pydantic configuration for DataConfig."""

        extra = "allow"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for TrainingArguments initialization."""
        return self.model_dump()
