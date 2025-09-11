from enum import Enum
from multiprocessing import cpu_count
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

WORKERS = max(1, cpu_count() // 2)


class FastModelArgs(BaseModel):
    """Configuration for Unsloth's FastModel initialization.

    Derived from unsloth/models/loader.py: FastModel.from_pretrained
    """

    model_name: str = Field(
        ..., 
        description="The model name or path to use.", 
        json_schema_extra={'cli_alias': 'model'}
    )
    max_seq_length: int = Field(
        4096, 
        description="Maximum sequence length for the model.", 
        json_schema_extra={'cli_alias': 'max-seq-length'}
    )
    load_in_4bit: bool = Field(
        True, 
        description="Load the model in 4-bit (QLoRA).", 
        json_schema_extra={'cli_alias': 'load-in-4bit'}
    )
    load_in_8bit: bool = Field(
        False, 
        description="Load the model in 8-bit.", 
        json_schema_extra={'cli_alias': 'load-in-8bit'}
    )
    full_finetuning: bool = Field(
        False, 
        description="Perform full fine-tuning instead of LoRA.", 
        json_schema_extra={'cli_alias': 'full-finetuning'}
    )
    use_gradient_checkpointing: str = "unsloth"
    fast_inference: bool = False
    max_lora_rank: int | None = None
    gpu_memory_utilization: float = 0.7

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

class LoraArgs(BaseModel):
    """Configuration for LoRA parameters in PEFT."""

    finetune_vision_layers: bool = False
    finetune_language_layers: bool = True
    finetune_attention_modules: bool = True
    finetune_mlp_modules: bool = True
    r: int = Field(8, description="LoRA rank (`r`).", json_schema_extra={'cli_alias': 'lora-r'})
    lora_alpha: int = Field(16, description="LoRA alpha.", json_schema_extra={'cli_alias': 'lora-alpha'})  # Updated default
    lora_dropout: float = Field(0.0, description="LoRA dropout.", json_schema_extra={'cli_alias': 'lora-dropout'})
    bias: str = "none"
    random_state: int = 3407
    target_modules: list[str] = Field(
        default_factory=_default_target_modules,
        description="List of target modules for LoRA application",
        json_schema_extra={'cli_alias': 'targets'}
    )
    use_rslora: bool = Field(False, description="Use RSLoRA (Rank-Stabilized LoRA).", json_schema_extra={'cli_alias': 'use-rslora'})



class OpenSlothConfig(BaseModel):
    """Main configuration class combining all sub-configurations."""

    data_cache_path: str = Field(
        description="Path to cache directory for datasets",
    )
    devices: list[int] = Field(default=[0], description="List of GPU indices to use")
    fast_model_args: FastModelArgs
    lora_args: LoraArgs | None = Field(default=None)
    pretrained_lora: str | None = Field(
        default=None,
        description="Path to pretrained LoRA model for continuous LoRA training",
        json_schema_extra={'cli_alias': 'pretrained-lora'}
    )
    sequence_packing: bool = Field(
        default=True,
        description="Disable packing of sequences for training",
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




OpenSlothConfig.model_rebuild()


class DatasetPrepConfig(BaseModel):
    """Configuration for dataset preparation.

    This mirrors the arguments used by the os-data CLI but provides
    a clean, typed interface for programmatic usage and auto-generation.
    """

    # Model/tokenizer
    tokenizer_name: str = Field(
        description="Tokenizer or model identifier/path",
        json_schema_extra={"cli_alias": "model"},
    )
    chat_template: str = Field("chatml", description="Chat template name to apply")

    # Dataset source
    dataset_name: str = Field(
        description="HF dataset 'repo' or path to a local JSON/JSONL file.",
        json_schema_extra={"cli_alias": "input"},
    )
    split: str = Field(default="train", description="Dataset split (for HF datasets)")

    # Processing
    num_samples: int = Field(
        default=-1,
        description="Number of samples to process (-1 for all)",
        json_schema_extra={"cli_alias": "samples"},
    )
    num_proc: int = Field(
        default=8,
        description="Workers for dataset map/tokenization",
        json_schema_extra={"cli_alias": "workers"},
    )
    gpus: int = Field(
        default=1, description="Number of GPU shards to create for the dataset."
    )
    output_dir: str | None = Field(
        default=None,
        description="Output directory for processed data.",
        json_schema_extra={"cli_alias": "data-output"},
    )

    # Labeling
    train_on_target_only: bool = Field(
        default=True,
        description="If True, mask non-assistant tokens (response-only training).",
    )
    instruction_part: str = Field(
        default="<|im_start|>user\n",
        description="Marker that begins a user/instruction turn",
    )
    response_part: str = Field(
        default="<|im_start|>assistant\n",
        description="Marker that begins an assistant/response turn",
    )
    max_seq_length: int = Field(
        4096,
        description="Maximum sequence length for tokenization.",
        json_schema_extra={"cli_alias": "max-seq-length"},
    )

    # Debug
    debug: int = Field(
        default=0, description="If >0, enable debug mode and dump samples"
    )

    # Authentication
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face token for accessing gated models/datasets",
    )

    class Config:
        extra = "allow"




class TrainingArguments(BaseModel):
    """Configuration for Hugging Face TrainingArguments."""

    output_dir: str = Field("saves/loras/", description="Output directory for checkpoints and logs.", json_schema_extra={'cli_alias': 'output'})
    per_device_train_batch_size: int = Field(2, description="Batch size per GPU.", json_schema_extra={'cli_alias': 'bs'})
    learning_rate: float = Field(2e-4, description="The initial learning rate.", json_schema_extra={'cli_alias': 'lr'})
    gradient_accumulation_steps: int = Field(4, description="Number of gradient accumulation steps.", json_schema_extra={'cli_alias': 'grad-accum'})
    logging_steps: int = 1
    num_train_epochs: int = Field(3, description="Total number of training epochs.", json_schema_extra={'cli_alias': 'epochs'})
    lr_scheduler_type: str = "linear"
    warmup_steps: int = Field(10, description="Warmup steps for LR scheduler.", json_schema_extra={'cli_alias': 'warmup'})
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
    
