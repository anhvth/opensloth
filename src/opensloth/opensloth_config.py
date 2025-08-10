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
    # vLLM-specific parameters for GRPO training
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


class DPOArgs(BaseModel):
    """Configuration for DPO training parameters."""
    
    beta: float = Field(default=0.1, description="DPO beta parameter for preference strength")
    max_length: int = Field(default=1024, description="Maximum sequence length for DPO")
    max_prompt_length: int = Field(default=512, description="Maximum prompt length for DPO")
    
    class Config:
        """Pydantic configuration for DPOArgs."""
        extra = "allow"


class GRPOArgs(BaseModel):
    """Configuration for GRPO (Group Relative Policy Optimization) with configurable rewards.

    This supports different task types (math, code, general) with appropriate reward functions.
    """
    group_size: int = Field(default=4, description="Number of sampled responses per prompt", json_schema_extra={'cli_alias': 'group-size'})
    max_new_tokens: int = Field(default=128, description="Max new tokens to sample per response", json_schema_extra={'cli_alias': 'max-new'})
    temperature: float = Field(default=1.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, description="Nucleus sampling top_p")
    top_k: int | None = Field(default=None, description="Top-k sampling (None = disabled)")
    min_p: float = Field(default=0.1, description="Minimum probability threshold")
    kl_coef: float = Field(default=0.05, description="KL coefficient vs reference policy", json_schema_extra={'cli_alias': 'beta'})
    
    # Task-specific configuration
    task_type: str = Field(default="general", description="Task type: 'math', 'code', 'general', 'reasoning'", json_schema_extra={'cli_alias': 'task'})
    reward_functions: list[str] = Field(default_factory=list, description="List of reward function names to apply (auto-selected if empty)", json_schema_extra={'cli_alias': 'rewards'})
    use_custom_chat_template: bool = Field(default=True, description="Use task-specific chat template")
    
    # Prompt processing
    max_prompt_length: int = Field(default=512, description="Maximum prompt token length (truncation boundary)", json_schema_extra={'cli_alias': 'max-prompt-len'})
    prompt_length_percentile: float = Field(default=0.9, description="Percentile for automatic prompt length filtering")
    
    # Training control  
    eval_interval: int = Field(default=200, description="Steps between live preview / eval events")
    save_interval: int = Field(default=500, description="Steps between adapter/model saves")
    print_sample_every: int = Field(default=5, description="Print sample generations every N steps")
    
    # vLLM sampling parameters
    stop_sequences: list[str] = Field(default_factory=list, description="Additional stop sequences for generation")
    include_stop_str_in_output: bool = Field(default=True, description="Include stop string in vLLM output")
    
    class Config:
        extra = "allow"


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
    grpo_args: GRPOArgs | None = Field(
        default=None,
        description="GRPO-specific configuration parameters",
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

    @model_validator(mode='after')
    def validate_training_type_args(self) -> "OpenSlothConfig":
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
        if self.training_type == "dpo":
            if self.dpo_args is None:
                self.dpo_args = DPOArgs()
            if self.grpo_args is not None:
                raise ValueError("grpo_args should not be set for DPO training")
        elif self.training_type == "grpo":
            if self.grpo_args is None:
                self.grpo_args = GRPOArgs()
            if self.dpo_args is not None:
                raise ValueError("dpo_args should not be set for GRPO training")
        else:
            # For SFT, ensure other args are not present.
            if self.dpo_args is not None or self.grpo_args is not None:
                import warnings
                warnings.warn("For SFT training, dpo_args and grpo_args will be ignored.")
            self.dpo_args = None
            self.grpo_args = None

        # Still gate not-yet-implemented types (kto, orpo) explicitly
        if self.training_type in ["kto", "orpo"]:
            raise NotImplementedError(
                f"Training type '{self.training_type}' is not yet implemented. Supported: sft, dpo, grpo"
            )
        
        # Ensure model_name is provided
        if self.fast_model_args and not self.fast_model_args.model_name:
             raise ValueError("fast_model_args.model_name must be specified.")

        return self


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
