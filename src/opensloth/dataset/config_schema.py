"""Pydantic schemas for dataset preparation and training UI."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DatasetPrepConfig(BaseModel):
	"""Configuration for dataset preparation.

	This mirrors the arguments used by the os-data CLI but provides
	a clean, typed interface for programmatic usage and auto-generation.
	"""
	# Model/tokenizer
	tok_name: str = Field(description="Tokenizer or model identifier/path", json_schema_extra={'cli_alias': 'model'})
	chat_template: str = Field("chatml", description="Chat template name to apply")

	# Dataset source
	dataset_name: str = Field(
		description="HF dataset 'repo' or path to a local JSON/JSONL file.",
        json_schema_extra={'cli_alias': 'input'}
	)
	split: str = Field(default="train", description="Dataset split (for HF datasets)")

	# Processing
	num_samples: int = Field(default=-1, description="Number of samples to process (-1 for all)", json_schema_extra={'cli_alias': 'samples'})
	num_proc: int = Field(default=8, description="Workers for dataset map/tokenization", json_schema_extra={'cli_alias': 'workers'})
	gpus: int = Field(default=1, description="Number of GPU shards to create for the dataset.")
	output_dir: str | None = Field(default=None, description="Output directory for processed data.", json_schema_extra={'cli_alias': 'data-output'})

	# Labeling
	train_on_target_only: bool = Field(
		default=True, description="If True, mask non-assistant tokens (response-only training)."
	)
	instruction_part: str = Field(
		default="<|im_start|>user\n", description="Marker that begins a user/instruction turn"
	)
	response_part: str = Field(
		default="<|im_start|>assistant\n", description="Marker that begins an assistant/response turn"
	)
	max_seq_length: int = Field(4096, description="Maximum sequence length for tokenization.", json_schema_extra={'cli_alias': 'max-seq-length'})
	training_type: str = Field("sft", description="The training method (sft, dpo, grpo).", json_schema_extra={'hidden': True})

	# Debug
	debug: int = Field(default=0, description="If >0, enable debug mode and dump samples")

	# Authentication
	hf_token: str | None = Field(default=None, description="Hugging Face token for accessing gated models/datasets")

	class Config:
		extra = "allow"


class TrainingUIConfig(BaseModel):
    """Configuration for training UI (renamed for clarity)."""
    # Placeholder for any existing Training UI config
    pass
