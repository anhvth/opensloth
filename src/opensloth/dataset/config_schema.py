"""Pydantic schemas for dataset preparation and training UI."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class DatasetPrepConfig(BaseModel):
	"""Configuration for dataset preparation.

	This mirrors the arguments used by BaseDatasetPreparer but provides
	a clean, typed interface for GUIs and programmatic usage.
	"""

	# Model/tokenizer
	tok_name: str = Field(description="Tokenizer or model identifier/path")
	chat_template: str = Field(description="Chat template name to apply")

	# Dataset source
	dataset_name: str = Field(
		description="HF dataset 'repo' or path to a local JSON/JSONL file with messages/conversations",
	)
	split: str = Field(default="train", description="Dataset split (for HF datasets)")

	# Processing
	num_samples: int = Field(default=-1, description="Number of samples to process (-1 for all)")
	num_proc: int = Field(default=8, description="Workers for dataset map/tokenization")
	output_dir: Optional[str] = Field(default=None, description="Output directory")

	# Labeling
	train_on_target_only: bool = Field(
		default=True, description="If True, mask non-assistant tokens (response-only training)."
	)
	instruction_part: str = Field(
		default="", description="Marker that begins a user/instruction turn"
	)
	response_part: str = Field(
		default="", description="Marker that begins an assistant/response turn"
	)

	# Debug
	debug: int = Field(default=0, description="If >0, enable debug mode and dump samples")

	# Authentication
	hf_token: Optional[str] = Field(default=None, description="Hugging Face token for accessing gated models/datasets")

	class Config:
		extra = "allow"
