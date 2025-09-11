#!/usr/bin/env python3
#type: ignore
"""
General dataset preparation script for fine-tuning language models.
Supports any model with configurable chat templates, tokenization, and formatting.
"""
import argparse
import contextlib
import os
import random
from typing import cast

import datasets

import unsloth
from unsloth.chat_templates import get_chat_template, standardize_data_formats

from transformers import AutoTokenizer

from pydantic import BaseModel, Field


from typing import Dict, Any

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
    input_file: str | None = Field(
        default=None,
        description="Path to local input file (overrides dataset_name if specified)",
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
        default=False,
        description="If True, mask non-assistant tokens (response-only training).",
    )
    instruction_part: str = Field(
        default="",
        description="Marker that begins a user/instruction turn",
    )
    response_part: str = Field(
        default="",
        description="Marker that begins an assistant/response turn",
    )
    max_seq_length: int = Field(
        4096,
        description="Maximum sequence length for tokenization.",
        json_schema_extra={"cli_alias": "max-seq-length"},
    )

    # Authentication
    hf_token: str | None = Field(
        default=None,
        description="Hugging Face token for accessing gated models/datasets",
    )

    class Config:
        extra = "allow"


def get_training_config_template(model_name: str, num_gpus: int = 1, max_seq_length: int = 4096) -> Dict[str, Any]:
    """Get a training configuration template."""
    return {
        "opensloth_config": {
            "data_cache_path": None,  # Will be set by dataset preparation
            "devices": list(range(num_gpus)),
            "fast_model_args": {
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "load_in_4bit": True,
                "load_in_8bit": False,
                "full_finetuning": False,
                "use_gradient_checkpointing": "unsloth",
                "fast_inference": False,
                "max_lora_rank": None,
                "gpu_memory_utilization": 0.7
            },
            "lora_args": {
                "finetune_vision_layers": False,
                "finetune_language_layers": True,
                "finetune_attention_modules": True,
                "finetune_mlp_modules": True,
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "bias": "none",
                "random_state": 3407,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "use_rslora": False
            },
            "pretrained_lora": None,
            "sequence_packing": True,
            "log_level": "info",
            "filter_overlength_samples": True
        },
        "training_args": {
            "output_dir": "saves/loras/",
            "per_device_train_batch_size": 2,
            "learning_rate": 2e-4,
            "gradient_accumulation_steps": 4,
            "logging_steps": 1,
            "num_train_epochs": 3,
            "lr_scheduler_type": "linear",
            "warmup_steps": 10,
            "save_total_limit": 2,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "save_only_model": False,
            "resume_from_checkpoint": None,
            "seed": 42,
            "report_to": "tensorboard",
            "eval_strategy": "no",
            "dataset_num_proc": 8
        }
    }


def prepare_dataset(config: DatasetPrepConfig):
    """Main function to prepare dataset for any model."""

    # Clean access to configuration parameters - no verbose dictionary parsing needed!
    tokenizer_name = config.tokenizer_name
    chat_template = config.chat_template
    dataset_name = config.dataset_name
    split = config.split
    num_samples = config.num_samples
    num_proc = config.num_proc
    gpus = config.gpus
    output_dir = config.output_dir
    max_seq_length = config.max_seq_length
    train_on_target_only = config.train_on_target_only
    instruction_part = config.instruction_part
    response_part = config.response_part

    # If input_file is specified via dataset_name parameter, use it as a file path
    if hasattr(config, 'input_file') and config.input_file:
        dataset_name = config.input_file
    elif os.path.exists(dataset_name):
        # dataset_name is actually a file path
        pass

    # Validate arguments
    if train_on_target_only:
        assert instruction_part, "instruction_part is required when train_on_target_only=True"
        assert response_part, "response_part is required when train_on_target_only=True"

    # Validate output directory
    if not output_dir:
        raise ValueError("output_dir must be specified")

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer = get_chat_template(tokenizer, chat_template=chat_template)

    # Load dataset
    if os.path.exists(dataset_name):
        data = load_local_file(dataset_name)
        dataset = datasets.Dataset.from_list(data)
    else:
        dataset = datasets.load_dataset(dataset_name, split=split)

        # Ensure we have a Dataset object, not IterableDataset
        if isinstance(dataset, datasets.IterableDataset):
            raise ValueError(
                "IterableDataset is not supported. Please use a dataset that can be loaded as Dataset. "
                "Try using a smaller dataset or check the dataset configuration."
            )

        # Cast to Dataset to help type checker
        dataset = cast(datasets.Dataset, dataset)

        dataset = standardize_data_formats(dataset)

    # Select samples
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))  # type: ignore

    # Format conversations
    formatting_func = create_formatting_func(
        tokenizer, train_on_target_only, instruction_part, response_part, tokenizer_name
    )
    dataset = dataset.map(formatting_func, batched=True)  # type: ignore

    # Tokenize and prepare labels
    processing_func = create_processing_func(
        tokenizer, train_on_target_only, instruction_part, response_part
    )
    data = dataset.map(processing_func, num_proc=num_proc)  # type: ignore

    # Filter out examples that are too long or have no training labels
    filter_func = create_filter_func(max_seq_length, train_on_target_only)

    original_count = len(data)  # type: ignore
    data = data.filter(filter_func, num_proc=num_proc)  # type: ignore
    filtered_count = len(data)  # type: ignore

    # Check if dataset is empty and provide helpful error
    if len(data) == 0:
        raise RuntimeError("Dataset preparation failed: No samples remaining after processing")

    # Save dataset
    os.makedirs(output_dir, exist_ok=True)

    num_shards = max(1, int(gpus))
    # Optional deterministic shuffle before sharding so each shard gets diverse samples
    if num_shards > 1:
        with contextlib.suppress(Exception):
            data = data.shuffle(seed=42)  # type: ignore

    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{i}")
        shard_dataset = data.shard(num_shards=num_shards, index=i)  # type: ignore
        shard_dataset.save_to_disk(shard_path)  # type: ignore

    # Save metadata and configurations using Pydantic object directly
    dataset_size = len(data)  # type: ignore
    save_dataset_metadata(output_dir, config, dataset_size)
    save_dataset_config(output_dir, config)
    # Save training configuration
    training_config_template = get_training_config_template(config.tokenizer_name, config.gpus, config.max_seq_length)
    save_training_config(output_dir, config, training_config_template)


def load_local_file(file_path: str) -> list:
    """Load data from a local JSON/JSONL file."""
    import json

    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        if file_path.endswith('.jsonl'):
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        else:
            data = json.load(f)

    return data


def create_formatting_func(tokenizer, train_on_target_only: bool, instruction_part: str, response_part: str, tokenizer_name: str):
    """Create formatting function for conversations."""
    def format_conversation(examples):
        """Format conversations for tokenization."""
        formatted_texts = []

        for messages in examples["conversations"]:
            # Apply chat template
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            formatted_texts.append(formatted_text)

        return {"text": formatted_texts}

    return format_conversation


def create_processing_func(tokenizer, train_on_target_only: bool, instruction_part: str, response_part: str):
    """Create processing function for tokenization and labeling."""
    def process_example(examples):
        """Process examples with tokenization and optional labeling."""
        # Tokenize the text
        tokenized = tokenizer(
            examples["text"],
            truncation=False,
            padding=False,
            add_special_tokens=True
        )

        # Prepare labels for training
        if train_on_target_only:
            labels = []
            for i, input_ids in enumerate(tokenized["input_ids"]):
                text = examples["text"][i]
                label_ids = [-100] * len(input_ids)  # Default to ignore

                # Find response parts and mark them for training
                response_start = text.find(response_part)
                if response_start != -1:
                    # Tokenize up to response start to find token position
                    prefix = text[:response_start]
                    prefix_tokens = tokenizer(prefix, add_special_tokens=True)["input_ids"]

                    # Mark response tokens for training
                    for j in range(len(prefix_tokens) - 1, len(input_ids)):
                        if j < len(label_ids):
                            label_ids[j] = input_ids[j]

                labels.append(label_ids)

            tokenized["labels"] = labels
        else:
            tokenized["labels"] = tokenized["input_ids"]

        return tokenized

    return process_example


def create_filter_func(max_seq_length: int, train_on_target_only: bool):
    """Create filter function for dataset."""
    def filter_example(example):
        """Filter out examples that are too long or have no training labels."""
        # Filter by sequence length
        if len(example["input_ids"]) > max_seq_length:
            return False

        # Filter by training labels if train_on_target_only
        if train_on_target_only:
            if "labels" not in example or not any(label != -100 for label in example["labels"]):
                return False

        return True

    return filter_example


def save_dataset_metadata(output_dir: str, config: DatasetPrepConfig, dataset_size: int):
    """Save dataset metadata."""
    import json
    from datetime import datetime

    metadata = {
        "created_at": datetime.now().isoformat(),
        "dataset_size": dataset_size,
        "config_hash": hash(str(config.model_dump())),
        "tokenizer_name": config.tokenizer_name,
        "max_seq_length": config.max_seq_length,
        "num_shards": config.gpus
    }

    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def save_dataset_config(output_dir: str, config: DatasetPrepConfig):
    """Save dataset configuration."""
    import json

    with open(os.path.join(output_dir, "dataset_config.json"), 'w') as f:
        json.dump(config.model_dump(), f, indent=2)


def save_training_config(output_dir: str, config: DatasetPrepConfig, training_config: dict):
    """Save training configuration template."""
    import json

    with open(os.path.join(output_dir, "training_config.json"), 'w') as f:
        json.dump(training_config, f, indent=2)


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning language models")

    # Model/tokenizer options
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer or model identifier/path")
    parser.add_argument("--chat_template", type=str, default="chatml", help="Chat template name to apply")

    # Dataset options
    parser.add_argument("--dataset_name", type=str, required=True, help="HF dataset repo or path to local JSON/JSONL file")
    parser.add_argument("--input_file", type=str, help="Path to local input file (overrides dataset_name)")
    parser.add_argument("--split", type=str, default="train", help="Dataset split (for HF datasets)")

    # Processing options
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples to process (-1 for all)")
    parser.add_argument("--num_proc", type=int, default=8, help="Workers for dataset map/tokenization")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPU shards to create")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for processed data")

    # Labeling options
    parser.add_argument("--train_on_target_only", action="store_true", help="Mask non-assistant tokens (response-only training)")
    parser.add_argument("--instruction_part", type=str, help="Marker that begins a user/instruction turn")
    parser.add_argument("--response_part", type=str, help="Marker that begins an assistant/response turn")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for tokenization")

    # Authentication
    parser.add_argument("--hf_token", type=str, help="Hugging Face token for accessing gated models/datasets")

    args = parser.parse_args()

    # Create configuration from command line arguments
    config = DatasetPrepConfig(
        tokenizer_name=args.tokenizer_name,
        chat_template=args.chat_template,
        dataset_name=args.dataset_name,
        input_file=args.input_file,
        split=args.split,
        num_samples=args.num_samples,
        num_proc=args.num_proc,
        gpus=args.gpus,
        output_dir=args.output_dir,
        train_on_target_only=args.train_on_target_only,
        instruction_part=args.instruction_part or "",
        response_part=args.response_part or "",
        max_seq_length=args.max_seq_length,
        hf_token=args.hf_token,
    )

    prepare_dataset(config)


if __name__ == "__main__":
    main()