#!/usr/bin/env python3
#type: ignore
"""
Standalone Qwen dataset preparation script.
Simplified version that prepares datasets for Qwen models with tokenization and formatting.
"""
import argparse
import contextlib
import os
import random
from typing import cast

import datasets

# Import unsloth first (before transformers)
import unsloth
from unsloth.chat_templates import get_chat_template, standardize_data_formats

# Now import transformers
from transformers import AutoTokenizer

# Import OpenSloth utilities and configurations
from opensloth.datasets_utils.qwen_dataset_utils import (
    compute_output_dir_from_args,
    save_dataset_metadata,
    save_dataset_config,
    save_training_config,
    create_formatting_func,
    create_processing_func,
    create_filter_func,
    load_local_file,
    post_process_text,
)
from opensloth.opensloth_config import DatasetPrepConfig
from opensloth.examples import qwen_config_1gpu


def prepare_qwen_dataset(config: DatasetPrepConfig | None = None):
    """Main function to prepare Qwen dataset."""
    if config is None:
        config = qwen_config_1gpu()
    
    # Clean access to configuration parameters - no verbose dictionary parsing needed!
    tokenizer_name = config.tokenizer_name
    chat_template = config.chat_template
    dataset_name = config.dataset_name
    split = config.split
    num_samples = config.num_samples
    num_proc = config.num_proc
    gpus = config.gpus
    max_seq_length = config.max_seq_length
    train_on_target_only = config.train_on_target_only
    instruction_part = config.instruction_part
    response_part = config.response_part
    debug = config.debug
    
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
    
    # Compute output directory
    output_dir = compute_output_dir_from_args(
        tokenizer_name, dataset_name, split, num_samples, max_seq_length, debug
    )
    
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
    elif debug > 0:
        indices = random.sample(range(len(dataset)), min(debug, len(dataset)))  # type: ignore
        dataset = dataset.select(indices)  # type: ignore
    
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
    
    # Debug visualization or save
    if debug > 0:
        from torch.utils.data import DataLoader
        data.set_format(type="torch", columns=["input_ids", "labels"])
        dataloader = DataLoader(cast(datasets.Dataset, data), batch_size=1, shuffle=False)  # type: ignore
        from opensloth._debug_dataloader import debug_chat_dataloader_for_training
        debug_chat_dataloader_for_training(dataloader, tokenizer, n_example=debug)
    else:
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
        
        try:
            save_dataset_metadata(output_dir, config, dataset_size)
        except Exception as e:
            pass

        try:
            save_dataset_config(output_dir, config)
        except Exception as e:
            pass

        # Save training configuration
        try:
            from opensloth.examples.qwen_configs import get_training_config_template
            training_config_template = get_training_config_template(config.gpus, config.max_seq_length)
            save_training_config(output_dir, config, training_config_template)
        except Exception as e:
            pass


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Prepare Qwen dataset for training")
    parser.add_argument("--config", type=str, help="Configuration to use (e.g., qwen_config_2gpus)")
    parser.add_argument("--tokenizer_name", type=str, help="Tokenizer name")
    parser.add_argument("--dataset_name", type=str, help="Dataset name")
    parser.add_argument("--num_samples", type=int, help="Number of samples")
    parser.add_argument("--gpus", type=int, help="Number of GPUs")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument("--debug", type=int, help="Debug mode")
    
    args = parser.parse_args()
    
    # Load base configuration
    if args.config:
        from opensloth.examples import qwen_config_1gpu, qwen_config_2gpus, qwen_config_4gpus, qwen_config_debug
        config_map = {
            "qwen_config_1gpu": qwen_config_1gpu,
            "qwen_config_2gpus": qwen_config_2gpus,
            "qwen_config_4gpus": qwen_config_4gpus,
            "qwen_config_debug": qwen_config_debug,
        }
        if args.config in config_map:
            config = config_map[args.config]()
        else:
            raise ValueError(f"Unknown config: {args.config}")
    else:
        config = qwen_config_1gpu()
    
    # Override config with command line arguments by creating a new model instance
    overrides = {}
    for key, value in vars(args).items():
        if value is not None and key != "config":
            overrides[key] = value
    
    if overrides:
        # Create new config with overrides
        config_dict = config.model_dump()
        config_dict.update(overrides)
        config = DatasetPrepConfig(**config_dict)
    
    prepare_qwen_dataset(config)


if __name__ == "__main__":
    main()