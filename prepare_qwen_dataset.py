#!/usr/bin/env python3
#type: ignore
"""
Standalone Qwen dataset preparation script.
Simplified version that prepares datasets for Qwen models with tokenization and formatting.
"""
import argparse
import contextlib
import hashlib
import json
import os
import random
import warnings
from pathlib import Path
from typing import cast

import datasets
from tabulate import tabulate
from transformers import AutoTokenizer

from opensloth.datasets_utils import (
    compute_output_dir,
    load_local_file,
    post_process_text,
    print_config_table,
    train_on_target_text_only,
)


def prepare_qwen_dataset():
    """Main function to prepare Qwen dataset."""
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Prepare Qwen dataset with tokenization and formatting.")
    
    # Model/tokenizer args
    parser.add_argument('--tokenizer_name', type=str, default='unsloth/Qwen2.5-0.5B-Instruct',
                       help='Path to the tokenizer/model directory')
    parser.add_argument('--chat_template', default="qwen-2.5",
                       help='Chat template to use')
    
    # Dataset args
    parser.add_argument('--dataset_name', type=str, default='mlabonne/FineTome-100k',
                       help='HuggingFace dataset name or local file path')
    parser.add_argument('--input_file', '-i', type=str, default=None,
                       help='Input JSON file with messages (alternative to dataset_name)')
    parser.add_argument('--split', type=str, default='train',
                       help='Dataset split to use (for HuggingFace datasets)')
    
    # Processing args
    parser.add_argument('--num_samples', '-n', type=int, default=-1,
                       help='Number of samples to process (use -1 for all)')
    parser.add_argument('--num_proc', '-wk', type=int, default=8,
                       help='Number of processes for mapping')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                       help='Output directory for the processed dataset')
    parser.add_argument('--gpus', type=int, default=1,
                       help='Number of shards (GPUs) to pre-split the dataset for')
    
    # Training configuration
    parser.add_argument('--max_seq_length', type=int, default=4096,
                       help='Maximum sequence length for tokenization')
    parser.add_argument('--train_on_target_only', action='store_true', default=True,
                       help='Whether to mask non-assistant tokens for response-only training')
    parser.add_argument('--instruction_part', type=str, default='<|im_start|>user\n',
                       help='Instruction part string')
    parser.add_argument('--response_part', type=str, default='<|im_start|>assistant\n',
                       help='Response part string')
    
    # Debug args
    parser.add_argument('--debug', type=int, default=0,
                       help='If >0, dump this many samples as HTML and use debug mode')
    
    args = parser.parse_args()
    
    # If input_file is specified, use it instead of dataset_name
    if args.input_file:
        args.dataset_name = args.input_file
    
    # Validate arguments
    if args.train_on_target_only:
        assert args.instruction_part, "instruction_part is required when train_on_target_only=True"
        assert args.response_part, "response_part is required when train_on_target_only=True"
    
    # Auto-compute output directory
    args.output_dir = compute_output_dir(args)
    
    # Setup and print configuration
    config_dict = {
        "Tokenizer/model": args.tokenizer_name,
        "Dataset": f"{args.dataset_name} [split: {args.split}]",
        "Chat template": args.chat_template,
        "Max sequence length": args.max_seq_length,
        "Train on target only": args.train_on_target_only,
        "Num samples": args.num_samples,
        "Num processes": args.num_proc,
        "Debug": args.debug,
        "Output directory": args.output_dir,
    }
    
    if args.input_file:
        config_dict["Input file"] = args.input_file
    
    if args.train_on_target_only:
        config_dict.update({
            "Instruction part": repr(args.instruction_part),
            "Response part": repr(args.response_part),
        })
    
    print_config_table(config_dict)
    
    # Setup tokenizer
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    print("[INFO] Patching tokenizer for chat template...")
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(tokenizer, chat_template=args.chat_template)
    
    # Load dataset
    print("[INFO] Loading dataset...")
    
    # Check if it's a local file
    if os.path.exists(args.dataset_name):
        print(f"[INFO] Loading local dataset from {args.dataset_name}")
        data = load_local_file(args.dataset_name)
        dataset = datasets.Dataset.from_list(data)
    else:
        print(f"[INFO] Loading HuggingFace dataset {args.dataset_name}...")
        dataset = datasets.load_dataset(args.dataset_name, split=args.split)
        
        # Ensure we have a Dataset object, not IterableDataset
        if isinstance(dataset, datasets.IterableDataset):
            raise ValueError(
                "IterableDataset is not supported. Please use a dataset that can be loaded as Dataset. "
                "Try using a smaller dataset or check the dataset configuration."
            )
        
        # Cast to Dataset to help type checker
        dataset = cast(datasets.Dataset, dataset)
        
        print("[INFO] Standardizing dataset format...")
        from unsloth.chat_templates import standardize_data_formats
        dataset = standardize_data_formats(dataset)
    
    print(f"[INFO] Dataset loaded: {len(dataset)} samples.")  # type: ignore
    
    # Select samples
    if args.num_samples > 0:
        print(f"[INFO] Selecting first {min(args.num_samples, len(dataset))} samples.")  # type: ignore
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))  # type: ignore
    elif args.debug > 0:
        print(f"[INFO] Selecting {min(args.debug, len(dataset))} random samples for debug mode.")  # type: ignore
        indices = random.sample(range(len(dataset)), min(args.debug, len(dataset)))  # type: ignore
        dataset = dataset.select(indices)  # type: ignore
    
    # Format conversations
    print("[INFO] Formatting conversations and removing <bos> tokens...")
    
    def formatting_prompts_func(examples):
        if "conversations" in examples:
            # HuggingFace format with conversations
            convos = examples["conversations"]
            texts = []
            missing_assistant = 0
            for convo in convos:
                text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                text = text.removeprefix('<bos>')
                if args.train_on_target_only and args.response_part not in text:
                    missing_assistant += 1
                texts.append(text)
            return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
        elif "messages" in examples:
            # Local format with messages
            all_messages = examples["messages"]
            texts = []
            missing_assistant = 0
            for messages in all_messages:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                text = post_process_text(text, args.tokenizer_name)
                if args.train_on_target_only and args.response_part not in text:
                    missing_assistant += 1
                texts.append(text)
            return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
        else:
            raise ValueError("Dataset must have either 'conversations' or 'messages' column")
    
    dataset = dataset.map(formatting_prompts_func, batched=True)  # type: ignore
    
    if dataset.column_names is not None and "missing_assistant" in dataset.column_names:  # type: ignore
        total_missing = sum(dataset["missing_assistant"])  # type: ignore
        if total_missing > 0:
            print(f"[WARN] {total_missing} conversation(s) missing assistant response part '{args.response_part}'.")
    
    print("[INFO] Formatting complete.")
    
    # Tokenize and prepare labels
    if args.train_on_target_only:
        print("[INFO] Tokenizing and masking labels for response-only training...")
        
        def process_one(example):
            text = example["text"]
            input_ids = tokenizer(text)["input_ids"]
            labels = train_on_target_text_only(
                input_ids, tokenizer, 
                args.instruction_part, args.response_part
            )
            return {
                "text": text,
                "input_ids": input_ids,
                "labels": labels,
                "all_masked": int(all(l == -100 for l in labels)),
            }
    else:
        print("[INFO] Tokenizing dataset...")
        
        def process_one(example):
            text = example["text"]
            input_ids = tokenizer(text)["input_ids"]
            return {
                "text": text,
                "input_ids": input_ids,
                "labels": input_ids.copy(),  # Use same as input_ids for full training
                "all_masked": 0,
            }
    
    data = dataset.map(process_one, num_proc=args.num_proc)  # type: ignore
    
    # Filter out examples that are too long or have no training labels
    max_length = args.max_seq_length
    
    def should_keep(example):
        # Filter out examples that are too long
        if len(example["input_ids"]) > max_length:
            return False
        
        # Filter out examples with no training labels (all -100)
        if args.train_on_target_only and "all_masked" in example:
            if example["all_masked"] == 1:  # All labels are -100
                return False
        
        return True
    
    print(f"[INFO] Filtering dataset: max_seq_length={max_length}")
    original_count = len(data)  # type: ignore
    data = data.filter(should_keep, num_proc=args.num_proc)  # type: ignore
    filtered_count = len(data)  # type: ignore
    removed_count = original_count - filtered_count
    
    if removed_count > 0:
        print(f"[INFO] Filtered out {removed_count}/{original_count} examples ({removed_count/original_count*100:.1f}%) that were too long (>{max_length} tokens) or had no training labels")
    
    if args.train_on_target_only and data.column_names is not None and "all_masked" in data.column_names:  # type: ignore
        total_all_masked = sum(data["all_masked"])  # type: ignore
        if total_all_masked > 0:
            print(f"[WARN] {total_all_masked} sample(s) have no assistant response to train on after masking.")
    
    print(f"[INFO] Tokenization complete. Dataset size: {len(data)}")  # type: ignore
    
    # Check if dataset is empty and provide helpful error
    if len(data) == 0:
        print("\n❌ [ERROR] No samples remain after processing!")
        print(f"📊 Original samples: {original_count}")
        print(f"🔍 After filtering: {len(data)}")  # type: ignore
        print("\n💡 This usually happens because:")
        print(f"   • All samples were longer than max_seq_length ({args.max_seq_length})")
        print("   • All samples had no training labels (when using --train-on-target-only)")
        print("   • Chat template or response patterns don't match the data")
        print("\n🛠️  Try these solutions:")
        print(f"   • Increase --max-seq-length (currently {args.max_seq_length})")
        print("   • Use a different --chat-template")
        print("   • Check --instruction-part and --response-part patterns")
        print("   • Try more samples with --num-samples")
        raise RuntimeError("Dataset preparation failed: No samples remaining after processing")
    
    # Debug visualization or save
    if args.debug > 0:
        from torch.utils.data import DataLoader
        print(f"[INFO] Debug mode enabled. Dumping {args.debug} samples to HTML and terminal...")
        data.set_format(type="torch", columns=["input_ids", "labels"])
        dataloader = DataLoader(cast(datasets.Dataset, data), batch_size=1, shuffle=False)  # type: ignore
        from opensloth._debug_dataloader import debug_chat_dataloader_for_training
        debug_chat_dataloader_for_training(dataloader, tokenizer, n_example=args.debug)
        print("[INFO] Debug HTML written to .log/dataloader_examples.html")
        print(f"\n🔍 [DEBUG] Debug mode completed with {len(data)} samples")  # type: ignore
    else:
        # Save dataset
        print(f"[INFO] Saving processed dataset (pre-sharded) to {args.output_dir} ...")
        os.makedirs(args.output_dir, exist_ok=True)

        num_shards = max(1, int(args.gpus))
        # Optional deterministic shuffle before sharding so each shard gets diverse samples
        if num_shards > 1:
            with contextlib.suppress(Exception):
                data = data.shuffle(seed=42)  # type: ignore

        for i in range(num_shards):
            shard_path = os.path.join(args.output_dir, f"shard_{i}")
            shard_dataset = data.shard(num_shards=num_shards, index=i)  # type: ignore
            shard_dataset.save_to_disk(shard_path)  # type: ignore
            print(f"[INFO]  • Saved shard {i}/{num_shards-1} -> {shard_path} ({len(shard_dataset)} samples)")  # type: ignore

        # Save metadata for reproducibility and GUI auto-match
        try:
            meta = {
                "config": {
                    "tokenizer_name": args.tokenizer_name,
                    "chat_template": args.chat_template,
                    "dataset_name": args.dataset_name,
                    "split": args.split,
                    "max_seq_length": args.max_seq_length,
                    "num_samples": args.num_samples,
                    "train_on_target_only": args.train_on_target_only,
                    "instruction_part": args.instruction_part if args.train_on_target_only else None,
                    "response_part": args.response_part if args.train_on_target_only else None,
                    "preparer_class": "QwenDatasetPreparer",
                },
                "size": len(data),  # type: ignore
            }
            payload = json.dumps(meta["config"], sort_keys=True, ensure_ascii=False)
            meta["config_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            
            with open(os.path.join(args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WARN] Failed to write metadata.json: {e}")

        # Save complete config for debugging and reuse
        try:
            complete_config = vars(args).copy()
            complete_config['gpus'] = num_shards
            complete_config['num_shards'] = num_shards
            complete_config['model_family'] = 'Qwen'

            with open(os.path.join(args.output_dir, "dataset_config.json"), "w", encoding="utf-8") as f:
                json.dump(complete_config, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Config saved to {args.output_dir}/dataset_config.json")
        except Exception as e:
            print(f"[WARN] Failed to write dataset_config.json: {e}")

        # Save complete training configuration for train.py with schema validation
        # Note: The schema is auto-generated from Pydantic models using scripts/generate_schema.py
        try:
            training_config = {
                "$schema": "./schemas/training_config.schema.json",
                "opensloth_config": {
                    "data_cache_path": args.output_dir,
                    "devices": list(range(num_shards)),  # Will be overridden by train.py
                    "fast_model_args": {
                        "model_name": args.tokenizer_name,
                        "max_seq_length": args.max_seq_length,
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
                    "training_type": "sft",
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
                },
                "dataset_prep_config": {
                    "tokenizer_name": args.tokenizer_name,
                    "chat_template": args.chat_template,
                    "dataset_name": args.dataset_name,
                    "split": args.split,
                    "num_samples": args.num_samples,
                    "num_proc": args.num_proc,
                    "gpus": num_shards,
                    "output_dir": args.output_dir,
                    "train_on_target_only": args.train_on_target_only,
                    "instruction_part": args.instruction_part if args.train_on_target_only else None,
                    "response_part": args.response_part if args.train_on_target_only else None,
                    "max_seq_length": args.max_seq_length,
                    "training_type": "sft",
                    "debug": args.debug,
                    "hf_token": None
                }
            }

            with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
                json.dump(training_config, f, ensure_ascii=False, indent=2)
            print(f"[INFO] Training config saved to {args.output_dir}/training_config.json")
            print(f"[INFO] ✨ Config includes JSON Schema for VS Code IntelliSense support")
        except Exception as e:
            print(f"[WARN] Failed to write training_config.json: {e}")

        print(f"[INFO] Dataset (with {num_shards} shard(s)) saved to {args.output_dir}")
        
        # Show final statistics
        print("\n🎉 [SUCCESS] Dataset preparation completed!")
        print(f"📁 Output: {args.output_dir}")
        print(f"📊 Final dataset contains: {len(data)} samples")  # type: ignore
        print(f"💾 Ready for training with {len(data)} high-quality samples")  # type: ignore


if __name__ == "__main__":
    prepare_qwen_dataset()