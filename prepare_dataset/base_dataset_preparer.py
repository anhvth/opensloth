"""
Base class for dataset preparation with common functionality.
Now also supports programmatic runs via `run_with_namespace` and `run_with_config`.
"""

import os
import argparse
import random
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Union
from pathlib import Path
from types import SimpleNamespace
import json
import hashlib

import datasets
from speedy_utils import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import train_on_target_text_only
from opensloth._debug_dataloader import debug_chat_dataloader_for_training
from config_printer import DatasetPreparationConfigPrinter

# Unsloth utilities
from unsloth.chat_templates import get_chat_template, standardize_data_formats


class BaseDatasetPreparer(ABC):
    """
    Base class for dataset preparation with common functionality.
    
    This class handles:
    - Argument parsing and configuration
    - Tokenizer loading and chat template setup
    - Dataset loading (from HuggingFace or local files)
    - Common processing pipeline
    - Debug visualization
    - Saving processed datasets
    """
    
    def __init__(self):
        self.args = None
        self.tokenizer = None
        self.config_dict = {}
    
    def create_argument_parser(self) -> argparse.ArgumentParser:
        """Create and configure argument parser with common arguments."""
        parser = argparse.ArgumentParser(description=self.get_description())
        
        # Model/tokenizer args
        parser.add_argument('--tok_name', type=str, default=self.get_default_tokenizer(), 
                          help='Path to the tokenizer/model directory')
        parser.add_argument('--chat_template', default=self.get_default_chat_template(), 
                          help='Chat template to use')
        
        # Dataset args
        parser.add_argument('--dataset_name', type=str, default=self.get_default_dataset_name(),
                          help='HuggingFace dataset name or local file path')
        parser.add_argument('--split', type=str, default='train', 
                          help='Dataset split to use (for HuggingFace datasets)')
        
        # Processing args
        parser.add_argument('--num_samples', '-n', type=int, default=-1, 
                          help='Number of samples to process (use -1 for all)')
        parser.add_argument('--num_proc', '-wk', type=int, default=8, 
                          help='Number of processes for mapping')
        parser.add_argument('--output_dir', '-o', type=str, default=None, 
                          help='Output directory for the processed dataset')
        
        # Training configuration
        parser.add_argument('--train_on_target_only', action='store_true', default=True,
                          help='Whether to mask non-assistant tokens for response-only training')
        parser.add_argument('--instruction_part', type=str, default=self.get_default_instruction_part(),
                          help='Instruction part string (required if train_on_target_only=True)')
        parser.add_argument('--response_part', type=str, default=self.get_default_response_part(),
                          help='Response part string (required if train_on_target_only=True)')
        
        # Debug args
        parser.add_argument('--debug', type=int, default=0, 
                          help='If >0, dump this many samples as HTML and use debug mode')
        
        self.add_custom_arguments(parser)
        return parser
    
    def add_custom_arguments(self, parser: argparse.ArgumentParser):
        """Override this method to add model-specific arguments."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return description for the argument parser."""
        pass
    
    @abstractmethod 
    def get_default_tokenizer(self) -> str:
        """Return default tokenizer name."""
        pass
    
    @abstractmethod
    def get_default_chat_template(self) -> str:
        """Return default chat template."""
        pass
    
    @abstractmethod
    def get_default_dataset_name(self) -> str:
        """Return default dataset name."""
        pass
    
    @abstractmethod
    def get_default_instruction_part(self) -> str:
        """Return default instruction part string."""
        pass
    
    @abstractmethod
    def get_default_response_part(self) -> str:
        """Return default response part string."""
        pass
    
    def sanitize_name(self, name: str) -> str:
        """Sanitize a name for use in file paths."""
        name = name.split("/")[-1] if "/" in name else name
        name = name.replace(" ", "_").replace("|", "_")
        return name[:32]  # limit length for sanity
    
    def compute_output_dir(self) -> str:
        """Auto-compute output directory if not specified."""
        if self.args.output_dir:
            return self.args.output_dir
            
        model_name = self.sanitize_name(self.args.tok_name)
        dataset_name = self.sanitize_name(self.args.dataset_name)
        split = self.sanitize_name(self.args.split)
        
        if self.args.num_samples > 0:
            count = f"n{self.args.num_samples}"
        elif self.args.debug > 0:
            count = f"debug{self.args.debug}"
        else:
            count = "all"
            
        return f"{model_name}-{dataset_name}-{split}-{count}-processed"
    
    def setup_configuration(self):
        """Setup and print configuration."""
        self.config_dict = {
            "Tokenizer/model": self.args.tok_name,
            "Dataset": f"{self.args.dataset_name} [split: {self.args.split}]",
            "Chat template": self.args.chat_template,
            "Train on target only": self.args.train_on_target_only,
            "Num samples": self.args.num_samples,
            "Num processes": self.args.num_proc,
            "Debug": self.args.debug,
            "Output directory": self.args.output_dir,
        }
        
        if self.args.train_on_target_only:
            self.config_dict.update({
                "Instruction part": repr(self.args.instruction_part),
                "Response part": repr(self.args.response_part),
            })
        
        self.add_custom_config_entries()
        DatasetPreparationConfigPrinter(self.config_dict).print_table()
    
    def add_custom_config_entries(self):
        """Override this method to add custom configuration entries."""
        pass
    
    def validate_arguments(self):
        """Validate command line arguments."""
        if self.args.train_on_target_only:
            assert self.args.instruction_part, "instruction_part is required when train_on_target_only=True"
            assert self.args.response_part, "response_part is required when train_on_target_only=True"
    
    def setup_tokenizer(self):
        """Load and setup tokenizer with chat template."""
        print("[INFO] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.tok_name)
        print("[INFO] Patching tokenizer for chat template...")
        self.tokenizer = get_chat_template(self.tokenizer, chat_template=self.args.chat_template)
    
    def load_dataset(self) -> datasets.Dataset:
        """Load dataset from HuggingFace or local file."""
        print("[INFO] Loading dataset...")
        
        # Check if it's a local file
        if os.path.exists(self.args.dataset_name):
            print(f"[INFO] Loading local dataset from {self.args.dataset_name}")
            data = load_by_ext(self.args.dataset_name, do_memoize=True)
            dataset = datasets.Dataset.from_list(data)
        else:
            print(f"[INFO] Loading HuggingFace dataset {self.args.dataset_name}...")
            dataset = datasets.load_dataset(self.args.dataset_name, split=self.args.split)
            print("[INFO] Standardizing dataset format...")
            dataset = standardize_data_formats(dataset)
        
        print(f"[INFO] Dataset loaded: {len(dataset)} samples.")
        return dataset
    
    def select_samples(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Select subset of samples based on num_samples or debug settings."""
        if self.args.num_samples > 0:
            print(f"[INFO] Selecting first {min(self.args.num_samples, len(dataset))} samples.")
            dataset = dataset.select(range(min(self.args.num_samples, len(dataset))))
        elif self.args.debug > 0:
            print(f"[INFO] Selecting {min(self.args.debug, len(dataset))} random samples for debug mode.")
            indices = random.sample(range(len(dataset)), min(self.args.debug, len(dataset)))
            dataset = dataset.select(indices)
        
        return dataset
    
    def format_conversations(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Format conversations using chat template."""
        print("[INFO] Formatting conversations and removing <bos> tokens...")
        
        def formatting_prompts_func(examples):
            if "conversations" in examples:
                # HuggingFace format with conversations
                convos = examples["conversations"]
                texts = []
                missing_assistant = 0
                for convo in convos:
                    text = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
                    text = text.removeprefix('<bos>')
                    if self.args.train_on_target_only and self.args.response_part not in text:
                        missing_assistant += 1
                    texts.append(text)
                return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
            elif "messages" in examples:
                # Local format with messages
                all_messages = examples["messages"]
                texts = []
                missing_assistant = 0
                for messages in all_messages:
                    text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    text = self.post_process_text(text)
                    if self.args.train_on_target_only and self.args.response_part not in text:
                        missing_assistant += 1
                    texts.append(text)
                return {"text": texts, "missing_assistant": [missing_assistant] * len(texts)}
            else:
                raise ValueError("Dataset must have either 'conversations' or 'messages' column")
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        
        if "missing_assistant" in dataset.column_names:
            total_missing = sum(dataset["missing_assistant"])
            if total_missing > 0:
                print(f"[WARN] {total_missing} conversation(s) missing assistant response part '{self.args.response_part}'.")
        
        print("[INFO] Formatting complete.")
        return dataset
    
    def post_process_text(self, text: str) -> str:
        """Post-process formatted text. Override for model-specific processing."""
        return text
    
    def tokenize_and_prepare_labels(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """Tokenize dataset and prepare labels for training."""
        if self.args.train_on_target_only:
            print("[INFO] Tokenizing and masking labels for response-only training...")
            
            def process_one(example):
                text = example["text"]
                input_ids = self.tokenizer(text)["input_ids"]
                labels = train_on_target_text_only(
                    input_ids, self.tokenizer, 
                    self.args.instruction_part, self.args.response_part
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
                input_ids = self.tokenizer(text)["input_ids"]
                return {
                    "text": text,
                    "input_ids": input_ids,
                    "labels": input_ids.copy(),  # Use same as input_ids for full training
                    "all_masked": 0,
                }
        
        data = dataset.map(process_one, num_proc=self.args.num_proc)

        if self.args.train_on_target_only and "all_masked" in data.column_names:
            total_all_masked = sum(data["all_masked"])
            if total_all_masked > 0:
                print(f"[WARN] {total_all_masked} sample(s) have no assistant response to train on after masking.")

        self._last_size = len(data)
        print(f"[INFO] Tokenization complete. Dataset size: {self._last_size}")
        return data
    
    def debug_visualization(self, data: datasets.Dataset):
        """Create debug visualization if debug mode is enabled."""
        if self.args.debug > 0:
            print(f"[INFO] Debug mode enabled. Dumping {self.args.debug} samples to HTML and terminal...")
            data.set_format(type="torch", columns=["input_ids", "labels"])
            dataloader = DataLoader(data, batch_size=1, shuffle=False)
            debug_chat_dataloader_for_training(dataloader, self.tokenizer, n_example=self.args.debug)
            print(f"[INFO] Debug HTML written to .log/dataloader_examples.html")
    
    def save_dataset(self, data: datasets.Dataset):
        """Save processed dataset to disk."""
        if self.args.debug <= 0:  # Don't save in debug mode
            print(f"[INFO] Saving processed dataset to {self.args.output_dir} ...")
            data.save_to_disk(self.args.output_dir)
            # save metadata for reproducibility and GUI auto-match
            try:
                meta = self._build_metadata()
                with open(os.path.join(self.args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to write metadata.json: {e}")
            print(f"[INFO] Dataset saved to {self.args.output_dir}")

    # ------------------------------
    # Metadata & hashing
    # ------------------------------
    def _prep_config_core(self) -> Dict[str, Any]:
        """Return the subset of args that define dataset content.

        This signature is used to determine if a preprocessed dataset matches
        the current configuration.
        """
        return {
            "tok_name": self.args.tok_name,
            "chat_template": self.args.chat_template,
            "dataset_name": self.args.dataset_name,
            "split": self.args.split,
            "num_samples": self.args.num_samples,
            "train_on_target_only": self.args.train_on_target_only,
            "instruction_part": self.args.instruction_part if self.args.train_on_target_only else None,
            "response_part": self.args.response_part if self.args.train_on_target_only else None,
            # exclude num_proc/debug/output_dir; they don't affect content
            "preparer_class": self.__class__.__name__,
        }

    def _hash_config(self, cfg: Dict[str, Any]) -> str:
        payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _build_metadata(self) -> Dict[str, Any]:
        core = self._prep_config_core()
        return {
            "config": core,
            "config_hash": self._hash_config(core),
            "size": getattr(self, "_last_size", None),
        }
    
    def run(self):
        """Main execution pipeline."""
        print("[INFO] Parsing arguments...")
        parser = self.create_argument_parser()
        self.args = parser.parse_args()
        
        # Validate arguments
        self.validate_arguments()
        
        # Auto-compute output directory
        self.args.output_dir = self.compute_output_dir()
        
        # Setup and print configuration
        self.setup_configuration()
        
        # Setup tokenizer
        self.setup_tokenizer()
        
        # Load and process dataset
        dataset = self.load_dataset()
        dataset = self.select_samples(dataset)
        dataset = self.format_conversations(dataset)
        data = self.tokenize_and_prepare_labels(dataset)
        
        # Debug visualization or save
        if self.args.debug > 0:
            self.debug_visualization(data)
        else:
            self.save_dataset(data)

    # ------------------------------
    # Programmatic API for GUIs/SDKs
    # ------------------------------
    def _namespace_from_dict(self, cfg: Dict[str, Any]) -> argparse.Namespace:
        """Convert a dict into an argparse.Namespace with defaults applied."""
        # build defaults from parser then update with cfg
        parser = self.create_argument_parser()
        defaults = vars(parser.parse_args([]))  # no CLI, just defaults
        defaults.update(cfg or {})
        return argparse.Namespace(**defaults)

    def run_with_namespace(self, ns: argparse.Namespace) -> str:
        """Run the pipeline using a prebuilt Namespace (no CLI parsing).

        Returns
        -------
        str
            The output directory containing the processed dataset (or where it
            would be saved if in debug mode).
        """
        self.args = ns

        # Validate arguments
        self.validate_arguments()

        # Auto-compute output directory if not provided
        self.args.output_dir = self.args.output_dir or self.compute_output_dir()

        # Setup and print configuration
        self.setup_configuration()

        # Setup tokenizer
        self.setup_tokenizer()

        # Load and process dataset
        dataset = self.load_dataset()
        dataset = self.select_samples(dataset)
        dataset = self.format_conversations(dataset)
        data = self.tokenize_and_prepare_labels(dataset)

        # Debug visualization or save
        if self.args.debug > 0:
            self.debug_visualization(data)
        else:
            self.save_dataset(data)

        return self.args.output_dir

    def run_with_config(self, cfg: Dict[str, Any]) -> str:
        """Run the pipeline using a plain dictionary config.

        This is a convenience wrapper around `run_with_namespace`.
        """
        ns = self._namespace_from_dict(cfg)
        return self.run_with_namespace(ns)
