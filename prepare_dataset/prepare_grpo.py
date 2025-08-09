"""
GRPO dataset preparer for math reasoning tasks.

This preparer formats conversational datasets for GRPO training by extracting
prompts and expected answers for reward computation.
"""

import json
import os
import re
import sys
from typing import Dict, Any
from pathlib import Path

# Import from src/opensloth/dataset
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from opensloth.dataset.base_dataset_preparer import BaseDatasetPreparer


class GRPODatasetPreparer(BaseDatasetPreparer):
    """
    GRPO dataset preparer that creates prompt-based datasets for reinforcement learning.
    
    Unlike SFT, GRPO training requires:
    1. Prompts (questions/instructions) 
    2. Expected answers for reward computation
    3. No pre-computed completions (generated during training)
    """

    def get_description(self) -> str:
        return "Prepare datasets for GRPO (Group Relative Policy Optimization) training"

    def get_default_tokenizer(self) -> str:
        return "unsloth/Qwen2.5-7B-Instruct"

    def get_default_chat_template(self) -> str:
        return "chatml"

    def get_default_dataset_name(self) -> str:
        return "open-r1/DAPO-Math-17k-Processed"

    def get_default_instruction_part(self) -> str:
        return "<|im_start|>user\n"

    def get_default_response_part(self) -> str:
        return "<|im_start|>assistant\n"

    def add_custom_arguments(self, parser):
        """Add GRPO-specific arguments."""
        parser.add_argument('--extract_answer_field', type=str, default="solution",
                          help='Field name containing the expected answer/solution')
        parser.add_argument('--system_prompt', type=str, 
                          default="You are given a problem. Think about the problem and provide your working out. "
                                  "Place it between <start_working_out> and <end_working_out>. "
                                  "Then, provide your solution between <SOLUTION></SOLUTION>",
                          help='System prompt for GRPO reasoning format')

    def add_custom_config_entries(self):
        """Add GRPO-specific configuration entries."""
        self.config_dict.update({
            "Extract answer field": self.args.extract_answer_field,
            "System prompt": repr(self.args.system_prompt[:100] + "..." if len(self.args.system_prompt) > 100 else self.args.system_prompt),
        })

    def load_dataset(self):
        """Load dataset from HuggingFace or local file, handling configs."""
        print("[INFO] Loading dataset...")
        
        # Check if it's a local file
        if os.path.exists(self.args.dataset_name):
            print(f"[INFO] Loading local dataset from {self.args.dataset_name}")
            from speedy_utils import load_by_ext
            data = load_by_ext(self.args.dataset_name, do_memoize=True)
            import datasets
            dataset = datasets.Dataset.from_list(data)
        else:
            print(f"[INFO] Loading HuggingFace dataset {self.args.dataset_name}...")
            import datasets
            try:
                # Try with split as config first (for datasets like open-r1/DAPO-Math-17k-Processed)
                dataset = datasets.load_dataset(self.args.dataset_name, self.args.split, split="train")
                print(f"[INFO] Loaded with config '{self.args.split}'")
            except Exception as e1:
                try:
                    # Fallback to split as actual split
                    dataset = datasets.load_dataset(self.args.dataset_name, split=self.args.split)
                    print(f"[INFO] Loaded with split '{self.args.split}'")
                except Exception as e2:
                    # Last fallback - try just the dataset name
                    try:
                        dataset = datasets.load_dataset(self.args.dataset_name)["train"]
                        print(f"[INFO] Loaded default train split")
                    except Exception as e3:
                        raise ValueError(f"Failed to load dataset: {e1}, {e2}, {e3}")
            
            print("[INFO] Standardizing dataset format...")
            from unsloth.chat_templates import standardize_data_formats
            dataset = standardize_data_formats(dataset)

        print(f"[INFO] Dataset loaded: {len(dataset)} samples.")
        return dataset

    def post_process_text(self, text: str) -> str:
        """Post-process formatted text for GRPO."""
        # Remove <bos> token if present
        text = text.removeprefix('<bos>')
        return text

    def format_conversations(self, dataset):
        """Format conversations for GRPO training."""
        print("[INFO] Formatting conversations for GRPO training...")
        
        def formatting_prompts_func(examples):
            prompts = []
            
            # Handle different dataset formats
            if "prompt" in examples and self.args.extract_answer_field in examples:
                # Direct prompt/solution format (like DAPO-Math)
                print(f"[INFO] Using direct prompt/{self.args.extract_answer_field} format")
                
                for prompt, solution in zip(examples["prompt"], examples[self.args.extract_answer_field]):
                    # Create a conversation with system prompt and user question
                    conversation = [
                        {"role": "system", "content": self.args.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    
                    # Format as prompt for GRPO (no assistant response - that's generated during training)
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        conversation, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    formatted_prompt = self.post_process_text(formatted_prompt)
                    prompts.append(formatted_prompt)
                
                return {"prompt": prompts}
            
            elif "conversations" in examples or "messages" in examples:
                # Standard conversation format - extract prompts
                conversations_key = "conversations" if "conversations" in examples else "messages"
                all_conversations = examples[conversations_key]
                
                for conversation in all_conversations:
                    # Extract just the user messages to create prompts
                    user_messages = []
                    for msg in conversation:
                        if msg.get("role") == "user" or msg.get("from") == "human":
                            user_messages.append({"role": "user", "content": msg["content"] or msg.get("value", "")})
                    
                    if user_messages:
                        # Add system prompt and format
                        full_conversation = [{"role": "system", "content": self.args.system_prompt}] + user_messages
                        formatted_prompt = self.tokenizer.apply_chat_template(
                            full_conversation, 
                            tokenize=False, 
                            add_generation_prompt=True
                        )
                        formatted_prompt = self.post_process_text(formatted_prompt)
                        prompts.append(formatted_prompt)
                    else:
                        prompts.append("")  # Empty prompt for malformed conversations
                
                return {"prompt": prompts}
            
            else:
                raise ValueError(f"Dataset must have either 'conversations'/'messages' or 'prompt' and "
                               f"'{self.args.extract_answer_field}' columns")
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        print("[INFO] GRPO formatting complete.")
        return dataset

    def tokenize_and_prepare_labels(self, dataset):
        """For GRPO, we only need tokenized prompts - no labels."""
        print("[INFO] Tokenizing prompts for GRPO training...")
        
        def process_one(example):
            prompt = example["prompt"]
            # Tokenize the prompt only
            tokenized = self.tokenizer(prompt, truncation=True, max_length=self.args.max_seq_length)
            
            return {
                "prompt": prompt,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
            }
        
        data = dataset.map(process_one, num_proc=self.args.num_proc)
        
        # Filter out prompts that are too long
        data = self.filter_dataset(data)
        
        self._last_size = len(data)
        print(f"[INFO] GRPO tokenization complete. Dataset size: {self._last_size}")
        return data

    def filter_dataset(self, data):
        """Filter out examples that are too long."""
        max_length = self.args.max_seq_length
        
        def should_keep(example):
            return len(example["input_ids"]) <= max_length
        
        print(f"[INFO] Filtering dataset: max_seq_length={max_length}")
        filtered_data = data.filter(should_keep, num_proc=self.args.num_proc)
        
        # Report filtering statistics
        original_count = len(data)
        filtered_count = len(filtered_data)
        removed_count = original_count - filtered_count
        
        if removed_count > 0:
            print(f"[INFO] Filtered out {removed_count}/{original_count} examples ({removed_count/original_count*100:.1f}%) that were too long (>{max_length} tokens)")
        
        return filtered_data

    def _build_metadata(self):
        """Build metadata for the GRPO dataset."""
        return {
            "model_family": "GRPO",
            "training_type": "grpo",
            "tokenizer": self.args.tok_name,
            "chat_template": self.args.chat_template,
            "max_seq_length": self.args.max_seq_length,
            "dataset_name": self.args.dataset_name,
            "split": self.args.split,
            "num_samples": getattr(self, '_last_size', -1),
            "extract_answer_field": self.args.extract_answer_field,
            "system_prompt": self.args.system_prompt,
            "train_on_target_only": False,  # GRPO doesn't use target masking
            "processing_complete": True
        }

def main():
    """Main function for standalone execution."""
    preparer = GRPODatasetPreparer()
    preparer.run()

if __name__ == "__main__":
    main()
