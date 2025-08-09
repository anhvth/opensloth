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

# Import from the same directory
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from base_dataset_preparer import BaseDatasetPreparer
from config_printer import DatasetPreparationConfigPrinter


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

    def extract_answer(self, text: str) -> str:
        """Extract answer from solution text."""
        if not text:
            return ""
        
        # Try to extract numeric answer
        # Remove common prefixes
        text = re.sub(r'^(the answer is|answer:|solution:)\s*', '', text.strip(), flags=re.IGNORECASE)
        
        # Look for #### pattern (GSM8K style)
        hash_match = re.search(r'####\s*(.*?)(?:\n|$)', text)
        if hash_match:
            return hash_match.group(1).strip()
        
        # Look for numbers in the text
        number_match = re.search(r'([-]?\d+(?:\.\d+)?(?:,\d{3})*)', text)
        if number_match:
            return number_match.group(1).replace(',', '')
        
        # Return first 50 chars as fallback
        return text[:50].strip()

    def format_conversations(self, dataset):
        """Format conversations for GRPO training."""
        print("[INFO] Formatting conversations for GRPO training...")
        
        def formatting_prompts_func(examples):
            if "conversations" in examples:
                # HuggingFace format
                convos = examples["conversations"]
                prompts = []
                answers = []
                
                for convo in convos:
                    # Extract user message as prompt
                    user_msg = None
                    assistant_msg = None
                    
                    for msg in convo:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            assistant_msg = msg.get("content", "")
                    
                    if user_msg:
                        # Create prompt with system message
                        prompt_msgs = [
                            {"role": "system", "content": self.args.system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                        prompt = self.tokenizer.apply_chat_template(
                            prompt_msgs, tokenize=False, add_generation_prompt=True
                        )
                        prompts.append(prompt)
                        
                        # Extract answer from assistant response
                        answer = self.extract_answer(assistant_msg or "")
                        answers.append(answer)
                    else:
                        prompts.append("")
                        answers.append("")
                
                return {"prompt": prompts, "answer": answers}
            
            elif "messages" in examples:
                # Local format
                all_messages = examples["messages"]
                prompts = []
                answers = []
                
                for messages in all_messages:
                    user_msg = None
                    assistant_msg = None
                    
                    for msg in messages:
                        if msg.get("role") == "user":
                            user_msg = msg.get("content", "")
                        elif msg.get("role") == "assistant":
                            assistant_msg = msg.get("content", "")
                    
                    if user_msg:
                        # Create prompt with system message
                        prompt_msgs = [
                            {"role": "system", "content": self.args.system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                        prompt = self.tokenizer.apply_chat_template(
                            prompt_msgs, tokenize=False, add_generation_prompt=True
                        )
                        prompts.append(prompt)
                        
                        # Extract answer from assistant response
                        answer = self.extract_answer(assistant_msg or "")
                        answers.append(answer)
                    else:
                        prompts.append("")
                        answers.append("")
                
                return {"prompt": prompts, "answer": answers}
            
            elif "prompt" in examples and self.args.extract_answer_field in examples:
                # Direct prompt/answer format
                prompts = []
                answers = []
                
                for prompt, solution in zip(examples["prompt"], examples[self.args.extract_answer_field]):
                    # Format prompt with system message
                    prompt_msgs = [
                        {"role": "system", "content": self.args.system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        prompt_msgs, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(formatted_prompt)
                    
                    # Extract answer
                    answer = self.extract_answer(solution or "")
                    answers.append(answer)
                
                return {"prompt": prompts, "answer": answers}
            
            else:
                raise ValueError(
                    "Dataset must have either 'conversations', 'messages', or 'prompt' + "
                    f"'{self.args.extract_answer_field}' columns"
                )
        
        dataset = dataset.map(formatting_prompts_func, batched=True)
        print("[INFO] GRPO formatting complete.")
        return dataset

    def tokenize_and_prepare_labels(self, dataset):
        """For GRPO, we only need prompts and answers - no pre-tokenization needed."""
        print("[INFO] Preparing GRPO dataset (prompts + answers only)...")
        
        # Filter out empty prompts/answers
        def filter_valid(example):
            return bool(example.get("prompt", "").strip()) and bool(example.get("answer", "").strip())
        
        data = dataset.filter(filter_valid)
        
        # Add some basic stats
        def add_stats(example):
            prompt_len = len(self.tokenizer(example["prompt"])["input_ids"])
            return {
                **example,
                "prompt_length": prompt_len,
                "answer_length": len(example["answer"]),
            }
        
        data = data.map(add_stats)
        
        self._last_size = len(data)
        print(f"[INFO] GRPO dataset ready. Valid samples: {self._last_size}")
        
        # Print some stats
        if self._last_size > 0:
            avg_prompt_len = sum(data["prompt_length"]) / len(data)
            avg_answer_len = sum(data["answer_length"]) / len(data)
            print(f"[INFO] Average prompt length: {avg_prompt_len:.1f} tokens")
            print(f"[INFO] Average answer length: {avg_answer_len:.1f} characters")
        
        return data

    def debug_visualization(self, data):
        """Create debug visualization for GRPO dataset."""
        if self.args.debug > 0:
            print(f"[INFO] Debug mode: showing {min(self.args.debug, len(data))} GRPO samples...")
            
            for i in range(min(self.args.debug, len(data))):
                sample = data[i]
                print(f"\n--- Sample {i+1} ---")
                print(f"Prompt ({sample.get('prompt_length', '?')} tokens):")
                print(sample["prompt"][:200] + "..." if len(sample["prompt"]) > 200 else sample["prompt"])
                print(f"\nExpected Answer: {sample['answer']}")
                print("-" * 50)

    def save_dataset(self, data):
        """Save GRPO dataset with metadata."""
        if self.args.debug <= 0:
            print(f"[INFO] Saving GRPO dataset to {self.args.output_dir} ...")
            data.save_to_disk(self.args.output_dir)
            
            # Save GRPO-specific metadata
            try:
                meta = self._build_metadata()
                meta.update({
                    "training_type": "grpo",
                    "extract_answer_field": self.args.extract_answer_field,
                    "system_prompt": self.args.system_prompt,
                })
                
                with open(os.path.join(self.args.output_dir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARN] Failed to write metadata.json: {e}")
            
            # Save complete config
            try:
                complete_config = vars(self.args).copy()
                complete_config['model_family'] = 'GRPO'
                complete_config['training_type'] = 'grpo'
                
                with open(os.path.join(self.args.output_dir, "dataset_config.json"), "w", encoding="utf-8") as f:
                    json.dump(complete_config, f, ensure_ascii=False, indent=2)
                print(f"[INFO] GRPO config saved to {self.args.output_dir}/dataset_config.json")
            except Exception as e:
                print(f"[WARN] Failed to write dataset_config.json: {e}")
            
            print(f"[INFO] GRPO dataset saved to {self.args.output_dir}")


if __name__ == "__main__":
    preparer = GRPODatasetPreparer()
    preparer.run()
