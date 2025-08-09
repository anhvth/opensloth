#!/usr/bin/env python3
"""
Dataset preparation for GRPO training.
Supports different task types with appropriate formatting.
"""

import os
import json
from datasets import Dataset, load_dataset
from typing import List, Dict, Any


def prepare_math_dataset(
    dataset_name: str = "open-r1/DAPO-Math-17k-Processed",
    subset: str = "en",
    num_samples: int = 1000,
    output_path: str = "./data/grpo_math"
) -> None:
    """Prepare math dataset for GRPO training."""
    print(f"Loading math dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, subset, split="train")
    
    # Take subset if specified
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format for GRPO
    def format_math_sample(sample):
        system_prompt = """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""
        
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample["prompt"]},
            ],
            "answer": sample.get("solution", "").strip(),
        }
    
    formatted_dataset = dataset.map(format_math_sample)
    
    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    formatted_dataset.save_to_disk(output_path)
    
    print(f"Saved {len(formatted_dataset)} math samples to {output_path}")
    
    # Print sample
    print("\\nSample:")
    sample = formatted_dataset[0]
    print("Prompt:", sample["prompt"])
    print("Answer:", sample["answer"][:100] + "..." if len(sample["answer"]) > 100 else sample["answer"])


def prepare_code_dataset(
    dataset_name: str = "codeparrot/github-code-clean",
    num_samples: int = 500,
    output_path: str = "./data/grpo_code"
) -> None:
    """Prepare code dataset for GRPO training."""
    print(f"Loading code dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Take subset and filter Python files
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        
        if sample.get("language") == "Python" and len(sample.get("code", "")) > 100:
            samples.append(sample)
    
    dataset = Dataset.from_list(samples)
    
    # Format for GRPO
    def format_code_sample(sample):
        # Extract function/class documentation as prompt
        code = sample["code"]
        lines = code.split("\\n")
        
        # Simple heuristic to find documentation
        prompt = "Write a Python function or class."
        for i, line in enumerate(lines[:10]):
            if '"""' in line or "'''" in line:
                # Found docstring
                doc_lines = []
                for j in range(i, min(i+10, len(lines))):
                    doc_lines.append(lines[j])
                    if j > i and ('"""' in lines[j] or "'''" in lines[j]):
                        break
                prompt = "\\n".join(doc_lines).strip()
                break
        
        return {
            "prompt": [
                {"role": "system", "content": "You are a helpful Python programming assistant. Write clean, efficient code."},
                {"role": "user", "content": f"Write Python code: {prompt}"},
            ],
            "answer": code.strip(),
        }
    
    formatted_dataset = dataset.map(format_code_sample)
    
    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    formatted_dataset.save_to_disk(output_path)
    
    print(f"Saved {len(formatted_dataset)} code samples to {output_path}")
    
    # Print sample
    print("\\nSample:")
    sample = formatted_dataset[0]
    print("Prompt:", sample["prompt"])
    print("Answer:", sample["answer"][:200] + "..." if len(sample["answer"]) > 200 else sample["answer"])


def prepare_general_qa_dataset(
    dataset_name: str = "Anthropic/hh-rlhf",
    subset: str = "helpful-base",
    num_samples: int = 500,
    output_path: str = "./data/grpo_general"
) -> None:
    """Prepare general QA dataset for GRPO training."""
    print(f"Loading general QA dataset: {dataset_name}")
    
    # Load dataset
    dataset = load_dataset(dataset_name, subset, split="train")
    
    # Take subset
    if num_samples > 0:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    # Format for GRPO
    def format_qa_sample(sample):
        # Parse conversation
        conversation = sample["chosen"]  # Use the preferred response
        
        # Simple parsing (this might need adjustment based on format)
        parts = conversation.split("Human:")
        if len(parts) < 2:
            return None
        
        human_part = parts[1].split("Assistant:")[0].strip()
        
        assistant_parts = conversation.split("Assistant:")
        if len(assistant_parts) < 2:
            return None
        
        assistant_part = assistant_parts[1].strip()
        
        return {
            "prompt": [
                {"role": "system", "content": "You are a helpful, harmless, and honest assistant."},
                {"role": "user", "content": human_part},
            ],
            "answer": assistant_part,
        }
    
    formatted_samples = []
    for sample in dataset:
        formatted = format_qa_sample(sample)
        if formatted:
            formatted_samples.append(formatted)
    
    formatted_dataset = Dataset.from_list(formatted_samples)
    
    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    formatted_dataset.save_to_disk(output_path)
    
    print(f"Saved {len(formatted_dataset)} QA samples to {output_path}")
    
    # Print sample
    print("\\nSample:")
    sample = formatted_dataset[0]
    print("Prompt:", sample["prompt"])
    print("Answer:", sample["answer"][:200] + "..." if len(sample["answer"]) > 200 else sample["answer"])


def prepare_custom_dataset(
    data_file: str,
    task_type: str = "general",
    output_path: str = "./data/grpo_custom"
) -> None:
    """Prepare custom dataset from JSON/JSONL file."""
    print(f"Loading custom dataset from: {data_file}")
    
    # Load data
    with open(data_file, 'r') as f:
        if data_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    # Format based on task type
    formatted_samples = []
    
    for sample in data:
        if task_type == "math":
            formatted = {
                "prompt": [
                    {"role": "system", "content": """You are given a problem.
Think about the problem and provide your working out.
Place it between <start_working_out> and <end_working_out>.
Then, provide your solution between <SOLUTION></SOLUTION>"""},
                    {"role": "user", "content": sample.get("question", sample.get("prompt", ""))},
                ],
                "answer": sample.get("answer", sample.get("solution", "")),
            }
        elif task_type == "code":
            formatted = {
                "prompt": [
                    {"role": "system", "content": "You are a helpful Python programming assistant. Write clean, efficient code."},
                    {"role": "user", "content": sample.get("instruction", sample.get("prompt", ""))},
                ],
                "answer": sample.get("code", sample.get("response", sample.get("answer", ""))),
            }
        else:  # general
            formatted = {
                "prompt": [
                    {"role": "system", "content": "You are a helpful, harmless, and honest assistant."},
                    {"role": "user", "content": sample.get("instruction", sample.get("prompt", sample.get("question", "")))},
                ],
                "answer": sample.get("response", sample.get("answer", sample.get("output", ""))),
            }
        
        formatted_samples.append(formatted)
    
    dataset = Dataset.from_list(formatted_samples)
    
    # Save dataset
    os.makedirs(output_path, exist_ok=True)
    dataset.save_to_disk(output_path)
    
    print(f"Saved {len(dataset)} custom samples to {output_path}")
    
    # Print sample
    print("\\nSample:")
    sample = dataset[0]
    print("Prompt:", sample["prompt"])
    print("Answer:", sample["answer"][:200] + "..." if len(sample["answer"]) > 200 else sample["answer"])


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for GRPO training")
    parser.add_argument("--task-type", choices=["math", "code", "general", "custom"], required=True,
                      help="Type of task to prepare dataset for")
    parser.add_argument("--num-samples", type=int, default=1000,
                      help="Number of samples to prepare")
    parser.add_argument("--output-path", type=str, required=True,
                      help="Output path for the prepared dataset")
    parser.add_argument("--custom-file", type=str,
                      help="Path to custom JSON/JSONL file (required for custom task)")
    
    args = parser.parse_args()
    
    if args.task_type == "math":
        prepare_math_dataset(
            num_samples=args.num_samples,
            output_path=args.output_path
        )
    elif args.task_type == "code":
        prepare_code_dataset(
            num_samples=args.num_samples,
            output_path=args.output_path
        )
    elif args.task_type == "general":
        prepare_general_qa_dataset(
            num_samples=args.num_samples,
            output_path=args.output_path
        )
    elif args.task_type == "custom":
        if not args.custom_file:
            print("Error: --custom-file is required for custom task type")
            exit(1)
        prepare_custom_dataset(
            data_file=args.custom_file,
            task_type="general",  # Default to general format
            output_path=args.output_path
        )
    
    print("\\nDataset preparation completed!")
