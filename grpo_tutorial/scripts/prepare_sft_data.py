#!/usr/bin/env python3
"""
SFT Data Preparation Script for OpenSloth GRPO Tutorial

This script prepares the initial SFT dataset by downloading raw data from
Hugging Face, applying custom formatting for math reasoning, and saving
it as a local JSONL file that os-data can process.
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
import json
import os
from pathlib import Path

def main():
    """
    This script prepares the initial SFT dataset by downloading raw data from
    Hugging Face, applying custom formatting for math reasoning, and saving
    it as a local JSONL file that os-data can process.
    """
    print("🔄 Preparing SFT dataset for reasoning format pre-finetuning...")

    # These are the custom markers for our reasoning format
    reasoning_start = "<start_working_out>"
    reasoning_end   = "<end_working_out>"
    solution_start  = "<SOLUTION>"
    solution_end    = "</SOLUTION>"

    system_prompt = (
        f"You are given a problem.\n"
        f"Think about the problem and provide your working out.\n"
        f"Place it between {reasoning_start} and {reasoning_end}.\n"
        f"Then, provide your solution between {solution_start}{solution_end}"
    )

    try:
        # Load the dataset from Hugging Face
        print("📥 Loading dataset from Hugging Face: unsloth/OpenMathReasoning-mini")
        # Dataset has a single split named 'cot'
        ds_dict = load_dataset("unsloth/OpenMathReasoning-mini")
        dataset = ds_dict["cot"]
        print(f"✅ Loaded {len(dataset)} examples (split: cot)")

        # Prepare the formatted data
        formatted_data = []
        
        for i, example in enumerate(dataset):
            # Extract the components from the example
            # The dataset should have 'problem' and 'solution' fields
            problem = example.get('problem', '')
            # 'generated_solution' contains step-by-step reasoning ending with answer, 'expected_answer' the concise final answer
            gen_solution = example.get('generated_solution', '')
            expected_answer = example.get('expected_answer', '')
            # Combine: keep original reasoning then ensure solution tag has concise expected answer if available
            solution = gen_solution if gen_solution else expected_answer
            
            if not problem or not solution:
                print(f"⚠️  Skipping example {i}: missing problem or solution")
                continue
            
            # Create the formatted response with reasoning tags
            # We'll structure it as: reasoning_start + working_out + reasoning_end + solution_start + final_answer + solution_end
            
            # For the SFT stage, we'll use the existing solution as both working out and final answer
            # In a real scenario, you might want to parse these separately
            formatted_response = f"{reasoning_start}\n{solution}\n{reasoning_end}\n{solution_start}{expected_answer or solution}{solution_end}"
            
            # Create the conversation format expected by OpenSloth
            conversation = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": problem},
                {"role": "assistant", "content": formatted_response}
            ]
            
            formatted_data.append({
                "conversations": conversation,
                "id": f"sft_math_{i}"
            })
            
            if (i + 1) % 100 == 0:
                print(f"✅ Processed {i + 1} examples")

        # Create output directory if it doesn't exist
        output_dir = Path("grpo_tutorial/data")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to JSONL format
        output_file = output_dir / "sft_openmath_raw.jsonl"
        print(f"💾 Saving formatted data to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Successfully saved {len(formatted_data)} formatted examples to {output_file}")
        print(f"📊 Dataset statistics:")
        print(f"   - Total examples: {len(formatted_data)}")
        print(f"   - Average problem length: {np.mean([len(item['conversations'][1]['content']) for item in formatted_data]):.1f} chars")
        print(f"   - Average response length: {np.mean([len(item['conversations'][2]['content']) for item in formatted_data]):.1f} chars")
        
        # Show a sample
        print("\n📖 Sample formatted example:")
        sample = formatted_data[0]
        print("System:", sample['conversations'][0]['content'][:100] + "...")
        print("User:", sample['conversations'][1]['content'][:100] + "...")
        print("Assistant:", sample['conversations'][2]['content'][:200] + "...")
        
    except Exception as e:
        print(f"❌ Error preparing SFT dataset: {e}")
        raise

if __name__ == "__main__":
    main()
