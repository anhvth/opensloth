"""
Simple dataset preparation for GRPO training.
Creates a minimal math reasoning dataset in the expected format.
"""

import os
from datasets import Dataset, DatasetDict

def create_grpo_test_dataset():
    """Create a minimal GRPO test dataset with math problems."""
    
    # Sample math problems with prompts and answers
    samples = [
        {
            "prompt": [
                {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"},
                {"role": "user", "content": "What is 15 + 27?"},
            ],
            "answer": "42"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"},
                {"role": "user", "content": "What is 8 × 9?"},
            ],
            "answer": "72"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"},
                {"role": "user", "content": "What is 144 ÷ 12?"},
            ],
            "answer": "12"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"},
                {"role": "user", "content": "What is 25 - 13?"},
            ],
            "answer": "12"
        },
        {
            "prompt": [
                {"role": "system", "content": "You are given a problem. Think about the problem and provide your working out. Place it between <start_working_out> and <end_working_out>. Then, provide your solution between <SOLUTION></SOLUTION>"},
                {"role": "user", "content": "What is 7 × 6 + 4?"},
            ],
            "answer": "46"
        },
    ]
    
    # Expand the dataset by repeating samples
    expanded_samples = samples * 20  # 100 total samples for testing
    
    return Dataset.from_list(expanded_samples)

def main():
    """Main function to create and save the dataset."""
    print("Creating GRPO test dataset...")
    
    # Create dataset
    dataset = create_grpo_test_dataset()
    
    # Save to disk
    output_path = "data/grpo_math_test"
    os.makedirs(output_path, exist_ok=True)
    
    dataset.save_to_disk(output_path)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset size: {len(dataset)} samples")
    print("Sample:")
    print(dataset[0])
    
    # Save dataset config for compatibility
    import json
    config = {
        "dataset_type": "grpo",
        "format": "chat_template",
        "max_seq_length": 2048,
        "created_for": "GRPO math reasoning training"
    }
    
    with open(os.path.join(output_path, "dataset_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print("Dataset preparation complete!")

if __name__ == "__main__":
    main()
