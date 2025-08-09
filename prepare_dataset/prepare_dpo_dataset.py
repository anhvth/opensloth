"""
Example script for preparing DPO datasets.
This script shows how to format and prepare datasets for DPO training.
"""

import json
import os
from datasets import Dataset
from pathlib import Path


def create_example_dpo_dataset():
    """
    Create an example DPO dataset with proper format.
    DPO requires: prompt, chosen, rejected columns.
    """
    
    # Example DPO data - preference pairs
    dpo_examples = [
        {
            "prompt": "What is the capital of France?",
            "chosen": "The capital of France is Paris. Paris is a beautiful city known for its art, culture, and the Eiffel Tower.",
            "rejected": "The capital of France is Lyon. Lyon is a nice city in France."
        },
        {
            "prompt": "Explain machine learning in simple terms.",
            "chosen": "Machine learning is a way for computers to learn and make decisions from data, just like how humans learn from experience. The computer looks at lots of examples and finds patterns to make predictions about new situations.",
            "rejected": "Machine learning is very complex computer stuff with algorithms and math that only experts can understand."
        },
        {
            "prompt": "How do you make a sandwich?",
            "chosen": "To make a sandwich: 1) Take two slices of bread, 2) Add your favorite fillings like meat, cheese, vegetables, 3) Add condiments if desired, 4) Put the slices together. Enjoy your sandwich!",
            "rejected": "Put stuff between bread and eat it."
        },
        {
            "prompt": "What's the best way to learn programming?",
            "chosen": "The best way to learn programming is through hands-on practice. Start with a beginner-friendly language like Python, work on small projects, read documentation, and don't be afraid to make mistakes. Consistency and patience are key.",
            "rejected": "Just read books about programming theory and memorize syntax. You don't need to actually write code."
        }
    ]
    
    return dpo_examples


def prepare_dpo_dataset_for_opensloth(output_dir: str):
    """
    Prepare a DPO dataset in the format expected by OpenSloth.
    """
    
    # Create example data
    dpo_data = create_example_dpo_dataset()
    
    # Create dataset
    dataset = Dataset.from_list(dpo_data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    dataset.save_to_disk(output_dir)
    
    # Save dataset configuration
    dataset_config = {
        "training_type": "dpo",
        "dataset_size": len(dpo_data),
        "columns": ["prompt", "chosen", "rejected"],
        "description": "Example DPO dataset for preference optimization training"
    }
    
    config_path = Path(output_dir) / "dataset_config.json"
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"DPO dataset prepared and saved to: {output_dir}")
    print(f"Dataset size: {len(dpo_data)} examples")
    print(f"Required columns: {dataset_config['columns']}")
    
    # Print example
    print("\nExample DPO entry:")
    example = dpo_data[0]
    print(f"Prompt: {example['prompt']}")
    print(f"Chosen: {example['chosen']}")
    print(f"Rejected: {example['rejected']}")
    
    return output_dir


def convert_existing_dataset_to_dpo(input_file: str, output_dir: str):
    """
    Convert an existing preference dataset to DPO format.
    Assumes input is a JSON file with preference pairs.
    """
    
    # Load existing data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Ensure the data has the required format
    required_keys = ['prompt', 'chosen', 'rejected']
    for i, example in enumerate(data):
        missing_keys = [key for key in required_keys if key not in example]
        if missing_keys:
            raise ValueError(f"Example {i} missing required keys: {missing_keys}")
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Prepare output directory
    return prepare_dpo_dataset_for_opensloth(output_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare DPO dataset for OpenSloth")
    parser.add_argument("--output_dir", default="data/dpo_dataset_cache", 
                        help="Output directory for the prepared dataset")
    parser.add_argument("--input_file", default=None,
                        help="Input JSON file with existing preference data")
    
    args = parser.parse_args()
    
    if args.input_file:
        # Convert existing dataset
        convert_existing_dataset_to_dpo(args.input_file, args.output_dir)
    else:
        # Create example dataset
        prepare_dpo_dataset_for_opensloth(args.output_dir)
    
    print(f"\nTo use this dataset for DPO training, set data_cache_path='{args.output_dir}' in your OpenSlothConfig")
