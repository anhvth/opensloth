# import argparse
# import os
# import sys
# import json
# import datetime
# from pathlib import Path
# from typing import Optional, Dict, Any, List

# # Import from the proper module structure
# from opensloth.dataset import (
#     DatasetPrepConfig,
#     BaseDatasetPreparer,
#     QwenDatasetPreparer,
#     GemmaDatasetPreparer
# )


# # Model family mappings
# MODEL_FAMILIES = {
#     "qwen": QwenDatasetPreparer,
#     "gemma": GemmaDatasetPreparer,
# }

# # Common model name patterns to auto-detect family
# MODEL_PATTERNS = {
#     "qwen": ["qwen", "qwen2", "qwen2.5", "qwen3"],
#     "gemma": ["gemma", "gemma-2", "gemma-3"],
#     "llama": ["llama", "llama-2", "llama-3", "code-llama"],
#     "mistral": ["mistral", "mixtral"],
# }

# # Default configurations for different model families
# DEFAULT_CONFIGS = {
#     "qwen": {
#         "tok_name": "unsloth/Qwen2.5-0.5B-Instruct",
#         "chat_template": "qwen-2.5",
#         "instruction_part": "<|im_start|>user\n",
#         "response_part": "<|im_start|>assistant\n",
#     },
#     "gemma": {
#         "tok_name": "unsloth/gemma-2-2b-it",
#         "chat_template": "gemma",
#         "instruction_part": "<start_of_turn>user\n",
#         "response_part": "<start_of_turn>model\n",
#     },
#     "llama": {
#         "tok_name": "unsloth/Llama-3.2-1B-Instruct",
#         "chat_template": "llama-3.1", 
#         "instruction_part": "<|start_header_id|>user<|end_header_id|>\n\n",
#         "response_part": "<|start_header_id|>assistant<|end_header_id|>\n\n",
#     },
# }


# def detect_model_family(model_name: str) -> Optional[str]:
#     """Auto-detect model family from model name."""
#     model_lower = model_name.lower()
#     for family, patterns in MODEL_PATTERNS.items():
#         if any(pattern in model_lower for pattern in patterns):
#             return family
#     return None


# def generate_output_dir(model_name: str, dataset_name: str, num_samples: int) -> str:
#     """Generate a descriptive output directory name."""
#     today = datetime.datetime.now().strftime("%m%d")
    
#     # Extract model family
#     family = detect_model_family(model_name) or "unknown"
    
#     # Clean dataset name
#     dataset_short = dataset_name.split('/')[-1].replace('-', '_').lower()
    
#     # Format sample count
#     if num_samples > 0:
#         count = f"n{num_samples}"
#     else:
#         count = "all"
    
#     return f"data/{family}_{dataset_short}_{count}_{today}"


# def list_presets() -> List[str]:
#     """List available preset configurations."""
#     preset_dir = Path(__file__).parent.parent / "dataset" / "presets" / "data"
    
#     if not preset_dir.exists():
#         return []
    
#     presets = []
#     for file in preset_dir.glob("*.json"):
#         try:
#             with open(file) as f:
#                 data = json.load(f)
#             description = data.get("description", "")
#             name = file.stem.replace("_", " ").title()
#             presets.append(f"{name}: {description}" if description else name)
#         except Exception:
#             presets.append(file.stem)
    
#     return presets


# def save_preset(name: str, config: Dict[str, Any]) -> None:
#     """Save configuration as a preset."""
#     preset_dir = Path(__file__).parent.parent / "dataset" / "presets" / "data"
#     preset_dir.mkdir(parents=True, exist_ok=True)
    
#     filename = name.lower().replace(" ", "_") + ".json"
#     filepath = preset_dir / filename
    
#     with open(filepath, 'w') as f:
#         json.dump(config, f, indent=2, ensure_ascii=False)
    
#     print(f"✅ Preset saved as '{filename}'")


# def load_preset(name: str) -> Optional[Dict[str, Any]]:
#     """Load a preset configuration."""
#     preset_dir = Path(__file__).parent.parent / "dataset" / "presets" / "data"
    
#     # Try exact filename first
#     filename = name.lower().replace(" ", "_") + ".json"
#     filepath = preset_dir / filename
    
#     if not filepath.exists():
#         # Try searching for partial matches
#         for file in preset_dir.glob("*.json"):
#             if name.lower() in file.stem.lower():
#                 filepath = file
#                 break
#         else:
#             return None
    
#     try:
#         with open(filepath) as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"❌ Error loading preset '{name}': {e}")
#         return None


# def create_parser() -> argparse.ArgumentParser:
#     """Create the main argument parser."""
#     parser = argparse.ArgumentParser(
#         description="OpenSloth Dataset Preparation CLI",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   # Quick start with HuggingFace dataset
#   opensloth-dataset --model unsloth/Qwen2.5-0.5B-Instruct --dataset mlabonne/FineTome-100k --samples 1000

#   # Local JSON/JSONL file
#   opensloth-dataset --model unsloth/gemma-2-2b-it --dataset ./my_data.jsonl --family gemma

#   # Use a preset configuration
#   opensloth-dataset --preset qwen_finetome --samples 5000

#   # Save current config as preset
#   opensloth-dataset --model qwen --dataset finetome --save-preset my_config

#   # List available presets
#   opensloth-dataset --list-presets

#   # Full configuration example
#   opensloth-dataset \\
#     --model unsloth/Qwen2.5-7B-Instruct \\
#     --dataset mlabonne/FineTome-100k \\
#     --chat-template qwen-2.5 \\
#     --samples 10000 \\
#     --workers 8 \\
#     --output data/qwen_finetome_10k \\
#     --train-on-target-only \\
#     --debug 5

# Dataset Sources:
#   - HuggingFace datasets: 'mlabonne/FineTome-100k', 'HuggingFaceH4/ultrachat_200k'
#   - Local files: './conversations.json', '/path/to/data.jsonl'
#   - Supported formats: JSON, JSONL with 'messages' or 'conversations' fields

# Model Families:
#   - Qwen: qwen, qwen2, qwen2.5, qwen3 models
#   - Gemma: gemma, gemma-2, gemma-3 models  
#   - Llama: llama, llama-2, llama-3, code-llama models
#   - Mistral: mistral, mixtral models
  
# For more information: https://github.com/anhvth/opensloth
#         """
#     )
    
#     # Dataset and model configuration
#     parser.add_argument(
#         "--model", "--tok-name", dest="model_name",
#         type=str, 
#         help="HuggingFace model identifier or local path (e.g., 'unsloth/Qwen2.5-0.5B-Instruct')"
#     )
    
#     parser.add_argument(
#         "--dataset", "--dataset-name", dest="dataset_name",
#         type=str,
#         help="HuggingFace dataset name or local file path (JSON/JSONL)"
#     )
    
#     parser.add_argument(
#         "--family", "--model-family", dest="model_family",
#         type=str, choices=list(MODEL_FAMILIES.keys()),
#         help="Model family (auto-detected if not specified)"
#     )
    
#     parser.add_argument(
#         "--chat-template", dest="chat_template",
#         type=str,
#         help="Chat template to use (auto-detected from model if not specified)"
#     )
    
#     # Processing parameters
#     parser.add_argument(
#         "--samples", "--num-samples", dest="num_samples",
#         type=int, default=-1,
#         help="Number of samples to process (-1 for all, recommended: 1000-10000 for testing)"
#     )
    
#     parser.add_argument(
#         "--split", dest="split",
#         type=str, default="train",
#         help="Dataset split to use for HuggingFace datasets (default: train)"
#     )
    
#     parser.add_argument(
#         "--workers", "--num-proc", dest="num_proc",
#         type=int, default=8,
#         help="Number of parallel workers for processing (default: 8)"
#     )
    
#     parser.add_argument(
#         "--output", "--output-dir", dest="output_dir",
#         type=str,
#         help="Output directory (auto-generated if not specified)"
#     )
    
#     # Training configuration
#     parser.add_argument(
#         "--train-on-target-only", dest="train_on_target_only",
#         action="store_true", default=True,
#         help="Only train on assistant responses (recommended for chat models)"
#     )
    
#     parser.add_argument(
#         "--train-on-all", dest="train_on_target_only",
#         action="store_false",
#         help="Train on all tokens (alternative to --train-on-target-only)"
#     )
    
#     parser.add_argument(
#         "--instruction-part", dest="instruction_part",
#         type=str,
#         help="Text that marks the start of user/instruction turns"
#     )
    
#     parser.add_argument(
#         "--response-part", dest="response_part", 
#         type=str,
#         help="Text that marks the start of assistant/response turns"
#     )
    
#     # Debug and utility
#     parser.add_argument(
#         "--debug", dest="debug",
#         type=int, default=0,
#         help="Enable debug mode and save N samples as HTML for inspection"
#     )
    
#     parser.add_argument(
#         "--hf-token", dest="hf_token",
#         type=str,
#         help="HuggingFace token for accessing gated models/datasets"
#     )
    
#     # Preset management
#     parser.add_argument(
#         "--preset", dest="preset_name",
#         type=str,
#         help="Load configuration from a preset"
#     )
    
#     parser.add_argument(
#         "--save-preset", dest="save_preset_name",
#         type=str,
#         help="Save current configuration as a preset"
#     )
    
#     parser.add_argument(
#         "--list-presets", dest="list_presets",
#         action="store_true",
#         help="List all available presets"
#     )
    
#     # Configuration file
#     parser.add_argument(
#         "--config", dest="config_file",
#         type=str,
#         help="Load configuration from JSON file"
#     )
    
#     parser.add_argument(
#         "--save-config", dest="save_config_file",
#         type=str,
#         help="Save current configuration to JSON file"
#     )
    
#     return parser


# def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
#     """Merge two configuration dictionaries."""
#     result = base.copy()
#     for key, value in override.items():
#         if value is not None:
#             result[key] = value
#     return result


# def args_to_config(args: argparse.Namespace) -> Dict[str, Any]:
#     """Convert argparse namespace to configuration dictionary."""
#     config = {}
    
#     # Direct mappings
#     direct_mappings = {
#         "model_name": "tok_name",
#         "dataset_name": "dataset_name", 
#         "chat_template": "chat_template",
#         "num_samples": "num_samples",
#         "split": "split",
#         "num_proc": "num_proc",
#         "output_dir": "output_dir",
#         "train_on_target_only": "train_on_target_only",
#         "instruction_part": "instruction_part",
#         "response_part": "response_part",
#         "debug": "debug",
#         "hf_token": "hf_token",
#     }
    
#     for arg_name, config_key in direct_mappings.items():
#         value = getattr(args, arg_name, None)
#         if value is not None:
#             config[config_key] = value
    
#     return config


# def validate_config(config: Dict[str, Any]) -> None:
#     """Validate the configuration."""
#     required_fields = ["tok_name", "dataset_name"]
    
#     for field in required_fields:
#         if not config.get(field):
#             raise ValueError(f"Missing required field: {field}")
    
#     if config.get("train_on_target_only") and (
#         not config.get("instruction_part") or not config.get("response_part")
#     ):
#         raise ValueError(
#             "instruction_part and response_part are required when train_on_target_only is enabled"
#         )


# def main():
#     """Main CLI entry point."""
#     parser = create_parser()
#     args = parser.parse_args()
    
#     try:
#         # Handle utility commands first
#         if args.list_presets:
#             presets = list_presets()
#             if presets:
#                 print("Available presets:")
#                 for preset in presets:
#                     print(f"  - {preset}")
#             else:
#                 print("No presets found.")
#             return
        
#         # Start with empty config
#         config = {}
        
#         # Load from config file if specified
#         if args.config_file:
#             if not os.path.exists(args.config_file):
#                 raise FileNotFoundError(f"Config file not found: {args.config_file}")
            
#             with open(args.config_file) as f:
#                 config = json.load(f)
#             print(f"📁 Loaded config from {args.config_file}")
        
#         # Load from preset if specified
#         if args.preset_name:
#             preset_config = load_preset(args.preset_name)
#             if preset_config is None:
#                 raise ValueError(f"Preset not found: {args.preset_name}")
            
#             config = merge_configs(config, preset_config)
#             print(f"🎯 Applied preset '{args.preset_name}'")
        
#         # Auto-detect model family and apply defaults
#         model_name = args.model_name or config.get("tok_name")
#         if model_name:
#             detected_family = args.model_family or detect_model_family(model_name)
#             if detected_family and detected_family in DEFAULT_CONFIGS:
#                 family_defaults = DEFAULT_CONFIGS[detected_family]
#                 config = merge_configs(family_defaults, config)
#                 print(f"🤖 Detected {detected_family} model family")
        
#         # Override with command line arguments
#         cli_config = args_to_config(args)
#         config = merge_configs(config, cli_config)
        
#         # Auto-generate output directory if not specified
#         if not config.get("output_dir") and config.get("tok_name") and config.get("dataset_name"):
#             config["output_dir"] = generate_output_dir(
#                 config["tok_name"], 
#                 config["dataset_name"], 
#                 config.get("num_samples", -1)
#             )
        
#         # Validate configuration
#         validate_config(config)
        
#         # Save config if requested
#         if args.save_config_file:
#             with open(args.save_config_file, 'w') as f:
#                 json.dump(config, f, indent=2, ensure_ascii=False)
#             print(f"💾 Saved config to {args.save_config_file}")
        
#         # Save as preset if requested
#         if args.save_preset_name:
#             save_preset(args.save_preset_name, config)
        
#         # Determine model family for preparer selection
#         family = args.model_family or detect_model_family(config["tok_name"])
#         if not family or family not in MODEL_FAMILIES:
#             # Default to Qwen if we can't detect
#             family = "qwen"
#             print(f"⚠️  Could not detect model family, defaulting to {family}")
        
#         # Create and run preparer
#         preparer_class = MODEL_FAMILIES[family]
#         preparer = preparer_class()
        
#         print(f"\n🚀 Starting dataset preparation with {family} preparer...")
#         print(f"📊 Model: {config['tok_name']}")
#         print(f"📁 Dataset: {config['dataset_name']}")
#         print(f"📝 Samples: {config.get('num_samples', -1)}")
#         print(f"💾 Output: {config['output_dir']}")
#         print()
        
#         # Run preparation
#         output_dir = preparer.run_with_config(config)
        
#         print(f"\n✅ Dataset preparation completed!")
#         print(f"📁 Output directory: {output_dir}")
        
#         if config.get("debug", 0) > 0:
#             debug_file = ".log/dataloader_examples.html"
#             if os.path.exists(debug_file):
#                 print(f"🔍 Debug visualization: {debug_file}")
        
#         print(f"\n💡 Next step: Use this dataset path in training:")
#         print(f"   opensloth-train --dataset {output_dir} --model {config['tok_name']}")
        
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         sys.exit(1)


# if __name__ == "__main__":
#     main()
