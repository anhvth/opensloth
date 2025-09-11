#!/usr/bin/env python3
"""
Auto-generate JSON Schema from OpenSloth Pydantic models.

This script creates a comprehensive JSON schema from the Pydantic configuration
classes in src/opensloth/opensloth_config.py. This ensures the schema is always
in sync with the actual code and provides accurate IntelliSense in VS Code.

Usage:
    python scripts/generate_schema.py
    # or with uv:
    uv run scripts/generate_schema.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

# Add src to path so we can import opensloth modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel

from opensloth.opensloth_config import (
    OpenSlothConfig,
    TrainingArguments, 
    DatasetPrepConfig,
    FastModelArgs,
    LoraArgs
)


def enhance_schema_with_examples(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Add practical examples and enhanced descriptions to the schema."""
    
    # Add examples for common model names
    if "properties" in schema:
        props = schema["properties"]
        
        # Enhanced opensloth_config
        if "opensloth_config" in props and "properties" in props["opensloth_config"]:
            osc_props = props["opensloth_config"]["properties"]
            
            # Add model examples
            if "fast_model_args" in osc_props and "properties" in osc_props["fast_model_args"]:
                fma_props = osc_props["fast_model_args"]["properties"]
                if "model_name" in fma_props:
                    fma_props["model_name"]["examples"] = [
                        "unsloth/Qwen2.5-0.5B-Instruct",
                        "unsloth/llama-3.2-1b-instruct", 
                        "unsloth/mistral-7b-instruct-v0.3",
                        "microsoft/DialoGPT-medium"
                    ]
            
            # Add LoRA target module enums
            if "lora_args" in osc_props and "properties" in osc_props["lora_args"]:
                lora_props = osc_props["lora_args"]["properties"]
                if "target_modules" in lora_props and "items" in lora_props["target_modules"]:
                    lora_props["target_modules"]["items"]["enum"] = [
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", 
                        "embed_tokens", "lm_head"
                    ]
        
        # Enhanced training_args
        if "training_args" in props and "properties" in props["training_args"]:
            ta_props = props["training_args"]["properties"]
            
            # Add optimizer enum
            if "optim" in ta_props:
                ta_props["optim"]["enum"] = [
                    "adamw_hf", "adamw_torch", "adamw_torch_fused", "adamw_apex_fused",
                    "adamw_anyprecision", "adafactor", "adamw_8bit", "adamw_bnb_8bit",
                    "lion_8bit", "lion_32bit", "paged_adamw_8bit", "paged_adamw_32bit",
                    "paged_lion_8bit", "paged_lion_32bit"
                ]
            
            # Add LR scheduler enum  
            if "lr_scheduler_type" in ta_props:
                ta_props["lr_scheduler_type"]["enum"] = [
                    "linear", "cosine", "cosine_with_restarts", "polynomial",
                    "constant", "constant_with_warmup", "inverse_sqrt", 
                    "reduce_lr_on_plateau"
                ]
        
        # Enhanced dataset_prep_config
        if "dataset_prep_config" in props and "properties" in props["dataset_prep_config"]:
            dpc_props = props["dataset_prep_config"]["properties"]
            
            # Add chat template enum
            if "chat_template" in dpc_props:
                dpc_props["chat_template"]["enum"] = [
                    "chatml", "qwen-2.5", "llama-3.1", "llama-3.2", "gemma",
                    "mistral", "phi-3", "zephyr", "vicuna", "alpaca"
                ]
            
            # Add dataset examples
            if "dataset_name" in dpc_props:
                dpc_props["dataset_name"]["examples"] = [
                    "mlabonne/FineTome-100k",
                    "HuggingFaceH4/ultrachat_200k",
                    "./local_dataset.json",
                    "microsoft/orca-math-word-problems-200k"
                ]
    
    return schema


def create_combined_schema() -> Dict[str, Any]:
    """Create a combined schema with all the configuration models."""
    
    # Get individual schemas
    opensloth_schema = OpenSlothConfig.model_json_schema()
    # training_args_schema = TrainingArguments.model_json_schema()
    from transformers import TrainingArguments
    from pydantic.dataclasses import dataclass
    from pydantic import TypeAdapter


    # Wrap Hugging Face TrainingArguments with Pydantic
    @dataclass
    class TrainingArgsSchema(TrainingArguments):
        pass


    adapter = TypeAdapter(TrainingArgsSchema)
    training_args_schema = adapter.json_schema()

    dataset_prep_schema = DatasetPrepConfig.model_json_schema()
    
    # Create the main combined schema
    combined_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "OpenSloth Training Configuration",
        "description": "Complete configuration schema for OpenSloth training with IntelliSense support",
        "type": "object",
        "properties": {
            "opensloth_config": opensloth_schema,
            "training_args": training_args_schema, 
            "dataset_prep_config": dataset_prep_schema
        },
        "required": ["opensloth_config"],
        "additionalProperties": True
    }
    
    # Enhance with examples and better descriptions
    enhanced_schema = enhance_schema_with_examples(combined_schema)
    
    return enhanced_schema


def main():
    """Generate the JSON schema and save it to the schemas directory."""
    
    print("🔧 Generating JSON Schema from OpenSloth Pydantic models...")
    
    try:
        # Generate the combined schema
        schema = create_combined_schema()
        
        # Ensure schemas directory exists
        schema_dir = Path(__file__).parent.parent / "schemas"
        schema_dir.mkdir(exist_ok=True)
        
        # Write the schema file
        schema_file = schema_dir / "training_config.schema.json"
        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Schema generated successfully!")
        print(f"📁 Saved to: {schema_file}")
        print(f"📊 Schema includes {len(schema.get('properties', {}))} top-level sections")
        
        # Show what's included
        sections = list(schema.get('properties', {}).keys())
        if sections:
            print(f"📋 Sections: {', '.join(sections)}")
        
        # Validate the schema by trying to create example models
        print("\n🧪 Validating schema by testing model creation...")
        
        # Test OpenSlothConfig
        test_config = OpenSlothConfig(
            data_cache_path="/tmp/test",
            fast_model_args={
                "model_name": "unsloth/Qwen2.5-0.5B-Instruct"
            }
        )
        print(f"   ✓ OpenSlothConfig validation passed")
        
        # Test TrainingArguments (now pure Pydantic)
        test_training = TrainingArguments()
        print(f"   ✓ TrainingArguments validation passed")
        
        # Test DatasetPrepConfig  
        test_dataset = DatasetPrepConfig(
            tokenizer_name="unsloth/Qwen2.5-0.5B-Instruct",
            dataset_name="mlabonne/FineTome-100k"
        )
        print(f"   ✓ DatasetPrepConfig validation passed")
        
        # Test HF conversion
        hf_training_args = test_training.to_hf_training_arguments()
        print(f"   ✓ HuggingFace TrainingArguments conversion passed")
        
        print("\n🎉 Schema generation completed successfully!")
        print("💡 VS Code will now provide IntelliSense for training config files")
        
    except Exception as e:
        print(f"❌ Error generating schema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()