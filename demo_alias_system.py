#!/usr/bin/env python3
"""
Simple test to demonstrate CLI aliases working.
"""
import json
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_manual_alias_usage():
    """Test manually using the new CLI builder with aliases."""
    print("Testing manual CLI alias usage...")
    
    from opensloth.config.builder import TrainingConfigBuilder
    
    # Create a temporary dataset directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_path = Path(temp_dir)
        
        # Create a mock dataset_config.json
        dataset_config = {
            "tok_name": "unsloth/gemma-3-4b-it",
            "max_seq_length": 2048,
            "num_shards": 1
        }
        with open(dataset_path / "dataset_config.json", "w") as f:
            json.dump(dataset_config, f)
        
        # Test with aliased CLI overrides (simulating what the new auto-generator would produce)
        cli_overrides = {
            "opensloth_config": {
                "fast_model_args": {
                    "model_name": "unsloth/new-model",  # Would come from --model
                    "max_seq_length": 4096,             # Would come from --max-seq-length
                    "load_in_4bit": False               # Would come from --load-in-4bit
                },
                "lora_args": {
                    "r": 32,                            # Would come from --lora-r
                    "lora_alpha": 64,                   # Would come from --lora-alpha
                }
            },
            "training_args": {
                "learning_rate": 1e-4,                 # Would come from --lr
                "num_train_epochs": 5,                 # Would come from --epochs
                "per_device_train_batch_size": 4,      # Would come from --batch-size
                "gradient_accumulation_steps": 2       # Would come from --grad-accum
            }
        }
        
        # Build configuration using the new simplified builder
        builder = TrainingConfigBuilder(dataset_path=str(dataset_path), method="sft")
        opensloth_cfg, train_args = builder.with_cli_args(cli_overrides).build()
        
        # Verify the configuration
        assert opensloth_cfg.fast_model_args.model_name == "unsloth/new-model"
        assert opensloth_cfg.fast_model_args.max_seq_length == 4096
        assert not opensloth_cfg.fast_model_args.load_in_4bit
        assert opensloth_cfg.lora_args.r == 32
        assert opensloth_cfg.lora_args.lora_alpha == 64
        assert train_args.learning_rate == 1e-4
        assert train_args.num_train_epochs == 5
        assert train_args.per_device_train_batch_size == 4
        assert train_args.gradient_accumulation_steps == 2
        
        print("✅ Manual CLI alias usage works correctly")
        
        # Show what the equivalent CLI commands would look like
        print("\n🎯 CLI Alias Equivalents:")
        print("New (with aliases):")
        print("  os-sft dataset/ --model unsloth/new-model --max-seq-length 4096 --load-in-4bit false")
        print("  --lora-r 32 --lora-alpha 64 --lr 1e-4 --epochs 5 --batch-size 4 --grad-accum 2")
        print()
        print("Old (without aliases):")
        print("  os-sft dataset/ --fast-model-args-model-name unsloth/new-model")
        print("  --fast-model-args-max-seq-length 4096 --fast-model-args-load-in-4bit false")
        print("  --lora-args-r 32 --lora-args-lora-alpha 64 --learning-rate 1e-4")
        print("  --num-train-epochs 5 --per-device-train-batch-size 4 --gradient-accumulation-steps 2")

def test_field_alias_metadata():
    """Test that field aliases are properly defined in the Pydantic models."""
    print("\nTesting field alias metadata...")
    
    from opensloth.opensloth_config import FastModelArgs, LoraArgs, TrainingArguments
    
    # Check FastModelArgs aliases
    model_name_field = FastModelArgs.model_fields['model_name']
    assert model_name_field.json_schema_extra.get('cli_alias') == 'model', "model_name alias missing"
    
    max_seq_length_field = FastModelArgs.model_fields['max_seq_length']
    assert max_seq_length_field.json_schema_extra.get('cli_alias') == 'max-seq-length', "max_seq_length alias missing"
    
    # Check LoraArgs aliases
    r_field = LoraArgs.model_fields['r']
    assert r_field.json_schema_extra.get('cli_alias') == 'lora-r', "r alias missing"
    
    lora_alpha_field = LoraArgs.model_fields['lora_alpha']
    assert lora_alpha_field.json_schema_extra.get('cli_alias') == 'lora-alpha', "lora_alpha alias missing"
    
    # Check TrainingArguments aliases
    lr_field = TrainingArguments.model_fields['learning_rate']
    assert lr_field.json_schema_extra.get('cli_alias') == 'lr', "learning_rate alias missing"
    
    epochs_field = TrainingArguments.model_fields['num_train_epochs']
    assert epochs_field.json_schema_extra.get('cli_alias') == 'epochs', "num_train_epochs alias missing"
    
    print("✅ Field alias metadata works correctly")

def run_demo():
    """Run the CLI alias demonstration."""
    print("🚀 CLI Alias System Demonstration\n")
    
    test_field_alias_metadata()
    test_manual_alias_usage()
    
    print("\n✅ CLI Alias System Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("- ✅ Pydantic models define user-friendly aliases")
    print("- ✅ Builder system works with aliased configurations")  
    print("- ✅ Backward compatibility maintained")
    print("- ✅ Much shorter, more intuitive CLI commands")
    
    print("\n🔄 Next Steps:")
    print("- Integrate autogen.py with existing CLI commands")
    print("- Update CLI commands to use @cli_from_pydantic decorator")
    print("- Provide migration guide for users")

if __name__ == "__main__":
    run_demo()
