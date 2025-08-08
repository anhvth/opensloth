#!/usr/bin/env python3
"""
Test script to verify that preset loading works correctly
"""

import os
import sys
import json

# Add the current directory to Python path
sys.path.insert(0, '/home/anhvth5/projects/opensloth/prepare_dataset')

from config_schema import DatasetPrepConfig

def test_preset_loading():
    """Test that preset files can be loaded correctly"""
    
    PRESETS_DATA_DIR = "/home/anhvth5/projects/opensloth/prepare_dataset/presets/data"
    
    def _load_json(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def _list_json_files(dir_path: str) -> list[str]:
        if not os.path.isdir(dir_path):
            return []
        names = [f for f in os.listdir(dir_path) if f.endswith(".json")]
        names.sort()
        return names
    
    def _apply_preset_file(fname):
        """Simulates the preset loading function from the app"""
        if not fname:
            return [None for _ in range(12)]
        try:
            path = os.path.join(PRESETS_DATA_DIR, fname)
            preset_data = _load_json(path)
            cfg = DatasetPrepConfig(**preset_data)
            return [
                preset_data.get("model_family", "Qwen"),  # model_family
                cfg.tok_name,                             # tok_name
                cfg.chat_template,                        # chat_template
                cfg.dataset_name,                         # dataset_name
                cfg.split,                                # split
                cfg.num_samples,                          # num_samples
                cfg.num_proc,                             # num_proc
                cfg.train_on_target_only,                 # train_on_target_only
                cfg.instruction_part,                     # instruction_part
                cfg.response_part,                        # response_part
                cfg.debug,                                # debug
                cfg.output_dir or "",                     # output_dir
            ]
        except Exception as e:
            print(f"Error loading preset {fname}: {e}")
            return [None for _ in range(12)]
    
    print("=== Testing Preset Loading ===")
    
    # Test 1: List available presets
    print("\n1. Available preset files:")
    preset_files = _list_json_files(PRESETS_DATA_DIR)
    for preset in preset_files:
        print(f"   - {preset}")
    
    # Test 2: Test loading each preset
    print("\n2. Testing preset loading:")
    for preset_file in preset_files:
        print(f"\n   Loading {preset_file}:")
        try:
            values = _apply_preset_file(preset_file)
            field_names = [
                "model_family", "tok_name", "chat_template", "dataset_name", 
                "split", "num_samples", "num_proc", "train_on_target_only",
                "instruction_part", "response_part", "debug", "output_dir"
            ]
            
            for field_name, value in zip(field_names, values):
                print(f"     {field_name}: {value}")
            
            print(f"     ✅ {preset_file} loaded successfully")
            
        except Exception as e:
            print(f"     ❌ Error loading {preset_file}: {e}")
    
    # Test 3: Verify specific preset values
    print("\n3. Verifying qwen_finetome.json specifically:")
    qwen_values = _apply_preset_file("qwen_finetome.json")
    expected_values = {
        0: "Qwen",  # model_family
        1: "unsloth/Qwen3-0.6B-Instruct",  # tok_name
        2: "qwen-3",  # chat_template
        5: 1000,  # num_samples
    }
    
    for index, expected in expected_values.items():
        actual = qwen_values[index]
        if actual == expected:
            print(f"   ✅ Field {index}: {actual} (expected: {expected})")
        else:
            print(f"   ❌ Field {index}: {actual} (expected: {expected})")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_preset_loading()
