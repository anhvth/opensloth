# DPO Training with OpenSloth

This document explains how to use OpenSloth for DPO (Direct Preference Optimization) training, as well as future support for other RL training methods like KTO, ORPO, and GRPO.

## Overview

OpenSloth now supports multiple training types beyond just SFT (Supervised Fine-Tuning):

- **SFT (Supervised Fine-Tuning)** - Traditional fine-tuning with input-output pairs
- **DPO (Direct Preference Optimization)** - Training with preference pairs (chosen vs rejected responses)
- **KTO, ORPO, GRPO** - Future RL methods (planned)

## DPO Training Setup

### 1. Dataset Preparation

DPO requires datasets with preference pairs. Your dataset should have three columns:
- `prompt`: The input/question
- `chosen`: The preferred response
- `rejected`: The less preferred response

#### Option A: Create Example Dataset
```bash
cd prepare_dataset
python prepare_dpo_dataset.py --output_dir data/dpo_dataset_cache
```

#### Option B: Convert Existing Dataset
```bash
cd prepare_dataset
python prepare_dpo_dataset.py --input_file your_preference_data.json --output_dir data/dpo_dataset_cache
```

Your input JSON should look like:
```json
[
  {
    "prompt": "What is the capital of France?",
    "chosen": "The capital of France is Paris. Paris is a beautiful city...",
    "rejected": "The capital of France is Lyon. Lyon is a nice city..."
  }
]
```

### 2. Configuration

Create a training script with DPO configuration:

```python
from opensloth.opensloth_config import (
    FastModelArgs, LoraArgs, OpenSlothConfig, 
    TrainingArguments, DPOArgs
)

# Configure DPO training
opensloth_config = OpenSlothConfig(
    data_cache_path='data/dpo_dataset_cache',
    devices=[0],
    training_type="dpo",  # Specify DPO training
    
    # Model configuration
    fast_model_args=FastModelArgs(
        model_name='unsloth/zephyr-sft-bnb-4bit',  # Use SFT model as base
        max_seq_length=2048,
        load_in_4bit=True,
    ),
    
    # LoRA configuration
    lora_args=LoraArgs(
        r=64,
        lora_alpha=64,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 
                       'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0,
    ),
    
    # DPO-specific parameters
    dpo_args=DPOArgs(
        beta=0.1,  # DPO beta parameter (preference strength)
        max_length=1024,  # Max sequence length for DPO
        max_prompt_length=512,  # Max prompt length
    ),
    
    # Disable sequence packing for DPO (recommended)
    sequence_packing=False,
)

# Training configuration
training_config = TrainingArguments(
    output_dir='outputs/dpo_training',
    per_device_train_batch_size=2,  # Smaller batch size for DPO
    learning_rate=5e-6,  # Lower learning rate for DPO
    num_train_epochs=3,
    warmup_ratio=0.1,
    optim="adamw_8bit",
)
```

### 3. Run Training

```python
from opensloth.scripts.opensloth_sft_trainer import run_mp_training, train_on_single_gpu

# Single GPU
train_on_single_gpu(
    gpu=0,
    opensloth_config=opensloth_config,
    hf_train_args=training_config,
)

# Multi-GPU
run_mp_training(
    gpus=opensloth_config.devices,
    opensloth_config=opensloth_config,
    training_config=training_config,
)
```

## Key Differences from SFT

### Dataset Format
- **SFT**: Requires `input_ids` and `labels` columns
- **DPO**: Requires `prompt`, `chosen`, and `rejected` columns

### Training Parameters
- **Learning Rate**: DPO typically uses lower learning rates (5e-6 vs 2e-4)
- **Batch Size**: DPO often needs smaller batch sizes due to memory requirements
- **Sequence Packing**: Recommended to disable for DPO

### Model Requirements
- DPO typically starts from an SFT-trained model, not a base model
- The SFT model serves as both the policy and reference model

## Configuration Options

### DPOArgs Parameters

```python
DPOArgs(
    beta=0.1,           # DPO beta parameter (0.1-0.5 typical range)
    max_length=1024,    # Maximum sequence length for DPO training
    max_prompt_length=512,  # Maximum prompt length
)
```

- **beta**: Controls the strength of the preference optimization. Higher values make the model more strongly prefer chosen over rejected responses.
- **max_length**: Total sequence length for preference pairs
- **max_prompt_length**: Maximum length of the prompt portion

## Example Scripts

See these example scripts:
- `train_scripts/train_dpo_example.py` - Complete DPO training example
- `prepare_dataset/prepare_dpo_dataset.py` - Dataset preparation utilities

## Troubleshooting

### Common Issues

1. **ImportError: PatchDPOTrainer not found**
   - Ensure you have the latest version of Unsloth installed
   - Run: `pip install --upgrade unsloth`

2. **Dataset validation errors**
   - Ensure your dataset has `prompt`, `chosen`, `rejected` columns
   - Check that all entries have these required fields

3. **Memory issues**
   - Reduce `per_device_train_batch_size`
   - Use gradient accumulation to maintain effective batch size
   - Enable 4-bit quantization: `load_in_4bit=True`

4. **Training divergence**
   - Lower the learning rate (try 1e-6 or 5e-7)
   - Reduce the DPO beta parameter
   - Ensure your preference data is high quality

## Future Training Types

The architecture is designed to easily support additional training methods:

- **KTO (Kahneman-Tversky Optimization)** - Planned
- **ORPO (Odds Ratio Preference Optimization)** - Planned  
- **GRPO (Group Relative Policy Optimization)** - Planned

These will follow the same pattern:
1. Add configuration class (e.g., `KTOArgs`)
2. Update `training_type` enum
3. Implement trainer factory method
4. Add dataset validation

## Technical Details

### Architecture Changes

The codebase now uses a factory pattern for trainer creation:

```python
# trainer_factory.py
def create_trainer_by_type(
    model, tokenizer, train_dataset, 
    opensloth_config, hf_train_args
):
    # Creates appropriate trainer based on training_type
```

This replaces the hardcoded SFTTrainer instantiation and allows for:
- Easy addition of new training types
- Type-specific dataset validation
- Flexible configuration per training method

### Backward Compatibility

All existing SFT training scripts will continue to work without modification:
- Default `training_type="sft"`
- Existing configuration fields preserved
- SFT behavior unchanged when `training_type` is not specified
