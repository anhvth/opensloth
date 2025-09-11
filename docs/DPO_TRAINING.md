# Training Types in OpenSloth

**Note: Currently, OpenSloth only supports SFT (Supervised Fine-Tuning). DPO and other training methods are not implemented.**

This document describes the current training capabilities and architecture for future extensibility.

## Current Support

OpenSloth currently supports only **SFT (Supervised Fine-Tuning)** training:

- Traditional fine-tuning with input-output pairs
- Requires `input_ids` and `labels` columns in dataset
- Uses sequence packing for efficiency
- Supports LoRA and full fine-tuning

## Future Training Types (Not Implemented)

The following training methods are planned but not currently supported:

- **DPO (Direct Preference Optimization)** - Training with preference pairs
- **KTO (Kahneman-Tversky Optimization)** - RL method
- **ORPO (Odds Ratio Preference Optimization)** - RL method

## Architecture for Extensibility

The codebase is designed with a factory pattern that could easily support additional training types when implemented:

```python
# trainer_factory.py
def create_sft_trainer(
    model, tokenizer, train_dataset,
    opensloth_config, hf_train_args
):
    # Creates SFT trainer (currently the only supported type)
```

### Adding New Training Types (Future)

When additional training types are implemented, the pattern would be:

1. Add configuration class (e.g., `DPOArgs`)
2. Implement trainer factory method
3. Add dataset validation for the new type
4. Update documentation

## Current SFT Configuration

```python
from opensloth.opensloth_config import (
    FastModelArgs, LoraArgs, OpenSlothConfig,
    TrainingArguments
)

# Configure SFT training (only supported type)
opensloth_config = OpenSlothConfig(
    data_cache_path='data/dataset_cache',
    devices=[0],

    # Model configuration
    fast_model_args=FastModelArgs(
        model_name='unsloth/Qwen2.5-0.5B-Instruct',
        max_seq_length=4096,
        load_in_4bit=True,
    ),

    # LoRA configuration
    lora_args=LoraArgs(
        r=8,
        lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj'],
        lora_dropout=0.0,
    ),

    # Sequence packing enabled for SFT
    sequence_packing=True,
)

# Training configuration
training_config = TrainingArguments(
    output_dir='outputs/sft_training',
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    optim="adamw_8bit",
)
```

## Dataset Requirements

**SFT Dataset Format:**

- Requires `input_ids` and `labels` columns
- Supports sequence packing for efficiency
- Can be prepared using `prepare_qwen_dataset.py`

## Technical Details

### Current Architecture

The trainer factory creates SFT trainers with appropriate configuration:

- Dataset validation for SFT requirements
- Sequence packing support
- Multi-GPU training coordination
- LoRA and full fine-tuning modes
