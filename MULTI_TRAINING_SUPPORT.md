# OpenSloth Multi-Training Type Support

OpenSloth now supports multiple training methodologies beyond traditional SFT (Supervised Fine-Tuning), including DPO (Direct Preference Optimization) and planned support for other RL-based training methods.

## 🚀 Supported Training Types

### ✅ Currently Available
- **SFT (Supervised Fine-Tuning)** - Traditional fine-tuning with input-output pairs
- **DPO (Direct Preference Optimization)** - Training with preference pairs (chosen vs rejected responses)

### 🔄 Coming Soon
- **KTO (Kahneman-Tversky Optimization)** - Planned
- **ORPO (Odds Ratio Preference Optimization)** - Planned  
- **GRPO (Group Relative Policy Optimization)** - Planned

## 🔧 Quick Start

### SFT Training (Default)
```python
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

config = OpenSlothConfig(
    data_cache_path='data/sft_dataset',
    training_type="sft",  # Default
    # ... other configs
)
```

### DPO Training (New!)
```python
from opensloth.opensloth_config import OpenSlothConfig, DPOArgs

config = OpenSlothConfig(
    data_cache_path='data/dpo_dataset',
    training_type="dpo",  # Specify DPO
    dpo_args=DPOArgs(
        beta=0.1,
        max_length=1024,
        max_prompt_length=512,
    ),
    # ... other configs
)
```

## 📊 Key Differences

| Aspect | SFT | DPO |
|--------|-----|-----|
| **Dataset Format** | `input_ids`, `labels` | `prompt`, `chosen`, `rejected` |
| **Model Base** | Base/Instruct model | SFT-trained model |
| **Learning Rate** | 2e-4 (higher) | 5e-6 (lower) |
| **Batch Size** | 8+ (larger) | 2-4 (smaller) |
| **Sequence Packing** | ✅ Enabled | ❌ Disabled |
| **LoRA Rank** | 8-16 (smaller) | 32-64 (larger) |

## 📖 Documentation

- **[DPO Training Guide](docs/DPO_TRAINING.md)** - Comprehensive DPO documentation
- **[SFT vs DPO Comparison](examples/sft_vs_dpo_comparison.py)** - Side-by-side configuration examples

## 🛠️ Example Scripts

### Dataset Preparation
```bash
# Prepare DPO dataset
python prepare_dataset/prepare_dpo_dataset.py --output_dir data/dpo_cache

# Convert existing preference data
python prepare_dataset/prepare_dpo_dataset.py --input_file preferences.json --output_dir data/dpo_cache
```

### Training Scripts
```bash
# SFT training (existing)
python train_scripts/train_qwen.py

# DPO training (new)
python train_scripts/train_dpo_example.py
```

## 🧪 Testing

Validate the implementation:
```bash
python test_dpo_implementation.py
```

## 🏗️ Architecture Changes

The codebase now uses a **factory pattern** for trainer creation:

```python
# Before (hardcoded SFT)
trainer = SFTTrainer(model, dataset, args, tokenizer)

# After (flexible factory)
trainer = create_trainer_by_type(
    model, tokenizer, dataset, 
    opensloth_config, training_args
)
```

### Key Components

1. **`opensloth_config.py`** - Added `training_type`, `DPOArgs` configuration
2. **`trainer_factory.py`** - New factory for creating different trainer types
3. **`init_modules.py`** - Updated to use trainer factory instead of hardcoded SFT
4. **Backward Compatibility** - All existing SFT scripts work unchanged

## 🔄 Migration Guide

### Existing SFT Users
No changes needed! All existing configurations continue to work:
- Default `training_type="sft"`
- Existing configuration fields preserved
- Same training behavior

### New DPO Users
1. Set `training_type="dpo"`
2. Prepare DPO dataset with `prompt`, `chosen`, `rejected` columns
3. Configure `dpo_args` for DPO-specific parameters
4. Use lower learning rates and smaller batch sizes

## 🎯 Future Extensions

The architecture is designed for easy extension:

```python
# Adding new training types (example)
class KTOArgs(BaseModel):
    # KTO-specific parameters
    pass

def _create_kto_trainer(...):
    # KTO trainer implementation
    pass

# Just add to factory map
trainer_factory_map["kto"] = _create_kto_trainer
```

## 📞 Support

- **GitHub Issues** - For bugs and feature requests
- **Documentation** - See `docs/DPO_TRAINING.md` for detailed guides
- **Examples** - Check `train_scripts/` and `examples/` directories

---

**OpenSloth**: Making advanced LLM training accessible with multiple training methodologies! 🦥✨
