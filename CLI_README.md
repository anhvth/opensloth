# OpenSloth CLI Quick Reference

## Installation

```bash
pip install -e .
```

## Basic Usage

### 1. Prepare Dataset
```bash
opensloth-dataset --model unsloth/Qwen2.5-0.5B-Instruct --dataset mlabonne/FineTome-100k --samples 1000
```

### 2. Train Model
```bash
opensloth-train --dataset data/qwen_finetome_1000_0808 --model unsloth/Qwen2.5-0.5B-Instruct
```

## Quick Examples

### Different Data Sources
```bash
# HuggingFace dataset
opensloth-dataset --model qwen --dataset mlabonne/FineTome-100k --samples 5000

# Local JSON/JSONL file  
opensloth-dataset --model gemma --dataset ./conversations.jsonl --family gemma

# Custom chat format
opensloth-dataset --model custom/model --dataset data.json \
  --instruction-part "<USER>" --response-part "<ASSISTANT>"
```

### Training Scenarios
```bash
# Quick test (50 steps)
opensloth-train --dataset data/my_dataset --preset quick_test

# Multi-GPU training
opensloth-train --dataset data/my_dataset --model qwen --gpus 0,1,2,3

# Large model optimization
opensloth-train --dataset data/my_dataset --preset large_model

# Full fine-tuning
opensloth-train --dataset data/my_dataset --full-finetune
```

### Model Families
```bash
# Qwen (auto-detected)
opensloth-dataset --model unsloth/Qwen2.5-7B-Instruct --dataset finetome

# Gemma 
opensloth-dataset --model unsloth/gemma-2-2b-it --dataset openhermes --family gemma

# Llama
opensloth-dataset --model unsloth/Llama-3.2-1B-Instruct --dataset ultrachat --family llama
```

## Help Commands

```bash
opensloth-dataset --help              # Full dataset help
opensloth-train --help                # Full training help
opensloth-dataset --list-presets      # Available dataset presets
opensloth-train --list-presets        # Available training presets
opensloth-train --list-datasets       # Available processed datasets
```

## Common Options

### Dataset Preparation
- `--model`: Model for tokenizer (e.g., `unsloth/Qwen2.5-0.5B-Instruct`)
- `--dataset`: HF dataset or local file (e.g., `mlabonne/FineTome-100k`)
- `--samples`: Number of samples (`-1` for all, `1000` for testing)
- `--output`: Output directory (auto-generated if not specified)
- `--debug N`: Create debug visualization with N samples

### Training
- `--dataset`: Path to processed dataset
- `--model`: Model to fine-tune  
- `--gpus`: GPU indices (e.g., `0,1,2,3`)
- `--preset`: Use predefined configuration
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size per GPU
- `--learning-rate`: Learning rate

## Presets

### Dataset Presets
Use `--preset <name>` or create custom presets with `--save-preset <name>`

### Training Presets
- `quick_test`: 50 steps for testing
- `small_model`: Optimized for < 3B models  
- `large_model`: Optimized for > 7B models
- `memory_efficient`: Minimal VRAM usage
- `full_finetune`: Full parameter training

## Workflow Example

```bash
# 1. Prepare dataset
opensloth-dataset \
  --model unsloth/Qwen2.5-0.5B-Instruct \
  --dataset mlabonne/FineTome-100k \
  --samples 2000

# 2. Train model
opensloth-train \
  --dataset data/qwen_finetome_2000_0808 \
  --preset small_model \
  --epochs 3

# 3. Use trained model
# Model saved in outputs/train_YYYYMMDD_HHMMSS/
```

For detailed documentation, see [docs/CLI_USAGE.md](docs/CLI_USAGE.md)
