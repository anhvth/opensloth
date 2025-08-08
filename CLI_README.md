# OpenSloth CLI Quick Reference

🎉 **New & Improved!** OpenSloth now features modern Typer-based CLI tools with beautiful output, smart defaults, and intuitive workflows.

## Installation

```bash
pip install -e .
```

## 🚀 Quick Start

### 1. Prepare Dataset
```bash
# Quick start with preset
opensloth-dataset prepare --preset qwen_chat --dataset mlabonne/FineTome-100k --samples 1000

# Or with defaults
opensloth-dataset prepare
```

### 2. Train Model  
```bash
# Quick training (interactive dataset selection)
opensloth-train train

# Or specify dataset
opensloth-train train --dataset data/qwen_finetome_100k_n1000_0808
```

## 📋 Available Commands

### Dataset Preparation: `opensloth-dataset`
```bash
# List presets and help
opensloth-dataset list-presets
opensloth-dataset prepare --help

# Quick preset-based preparation
opensloth-dataset prepare --preset qwen_chat --dataset my/dataset
opensloth-dataset prepare --preset llama_chat --dataset mlabonne/FineTome-100k --samples 5000

# Custom configuration
opensloth-dataset prepare \
  --model unsloth/Qwen2.5-7B-Instruct \
  --dataset mlabonne/FineTome-100k \
  --samples 5000 \
  --workers 8 \
  --target-only

# List and inspect datasets
opensloth-dataset list-datasets
opensloth-dataset info data/my_dataset
```

### Model Training: `opensloth-train`
```bash
# List presets and help  
opensloth-train list-presets
opensloth-train train --help

# Quick training with presets
opensloth-train train --dataset data/my_dataset --preset small_model
opensloth-train train --dataset data/my_dataset --preset large_model --gpus 0,1,2,3

# Multi-GPU training
opensloth-train train \
  --dataset data/my_dataset \
  --model unsloth/Qwen2.5-7B-Instruct \
  --gpus 0,1,2,3 \
  --epochs 3 \
  --batch-size 2

# Full fine-tuning
opensloth-train train \
  --dataset data/my_dataset \
  --full-finetune \
  --gpus 0,1,2,3 \
  --preset full_finetune

# Memory-efficient training
opensloth-train train --dataset data/my_dataset --preset memory_efficient
```

## 🎯 Presets

### Dataset Presets
- `qwen_chat` - Qwen models with chat templates  
- `llama_chat` - Llama models with chat templates
- `gemma_chat` - Gemma models with chat templates
- `mistral_chat` - Mistral models with chat templates

### Training Presets  
- `quick_test` - Fast validation run (50 steps)
- `small_model` - Optimized for <3B parameters
- `large_model` - Optimized for >7B parameters
- `full_finetune` - Full parameter training
- `memory_efficient` - Lowest memory usage

## 💡 Advanced Examples

### Custom Data Sources
```bash
# HuggingFace dataset with specific split
opensloth-dataset prepare --dataset mlabonne/FineTome-100k --split train --samples 10000

# Local JSON/JSONL file
opensloth-dataset prepare --dataset ./my_conversations.jsonl --preset qwen_chat

# Custom chat format
opensloth-dataset prepare \
  --model custom/model \
  --dataset data.json \
  --instruction-part "<USER>" \
  --response-part "<ASSISTANT>" \
  --target-only

# Debug dataset (preview samples)
opensloth-dataset prepare --dataset my/dataset --debug 5 --dry-run
```

### Advanced Training Options
```bash
# Custom hyperparameters
opensloth-train train \
  --dataset data/my_dataset \
  --epochs 5 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --learning-rate 2e-4 \
  --warmup-steps 100 \
  --weight-decay 0.01

# LoRA customization  
opensloth-train train \
  --dataset data/my_dataset \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1

# Resume from checkpoint
opensloth-train train \
  --dataset data/my_dataset \
  --resume outputs/my_model/checkpoint-500

# Different optimizers and schedulers
opensloth-train train \
  --dataset data/my_dataset \
  --optimizer adamw_torch \
  --scheduler cosine \
  --report-to wandb
```

### Workflow Examples
```bash
# Complete workflow
opensloth-dataset prepare --preset qwen_chat --dataset mlabonne/FineTome-100k --samples 5000
opensloth-train train --dataset data/qwen_finetome_100k_n5000_0808 --preset large_model --gpus 0,1

# Quick experimentation
opensloth-dataset prepare --preset qwen_chat --dataset small/dataset --samples 100  
opensloth-train train --dataset data/qwen_small_dataset_n100_0808 --preset quick_test

# Production training
opensloth-dataset prepare --preset llama_chat --dataset large/dataset --samples 50000
opensloth-train train --dataset data/llama_large_dataset_n50000_0808 --preset large_model --gpus 0,1,2,3,4,5,6,7
```

## 🛠️ Features

### New in Modern CLI
- 🎨 **Rich output** - Beautiful, color-coded interface
- 🚀 **Fast startup** - Optimized loading with lazy imports  
- 🧠 **Smart defaults** - Reasonable presets for common scenarios
- 🔧 **Type safety** - Built on Python type hints
- 📋 **Presets** - Quick configurations for different model families
- 🔍 **Interactive** - Auto-discovery and guided workflows
- 📊 **Rich logging** - Real-time progress with structured output
- ✅ **Validation** - Configuration checking before execution

### Backwards Compatibility
- Legacy CLIs available as `opensloth-dataset-simple` and `opensloth-train-simple`
- Same underlying functionality with enhanced interface
- Non-breaking changes to existing workflows

## 🚦 Migration Guide

### From Legacy CLI
```bash
# Old way
opensloth-dataset --model qwen --dataset mlabonne/FineTome-100k --samples 1000

# New way (equivalent)
opensloth-dataset prepare --model unsloth/Qwen2.5-0.5B-Instruct --dataset mlabonne/FineTome-100k --samples 1000

# New way (with preset)  
opensloth-dataset prepare --preset qwen_chat --dataset mlabonne/FineTome-100k --samples 1000
```

### Benefits of Migration
- Better error messages and validation
- Rich, colorful output
- Preset-based quick start
- Auto-completion support
- Interactive features

## 📚 Tips & Tricks

- Use `--help` on any command for detailed help with examples
- Use `--dry-run` to preview configuration without execution  
- Use presets as starting points and override specific options
- Enable shell completion: `opensloth-train --install-completion`
- Use `list-datasets` and `list-presets` to explore options

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
