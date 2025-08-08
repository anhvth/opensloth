# OpenSloth CLI Documentation

OpenSloth provides two powerful command-line interfaces for dataset preparation and model training:

- `opensloth-dataset`: Prepare datasets for fine-tuning
- `opensloth-train`: Train models with advanced features

## Quick Start

### 1. Prepare a Dataset

```bash
# Basic usage - prepare 1000 samples from FineTome
opensloth-dataset --model unsloth/Qwen2.5-0.5B-Instruct --dataset mlabonne/FineTome-100k --samples 1000

# Local JSON/JSONL file
opensloth-dataset --model unsloth/gemma-2-2b-it --dataset ./my_conversations.jsonl --family gemma

# Full dataset processing
opensloth-dataset --model unsloth/Qwen2.5-7B-Instruct --dataset mlabonne/FineTome-100k --samples -1
```

### 2. Train a Model

```bash
# Basic training
opensloth-train --dataset data/qwen_finetome_1000_0808 --model unsloth/Qwen2.5-0.5B-Instruct

# Multi-GPU training
opensloth-train --dataset data/my_dataset --model unsloth/Qwen2.5-7B-Instruct --gpus 0,1,2,3

# Use presets for common scenarios
opensloth-train --dataset data/my_dataset --preset large_model
```

## Dataset Preparation (`opensloth-dataset`)

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--model` | Model for tokenizer | `unsloth/Qwen2.5-0.5B-Instruct` |
| `--dataset` | Dataset source | `mlabonne/FineTome-100k` or `./data.jsonl` |
| `--samples` | Number of samples | `1000` (use `-1` for all) |
| `--family` | Model family | `qwen`, `gemma`, `llama` |
| `--output` | Output directory | `data/my_processed_dataset` |

### Dataset Sources

#### HuggingFace Datasets
```bash
# Popular instruction datasets
opensloth-dataset --model qwen --dataset mlabonne/FineTome-100k --samples 5000
opensloth-dataset --model llama --dataset HuggingFaceH4/ultrachat_200k --samples 10000
opensloth-dataset --model gemma --dataset teknium/OpenHermes-2.5 --samples 2000
```

#### Local Files
```bash
# JSON file with conversations
opensloth-dataset --model qwen --dataset ./conversations.json

# JSONL file (one conversation per line)
opensloth-dataset --model gemma --dataset ./chat_logs.jsonl

# Specify custom chat template markers
opensloth-dataset \
  --model custom/model \
  --dataset ./data.json \
  --instruction-part "<|user|>" \
  --response-part "<|assistant|>"
```

### Advanced Usage

#### Response-only Training (Recommended)
```bash
# Train only on assistant responses (default)
opensloth-dataset --model qwen --dataset finetome --train-on-target-only

# Train on all tokens
opensloth-dataset --model qwen --dataset finetome --train-on-all
```

#### Debug Mode
```bash
# Generate debug visualization
opensloth-dataset --model qwen --dataset finetome --samples 100 --debug 5
# Creates .log/dataloader_examples.html for inspection
```

#### Using Presets
```bash
# List available presets
opensloth-dataset --list-presets

# Use a preset and modify
opensloth-dataset --preset qwen_finetome --samples 5000

# Save current config as preset
opensloth-dataset --model qwen --dataset finetome --save-preset my_config
```

### Model Family Examples

#### Qwen Models
```bash
opensloth-dataset \
  --model unsloth/Qwen2.5-0.5B-Instruct \
  --dataset mlabonne/FineTome-100k \
  --chat-template qwen-2.5 \
  --instruction-part "<|im_start|>user\n" \
  --response-part "<|im_start|>assistant\n"
```

#### Gemma Models
```bash
opensloth-dataset \
  --model unsloth/gemma-2-2b-it \
  --dataset teknium/OpenHermes-2.5 \
  --chat-template gemma \
  --instruction-part "<start_of_turn>user\n" \
  --response-part "<start_of_turn>model\n"
```

#### Llama Models
```bash
opensloth-dataset \
  --model unsloth/Llama-3.2-1B-Instruct \
  --dataset HuggingFaceH4/ultrachat_200k \
  --chat-template llama-3.1 \
  --instruction-part "<|start_header_id|>user<|end_header_id|>\n\n" \
  --response-part "<|start_header_id|>assistant<|end_header_id|>\n\n"
```

## Model Training (`opensloth-train`)

### Common Options

| Option | Description | Example |
|--------|-------------|---------|
| `--dataset` | Processed dataset path | `data/qwen_finetome_1000_0808` |
| `--model` | Model to train | `unsloth/Qwen2.5-0.5B-Instruct` |
| `--gpus` | GPU indices | `0,1,2,3` |
| `--output` | Output directory | `outputs/my_model` |
| `--epochs` | Training epochs | `3` |
| `--batch-size` | Batch size per GPU | `2` |

### Training Presets

#### Quick Test
```bash
opensloth-train --dataset data/my_dataset --preset quick_test
# 50 steps, minimal configuration for testing
```

#### Small Models (< 3B parameters)
```bash
opensloth-train --dataset data/my_dataset --preset small_model
# Optimized for 0.5B-3B models: higher batch size, LoRA rank 16
```

#### Large Models (> 7B parameters)
```bash
opensloth-train --dataset data/my_dataset --preset large_model
# Optimized for 7B+ models: lower batch size, LoRA rank 8
```

#### Memory Efficient
```bash
opensloth-train --dataset data/my_dataset --preset memory_efficient
# Minimal memory usage: small batch size, low LoRA rank
```

#### Full Fine-tuning
```bash
opensloth-train --dataset data/my_dataset --preset full_finetune
# Full parameter training instead of LoRA
```

### Hardware Configuration

#### Single GPU
```bash
opensloth-train --dataset data/my_dataset --model qwen --gpus 0
```

#### Multi-GPU
```bash
# Use specific GPUs
opensloth-train --dataset data/my_dataset --model qwen --gpus 0,1,2,3

# Use all available GPUs
opensloth-train --dataset data/my_dataset --model qwen
```

### Training Parameters

#### Learning Rate and Schedule
```bash
opensloth-train \
  --dataset data/my_dataset \
  --learning-rate 2e-4 \
  --scheduler linear \
  --warmup-steps 100
```

#### Batch Size and Accumulation
```bash
opensloth-train \
  --dataset data/my_dataset \
  --batch-size 2 \
  --accumulation-steps 8
# Effective batch size = 2 * 8 * num_gpus
```

#### Training Duration
```bash
# Train for specific number of epochs
opensloth-train --dataset data/my_dataset --epochs 3

# Train for specific number of steps
opensloth-train --dataset data/my_dataset --max-steps 1000
```

### LoRA Configuration

#### Basic LoRA
```bash
opensloth-train \
  --dataset data/my_dataset \
  --lora-r 16 \
  --lora-alpha 32 \
  --lora-dropout 0.1
```

#### Advanced LoRA
```bash
opensloth-train \
  --dataset data/my_dataset \
  --lora-r 8 \
  --lora-alpha 16 \
  --target-modules "q_proj,k_proj,v_proj,o_proj" \
  --use-rslora
```

#### Full Fine-tuning
```bash
opensloth-train \
  --dataset data/my_dataset \
  --full-finetune \
  --learning-rate 5e-6 \
  --batch-size 1 \
  --accumulation-steps 32
```

### Quantization Options

```bash
# 4-bit quantization (default)
opensloth-train --dataset data/my_dataset --load-in-4bit

# 8-bit quantization
opensloth-train --dataset data/my_dataset --load-in-8bit

# No quantization (requires more VRAM)
opensloth-train --dataset data/my_dataset --no-quantization
```

### Monitoring and Logging

#### TensorBoard
```bash
opensloth-train --dataset data/my_dataset --report-to tensorboard
# View with: tensorboard --logdir outputs/
```

#### Weights & Biases
```bash
opensloth-train --dataset data/my_dataset --report-to wandb
```

#### Custom Logging
```bash
opensloth-train \
  --dataset data/my_dataset \
  --logging-steps 10 \
  --save-steps 500 \
  --save-total-limit 3
```

### Checkpointing

#### Resume Training
```bash
opensloth-train \
  --dataset data/my_dataset \
  --resume outputs/my_model/checkpoint-500
```

#### Continue LoRA Training
```bash
opensloth-train \
  --dataset data/my_dataset \
  --pretrained-lora outputs/previous_model/adapter_model
```

## Complete Workflow Examples

### Example 1: Quick Experimentation
```bash
# 1. Prepare small dataset for testing
opensloth-dataset \
  --model unsloth/Qwen2.5-0.5B-Instruct \
  --dataset mlabonne/FineTome-100k \
  --samples 500

# 2. Quick training run
opensloth-train \
  --dataset data/qwen_finetome_500_0808 \
  --preset quick_test
```

### Example 2: Production Training
```bash
# 1. Prepare full dataset
opensloth-dataset \
  --model unsloth/Qwen2.5-7B-Instruct \
  --dataset mlabonne/FineTome-100k \
  --samples 50000

# 2. Multi-GPU training with large model preset
opensloth-train \
  --dataset data/qwen_finetome_50000_0808 \
  --preset large_model \
  --gpus 0,1,2,3 \
  --epochs 2 \
  --output outputs/qwen_7b_finetome
```

### Example 3: Custom Local Dataset
```bash
# 1. Prepare custom dataset
opensloth-dataset \
  --model unsloth/gemma-2-2b-it \
  --dataset ./my_conversations.jsonl \
  --family gemma \
  --debug 5

# 2. Train with custom settings
opensloth-train \
  --dataset data/gemma_my_conversations_all_0808 \
  --model unsloth/gemma-2-2b-it \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --lora-r 32 \
  --epochs 5
```

## Tips and Best Practices

### Dataset Preparation
- Start with small samples (500-1000) for testing
- Use `--debug 5` to inspect data formatting
- Verify chat templates match your model
- Check that `instruction_part` and `response_part` are correct

### Training
- Start with presets, then customize
- Monitor training with TensorBoard
- Use sequence packing for efficiency (enabled by default)
- Save checkpoints regularly for long training runs

### Memory Management
- Use 4-bit quantization for most cases
- Reduce `max_seq_length` if running out of memory
- Lower `batch_size` and increase `accumulation_steps`
- Use `memory_efficient` preset for very limited VRAM

### Multi-GPU Training
- Ensure all GPUs have similar memory
- Total batch size = `batch_size` × `accumulation_steps` × `num_gpus`
- Use `--log-level debug` for detailed multi-GPU logs

## Troubleshooting

### Common Issues

#### "Dataset directory does not exist"
```bash
# Make sure to use the output from dataset preparation
opensloth-dataset --model qwen --dataset finetome --samples 1000
# Use the printed output path in training
opensloth-train --dataset data/qwen_finetome_1000_0808 --model qwen
```

#### "CUDA out of memory"
```bash
# Reduce memory usage
opensloth-train \
  --dataset data/my_dataset \
  --preset memory_efficient \
  --max-length 1024 \
  --batch-size 1
```

#### "Model family not detected"
```bash
# Explicitly specify model family
opensloth-dataset --model custom/model --dataset data.json --family qwen
```

### Getting Help

```bash
# Detailed help for dataset preparation
opensloth-dataset --help

# Detailed help for training
opensloth-train --help

# List available options
opensloth-dataset --list-presets
opensloth-train --list-presets
opensloth-train --list-datasets
```

## Integration with Existing Workflows

The CLI commands are designed to work alongside existing OpenSloth workflows:

- Generated configs are compatible with the Gradio interface
- Presets can be shared between CLI and GUI
- Output directories follow the same conventions
- All underlying APIs remain the same

For more information and updates, visit: https://github.com/anhvth/opensloth
