#!/bin/bash

# OpenSloth GRPO Tutorial - Step 2: Run SFT Training
# This script runs supervised fine-tuning to teach the model reasoning format

set -e  # Exit on any error

echo "🚀 Starting SFT Training"
echo "======================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/01_train_sft.sh"
    exit 1
fi

# Check if SFT dataset exists
if [ ! -d "data/sft_openmath_prepared" ]; then
    echo "❌ Error: SFT dataset not found at data/sft_openmath_prepared"
    echo "Please run: bash grpo_tutorial/scripts/00_prepare_sft_dataset.sh"
    exit 1
fi

echo "📊 Step 1: Verifying SFT dataset..."
echo "✅ SFT dataset found at data/sft_openmath_prepared"

# Create output directory
mkdir -p outputs/sft_qwen_reasoning_model

# Step 2: Run SFT training
echo "🎯 Step 2: Starting SFT training..."
echo "Goal: Teaching the model reasoning format (not mathematical correctness yet)"

# Run os-sft with appropriate parameters for format learning
echo "🔄 Running os-sft training..."
GPU_COUNT=${GPU_COUNT:-$(python -c 'import torch,os;print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}
echo "🔍 Detected GPUs: $GPU_COUNT"

# For multi-GPU, the dataset was pre-sharded; os-sft derives devices from dataset_config.json
os-sft \
    data/sft_openmath_prepared \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --output outputs/sft_qwen_reasoning_model \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 4 \
    --lr 2e-4 \
    --lora-r 16 \
    --lora-alpha 32

# Verify the training completed successfully
if [ ! -d "outputs/sft_qwen_reasoning_model" ] || [ -z "$(ls -A outputs/sft_qwen_reasoning_model)" ]; then
    echo "❌ Error: SFT training failed - output directory is empty"
    exit 1
fi

# Check for adapter files
if [ ! -f "outputs/sft_qwen_reasoning_model/adapter_model.safetensors" ] && [ ! -f "outputs/sft_qwen_reasoning_model/pytorch_model.bin" ]; then
    echo "❌ Error: SFT training failed - no adapter files found"
    exit 1
fi

echo "✅ SFT training completed successfully!"
echo ""
echo "📊 Training Summary:"
echo "  - Model: Qwen/Qwen2.5-3B-Instruct"
echo "  - Training type: Supervised Fine-Tuning (SFT)"
echo "  - Purpose: Learning reasoning format"
echo "  - Epochs: 3"
echo "  - LoRA rank: 16"
echo "  - Output: outputs/sft_qwen_reasoning_model/"
echo ""
echo "🎯 What the model learned:"
echo "  - How to structure responses with <start_working_out> tags"
echo "  - How to place solutions in <SOLUTION> tags"
echo "  - Basic reasoning format conventions"
echo ""
echo "🔜 Next step: Run bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh"
