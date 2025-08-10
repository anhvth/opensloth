#!/bin/bash

# OpenSloth GRPO Tutorial - Step 1: End-to-End SFT Training
# This script prepares the SFT dataset and runs supervised fine-tuning in one command

set -e  # Exit on any error

echo "🚀 Starting End-to-End SFT Training"
echo "==================================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/01_sft_end_to_end.sh"
    exit 1
fi

echo "📝 Step 1: Preparing raw SFT data..."

# Step 1: Run the Python data preparation script to create raw JSONL
python grpo_tutorial/scripts/prepare_sft_data.py

# Check if the raw data was created successfully
if [ ! -f "grpo_tutorial/data/sft_openmath_raw.jsonl" ]; then
    echo "❌ Error: Failed to create raw SFT data"
    exit 1
fi

echo "✅ Raw SFT data prepared successfully"

# Step 2: Run end-to-end SFT training with os-train
echo "🎯 Step 2: Starting end-to-end SFT training..."
echo "Goal: Teaching the model reasoning format (not mathematical correctness yet)"

# Detect GPU count
GPU_COUNT=${GPU_COUNT:-$(python -c 'import torch,os;print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}
echo "🔍 Detected GPUs: $GPU_COUNT"

# Set sample count for tutorial (can be overridden)
SFT_SAMPLES=${SFT_SAMPLES:-4000}

# Create output directory
mkdir -p outputs/sft_qwen_reasoning_model

echo "🔄 Running os-train sft (end-to-end: data prep + training)..."
os-train sft grpo_tutorial/data/sft_openmath_raw.jsonl \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --output outputs/sft_qwen_reasoning_model \
    --samples ${SFT_SAMPLES} \
    --gpus ${GPU_COUNT} \
    --max-seq-length 2048 \
    --epochs 3 \
    --batch-size 2 \
    --lr 2e-4

# Verify the training completed successfully
if [ ! -d "outputs/sft_qwen_reasoning_model" ] || [ -z "$(ls -A outputs/sft_qwen_reasoning_model)" ]; then
    echo "❌ Error: SFT training failed - output directory is empty"
    exit 1
fi

# Check for adapter files or model files
if [ ! -f "outputs/sft_qwen_reasoning_model/adapter_model.safetensors" ] && 
   [ ! -f "outputs/sft_qwen_reasoning_model/pytorch_model.bin" ] &&
   [ ! -f "outputs/sft_qwen_reasoning_model/model.safetensors" ]; then
    echo "❌ Error: SFT training failed - no model files found"
    exit 1
fi

echo "✅ End-to-end SFT training completed successfully!"
echo ""
echo "📊 Training Summary:"
echo "  - Model: Qwen/Qwen2.5-3B-Instruct"
echo "  - Training type: Supervised Fine-Tuning (SFT)"
echo "  - Purpose: Learning reasoning format"
echo "  - Dataset samples: ${SFT_SAMPLES}"
echo "  - Epochs: 3"
echo "  - GPUs: ${GPU_COUNT}"
echo "  - Output: outputs/sft_qwen_reasoning_model/"
echo ""
echo "🎯 What the model learned:"
echo "  - How to structure responses with <start_working_out> tags"
echo "  - How to place solutions in <SOLUTION> tags"
echo "  - Basic reasoning format conventions"
echo ""
echo "💡 Note: os-train automatically:"
echo "  - Prepared and tokenized the dataset"
echo "  - Sharded data across ${GPU_COUNT} GPU(s)"
echo "  - Ran training with consistent GPU allocation"
echo ""
echo "🔜 Next step: Run bash grpo_tutorial/scripts/02_grpo_end_to_end.sh"
