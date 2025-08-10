#!/bin/bash

# OpenSloth GRPO Tutorial - Step 2: End-to-End GRPO Training
# This script prepares the GRPO dataset and runs preference optimization in one command

set -e  # Exit on any error

echo "🚀 Starting End-to-End GRPO Training"
echo "===================================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/02_grpo_end_to_end.sh"
    exit 1
fi

# Check if SFT training was completed
if [ ! -d "outputs/sft_qwen_reasoning_model" ]; then
    echo "❌ Error: SFT model not found at outputs/sft_qwen_reasoning_model"
    echo "Please run: bash grpo_tutorial/scripts/01_sft_end_to_end.sh"
    exit 1
fi

echo "📊 Step 1: Verifying SFT model..."
echo "✅ SFT model found at outputs/sft_qwen_reasoning_model"

# Step 2: Run end-to-end GRPO training with os-train
echo "🎯 Step 2: Starting end-to-end GRPO training..."
echo "Goal: Improving mathematical reasoning quality using preference optimization"

# Detect GPU count
GPU_COUNT=${GPU_COUNT:-$(python -c 'import torch,os;print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}
echo "🔍 Detected GPUs: $GPU_COUNT"

# Set sample count for tutorial (can be overridden)
GRPO_SAMPLES=${GRPO_SAMPLES:-4000}

# Create output directory
mkdir -p outputs/grpo_final_model

echo "🔄 Running os-train grpo (end-to-end: data prep + training)..."
echo "📥 Dataset: open-r1/DAPO-Math-17k-Processed (HuggingFace)"
echo "🔗 Using SFT model as starting point for preference optimization"

# Use tmux for multi-GPU if more than 1 GPU is available
TMUX_FLAG=""
if [ "$GPU_COUNT" -gt 1 ]; then
    TMUX_FLAG="--tmux"
    echo "🖥️  Multi-GPU detected: Using tmux for training"
fi

# Run os-train grpo with the SFT model as the base
os-train grpo open-r1/DAPO-Math-17k-Processed \
    --model "outputs/sft_qwen_reasoning_model" \
    --output outputs/grpo_final_model \
    --samples ${GRPO_SAMPLES} \
    --gpus ${GPU_COUNT} \
    --max-seq-length 4096 \
    --epochs 1 \
    --batch-size 64 \
    --lr 5e-5 
    ${TMUX_FLAG}

# Verify the training completed successfully
if [ ! -d "outputs/grpo_final_model" ] || [ -z "$(ls -A outputs/grpo_final_model)" ]; then
    echo "❌ Error: GRPO training failed - output directory is empty"
    exit 1
fi

# Check for adapter files or model files
if [ ! -f "outputs/grpo_final_model/adapter_model.safetensors" ] && 
   [ ! -f "outputs/grpo_final_model/pytorch_model.bin" ] &&
   [ ! -f "outputs/grpo_final_model/model.safetensors" ]; then
    echo "❌ Error: GRPO training failed - no model files found"
    exit 1
fi

echo "🎉 End-to-end GRPO training completed successfully!"
echo ""
echo "📊 Training Summary:"
echo "  - Base model: outputs/sft_qwen_reasoning_model (SFT checkpoint)"
echo "  - Training type: Group Relative Policy Optimization (GRPO)"
echo "  - Purpose: Improving mathematical reasoning accuracy"
echo "  - Dataset: open-r1/DAPO-Math-17k-Processed"
echo "  - Dataset samples: ${GRPO_SAMPLES}"
echo "  - Epochs: 1"
echo "  - GPUs: ${GPU_COUNT}"
echo "  - Output: outputs/grpo_final_model/"
echo ""
echo "🎯 What the model learned:"
echo "  - Improved mathematical reasoning accuracy"
echo "  - Better problem-solving strategies"
echo "  - Enhanced correctness in solutions"
echo "  - Maintained formatting from SFT stage"
echo ""
echo "💡 Note: os-train automatically:"
echo "  - Downloaded and processed the GRPO dataset"
echo "  - Tokenized and sharded data across ${GPU_COUNT} GPU(s)"
echo "  - Ran GRPO training with consistent GPU allocation"
echo "  - Applied built-in math reward functions"
echo ""
echo "✅ Tutorial Complete!"
echo "===================="
echo ""
echo "🎊 Congratulations! You have successfully created a math reasoning model using:"
echo "  1. ✅ End-to-end SFT for format learning"
echo "  2. ✅ End-to-end GRPO for reasoning improvement"
echo ""
echo "📁 Your final model is ready at: outputs/grpo_final_model/"
echo ""
echo "🚀 Next Steps:"
echo "  - Test the model: python grpo_tutorial/scripts/test_final_model.py"
echo "  - Use the model for inference (see README.md for examples)"
echo "  - Experiment with different hyperparameters"
echo "  - Try the workflow on your own datasets"
echo "  - Deploy the model in production applications"
echo ""
echo "📖 For inference examples and more details, see:"
echo "   grpo_tutorial/README.md"
