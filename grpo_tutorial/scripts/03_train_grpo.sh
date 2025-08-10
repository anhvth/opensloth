#!/bin/bash

# OpenSloth GRPO Tutorial - Step 4: Run GRPO Training
# This script runs Group Relative Policy Optimization to improve reasoning

set -e  # Exit on any error

echo "🚀 Starting GRPO Training"
echo "========================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/03_train_grpo.sh"
    exit 1
fi

# Check if GRPO dataset exists
if [ ! -d "data/grpo_dapo_prepared" ]; then
    echo "❌ Error: GRPO dataset not found at data/grpo_dapo_prepared"
    echo "Please run: bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh"
    exit 1
fi

# Check if SFT model exists
if [ ! -d "outputs/sft_qwen_reasoning_model" ]; then
    echo "❌ Error: SFT model not found at outputs/sft_qwen_reasoning_model"
    echo "Please run: bash grpo_tutorial/scripts/01_train_sft.sh"
    exit 1
fi

echo "📊 Step 1: Verifying prerequisites..."
echo "✅ GRPO dataset found at data/grpo_dapo_prepared"
echo "✅ SFT model found at outputs/sft_qwen_reasoning_model"

# Create output directory
mkdir -p outputs/grpo_final_model

# Step 2: Run GRPO training
echo "🎯 Step 2: Starting GRPO training..."
echo "Goal: Improving mathematical reasoning quality using preference optimization"

# Generate a unique run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="grpo_tutorial_${TIMESTAMP}"

# Run os-grpo with the SFT model as the base
echo "🔄 Running os-grpo training..."
echo "🔗 Using SFT model as starting point for preference optimization"
rm -rf outputs/grpo_final_model
os-grpo data/grpo_dapo_prepared \
    --model "outputs/sft_qwen_reasoning_model" \
    --output outputs/grpo_final_model \
    --task math \
    --epochs 10 \
    --batch-size 128 \
    --grad-accum 1 \
    --lr 5e-5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --beta 0.1 \
    --group-size 8 \
    --max-new 2048 \
    --max-prompt-len 1024 \
    --max-seq-length 4096 \
    --temperature 1.0 \
    --top-p 0.9 \
    --logging-steps 10 \
    --save-total-limit 2

# Verify the training completed successfully
if [ ! -d "outputs/grpo_final_model" ] || [ -z "$(ls -A outputs/grpo_final_model)" ]; then
    echo "❌ Error: GRPO training failed - output directory is empty"
    exit 1
fi

# Check for adapter files
if [ ! -f "outputs/grpo_final_model/adapter_model.safetensors" ] && [ ! -f "outputs/grpo_final_model/pytorch_model.bin" ]; then
    echo "❌ Error: GRPO training failed - no adapter files found"
    exit 1
fi

echo "🎉 GRPO training completed successfully!"
echo ""
echo "📊 Training Summary:"
echo "  - Base model: outputs/sft_qwen_reasoning_model (SFT checkpoint)"
echo "  - Training type: Group Relative Policy Optimization (GRPO)"
echo "  - Purpose: Improving mathematical reasoning accuracy"
echo "  - Epochs: 2"
echo "  - Beta: 0.1"
echo "  - Reward functions preset: math (math_format, math_answer, math_number, demo_reward)"
echo "  - Output: outputs/grpo_final_model/"
echo "  - Run name: ${RUN_NAME}"
echo ""
echo "🎯 What the model learned:"
echo "  - Improved mathematical reasoning accuracy"
echo "  - Better problem-solving strategies"
echo "  - Enhanced correctness in solutions"
echo "  - Maintained formatting from SFT stage"
echo ""
echo "✅ Tutorial Complete!"
echo "===================="
echo ""
echo "🎊 Congratulations! You have successfully created a math reasoning model using:"
echo "  1. ✅ SFT for format learning"
echo "  2. ✅ GRPO for reasoning improvement"
echo ""
echo "📁 Your final model is ready at: outputs/grpo_final_model/"
echo ""
echo "🚀 Next Steps:"
echo "  - Use the model for inference (see README.md for examples)"
echo "  - Experiment with different hyperparameters"
echo "  - Try the workflow on your own datasets"
echo "  - Deploy the model in production applications"
echo ""
echo "📖 For inference examples and more details, see:"
echo "   grpo_tutorial/README.md"
