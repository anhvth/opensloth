#!/bin/bash

# OpenSloth GRPO Tutorial - Step 3: Prepare GRPO Dataset
# This script prepares the GRPO dataset for preference optimization training

set -e  # Exit on any error

echo "🚀 Starting GRPO Dataset Preparation"
echo "===================================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh"
    exit 1
fi

# Check if SFT training was completed
if [ ! -d "outputs/sft_qwen_reasoning_model" ]; then
    echo "❌ Error: SFT model not found at outputs/sft_qwen_reasoning_model"
    echo "Please run: bash grpo_tutorial/scripts/01_train_sft.sh"
    exit 1
fi

echo "📊 Step 1: Verifying SFT model..."
echo "✅ SFT model found at outputs/sft_qwen_reasoning_model"

# Create output directory for GRPO dataset
mkdir -p data/grpo_dapo_prepared

# Step 2: Prepare GRPO dataset using os-data
echo "📥 Step 2: Preparing GRPO dataset from open-r1/DAPO-Math-17k-Processed..."
echo "This dataset contains mathematical problems for preference optimization"

# Run os-data with GRPO method to process the preference dataset
echo "🔄 Running os-data to prepare GRPO dataset..."
GPU_COUNT=${GPU_COUNT:-$(python -c 'import torch,os;print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}
echo "🔍 Detected GPUs: $GPU_COUNT (used for sharding)"

REPROCESS=0
if [ -d data/grpo_dapo_prepared ]; then
    if [ ! -f data/grpo_dapo_prepared/dataset_config.json ]; then
        echo "⚠️  Existing grpo_dapo_prepared lacks dataset_config.json; will reprocess."; REPROCESS=1
    elif [ -z "$(ls -A data/grpo_dapo_prepared 2>/dev/null | grep shard_)" ]; then
        echo "⚠️  Existing grpo_dapo_prepared has no shards; will reprocess."; REPROCESS=1
    else
        echo "✅ Existing GRPO dataset present; skipping tokenization."; REPROCESS=2
    fi
else
    REPROCESS=1
fi

if [ "$REPROCESS" = "1" ]; then
    GRPO_SAMPLES=${GRPO_SAMPLES:-4000}
    echo "🔁 Tokenizing GRPO dataset (samples=${GRPO_SAMPLES})..."
    rm -rf data/grpo_dapo_prepared
    os-data \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        open-r1/DAPO-Math-17k-Processed \
        --method grpo \
        --output data/grpo_dapo_prepared \
        --max-seq-length 2048 \
        --gpus $GPU_COUNT \
        --workers 4 \
        --samples ${GRPO_SAMPLES} || { echo "❌ os-data failed"; exit 1; }
fi

if [ ! -f data/grpo_dapo_prepared/dataset_config.json ]; then
    echo "❌ Error: Failed to prepare GRPO dataset"; exit 1; fi

echo "✅ GRPO dataset preparation completed successfully!"
echo ""
echo "📊 Dataset Summary:"
echo "  - Source: open-r1/DAPO-Math-17k-Processed (HuggingFace)"
echo "  - Processed data: data/grpo_dapo_prepared/"
echo "  - Method: GRPO (Group Relative Policy Optimization)"
echo "  - Max samples: 10,000 (for tutorial purposes)"
echo "  - Ready for GRPO training!"
echo ""
echo "🎯 Dataset purpose:"
echo "  - Contains mathematical problems for preference optimization"
echo "  - Will be used to improve reasoning quality beyond format"
echo "  - Works with OpenSloth's built-in reward functions"
echo ""
echo "🔜 Next step: Run bash grpo_tutorial/scripts/03_train_grpo.sh"
