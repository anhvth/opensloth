#!/bin/bash

# OpenSloth GRPO Tutorial - Step 1: Prepare SFT Dataset
# This script prepares the SFT dataset for format pre-training

set -e  # Exit on any error

echo "🚀 Starting SFT Dataset Preparation"
echo "=================================="

# Ensure we're in the correct directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/opensloth" ]; then
    echo "❌ Error: This script must be run from the root of the opensloth project"
    echo "Please cd to the opensloth directory and run:"
    echo "bash grpo_tutorial/scripts/00_prepare_sft_dataset.sh"
    exit 1
fi

# Step 1: Run the Python data preparation script
echo "📝 Step 1: Running Python data preparation script..."
python grpo_tutorial/scripts/prepare_sft_data.py

# Check if the raw data was created successfully
if [ ! -f "grpo_tutorial/data/sft_openmath_raw.jsonl" ]; then
    echo "❌ Error: Failed to create raw SFT data"
    exit 1
fi

echo "✅ Raw SFT data prepared successfully"

# Step 2: Use os-data CLI to process the dataset
echo "📊 Step 2: Processing dataset with os-data CLI..."

# Create the output directory for processed data
mkdir -p data/sft_openmath_prepared

# Run os-data to tokenize and prepare the dataset
echo "🔄 Running os-data to tokenize and prepare SFT dataset..."
GPU_COUNT=${GPU_COUNT:-$(python -c 'import torch,os;print(torch.cuda.device_count() if torch.cuda.is_available() else 1)')}
echo "🔍 Detected GPUs: $GPU_COUNT (used for sharding)"

REPROCESS=0
if [ -d data/sft_openmath_prepared ]; then
  if [ ! -f data/sft_openmath_prepared/dataset_config.json ]; then
      echo "⚠️  Existing directory lacks dataset_config.json; will reprocess (force)."
      REPROCESS=1
  elif [ -z "$(ls -A data/sft_openmath_prepared 2>/dev/null | grep shard_)" ]; then
      echo "⚠️  Existing directory has no shard_* subdirectories; will reprocess (force)."
      REPROCESS=1
  else
      echo "✅ Existing processed dataset detected; skipping tokenization."
  fi
else
  REPROCESS=1
fi

if [ "$REPROCESS" = "1" ]; then
  SFT_SAMPLES=${SFT_SAMPLES:-4000}
  echo "🔁 Tokenizing SFT dataset (samples=${SFT_SAMPLES})..."
  rm -rf data/sft_openmath_prepared
  os-data \
        --model "Qwen/Qwen2.5-3B-Instruct" \
        grpo_tutorial/data/sft_openmath_raw.jsonl \
        --method sft \
        --output data/sft_openmath_prepared \
        --max-seq-length 2048 \
        --gpus $GPU_COUNT \
        --workers 4 \
        --samples ${SFT_SAMPLES} || { echo "❌ os-data failed"; exit 1; }
fi

# Verify the processed dataset was created
if [ ! -d "data/sft_openmath_prepared" ] || [ -z "$(ls -A data/sft_openmath_prepared)" ]; then
    echo "❌ Error: Failed to create processed SFT dataset"
    exit 1
fi

echo "✅ SFT dataset preparation completed successfully!"
echo ""
echo "📊 Dataset Summary:"
echo "  - Raw data: grpo_tutorial/data/sft_openmath_raw.jsonl"
echo "  - Processed data: data/sft_openmath_prepared/"
echo "  - Ready for SFT training!"
echo ""
echo "🔜 Next step: Run bash grpo_tutorial/scripts/01_train_sft.sh"
