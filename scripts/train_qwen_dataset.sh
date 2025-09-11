#!/usr/bin/env fish
# Qwen Training Script
# Pre-configured for Qwen models with optimal settings

set TRAIN_SCRIPT "opensloth-sft-train"

# Default configuration for Qwen training
set DEFAULT_MODEL "hf-models/unsloth/Qwen3-14B-bnb-4bit"
set DEFAULT_INPUT "./data/x1.jsonl"
set DEFAULT_DEVICES "0,1,2,3"
set DEFAULT_OUTPUT_DIR "./outputs/test_unified/"

function usage
    echo "Usage: train_qwen_dataset.sh [OPTIONS]"
    echo ""
    echo "Train a Qwen model using prepared dataset"
    echo ""
    echo "Optional:"
    echo "  --model MODEL              Model path (default: $DEFAULT_MODEL)"
    echo "  --input INPUT              Input dataset path (default: $DEFAULT_INPUT)"
    echo "  --devices DEVICES          GPU devices (default: $DEFAULT_DEVICES)"
    echo "  --output-dir OUTPUT_DIR    Output directory (default: $DEFAULT_OUTPUT_DIR)"
    echo "  --hf_token TOKEN           Hugging Face token for gated models"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with defaults"
    echo "  ./train_qwen_dataset.sh"
    echo ""
    echo "  # Custom model and input"
    echo "  ./train_qwen_dataset.sh --model unsloth/Qwen2.5-1.5B-Instruct --input ./data/my_dataset.jsonl --output-dir ./outputs/custom_train"
    echo ""
    echo "  # Single GPU training"
    echo "  ./train_qwen_dataset.sh --devices 0 --output-dir ./outputs/single_gpu"
end

# Parse arguments
set -l model $DEFAULT_MODEL
set -l input $DEFAULT_INPUT
set -l devices $DEFAULT_DEVICES
set -l output_dir $DEFAULT_OUTPUT_DIR
set -l hf_token ""

while set -q argv[1]
    switch $argv[1]
        case --model
            set model $argv[2]
            set -e argv[1 2]
        case --input
            set input $argv[2]
            set -e argv[1 2]
        case --devices
            set devices $argv[2]
            set -e argv[1 2]
        case --output-dir
            set output_dir $argv[2]
            set -e argv[1 2]
        case --hf_token
            set hf_token $argv[2]
            set -e argv[1 2]
        case -h --help
            usage
            exit 0
        case '*'
            echo "Unknown option: $argv[1]"
            usage
            exit 1
    end
end

# Build command
set cmd uv run $TRAIN_SCRIPT \
    --model $model \
    --input $input \
    --devices $devices \
    --output-dir $output_dir

if test -n "$hf_token"
    set cmd $cmd --hf_token $hf_token
end

# Print command for debugging
echo "Running command:"
echo $cmd
echo ""

# Execute command
eval $cmd