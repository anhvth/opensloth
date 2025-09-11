#!/usr/bin/env fish
# Qwen Dataset Preparation Script
# Pre-configured for Qwen models with optimal settings

set PREPARE_SCRIPT "opensloth-make-data"

# Default configuration for Qwen models
set DEFAULT_MODEL "unsloth/Qwen2.5-0.5B-Instruct"
set DEFAULT_CHAT_TEMPLATE "qwen-2.5"
set DEFAULT_DATASET "mlabonne/FineTome-100k"
set DEFAULT_SPLIT "train"
set DEFAULT_MAX_SEQ_LENGTH 16096
set DEFAULT_NUM_PROC 8
set DEFAULT_GPUS 1

function usage
    echo "Usage: gen_qwen_dataset.sh [OPTIONS] --output_dir OUTPUT_DIR"
    echo ""
    echo "Prepare dataset for Qwen model fine-tuning"
    echo ""
    echo "Required:"
    echo "  --output_dir OUTPUT_DIR    Output directory for processed data"
    echo ""
    echo "Optional:"
    echo "  --model MODEL              Model name (default: $DEFAULT_MODEL)"
    echo "  --dataset DATASET          Dataset name or path (default: $DEFAULT_DATASET)"
    echo "  --chat_template TEMPLATE   Chat template (default: $DEFAULT_CHAT_TEMPLATE)"
    echo "  --split SPLIT              Dataset split (default: $DEFAULT_SPLIT)"
    echo "  --num_samples N            Number of samples (-1 for all, default: -1)"
    echo "  --max_seq_length LENGTH    Maximum sequence length (default: $DEFAULT_MAX_SEQ_LENGTH)"
    echo "  --num_proc N               Number of workers (default: $DEFAULT_NUM_PROC)"
    echo "  --gpus N                   Number of GPU shards (default: $DEFAULT_GPUS)"
    echo "  --hf_token TOKEN           Hugging Face token for gated models"
    echo "  --train_on_target_only     Train only on assistant responses"
    echo "  --instruction_part MARKER  Instruction marker (required if train_on_target_only)"
    echo "  --response_part MARKER     Response marker (required if train_on_target_only)"
    echo ""
    echo "Examples:"
    echo "  # Basic usage with defaults"
    echo "  ./gen_qwen_dataset.sh --output_dir ./data/qwen_finetome"
    echo ""
    echo "  # Custom model and dataset"
    echo "  ./gen_qwen_dataset.sh --model unsloth/Qwen2.5-1.5B-Instruct --dataset my-dataset --output_dir ./data/custom"
    echo ""
    echo "  # Response-only training"
    echo "  ./gen_qwen_dataset.sh --train_on_target_only --instruction_part '<|im_start|>user' --response_part '<|im_start|>assistant' --output_dir ./data/qwen_response_only"
end

# Parse arguments
set -l output_dir ""
set -l model $DEFAULT_MODEL
set -l dataset $DEFAULT_DATASET
set -l chat_template $DEFAULT_CHAT_TEMPLATE
set -l split $DEFAULT_SPLIT
set -l num_samples -1
set -l max_seq_length $DEFAULT_MAX_SEQ_LENGTH
set -l num_proc $DEFAULT_NUM_PROC
set -l gpus $DEFAULT_GPUS
set -l hf_token ""
set -l train_on_target_only false
set -l instruction_part ""
set -l response_part ""

while set -q argv[1]
    switch $argv[1]
        case --output_dir
            set output_dir $argv[2]
            set -e argv[1 2]
        case --model
            set model $argv[2]
            set -e argv[1 2]
        case --dataset
            set dataset $argv[2]
            set -e argv[1 2]
        case --chat_template
            set chat_template $argv[2]
            set -e argv[1 2]
        case --split
            set split $argv[2]
            set -e argv[1 2]
        case --num_samples
            set num_samples $argv[2]
            set -e argv[1 2]
        case --max_seq_length
            set max_seq_length $argv[2]
            set -e argv[1 2]
        case --num_proc
            set num_proc $argv[2]
            set -e argv[1 2]
        case --gpus
            set gpus $argv[2]
            set -e argv[1 2]
        case --hf_token
            set hf_token $argv[2]
            set -e argv[1 2]
        case --train_on_target_only
            set train_on_target_only true
            set -e argv[1]
        case --instruction_part
            set instruction_part $argv[2]
            set -e argv[1 2]
        case --response_part
            set response_part $argv[2]
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

# Validate required arguments
if test -z "$output_dir"
    echo "Error: --output_dir is required"
    usage
    exit 1
end

# Set default markers for response-only training if enabled
if $train_on_target_only
    if test -z "$instruction_part"
        set instruction_part "<|im_start|>user\n"
    end
    if test -z "$response_part"
        set response_part "<|im_start|>assistant\n"
    end
end

# Build command
set cmd uv run $PREPARE_SCRIPT \
    --tokenizer_name $model \
    --chat_template $chat_template \
    --dataset_name $dataset \
    --split $split \
    --num_samples $num_samples \
    --max_seq_length $max_seq_length \
    --num_proc $num_proc \
    --gpus $gpus \
    --output_dir $output_dir

if test -n "$hf_token"
    set cmd $cmd --hf_token $hf_token
end

if $train_on_target_only
    set cmd $cmd --train_on_target_only \
        --instruction_part $instruction_part \
        --response_part $response_part
end

# Print command for debugging
echo "Running command:"
echo $cmd
echo ""

# Execute command
eval $cmd