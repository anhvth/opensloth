# GRPO Tutorial Quick Start

## Overview

This tutorial demonstrates a two-stage training pipeline for creating a math reasoning model:

1. **SFT Stage**: Teaching format and structure
2. **GRPO Stage**: Improving reasoning quality

## File Structure

```
grpo_tutorial/
├── README.md                           # Complete tutorial documentation
├── data/                              # Generated datasets (created during execution)
└── scripts/
    ├── prepare_sft_data.py            # Python helper for SFT data formatting
    ├── 00_prepare_sft_dataset.sh      # Step 1: Prepare SFT dataset
    ├── 01_train_sft.sh               # Step 2: Run SFT training
    ├── 02_prepare_grpo_dataset.sh    # Step 3: Prepare GRPO dataset
    └── 03_train_grpo.sh              # Step 4: Run GRPO training
```

## Quick Run

```bash
# From the opensloth project root:
bash grpo_tutorial/scripts/00_prepare_sft_dataset.sh
bash grpo_tutorial/scripts/01_train_sft.sh
bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh
bash grpo_tutorial/scripts/03_train_grpo.sh
```

## Expected Outputs

- `data/sft_openmath_prepared/` - Tokenized SFT dataset
- `outputs/sft_qwen_reasoning_model/` - SFT LoRA adapter
- `data/grpo_dapo_prepared/` - Processed GRPO dataset  
- `outputs/grpo_final_model/` - Final reasoning model

## Training Pipeline

```
unsloth/OpenMathReasoning-mini → SFT Dataset → SFT Training → SFT LoRA
                                                                  ↓
open-r1/DAPO-Math-17k-Processed → GRPO Dataset → GRPO Training → Final Model
```

For detailed documentation, see `README.md`.
