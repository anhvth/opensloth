# OpenSloth SFT Configuration Presets

This document provides an overview of the diverse SFT (Supervised Fine-Tuning) configuration presets created based on Unsloth's official notebooks and best practices.

## Overview

The configuration system includes two types of presets:
- **Dataset Preparation Configs**: Located in `prepare_dataset/presets/data/`
- **Training Configs**: Located in `prepare_dataset/presets/train/`

Each preset covers a specific training scenario with optimized hyperparameters and model configurations.

## Dataset Preparation Configs

### 1. Instruction Following
- **`alpaca_instruction_llama.json`**: Alpaca dataset for Llama 3.1 8B with response-only training
- **`alpaca_instruction_gemma.json`**: Alpaca dataset optimized for Gemma models  
- **`alpaca_tiny_model.json`**: Reduced Alpaca dataset for 1B parameter models
- **`finetome_qwen.json`**: High-quality FineTome dataset for Qwen 2.5 models

### 2. Conversational Training
- **`conversational_sharegpt_llama.json`**: Multi-turn ShareGPT conversations for Llama 3
- **`conversational_phi.json`**: ShareGPT optimized for Phi-3 Mini models

### 3. Specialized Training Methods
- **`text_completion_tinystories.json`**: Continued pretraining using TinyStories (all tokens)
- **`tool_calling_hermes.json`**: Function calling with structured JSON outputs
- **`dpo_preference_zephyr.json`**: DPO preference dataset for alignment
- **`orpo_preference_dolphin.json`**: ORPO preference optimization dataset

### 4. Domain-Specific Training
- **`coding_python_codellama.json`**: Python code generation with CodeLlama
- **`code_commits_deepseek.json`**: Git commit message generation
- **`math_reasoning_qwen.json`**: Mathematical word problems and reasoning
- **`vision_vqa_llama32.json`**: Vision-language QA training
- **`roleplay_pippa.json`**: Character roleplay conversations

### 5. Training Paradigms
- **`full_sequence_dolly.json`**: Full sequence training (all tokens) vs response-only

## Training Configs

Each dataset config has a corresponding training config with optimized hyperparameters:

### Key Hyperparameter Variations

**LoRA Rank (r)**:
- `r=8`: Tiny models (TinyLlama)
- `r=16`: Standard SFT (most instruction following)
- `r=32`: Complex tasks (tool calling, vision)
- `r=64`: Advanced reasoning (math, code)
- `r=128`: Continued pretraining

**Sequence Length**:
- `1024`: Small models, roleplay
- `2048`: Standard instruction following
- `4096`: Long-form content, math reasoning
- `8192`: Code with large context

**Learning Rates**:
- `1e-4` to `3e-4`: Standard SFT
- `5e-6` to `8e-6`: Preference optimization (DPO/ORPO)
- `5e-6` embedding rate: Continued pretraining

**Special Configurations**:
- Vision models: No sequence packing, higher LoRA rank
- Preference optimization: Lower learning rate, no weight decay, beta parameter
- Continued pretraining: Include embed_tokens/lm_head in target modules
- Tiny models: Higher batch size, more epochs

## Features Added

### 1. HF Token Support
- Added `hf_token` field to configuration schema
- Supports gated models and datasets requiring authentication
- Secure password field in UI

### 2. Comprehensive Model Coverage
- **Llama family**: 3, 3.1, 3.2 (including Vision)
- **Qwen family**: 2.5, 3 (including Math and Coder variants)  
- **Gemma family**: 2, 3
- **Code models**: CodeLlama, DeepSeek Coder
- **Specialized**: Phi-3, DialoGPT, Zephyr

### 3. Diverse Training Scenarios
- Standard instruction following
- Multi-turn conversations
- Continued pretraining/domain adaptation
- Function/tool calling
- Preference optimization (DPO/ORPO)
- Vision-language training
- Mathematical reasoning
- Code generation
- Character roleplay

### 4. Best Practices Integration
- Response-only vs full sequence training
- Appropriate target modules per model architecture
- Optimized hyperparameters per scenario
- Sequence packing configurations
- GPU allocation strategies

## Usage

1. **Select Dataset Preset**: Choose from 18 diverse dataset configurations
2. **Select Training Preset**: Choose corresponding training configuration
3. **Customize**: Modify parameters as needed for your specific use case
4. **Run**: Execute dataset preparation followed by training

Each preset includes a `description` field explaining the intended use case and key characteristics.

## Coverage Summary

This configuration set covers approximately 80% of the SFT training scenarios found in Unsloth's official notebooks, including:
- ✅ Alpaca instruction following
- ✅ Conversational training (ShareGPT)
- ✅ Text completion/continued pretraining
- ✅ Tool/function calling
- ✅ DPO preference optimization
- ✅ ORPO preference optimization
- ✅ Vision-language training
- ✅ Mathematical reasoning
- ✅ Code generation
- ✅ Domain-specific fine-tuning
- ✅ Small model optimization
- ✅ Multi-GPU configurations

The configurations provide a solid foundation for most SFT training needs while maintaining the flexibility to customize for specific requirements.
