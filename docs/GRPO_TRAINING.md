# GRPO Training with OpenSloth

This document explains how to use OpenSloth for GRPO (Group Relative Policy Optimization) training with configurable reward functions.

## Overview

OpenSloth now supports GRPO training using Unsloth's native implementation with:
- **Configurable reward functions** for different task types
- **Multi-GPU support** with gradient synchronization  
- **Task-specific chat templates** (math, code, general, reasoning)
- **CLI integration** for easy usage

## Quick Start

### 1. Prepare Dataset

```bash
# For math problems
python prepare_dataset/prepare_grpo_dataset.py --task-type math --num-samples 1000 --output-path ./data/grpo_math

# For code generation
python prepare_dataset/prepare_grpo_dataset.py --task-type code --num-samples 500 --output-path ./data/grpo_code

# For general QA
python prepare_dataset/prepare_grpo_dataset.py --task-type general --num-samples 1000 --output-path ./data/grpo_general

# Custom dataset
python prepare_dataset/prepare_grpo_dataset.py --task-type custom --custom-file my_data.json --output-path ./data/grpo_custom
```

### 2. Train with CLI

```bash
# Basic math GRPO training
os train data/grpo_math --method grpo --grpo-task-type math

# Multi-GPU training
os train data/grpo_math --method grpo --grpo-task-type math --gpus 0,1,2,3

# Custom settings
os train data/grpo_code --method grpo \
  --grpo-task-type code \
  --grpo-group-size 4 \
  --grpo-max-tokens 256 \
  --grpo-temp 0.8 \
  --max-steps 500

# Explicit reward functions
os train data/my_dataset --method grpo \
  --grpo-rewards math_format,math_answer,length_penalty
```

### 3. Train with Python Script

```python
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

# Configure GRPO training
config = OpenSlothConfig(
    devices=[0, 1],  # Multi-GPU
    training_type="grpo",
    data_cache_path="./data/grpo_math",
    
    # Model settings
    fast_model_args={
        "model_name": "unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit",
        "max_seq_length": 2048,
        "load_in_4bit": True,
    },
    
    # GRPO-specific settings
    grpo_args={
        "task_type": "math",  # Auto-selects appropriate rewards
        "group_size": 4,
        "max_new_tokens": 256,
        "temperature": 1.0,
        # reward_functions: [],  # Auto-selected based on task_type
    }
)

# Training configuration  
training_args = TrainingArguments(
    output_dir="./outputs/grpo_training",
    max_steps=200,
    per_device_train_batch_size=1,
    learning_rate=5e-6,
    # ... other standard training args
)
```

## Reward Functions

### Available Reward Functions

List available reward functions:
```bash
os list-rewards
```

**Individual Functions:**
- `length_penalty`: Penalizes too short/long responses
- `math_format`: Rewards proper math reasoning format  
- `math_answer`: Rewards correct mathematical answers
- `math_number`: Rewards correct numerical answers
- `code_correctness`: Rewards syntactically correct code

**Task Presets:**
- `math`: `[math_format, math_answer, math_number]`
- `code`: `[code_correctness, length_penalty]` 
- `general`: `[length_penalty]`
- `reasoning`: `[math_format, length_penalty]`

### Custom Reward Functions

Create your own reward function:

```python
from opensloth.grpo_rewards import RewardFunction, register_reward_function

class MyRewardFunction(RewardFunction):
    def __init__(self):
        super().__init__("my_reward", "Description of my reward function")
    
    def __call__(self, prompts, completions, **kwargs):
        scores = []
        for completion in completions:
            content = completion[0]["content"]
            # Your scoring logic here
            score = len(content) * 0.01  # Simple example
            scores.append(score)
        return scores

# Register it
register_reward_function(MyRewardFunction())
```

## Multi-GPU Training

GRPO supports multi-GPU training with gradient synchronization:

```bash
# Multi-GPU with tmux (recommended)
os train data/grpo_math --method grpo --gpus 0,1,2,3 --use-tmux

# Multi-GPU with multiprocessing  
os train data/grpo_math --method grpo --gpus 0,1,2,3
```

The system automatically:
- Distributes training across GPUs
- Synchronizes gradients using NCCL
- Saves models only from rank 0

## Configuration Options

### GRPO-Specific CLI Options

```bash
--grpo-task-type        # Task type: math|code|general|reasoning
--grpo-group-size       # Number of responses per prompt (default: 4)
--grpo-temp             # Sampling temperature (default: 1.0)  
--grpo-max-tokens       # Max new tokens per response (default: 256)
--grpo-rewards          # Comma-separated reward function names
```

### Chat Templates

Each task type uses an appropriate chat template:

**Math Template:**
```
System: You are given a problem. Think about it and provide working.
Place it between <start_working_out> and <end_working_out>.
Then provide solution between <SOLUTION></SOLUTION>

User: [math problem]
Assistant: <start_working_out>[reasoning]<end_working_out><SOLUTION>[answer]</SOLUTION>
```

**Code Template:**
```
System: You are a helpful Python programming assistant.

User: [coding request]  
Assistant: [code solution]
```

## Advanced Usage

### Custom Dataset Format

Your dataset should have this structure:
```json
{
  "prompt": [
    {"role": "system", "content": "System prompt"},
    {"role": "user", "content": "User question"}
  ],
  "answer": "Expected answer (for reward calculation)"
}
```

### Performance Tips

1. **Batch Size**: Start with `per_device_train_batch_size=1` for GRPO
2. **Group Size**: Use 4-8 responses per prompt (higher = better but more VRAM)
3. **Temperature**: 0.8-1.2 for good diversity
4. **Max Tokens**: Match your expected response length

### Monitoring

GRPO training logs:
- Reward scores per response
- Sample generations every N steps
- Training metrics (loss, learning rate, etc.)

Check TensorBoard logs:
```bash
tensorboard --logdir outputs/your_training_dir/runs
```

## Troubleshooting

**Common Issues:**

1. **CUDA OOM**: Reduce `group_size` or `max_new_tokens`
2. **Low Rewards**: Check reward functions match your task
3. **Generation Issues**: Verify chat template is appropriate
4. **Multi-GPU Issues**: Ensure NCCL is properly configured

**Debug Commands:**
```bash
# Test dataset loading
python -c "from datasets import load_from_disk; print(load_from_disk('data/my_dataset')[0])"

# Test reward functions  
os list-rewards

# Dry run to check config
os train data/my_dataset --method grpo --dry-run
```

## Examples

See the `train_scripts/` directory for complete examples:
- `train_grpo_qwen.py`: Basic GRPO training script
- `train_scripts/train_grpo_math.py`: Math-specific configuration
- `train_scripts/train_grpo_code.py`: Code generation setup
