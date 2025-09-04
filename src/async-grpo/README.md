# Async-GRPO: Distributed GRPO Training with CUDA IPC

## Table of Contents
- [Project Overview](#project-overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Training Modes](#training-modes)
- [Configuration](#configuration)
- [Monitoring and Debugging](#monitoring-and-debugging)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)
- [File Structure](#file-structure)

## Project Overview

Async-GRPO is a distributed GRPO (Generalized Reward Policy Optimization) training system that uses CUDA IPC (Inter-Process Communication) for zero-copy GPU-to-GPU memory sharing. The system is designed to efficiently train large language models using reward-based optimization while distributing the computational load across multiple GPUs.

### Key Features
- **Zero-Copy Memory Sharing**: Uses CUDA IPC for direct GPU-to-GPU weight synchronization
- **Distributed Architecture**: Parameter server handles training while workers process inference
- **Atomic File Operations**: Robust job queue system with fault tolerance
- **Multi-GPU Support**: Scales across multiple CUDA devices
- **GSM8K Integration**: Pre-configured for mathematical reasoning tasks

## System Requirements

### Hardware
- **GPUs**: 2+ CUDA-compatible GPUs (minimum Pascal architecture)
- **Memory**: 16GB+ system RAM, 8GB+ GPU memory per device
- **Storage**: 10GB+ free disk space for model caching and outputs

### Software
- **OS**: Linux (Ubuntu 18.04+ recommended)
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or higher
- **Driver**: NVIDIA driver 450.80.02+

### Dependencies
- PyTorch
- CuPy (CUDA 12.x)
- Transformers
- TRL (Transformer Reinforcement Learning)
- Unsloth (for efficient model loading)
- VLLM (for fast inference)
- Datasets (HuggingFace)

## Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd async-grpo
```

### 2. Install Dependencies
```bash
# Install basic requirements
pip install -r requirements.txt

# Install additional dependencies
pip install transformers trl datasets unsloth vllm loguru
```

### 3. Download Models
Ensure you have access to the required models:
```bash
# The system expects models at:
# /data/hf-models/Qwen/Qwen3-1.7B-Base
# 
# You may need to download or symlink the model directory
```

### 4. Verify CUDA Setup
```python
import torch
import cupy as cp

# Check CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")

# Check CuPy
print(f"CuPy version: {cp.__version__}")
```

## Quick Start

### Single GPU Training (Simplest)
```bash
# Train on a single GPU (recommended for testing)
python src/cli/train.py --device 0 --steps 50
```

### Distributed Training (Multi-GPU)
```bash
# Terminal 1: Start Parameter Server
python src/cli/train.py --distributed
```

This will automatically start:
- Parameter server on GPU 0
- Workers on GPUs 1, 2, 3
- All with proper logging and monitoring

### Custom Configuration
```bash
# Use custom trainer configuration
python src/cli/train.py src/app/trainer_setup_gsmk.py --device 0 --steps 100
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Parameter      │    │     Worker      │    │     Worker      │
│  Server (GPU 0) │    │    (GPU 1)      │    │    (GPU 2)      │
│                 │    │                 │    │                 │
│ ┌─────────────┐ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ GRPO Trainer│ │    │ │Model Replica│ │    │ │Model Replica│ │
│ │             │ │    │ │             │ │    │ │             │ │
│ │ - Training  │ │    │ │ - Inference │ │    │ │ - Inference │ │
│ │ - Optimizer │ │    │ │ - Job Proc. │ │    │ │ - Job Proc. │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌────────┴───────┐              │
         │              │                │              │
         ▼              ▼                ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CUDA IPC Memory                         │
│              (Shared Weight Tensors)                       │
└─────────────────────────────────────────────────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   File-based Job Queue                     │
│     ./worker/queue/                                         │
│     - {job_id}_pending.pt                                  │
│     - {job_id}_processing_by_dev{N}.pt                     │
│     - {job_id}_complete_from_dev{N}.pt                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Parameter Server (PS)**
   - Runs on GPU 0
   - Handles model training and optimization
   - Creates and distributes jobs
   - Collects results for training steps

2. **Workers**
   - Run on GPU 1+
   - Process inference jobs
   - Sync weights from PS via CUDA IPC
   - Return results to job queue

3. **CUDA IPC System**
   - Zero-copy weight sharing between GPUs
   - Automatic peer access management
   - Dtype-bucketed serialization for efficiency

4. **Job Queue**
   - File-based atomic operations
   - Fault-tolerant job distribution
   - Automatic cleanup and archival

## Training Modes

### 1. Single GPU Mode
Best for development and testing:
```bash
python src/cli/train.py --device 0
```

### 2. Distributed Mode  
For production multi-GPU training:
```bash
python src/cli/train.py --distributed
```

### 3. Manual Distributed Setup
For custom control:
```bash
# Terminal 1: Parameter Server
CUDA_VISIBLE_DEVICES=0 python src/cli/parameter_server.py

# Terminal 2: Worker 1
CUDA_VISIBLE_DEVICES=1,0 WORKER_DEVICE=1 python src/cli/worker.py -d 1

# Terminal 3: Worker 2  
CUDA_VISIBLE_DEVICES=2,0 WORKER_DEVICE=2 python src/cli/worker.py -d 2
```

## Configuration

### Trainer Configuration
Create custom trainer setups by modifying `src/app/trainer_setup_gsmk.py`:

```python
def get_trainer():
    # Model configuration
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/path/to/your/model",
        max_seq_length=4096,
        load_in_4bit=False,
        fast_inference=True,
    )
    
    # LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # LoRA rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    
    # Training arguments
    training_args = GRPOConfig(
        learning_rate=5e-6,
        max_steps=1000,
        per_device_train_batch_size=1,
        num_generations=4,
        output_dir="outputs",
    )
    
    return GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[your_reward_functions],
        args=training_args,
        train_dataset=dataset,
    )
```

### Environment Variables
```bash
# CUDA device visibility
export CUDA_VISIBLE_DEVICES="0,1,2,3"

# Worker device assignment
export WORKER_DEVICE=1

# Unsloth cache location (per worker)
export UNSLOTH_COMPILE_LOCATION=".cache/UNSLOTH_CACHE_DIR_1"

# IPC handle path
export CUDA_IPC_HANDLE_PATH="./worker/ps_ipc_handle.json"

# Logging level
export LOG_LEVEL="INFO"

# Fast inference mode
export FAST_INFERENCE="1"
```

## Monitoring and Debugging

### Real-time Monitoring
```bash
# Watch job queue activity
watch -n 1 'ls -la worker/queue/ | head -20'

# Monitor logs
tail -f worker/log_parameter_server.txt
tail -f worker/log_worker_1.txt

# Check performance
python src/cli/show_board.py ./outputs/tensorboard/
```

### Key Files to Monitor
- `worker/global_step.txt` - Current training step
- `worker/ps_ready.signal` - PS initialization status
- `worker/ps_ipc_handle.json` - CUDA IPC configuration
- `worker/ps_ipc_handle_version.json` - Weight version tracking

### TensorBoard Visualization
```bash
# Generate TensorBoard logs from results
python src/cli/show_board.py ./outputs/tensorboard/

# Launch TensorBoard
tensorboard --logdir ./outputs/tensorboard/
```

### Debug Commands
```python
# Check IPC connection
from async_grpo_ipc import open_remote_memory
remote = open_remote_memory("./worker/ps_ipc_handle.json")
print(f"Connected: {remote}")

# Verify model loading
from src.app.trainer_setup_gsmk import get_trainer
trainer = get_trainer()
print(f"Model device: {next(trainer.model.parameters()).device}")
```

## Performance Tuning

### GPU Memory Optimization
```python
# Reduce model size
training_args = GRPOConfig(
    max_prompt_length=2048,      # Reduce from 4096
    max_completion_length=2048,   # Reduce from 4096
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2, # Simulate larger batch
)

# Enable gradient checkpointing
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing="unsloth",
    r=8,  # Smaller LoRA rank
)
```

### Job Queue Tuning
```python
# In ps_utils.py:throw_job_data()
max_input_in_folder = 16   # Increase for higher throughput
max_pending_output = 16    # Increase for more buffering

# Worker sleep times
base_sleep = 0.005  # Reduce for faster polling
max_sleep = 0.02    # Reduce for lower latency
```

### CUDA IPC Optimization
```python
# Enable peer access for all GPU pairs
import cupy as cp
device_count = cp.cuda.runtime.getDeviceCount()
for i in range(device_count):
    for j in range(device_count):
        if i != j:
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(j, 0)
            except:
                pass
```

## Troubleshooting

### Common Issues

#### 1. CUDA IPC Connection Failures
```bash
# Error: "Could not connect to PS after N attempts"
# Solution: Check GPU peer access
nvidia-smi topo -m

# Ensure GPUs can communicate
python -c "
import cupy as cp
print('Device 0 can access 1:', cp.cuda.runtime.deviceCanAccessPeer(0, 1))
print('Device 1 can access 0:', cp.cuda.runtime.deviceCanAccessPeer(1, 0))
"
```

#### 2. Model Loading Issues
```bash
# Error: "Model not found"
# Solution: Verify model path
ls -la /data/hf-models/Qwen/Qwen3-1.7B-Base/

# Alternative: Use different model
export MODEL_PATH="/path/to/your/model"
```

#### 3. Out of Memory Errors
```bash
# Error: "CUDA out of memory"
# Solution: Reduce memory usage
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# In trainer config:
# - Reduce max_seq_length
# - Increase gradient_accumulation_steps  
# - Use smaller LoRA rank
# - Enable 4-bit quantization
```

#### 4. File Permission Issues
```bash
# Error: "Permission denied" 
# Solution: Check file permissions
chmod -R 755 worker/
mkdir -p worker/queue worker/done

# Ensure atomic file operations work
python -c "
from utils.atomic_file_ops import AtomicFileOps
AtomicFileOps.write_text('test.txt', 'hello')
print(AtomicFileOps.read_text('test.txt'))
"
```

#### 5. Unsloth Import Errors
```python
# Error: Unsloth not found or incompatible
# Solution: Ensure proper installation order
# 1. Install PyTorch first
# 2. Install Unsloth
# 3. Restart Python session

# Verify installation
import unsloth
print(f"Unsloth version: {unsloth.__version__}")
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with Python debugger
python -m pdb src/cli/train.py --device 0 --steps 10

# Enable CUDA debugging
export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## Advanced Usage

### Custom Reward Functions
```python
def custom_reward(prompts, completions, answer, **kwargs):
    """Custom reward function for your task."""
    scores = []
    for completion, true_answer in zip(completions, answer):
        response = completion[0]["content"]
        
        # Your scoring logic here
        if "correct_pattern" in response:
            score = 5.0
        elif "partial_pattern" in response:
            score = 2.0
        else:
            score = -1.0
            
        scores.append(score)
    return scores

# Add to trainer
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[custom_reward],  # Use your function
    args=training_args,
    train_dataset=dataset,
)
```

### Custom Dataset Loading
```python
def prepare_custom_dataset():
    """Load and format your custom dataset."""
    from datasets import Dataset
    
    # Your data loading logic
    data = [
        {
            "prompt": [{"role": "user", "content": "Question 1"}],
            "answer": "Answer 1"
        },
        # ... more examples
    ]
    
    return Dataset.from_list(data)

# In trainer setup
dataset = prepare_custom_dataset()
```

### Multiple Worker Types
```bash
# Start specialized workers for different tasks
CUDA_VISIBLE_DEVICES=1,0 python src/cli/worker.py -d 1 --config config_fast.py &
CUDA_VISIBLE_DEVICES=2,0 python src/cli/worker.py -d 2 --config config_quality.py &
```

### Checkpoint Management
```python
# Save intermediate checkpoints
trainer.save_model("checkpoints/step_1000")

# Resume from checkpoint
trainer = GRPOTrainer.from_pretrained(
    "checkpoints/step_1000",
    # ... other arguments
)
```

## File Structure

```
async-grpo/
├── src/
│   ├── cli/
│   │   ├── train.py              # Main training CLI
│   │   ├── parameter_server.py   # Distributed PS
│   │   ├── worker.py             # Distributed worker
│   │   └── show_board.py         # TensorBoard generation
│   ├── app/
│   │   └── trainer_setup_gsmk.py # GSM8K configuration
│   ├── utils/
│   │   ├── worker_utils.py       # Worker helper functions
│   │   ├── ps_utils.py           # PS helper functions
│   │   ├── atomic_file_ops.py    # File operations
│   │   ├── job_queue.py          # Job management
│   │   ├── filesystem.py         # File utilities
│   │   ├── logger.py             # Logging setup
│   │   ├── lock.py               # File locking
│   │   └── performance.py        # Metrics collection
│   └── async_grpo_ipc/
│       ├── __init__.py           # IPC exports
│       ├── ipc.py                # CUDA IPC implementation
│       └── process_tensors.py    # Tensor serialization
├── docs/
│   ├── file_lifecycle_documentation.md
│   └── minimal_grpo.md
├── scripts/
│   └── run_worker.sh             # Worker startup script
├── worker/                       # Created at runtime
│   ├── queue/                    # Job files
│   ├── done/                     # Completed jobs
│   └── logs/                     # Log files
├── outputs/                      # Training outputs
├── requirements.txt
├── pyproject.toml
└── README.md                     # This file
```

### Key Directories
- **src/cli/**: Command-line entry points
- **src/app/**: Application-specific configurations
- **src/utils/**: Shared utility modules  
- **async_grpo_ipc/**: CUDA IPC implementation
- **worker/**: Runtime job queue and logs
- **outputs/**: Training checkpoints and results

## Getting Help

### Documentation
- `docs/file_lifecycle_documentation.md` - Detailed system internals
- `docs/minimal_grpo.md` - Simplified GRPO example
- Inline code documentation and comments

### Common Workflows
1. **First Time Setup**: Follow installation → quick start → single GPU test
2. **Development**: Use single GPU mode with small datasets
3. **Production**: Use distributed mode with full datasets
4. **Debugging**: Check logs, monitor files, use debug mode
5. **Optimization**: Tune parameters based on performance metrics

### Support
- Check existing issues and documentation
- Enable debug logging for detailed error information
- Use TensorBoard for training visualization
- Monitor system resources (GPU memory, disk space)

This system provides a robust foundation for distributed GRPO training with excellent fault tolerance and monitoring capabilities. Start with single GPU training to understand the system, then scale to distributed training for production workloads.