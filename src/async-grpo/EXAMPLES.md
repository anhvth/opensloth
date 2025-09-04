# Usage Examples - Async-GRPO

## Basic Training Examples

### Example 1: First Time Training
```bash
# Start with minimal training to test the system
python src/cli/train.py --device 0 --steps 5

# Expected output:
# - Model loads on GPU 0
# - Training runs for 5 steps
# - Creates outputs/ directory with checkpoints
```

### Example 2: Production Training
```bash
# Full distributed training on 4 GPUs
python src/cli/train.py --distributed

# This automatically:
# - Starts parameter server on GPU 0
# - Launches workers on GPUs 1, 2, 3
# - Trains for default 10,000 steps
# - Saves checkpoints every 100 steps
```

### Example 3: Custom Configuration
```bash
# Train with custom settings
python src/cli/train.py src/app/trainer_setup_gsmk.py --device 0 --steps 500

# Uses GSM8K configuration for mathematical reasoning
# Trains for 500 steps on GPU 0
```

## Configuration Examples

### Example 4: Custom Reward Function
```python
# Create custom_trainer.py
def get_trainer():
    from unsloth import FastLanguageModel
    from trl import GRPOConfig, GRPOTrainer
    
    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/data/hf-models/Qwen/Qwen3-1.7B-Base",
        max_seq_length=2048,
        fast_inference=True,
    )
    
    # Custom reward function
    def reward_creativity(prompts, completions, answer, **kwargs):
        scores = []
        for completion in completions:
            response = completion[0]["content"]
            
            # Reward creative responses
            creativity_words = ["creative", "novel", "innovative", "unique"]
            score = sum(2.0 for word in creativity_words if word in response.lower())
            scores.append(score)
        return scores
    
    # Configure trainer
    training_args = GRPOConfig(
        learning_rate=3e-6,
        max_steps=100,
        per_device_train_batch_size=1,
        num_generations=4,
        output_dir="creative_outputs",
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_creativity],
        args=training_args,
        train_dataset=your_dataset,
    )
    
    return trainer

# Run with custom trainer
# python src/cli/train.py custom_trainer.py --device 0
```

### Example 5: Memory-Optimized Training
```python
# Create memory_efficient_trainer.py
def get_trainer():
    # ... model loading ...
    
    # Memory-efficient configuration
    training_args = GRPOConfig(
        learning_rate=5e-6,
        max_prompt_length=1024,      # Reduced from 4096
        max_completion_length=1024,   # Reduced from 4096
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4, # Simulate larger batch
        num_generations=2,            # Reduced from 4
        max_steps=200,
        save_steps=50,
        output_dir="memory_efficient_outputs",
    )
    
    # Smaller LoRA configuration
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,           # Reduced rank
        lora_alpha=16, # Smaller alpha
        target_modules=["q_proj", "v_proj"],  # Fewer modules
        use_gradient_checkpointing="unsloth",
    )
    
    return trainer
```

## Monitoring Examples

### Example 6: Real-time Monitoring
```bash
# Terminal 1: Start training
python src/cli/train.py --distributed

# Terminal 2: Monitor job queue
watch -n 2 'echo "=== Job Queue ===" && ls -la worker/queue/ | head -10 && echo "=== Global Step ===" && cat worker/global_step.txt 2>/dev/null || echo "Not started"'

# Terminal 3: Monitor logs
tail -f worker/log_parameter_server.txt

# Terminal 4: Monitor GPU usage
watch -n 1 nvidia-smi
```

### Example 7: TensorBoard Analysis
```bash
# After training, generate TensorBoard logs
python src/cli/show_board.py ./outputs/tensorboard/

# Launch TensorBoard
tensorboard --logdir ./outputs/tensorboard/ --port 6006

# Open browser to http://localhost:6006
# View metrics:
# - Training loss curves
# - Reward function scores
# - Job processing times
# - GPU utilization
```

## Debugging Examples

### Example 8: Debug Mode Training
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
export CUDA_LAUNCH_BLOCKING=1

# Run with Python debugger
python -m pdb src/cli/train.py --device 0 --steps 3

# Useful pdb commands:
# (Pdb) c          # continue
# (Pdb) n          # next line
# (Pdb) pp vars()  # print variables
# (Pdb) l          # list current code
```

### Example 9: System Verification
```python
# Create verify_system.py
import torch
import cupy as cp
from async_grpo_ipc import open_remote_memory, create_stream

def verify_system():
    print("=== System Verification ===")
    
    # Check CUDA
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    
    # Check CuPy
    print(f"CuPy version: {cp.__version__}")
    
    # Check GPU peer access
    if torch.cuda.device_count() >= 2:
        can_access = cp.cuda.runtime.deviceCanAccessPeer(0, 1)
        print(f"GPU 0 can access GPU 1: {can_access}")
    
    # Check model loading
    try:
        from src.app.trainer_setup_gsmk import get_trainer
        trainer = get_trainer()
        device = next(trainer.model.parameters()).device
        print(f"Model loaded on device: {device}")
        print("✅ System verification passed!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

if __name__ == "__main__":
    verify_system()

# Run verification:
# python verify_system.py
```

## Performance Examples

### Example 10: Benchmark Training Speed
```bash
# Benchmark single GPU
time python src/cli/train.py --device 0 --steps 20

# Benchmark distributed
time python src/cli/train.py --distributed # (interrupt after ~50 steps)

# Compare job processing rates
grep "Completed job" worker/log_worker_*.txt | wc -l
```

### Example 11: Profile Memory Usage
```python
# Create profile_memory.py
import torch
import time
from src.app.trainer_setup_gsmk import get_trainer

def profile_memory():
    torch.cuda.empty_cache()
    
    print("Memory before model loading:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    trainer = get_trainer()
    
    print("\nMemory after model loading:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
    
    # Simulate training step
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    
    print("\nMemory after batch loading:")
    print(f"Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

if __name__ == "__main__":
    profile_memory()
```

## Error Handling Examples

### Example 12: Graceful Error Recovery
```bash
# Handle CUDA out of memory
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Reduce memory usage and retry
python src/cli/train.py --device 0 --steps 10

# If still failing, try minimal config:
# Edit trainer config to use:
# - max_seq_length=1024
# - r=4 (LoRA rank)
# - num_generations=2
```

### Example 13: Model Path Issues
```bash
# If model path doesn't exist
ls -la /data/hf-models/Qwen/Qwen3-1.7B-Base/

# Download/symlink model
# mkdir -p /data/hf-models/Qwen/
# ln -s /your/actual/model/path /data/hf-models/Qwen/Qwen3-1.7B-Base

# Or modify trainer config to use different path
sed -i 's|/data/hf-models/Qwen/Qwen3-1.7B-Base|/your/model/path|g' src/app/trainer_setup_gsmk.py
```

## Integration Examples

### Example 14: Custom Dataset Integration
```python
# Create custom_dataset_trainer.py
from datasets import Dataset

def prepare_coding_dataset():
    """Example: Code generation dataset"""
    data = [
        {
            "prompt": [{"role": "user", "content": "Write a Python function to reverse a string"}],
            "answer": "def reverse_string(s): return s[::-1]"
        },
        {
            "prompt": [{"role": "user", "content": "Create a function to find factorial"}], 
            "answer": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
        },
        # Add more examples...
    ]
    return Dataset.from_list(data)

def coding_reward(prompts, completions, answer, **kwargs):
    """Reward function for code quality"""
    scores = []
    for completion, expected in zip(completions, answer):
        code = completion[0]["content"]
        
        score = 0
        # Basic syntax checking
        if "def " in code: score += 2
        if "return " in code: score += 2
        if expected.split("(")[0] in code: score += 3  # Function name match
        
        scores.append(score)
    return scores

# Use in trainer configuration...
```

## Troubleshooting Workflow

### Example 15: Step-by-Step Debugging
```bash
# 1. Verify system
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# 2. Test minimal training
python src/cli/train.py --device 0 --steps 1

# 3. Check logs for errors
cat worker/log_parameter_server.txt | grep -i error

# 4. Monitor resources
nvidia-smi
df -h  # disk space

# 5. Clean up if needed
rm -rf worker/ outputs/ .cache/
```

These examples cover the most common use cases and should help new users get started quickly while providing templates for advanced customization.