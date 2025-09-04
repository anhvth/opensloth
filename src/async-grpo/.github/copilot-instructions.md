# Copilot Coding Agent Instructions for async-grpo

## Project Overview
Distributed GRPO (Generalized Reward Policy Optimization) training system using CUDA IPC for zero-copy GPU-to-GPU memory sharing. The parameter server (PS) trains a GRPO model on GPU 0 while workers on GPU 1+ handle compute-intensive inference tasks, with direct memory sharing for weight synchronization.

## Architecture & Data Flow
- **ps.py**: GRPO trainer on GPU 0, exports weight tensors via CUDA IPC, creates jobs for workers, handles optimization steps
- **worker.py**: Claims jobs from `./worker/inputs/`, syncs weights from PS via IPC, processes inference using local model copy
- **grpo/base.py**: Shared environment setup creating GRPO trainers with Qwen3-0.6B model + LoRA adapters
- **async_grpo_ipc/**: Core IPC abstraction - serializes/deserializes trainable parameters, handles D2D memory copies

**Job Flow**: PS creates `{job_id}.input.pt` → Worker claims via atomic rename → Processes with synced weights → Saves `{job_id}.result.pt` → PS consumes results

## Key Patterns & Conventions
- **Trainer Setup**: Use `grpo.base.setup_environment(device_id)` to create consistent GRPO trainers across PS/workers
- **Weight Sync**: PS serializes only trainable parameters (LoRA weights) into flat tensor + metadata, workers deserialize and apply via `set_trainable_params()`
- **Job System**: Atomic file operations for job distribution - rename for claiming, torch.save/load for data transfer
- **CUDA Devices**: PS always GPU 0, workers start from GPU 1. Set `CUDA_VISIBLE_DEVICES="1,0"` in worker so Unsloth allocates model on GPU 1
- **IPC Handles**: Serialization metadata stored in handle JSON enables parameter-level sync rather than full model sync

## Developer Workflows
- **Setup Dependencies**: Requires Unsloth, TRL, VLLM, specific model path `/data/hf-models/unsloth/Qwen3-0.6B/`
- **Run Training**:
  ```zsh
  # Terminal 1: PS trains and creates jobs
  CUDA_VISIBLE_DEVICES=0 python ps.py
  
  # Terminal 2: Worker processes inference jobs  
  CUDA_VISIBLE_DEVICES=0,1 WORKER_DEVICE=1 python worker.py
  ```
- **Testing**: Use `test_job_system.py` for job distribution tests, `dev_test.py` for weight sync simulation
- **Debugging**: Check `./worker/global_step.txt` for sync status, `ipdb` breakpoints common in IPC code for device debugging

## Integration Points & Gotchas
- **Unsloth Cache**: Worker sets separate cache dir to avoid conflicts: `UNSLOTH_COMPILE_LOCATION=".cache/UNSLOTH_CACHE_DIR_1"`
- **CUDA Compatibility**: PyTorch/CuPy must share same CUDA runtime, peer access required between GPUs
- **IPC Memory**: CuPy allocation used for IPC-compatible tensors, fallback to host memory if D2D copy fails
- **Trainer Patching**: PS overrides `trainer._prepare_inputs()` to delegate inference to workers, uses `TrainerCallback` for weight updates post-optimization

## Model & Data Specifics
- **Model**: Qwen3-0.6B with LoRA (r=16, alpha=32) on specific attention/MLP layers
- **Dataset**: GSM8K math problems, reward function extracts numeric answers after '####'
- **Training**: GRPO with VLLM sampling, 100 steps, batch_size=1, 4 generations per prompt

## File Structure Navigation
- `grpo/`: GRPO-specific trainer setup and patches
- `async_grpo_ipc/`: Reusable IPC module with tensor serialization
- `worker/inputs/`, `worker/results/`: Job queue directories
- `scripts/`: Standalone training examples
- `test_*.py`: Various testing approaches for different scenarios

---

_Core insight: This is a distributed RL training system disguised as a CUDA IPC demo. The "worker" is actually offloading GRPO inference while keeping optimization centralized._


RULE: 
- PS always run on first device "0"
- Worker run on device "x,0", the cuda visible devices are set at the begining of script before import unsloth
- Unsloth must import before anything torch related
