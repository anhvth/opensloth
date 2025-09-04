# Quick Start Guide - Async-GRPO

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install transformers trl datasets unsloth vllm loguru
```

### 2. Test Single GPU Training
```bash
# Quick test with 10 steps
python src/cli/train.py --device 0 --steps 10
```

### 3. Run Distributed Training
```bash
# Automatic multi-GPU setup
python src/cli/train.py --distributed
```

## 📊 Monitor Progress

```bash
# Watch job queue in real-time
watch -n 1 'ls -la worker/queue/ | head -10'

# Check logs
tail -f worker/log_parameter_server.txt
tail -f worker/log_worker_1.txt
```

## 🔧 Common Commands

### Single GPU Training
```bash
# Basic training
python src/cli/train.py

# Custom steps
python src/cli/train.py --steps 100

# Different GPU
python src/cli/train.py --device 1
```

### Distributed Training
```bash
# Automatic (recommended)
python src/cli/train.py --distributed

# Manual setup
# Terminal 1:
CUDA_VISIBLE_DEVICES=0 python src/cli/parameter_server.py

# Terminal 2:  
CUDA_VISIBLE_DEVICES=1,0 WORKER_DEVICE=1 python src/cli/worker.py -d 1
```

### Monitoring & Debugging
```bash
# TensorBoard visualization
python src/cli/show_board.py ./outputs/tensorboard/
tensorboard --logdir ./outputs/tensorboard/

# Debug mode
LOG_LEVEL=DEBUG python src/cli/train.py --device 0 --steps 5
```

## ⚠️ Troubleshooting

### GPU Issues
```bash
# Check CUDA
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Check peer access
python -c "import cupy as cp; print('P2P:', cp.cuda.runtime.deviceCanAccessPeer(0, 1))"
```

### Model Issues
```bash
# Verify model path
ls -la /data/hf-models/Qwen/Qwen3-1.7B-Base/

# Use different model path
MODEL_PATH="/your/model/path" python src/cli/train.py
```

### Memory Issues
```bash
# Reduce memory usage
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# Small training config
python src/cli/train.py --steps 10 --device 0
```

## 📁 Key Files
- `README.md` - Full documentation
- `src/cli/train.py` - Main training script
- `src/app/trainer_setup_gsmk.py` - Model configuration
- `worker/queue/` - Job files (created during training)
- `outputs/` - Training checkpoints

## 🎯 Next Steps
1. Start with single GPU training
2. Verify system works with small steps
3. Scale to distributed training
4. Customize reward functions and datasets
5. Monitor with TensorBoard

See `README.md` for detailed documentation!