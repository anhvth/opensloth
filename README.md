# Python Version Compatibility

**Important:** This project only supports Python 3.10.x.

Due to strict dependency requirements (notably from `unsloth` and `cut-cross-entropy`), you must use Python 3.10 for all development and production environments. Other Python versions (3.9, 3.11, 3.12, etc.) are not compatible and will result in installation errors.

**How to check your Python version:**

```sh
python --version
```

If you are not using Python 3.10.x, please create a new virtual environment with Python 3.10 before installing dependencies.

<p align="center">
    <img src="images/opensloth.png" alt="opensloth Logo" width="200" />
</p>

# OpenSloth

A multi-GPU training framework that combines [Unsloth](https://github.com/unslothai/unsloth) with multi-GPU support and [sequence packing](https://huggingface.co/blog/sirluk/llm-sequence-packing) optimizations.

**Core Components:**
- **Unsloth**: 2x faster training with 75% VRAM savings
- **Multi-GPU**: Distributed training across multiple GPUs  
- **Sequence Packing**: Smart batching that reduces padding waste by up to 40%

**The Result:** Unsloth's efficiency × GPU count × sequence packing optimizations = speedups that often exceed theoretical maximums.

## 💾 Installation

```bash
conda create --name opensloth_env python=3.11
pip install uv
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install unsloth xformers opensloth
# or from source pip install git+https://github.com/anhvth/opensloth.git
```

## ⚡ Quickstart

```bash
# Basic multi-GPU training
python scripts/train.py
```

| Example | Description | Link/Command |
|---------|-------------|--------------|
| **Kaggle Notebook (T4x2)** | Live training example on Kaggle's dual T4 GPU environment | [🔗 Qwen3 OpenSloth 2GPUs](https://www.kaggle.com/code/anhvth226/qwen3-opensloth-2gpus?scriptVersionId=246662898) |
| **Local Training Script** | Check out the training script for configuration examples | `python scripts/train.py` |
| **Local Jupyter Notebook** | Interactive training notebook for local development | [`notebooks/train.ipynb`](notebooks/train.ipynb) |

## ⚡ Performance Benchmarks

**[📊 View Full WandB Comparison](https://wandb.ai/anhvth/CompareUnsloth)**

### opensloth vs Unsloth Direct Comparison

Controlled comparison with identical configurations:
- **Model**: Qwen3-8B-bnb-4bit  
- **Training Steps**: 100 steps
- **Global Batch Size**: 32

**Results:**
- **opensloth (2 GPUs)**: 8m 28s ⚡
- **Unsloth (1 GPU)**: 19m 34s
- **Performance Gain**: ~2.3x faster 

**Why 2.3x Speedup on 2 GPUs?**

OpenSloth achieves **2.3x speedup** through three optimizations:
- ✅ **Sequence packing**: Smart batching reduces padding waste ([learn more](https://huggingface.co/blog/sirluk/llm-sequence-packing))
- ✅ **Multi-GPU scaling**: Distributed training across GPUs
- ✅ **Load balancing**: Even workload distribution across GPUs

**Scaling Expectations:**
- **2 GPUs**: ~2.3x faster than single GPU
- **4 GPUs**: ~4.6x faster than single GPU
- **8 GPUs**: ~9.2x faster than single GPU


## 🔧 Quick Tips
- Enable packing, set bz=1, long sequence length (8k, 16k, etc.) with larger gradient accumulation steps (64, 128). Unsloth's will automatically handle sequence packing on global batch to optimize gpu utilization.

**For faster iteration:**
- Start with smaller models: `unsloth/Qwen3-0.6b-bnb-4bit`
- Test single GPU first: modify `gpus=[0]` in script
- Use fewer samples for quick testing

**Recommended Configuration:**
```python
# Optimize for sequence packing and multi-GPU efficiency
TrainingConfig(
    per_device_train_batch_size=4,      # Larger batches per GPU
    gradient_accumulation_steps=8,       # Fewer gradient sync operations
    # Effective batch size = 4 * 8 * num_gpus
)
```

## 🔧 Troubleshooting


### Triton Kernel Compilation Error: Missing Python.h

If you encounter an error like:

```
/tmp/tmpy04kmz_t/main.c:4:10: fatal error: Python.h: No such file or directory
    4 | #include <Python.h>
```

This means a C extension (such as Triton, used by Unsloth) is trying to compile native code and cannot find the Python development headers (`Python.h`).

#### How to Fix

Install the Python development package for your Python version:

**Ubuntu/Debian:**

```bash
sudo apt update
sudo apt install python3.10-dev
```

Replace `python3.10-dev` with your Python version if needed (e.g., `python3.11-dev`).

**Notes:**
- This is common for packages that compile native extensions (Triton, PyTorch custom kernels, etc.).
- After installing, you do **not** need to reinstall your virtual environment—just re-run your script.

If you are on a different OS (CentOS, macOS, etc.), please consult your system's package manager or ask for specific instructions.

**Single GPU Testing:**
```python
# In your training script, change:
gpus = [0]  # Use only first GPU for debugging
```

## 📖 Documentation

- See [CHANGELOG.md](CHANGELOG.md) for recent updates and breaking changes.

## How to Prepare and Store a Trainer Dataset

Follow these steps to extract and save a dataset from an Unsloth notebook:

1. Visit the [Unsloth Notebooks Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks).
2. Select the notebook for your target model.
3. Export the notebook to a Python script.
4. Copy all code up to (but not including) `trainer.train()`.
5. Run the code to initialize the trainer.
6. Save the trainer's dataset:

   ```python
   trainer.train_dataset.save_to_disk("data/cache_qwen3_dataset")
   ```

7. Place it here
```python
....
opensloth_config = OpenSlothConfig(
    data_cache_path="data/cache_qwen3_dataset/",
    devices=DEVICES,
    ...
        
```


This will store the processed dataset for later use.
