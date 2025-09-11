# OpenSloth Memory Management Fixes

## Problem Analysis

The "Fatal Python error: none_dealloc: deallocating None" error indicates a critical reference counting bug in C extensions, likely from Unsloth's CUDA kernels. This type of error happens when:

1. **Reference counting bugs**: Improper handling of Python object references in C extensions
2. **Memory leaks**: Accumulated tensors and gradients not properly cleaned up
3. **Thread safety issues**: Multiple threads (tqdm monitors, autograd) accessing shared resources
4. **Import order conflicts**: Torch modules imported before Unsloth causing C extension conflicts

## Fixes Implemented

### 1. NCCL Gradient Synchronization (`nccl_grad_sync.py`)

**Problem**: In-place gradient modifications can cause reference counting issues.

**Fix**:

- Added gradient validation before synchronization
- Use tensor copies instead of in-place operations
- Added explicit cleanup of intermediate tensors
- Added error handling with forced garbage collection

### 2. Training Loop Memory Management (`inner_training_loop.py`)

**Problem**: No memory cleanup during training leads to accumulation and reference bugs.

**Fix**:

- Added memory monitoring and health checks
- Implemented periodic garbage collection (every 50 steps)
- Added CUDA cache clearing (every 10 steps)
- Added leak detection for None parameters
- Wrapped training_step with error handling for C extension errors

### 3. Model Initialization Safety (`model_init.py`)

**Problem**: Unsloth import order violations and lack of error handling.

**Fix**:

- Added import order validation (Unsloth before torch)
- Added comprehensive error handling around model/LoRA loading
- Added state validation after each operation
- Added cleanup on initialization failures

### 4. Memory Monitoring System (`memory_monitor.py`)

**New Component**: Proactive memory monitoring to prevent issues.

**Features**:

- Real-time system and GPU memory tracking
- Configurable warning and critical thresholds
- Automatic cleanup when memory usage is critical
- Memory leak detection for model parameters
- Safe tensor operation wrapper

## Testing and Monitoring

### 1. Immediate Testing

```bash
# Test with the same configuration that failed
uv run scripts/opensloth_trainer.py config.py

# Monitor for these log messages:
# - "Training start" memory summary
# - Periodic memory cleanup logs
# - Any "CRITICAL" or "WARNING" memory alerts
```

### 2. Memory Monitoring

The new system will log:

- Initial memory state at training start
- Memory summaries every 50 steps
- Warnings when memory usage > 85%
- Critical alerts when memory usage > 95%

### 3. Early Warning Signs

Watch for these log messages that indicate potential issues:

- "CRITICAL: System/GPU memory at X%"
- "Found None parameters" - indicates reference counting bugs
- "Invalid gradient detected" - may precede C extension errors
- "Memory-related error in operation" - C extension conflicts

### 4. Performance Impact

The fixes are designed to be minimally invasive:

- Memory checks every step (lightweight)
- Garbage collection every 50 steps
- CUDA cache clearing every 10 steps
- Deep monitoring only when issues detected

## Emergency Procedures

If the error still occurs:

1. **Immediate**: The training will attempt automatic cleanup and continue
2. **Manual intervention**: Reduce batch size or gradient accumulation steps
3. **Diagnostic**: Check logs for memory patterns before crash
4. **Fallback**: Use single GPU mode to isolate the issue

## Environment Variables for Debugging

```bash
# Enable verbose NCCL logging
export NCCL_DEBUG=INFO

# Use tmux mode for better process isolation
export USE_TMUX=1

# Enable Python memory debugging
export PYTHONMALLOC=malloc_debug
```

## Long-term Monitoring

Monitor these metrics across training runs:

1. Peak memory usage patterns
2. Frequency of memory warnings
3. Training step where issues typically occur
4. Correlation with specific operations (gradient sync, checkpointing)

The comprehensive fixes should significantly reduce the likelihood of the `none_dealloc` error and provide early warning systems to prevent future occurrences.
