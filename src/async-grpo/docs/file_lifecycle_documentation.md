# File Lifecycle in async-grpo: Generation to Consumption

## Overview

The async-grpo system implements a sophisticated distributed training architecture where files serve as the primary communication mechanism between the Parameter Server (PS) and multiple Workers. This document provides a comprehensive analysis of how files flow through the system from generation to consumption.

## Architecture Summary

- **Parameter Server (PS)**: Runs on GPU 0, handles model training and optimization
- **Workers**: Run on GPU 1+, handle compute-intensive inference tasks  
- **Communication**: File-based job queue system with atomic operations
- **Synchronization**: CUDA IPC for weight sharing, file locks for coordination

## File Lifecycle Stages

### 1. Job Creation (PS → File System)

#### Location: `ps_utils.py:throw_job_data()`

The Parameter Server continuously creates jobs to keep workers busy:

```
PS DataLoader → generation_batch → {job_id}.pending.pt
```

**File Generation Process:**
1. **Data Preparation**: PS extracts training batch from dataloader
2. **Job ID Generation**: Composite format `{global_step:06d}_{job_index:03d}` (e.g., `000042_003`)
3. **Job Data Structure**:
   ```python
   job_data = {
       "generation_batch": generation_batch,
       "job_id": job_id,
       "created_at": time.time()
   }
   ```
4. **Atomic Write**: Uses `AtomicFileOps.write_torch()` to prevent corruption
5. **Queue Management**: Maintains max 8 pending input files, max 8 pending outputs

**File Naming Convention:**
- Initial: `{job_id}_pending.pt`
- Location: `./worker/queue/`

### 2. Job Claiming (Worker → File System)

#### Location: `job_queue.py:claim_job()`

Workers atomically claim available jobs using file rename operations:

```
{job_id}_pending.pt → {job_id}_processing_by_dev{N}.pt
```

**Claiming Process:**
1. **Discovery**: Workers scan `./worker/queue/` for `*_pending.pt` files
2. **FIFO Ordering**: Files sorted by modification time for fair processing
3. **Atomic Claiming**: `AtomicFileOps.claim_file()` uses `os.rename()` for atomicity
4. **Conflict Resolution**: If rename fails, another worker claimed the file
5. **Metadata Addition**:
   ```python
   job_data["claimed_at"] = time.time()
   job_data["worker_device"] = worker_device
   ```

**State Transition:**
- `000042_003_pending.pt` → `000042_003_processing_by_dev2.pt`

### 3. Job Processing (Worker Internal)

#### Location: `worker_utils.py:process_job()`

Workers process the claimed job with current model weights:

```
processing file + model weights → computation → result data
```

**Processing Steps:**
1. **Weight Synchronization**: Update local model from PS via CUDA IPC
2. **Input Preparation**: Extract `generation_batch` from job data
3. **Model Inference**: Execute `trainer._prepare_inputs(generation_batch)`
4. **Result Preparation**:
   ```python
   result_data = {
       'result': result_cpu,
       'job_id': job_id,
       'processed_at_step': current_global_step,
       'worker_device': worker_device,
       'job_created_step': job_global_step,
       'job_age': job_age,
       'queue_time': queue_time,
       'processing_time': processing_time,
       'metrics': metrics_cpu
   }
   ```

### 4. Result Generation (Worker → File System)

Workers save processing results atomically:

```
result_data → {job_id}_complete_from_dev{N}.pt
```

**Result File Creation:**
1. **CPU Migration**: All tensors moved to CPU to avoid device conflicts
2. **Atomic Write**: `AtomicFileOps.write_torch()` ensures consistency  
3. **File Cleanup**: Original processing file archived to `./worker/done/`

**State Transition:**
- `000042_003_processing_by_dev2.pt` → `000042_003_complete_from_dev2.pt`

### 5. Result Collection (PS ← File System)

#### Location: `ps_utils.py:collect_results()`

PS continuously scans for and collects completed results:

```
{job_id}_complete_from_dev{N}.pt → {job_id}_collected.pt → memory cache
```

**Collection Process:**
1. **Discovery**: PS scans for `*_complete_from_dev*.pt` files
2. **Atomic Claiming**: Rename to `*_collected.pt` to prevent conflicts
3. **Data Loading**: `AtomicFileOps.read_torch()` with error handling
4. **Validation**: Verify job_id matches filename
5. **Caching**: Store in `result_cache` with thread-safe access

### 6. Result Consumption (PS Internal)

#### Location: `ps_utils.py:get_next_job_result()`

PS consumes cached results for training:

```
result_cache → training batch → model optimization
```

**Consumption Process:**
1. **Queue Matching**: Find cached results for jobs in `job_queue`
2. **Result Extraction**: Remove from cache and queue atomically
3. **File Archival**: Move collected file to `./worker/done/` with reason
4. **Training Integration**: Return result to trainer for optimization step

## File State Transitions

```
[PS] generation_batch
  ↓ (create_job)
{job_id}_pending.pt
  ↓ (claim_job)  
{job_id}_processing_by_dev{N}.pt
  ↓ (process_job)
{job_id}_complete_from_dev{N}.pt  
  ↓ (collect_results)
{job_id}_collected.pt
  ↓ (get_next_job_result)
./worker/done/{job_id}_*_consumed_{job_id}.pt
```

## Atomic Operations & Error Handling

### File Locking System

The system uses `FileLock` class with `fcntl` for cross-process coordination:

```python
class FileLock:
    - Exclusive locks for write operations
    - Shared locks for read operations  
    - Configurable timeout (default 10s)
    - Automatic cleanup on context exit
```

### Atomic Write Pattern

All file operations use atomic write patterns:

```python
def _atomic_write_context():
    1. Create temporary file in same directory
    2. Write content to temporary file
    3. Atomic rename to final location
    4. Cleanup temporary file on error
```

### Error Recovery

- **Orphaned Files**: Cleanup threads remove abandoned temporary files
- **Corrupted Data**: Files with load errors are archived with "failed_load" reason
- **Duplicate Results**: Multiple results for same job_id are deduplicated
- **Missing Files**: Non-existent files handled gracefully

## Performance Characteristics

### Throughput Metrics

From the queue directory analysis:
- **Active Job Range**: Steps 000039 through 000170+ 
- **Concurrent Jobs**: ~8 pending + ~8 processing + ~8 complete per step
- **Worker Distribution**: Round-robin across devices 1, 2, 3
- **Job Frequency**: 6-10 jobs per global step

### File Lifecycle Timing

Typical file lifetimes observed:
- **Pending Duration**: Seconds to minutes (depends on worker availability)
- **Processing Duration**: Variable (model inference time)
- **Collection Latency**: Near-immediate (PS polls every 50ms)
- **Archive Retention**: Indefinite in `./worker/done/`

## Directory Structure

```
./worker/
├── queue/                    # Active job processing
│   ├── *_pending.pt         # Awaiting worker claim
│   ├── *_processing_by_dev*.pt # Currently being processed  
│   ├── *_complete_from_dev*.pt # Awaiting PS collection
│   └── *_collected.pt       # Claimed by PS
├── done/                    # Archived completed files
│   ├── *_processed.pt       # Successfully processed jobs
│   ├── *_consumed_*.pt      # Consumed by PS
│   ├── *_failed_*.pt        # Processing failures
│   └── *_orphaned_*.pt      # Cleanup artifacts
└── global_step.txt          # PS training progress
```

## Synchronization Points

### Weight Updates
- **Trigger**: After each optimizer step in PS
- **Method**: CUDA IPC handles for zero-copy weight sharing
- **Version Control**: `ps_ipc_handle_version.json` with timestamps

### Job Queue Coordination  
- **Flow Control**: Limit pending jobs to prevent memory issues
- **Load Balancing**: FIFO job claiming across multiple workers
- **Backpressure**: PS throttles job creation based on queue depth

## Failure Modes & Recovery

### Worker Failures
- **Detection**: Processing files without corresponding complete files
- **Recovery**: PS timeout mechanism archives stale processing files
- **Restart**: Workers can restart and immediately claim new jobs

### File System Issues
- **Corruption**: Atomic operations minimize corruption windows
- **Disk Full**: Graceful degradation with error logging
- **Permission Errors**: Comprehensive error handling and logging

## Monitoring & Debugging

### Key Metrics
- **Queue Depth**: Number of files in each state
- **Processing Latency**: Time from pending to complete
- **Worker Utilization**: Distribution of jobs across devices
- **Error Rates**: Failed/orphaned file counts

### Debug Files
- **Log Files**: `log_parameter_server.txt`, `log_worker_{N}.txt`
- **State Files**: `global_step.txt`, version tracking files
- **Archive Analysis**: `./worker/done/` for historical job data

## Best Practices

### For Development
1. **Monitor Queue Directories**: Watch for growing backlogs
2. **Check Log Files**: Regular scanning for error patterns  
3. **Validate Atomicity**: Ensure all file operations use `AtomicFileOps`
4. **Test Error Scenarios**: Simulate failures to verify recovery

### For Production
1. **Disk Space Management**: Regular cleanup of `./worker/done/`
2. **Performance Monitoring**: Track job processing latencies
3. **Error Alerting**: Monitor for increased failure rates
4. **Graceful Shutdown**: Ensure workers finish processing before exit

## Conclusion

The async-grpo file lifecycle implements a robust, fault-tolerant communication system between distributed training components. The combination of atomic file operations, comprehensive error handling, and systematic state management enables reliable high-throughput distributed training while maintaining data consistency and system resilience.

The file-based approach provides several advantages:
- **Transparency**: Easy to monitor and debug
- **Persistence**: Jobs survive process restarts  
- **Scalability**: Add workers without coordination complexity
- **Fault Tolerance**: Atomic operations prevent corruption
- **Simplicity**: No complex networking or message passing protocols

This design demonstrates how file systems can serve as effective coordination mechanisms for distributed machine learning workloads when implemented with proper atomic operations and error handling.