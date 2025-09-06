from typing import List, Dict, Any
import torch
from utils.logger import logger


def get_trainable_params(model):
    """Get all trainable parameters from a model as a list of tensors."""
    trainable_params = []
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param.data)
    return trainable_params


def serialize_tensors(tensors: List):
    """
    Serialize tensors into per-dtype buckets using CuPy allocations for IPC compatibility.
    Returns (buckets: Dict[str, torch.Tensor], meta: List[dict]).
    """
    import cupy as cp
    if not tensors:
        return {}, []
    
    def _dtype_key(t):
        """Convert tensor dtype to string key for bucketing."""
        return str(t.dtype).replace("torch.", "")  # e.g., "bfloat16"
    
    # Group by dtype
    groups = {}
    for idx, t in enumerate(tensors):
        dtype_key = _dtype_key(t)
        if dtype_key not in groups:
            groups[dtype_key] = []
        groups[dtype_key].append((idx, t))

    # Allocate flat per dtype using CuPy for IPC compatibility, then convert to PyTorch
    buckets = {}
    sizes = {k: sum(x.numel() for _, x in lst) for k, lst in groups.items()}
    devices = {k: lst[0][1].device for k, lst in groups.items()}
    
    for k, total_numel in sizes.items():
        device = devices[k]
        device_id = device.index
        
        # Use CuPy to allocate IPC-compatible memory
        with cp.cuda.Device(device_id):
            # Map PyTorch dtypes to CuPy dtypes for allocation
            if k == "bfloat16":
                # CuPy doesn't have bfloat16, use uint16 and reinterpret
                cupy_array = cp.empty(total_numel, dtype=cp.uint16)
            elif k == "float16":
                cupy_array = cp.empty(total_numel, dtype=cp.float16)
            elif k == "float32":
                cupy_array = cp.empty(total_numel, dtype=cp.float32)
            elif k == "float64":
                cupy_array = cp.empty(total_numel, dtype=cp.float64)
            else:
                # Default fallback to float32
                cupy_array = cp.empty(total_numel, dtype=cp.float32)
            
            # Convert CuPy array to PyTorch tensor using DLPack for zero-copy
            torch_tensor = torch.from_dlpack(cupy_array.toDlpack()) # type: ignore
            
            if k == "bfloat16":
                # Reinterpret uint16 as bfloat16 BEFORE fill to avoid dtype-cast copies
                torch_tensor = torch_tensor.view(torch.bfloat16)
            elif k != "float32":  # Ensure correct dtype
                target_dtype = getattr(torch, k)
                torch_tensor = torch_tensor.to(dtype=target_dtype)
                
            buckets[k] = torch_tensor
        
        logger.debug(f"Allocated {k} bucket: {total_numel} elements on {device} (via CuPy+DLPack)")

    # Fill buckets and build meta with device info
    meta: List[Dict[str, Any]] = [{}] * len(tensors)  # Initialize with empty dicts
    with torch.no_grad():
        for k, lst in groups.items():
            offset = 0
            flat = buckets[k]
            device = devices[k]
            for idx, t in lst:
                n = t.numel()
                flat[offset:offset+n].copy_(t.flatten())
                meta[idx] = {
                    'shape': list(t.shape), 
                    'dtype': k, 
                    'numel': n, 
                    'bucket': k, 
                    'start': offset,
                    'device': str(device)  # Record source device for completeness
                }
                offset += n

    logger.debug(f"Created {len(buckets)} dtype buckets for {len(tensors)} tensors")
    return buckets, meta


def deserialize_from_buckets(buckets, meta, target_device=None):
    """
    Deserialize tensors from per-dtype buckets back to original tensor list.
    """
    if not meta:
        return []
    
    # For legacy cases with a single unnamed bucket.
    single_bucket_key = list(buckets.keys())[0] if len(buckets) == 1 else None

    tensors = []
    for tensor_meta in meta:
        # Use 'bucket' key if present, otherwise fall back to the single bucket key
        bucket_key = tensor_meta.get('bucket', single_bucket_key)
        if bucket_key is None:
            raise ValueError("Cannot determine bucket for tensor: meta is missing 'bucket' key and more than one bucket was provided.")
        
        start = tensor_meta['start']
        numel = tensor_meta['numel']
        shape = tensor_meta['shape']
        
        # Extract from bucket
        if bucket_key not in buckets:
            raise ValueError(f"Bucket '{bucket_key}' not found in provided buckets: {list(buckets.keys())}")
        bucket = buckets[bucket_key]
        tensor_data = bucket[start:start + numel]
        
        # Reshape and move to target device if specified
        tensor = tensor_data.reshape(shape)
        if target_device is not None:
            tensor = tensor.to(device=target_device)
        
        tensors.append(tensor)
    
    return tensors


# Element sizes for different dtypes (in bytes)
_ELEM_SIZE = {
    "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "bool": 1,
}


def stream_set_trainable_params(model, remote_memory_or_bundle, stream):
    """
    Zero-copy streaming: copy directly from IPC-mapped bucket(s) into model param storage.
    Supports RemoteBundle (preferred) and legacy RemoteMemory (requires contiguous, single-dtype meta).
    """
    from async_grpo_ipc.ipc import memcpy_d2d_async
    import torch

    # Extract meta & bucket ptrs
    if hasattr(remote_memory_or_bundle, "buffers"):  # RemoteBundle
        meta = remote_memory_or_bundle.serialization_meta
        bucket_ptrs = {k: rm.ptr for k, rm in remote_memory_or_bundle.buffers.items()}
    else:  # RemoteMemory legacy single buffer
        meta = remote_memory_or_bundle.serialization_meta
        bucket_ptrs = {"__single__": remote_memory_or_bundle.ptr}
        # If meta lacks 'bucket'/'start', we fall back to cumulative offsets below.

    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) != len(meta):
        raise ValueError(f"Param/meta length mismatch: {len(params)} vs {len(meta)}")

    cumulative_bytes = 0  # legacy path
    with torch.no_grad():
        for p, m in zip(params, meta):
            dtype_key = m.get("bucket", m.get("dtype", "__single__"))
            elem_size = _ELEM_SIZE.get(dtype_key, _ELEM_SIZE.get(str(p.dtype).replace("torch.", ""), 4))
            nbytes = p.numel() * elem_size

            if "start" in m and dtype_key in bucket_ptrs:
                # New bucketed format: use per-param start offset
                src_ptr = bucket_ptrs[dtype_key] + m["start"] * elem_size
            else:
                # Legacy linear layout (no per-param start); assume contiguous in meta order
                src_ptr = bucket_ptrs["__single__"] + cumulative_bytes

            dst_ptr = p.data.data_ptr()
            memcpy_d2d_async(dst_ptr, src_ptr, nbytes, stream)

            if "start" not in m:
                cumulative_bytes += nbytes

    return len(params)