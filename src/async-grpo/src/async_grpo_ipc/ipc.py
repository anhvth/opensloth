import os
import json
import base64
from dataclasses import dataclass
from typing import Tuple, Optional, Sequence

import torch
import cupy as cp
from utils.logger import logger


DEFAULT_HANDLE_PATH = os.environ.get("CUDA_IPC_HANDLE_PATH", "./worker/ps_ipc_handle.json")

# CUDA memcpy kind enum value for device-to-device; stable across versions
CUDA_MEMCPY_DEVICE_TO_DEVICE = 3


@dataclass
class RemoteBundle:
    """Bundle of multiple RemoteMemory objects for multi-dtype tensor storage."""
    buffers: dict         # dtype_key -> RemoteMemory
    serialization_meta: list

    def close(self) -> None:
        for rm in self.buffers.values():
            rm.close()

    def pull_serialized_tensors(self, stream):
        """Pull and deserialize tensors using fixed deserialize_from_buckets."""
        from .process_tensors import deserialize_from_buckets
        local = {}
        target_device = None
        
        for k, rm in self.buffers.items():
            target_device = rm.opened_on_device  # Use worker's device
            t = torch.empty(rm.shape, dtype=rm.dtype, device=f"cuda:{target_device}")
            rm.pull_to_tensor(t, stream)
            local[k] = t
            
        return deserialize_from_buckets(local, self.serialization_meta, target_device=f"cuda:{target_device}")


def _tensor_nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def export_serialized_tensors_handle(flat_or_buckets, meta, path: str = DEFAULT_HANDLE_PATH) -> None:
    """
    Export a CUDA IPC memory handle for a flat tensor or dict of dtype buckets with serialization metadata.
    - For dict input: Export bundle format with per-dtype buffers  
    - For tensor input: Export legacy single flat tensor format
    - Metadata contains information to deserialize back to original tensors.
    """
    def _encode(t: torch.Tensor) -> dict:
        assert t.is_cuda, "export_serialized_tensors_handle expects a CUDA tensor"
        ptr = t.data_ptr()
        handle_bytes = cp.cuda.runtime.ipcGetMemHandle(ptr)
        logger.debug(f"Successfully created IPC handle for tensor on device {t.device.index}")
        handle_b64 = base64.b64encode(handle_bytes).decode("ascii")
        return {
            "handle_b64": handle_b64,
            "nbytes": _tensor_nbytes(t),
            "dtype": str(t.dtype).replace("torch.", ""),
            "shape": list(t.shape),
            "device_owner": t.device.index,
        }

    if isinstance(flat_or_buckets, dict):
        # Bundle format: multiple dtype buckets
        payload = {
            "buffers": {k: _encode(v) for k, v in flat_or_buckets.items()},
            "serialization_meta": meta
        }
        logger.info(f"Exporting bundle with {len(flat_or_buckets)} dtype buckets: {list(flat_or_buckets.keys())}")
    else:
        # Legacy format: single flat tensor
        payload = {**_encode(flat_or_buckets), "serialization_meta": meta}
        logger.info("Exporting legacy single tensor format")

    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _open_ipc_handle(path: str) -> Tuple[bytes, int, int, Sequence[int], torch.dtype, Optional[list]]:
    with open(path, "r") as f:
        meta = json.load(f)
    handle_b64 = meta["handle_b64"]
    nbytes = int(meta["nbytes"])
    owner_dev = int(meta.get("device_owner", 0))
    handle_bytes = base64.b64decode(handle_b64)
    shape = tuple(meta.get("shape", []))
    dtype_str = meta.get("dtype", "float32")
    torch_dtype = getattr(torch, dtype_str)
    serialization_meta = meta.get("serialization_meta", None)
    return handle_bytes, nbytes, owner_dev, shape, torch_dtype, serialization_meta


def create_stream(non_blocking: bool = True, device: Optional[int] = None) -> cp.cuda.Stream:
    """Create a CUDA stream on the specified device."""
    if device is not None:
        with cp.cuda.Device(device):
            return cp.cuda.Stream(non_blocking=non_blocking)
    return cp.cuda.Stream(non_blocking=non_blocking)


def memcpy_d2d_async(dst_ptr: int, src_ptr: int, nbytes: int, stream: cp.cuda.Stream) -> None:
    """
    Device-to-device async memory copy with peer access enabling and host fallback.
    """
    try:
        cp.cuda.runtime.memcpyAsync(dst_ptr, src_ptr, nbytes, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.ptr)
    except Exception as e:
        # If direct D2D copy fails, try enabling peer access or fall back to host copy
        logger.warning(f"Direct D2D copy failed (dst=0x{dst_ptr:x}, src=0x{src_ptr:x}, {nbytes}B): {e}")
        logger.info("Attempting peer access setup...")
        
        # Try to enable peer access between devices
        try:
            current_device = cp.cuda.runtime.getDevice()
            device_count = cp.cuda.runtime.getDeviceCount()
            
            for device_id in range(device_count):
                if device_id != current_device:
                    try:
                        can_access = cp.cuda.runtime.deviceCanAccessPeer(current_device, device_id)
                        if can_access:
                            cp.cuda.runtime.deviceEnablePeerAccess(device_id, 0)
                            logger.info(f"Enabled peer access from device {current_device} to {device_id}")
                    except Exception as peer_e:
                        if "peer access is already enabled" in str(peer_e).lower():
                            logger.info(f"Peer access already enabled: {current_device} -> {device_id}")
                        else:
                            logger.warning(f"Failed to enable peer access {current_device} -> {device_id}: {peer_e}")
            
            # Retry the D2D copy
            cp.cuda.runtime.memcpyAsync(dst_ptr, src_ptr, nbytes, CUDA_MEMCPY_DEVICE_TO_DEVICE, stream.ptr)
            logger.success(f"D2D copy succeeded after peer access setup ({nbytes}B)")
            
        except Exception as e2:
            logger.error(f"D2D copy still failed after peer access attempt: {e2}")
            logger.warning(f"Falling back to host-pinned copy ({nbytes}B)...")
            
            # Fall back to host-pinned copy (slower but should work)
            try:
                # Use canonical CUDA runtime names
                host_buffer = cp.cuda.runtime.hostAlloc(nbytes, 0)  # flags=0 for default
                try:
                    # Device to host
                    cp.cuda.runtime.memcpyAsync(host_buffer, src_ptr, nbytes, 2, stream.ptr)  # D2H = 2
                    stream.synchronize()
                    
                    # Host to device  
                    cp.cuda.runtime.memcpyAsync(dst_ptr, host_buffer, nbytes, 1, stream.ptr)  # H2D = 1
                    logger.success(f"Completed host-pinned fallback copy ({nbytes}B)")
                finally:
                    cp.cuda.runtime.freeHost(host_buffer)
            except Exception as e3:
                logger.error(f"Host-pinned fallback also failed: {e3}")
                raise RuntimeError(f"All memory copy strategies failed: D2D={e}, retry={e2}, host={e3}")


@dataclass
class RemoteMemory:
    """Represents an IPC-mapped remote memory block."""

    ptr: int
    nbytes: int
    owner_device: int
    opened_on_device: int
    shape: Tuple[int, ...]
    dtype: torch.dtype
    serialization_meta: Optional[list] = None

    def close(self) -> None:
        try:
            with cp.cuda.Device(self.opened_on_device):
                cp.cuda.runtime.ipcCloseMemHandle(self.ptr)
        except Exception as e:
            # Ignore "already closed" or similar errors
            if "invalid handle" not in str(e).lower():
                logger.warning(f"Warning during IPC close: {e}")

    def push_from_tensor(self, src: torch.Tensor, stream: cp.cuda.Stream) -> None:
        assert src.is_cuda, "push_from_tensor expects CUDA tensor"
        assert src.element_size() * src.numel() == self.nbytes, "size mismatch"
        memcpy_d2d_async(self.ptr, src.data_ptr(), self.nbytes, stream)

    def pull_to_tensor(self, dst: torch.Tensor, stream: cp.cuda.Stream) -> None:
        assert dst.is_cuda, "pull_to_tensor expects CUDA tensor"
        assert dst.element_size() * dst.numel() == self.nbytes, "size mismatch"
        memcpy_d2d_async(dst.data_ptr(), self.ptr, self.nbytes, stream)

    def push_serialized_tensors(self, tensors: list, stream: cp.cuda.Stream) -> None:
        """Push a list of tensors by serializing them first."""
        from .process_tensors import serialize_tensors
        flat_tensor, _ = serialize_tensors(tensors)
        self.push_from_tensor(flat_tensor, stream) # type: ignore

    def pull_serialized_tensors(self, stream: cp.cuda.Stream) -> list:
        """Pull and deserialize tensors back to a list."""
        from .process_tensors import deserialize_from_buckets
        
        assert self.serialization_meta is not None, "No serialization metadata available"
        
        # Create a flat tensor to receive the data
        flat_tensor = torch.empty(self.shape, dtype=self.dtype, device=f"cuda:{self.opened_on_device}")
        self.pull_to_tensor(flat_tensor, stream)
        
        # Create a single-entry bucket dict for deserialization
        dtype_key = str(self.dtype).replace("torch.", "")
        buckets = {dtype_key: flat_tensor}

        # Deserialize back to original tensors
        return deserialize_from_buckets(buckets, self.serialization_meta)


def open_remote_memory(path: str = DEFAULT_HANDLE_PATH, target_device: Optional[int] = None):
    """
    Open remote IPC memory, supporting both bundle and legacy formats.
    Returns RemoteBundle for bundle format or RemoteMemory for legacy format.
    Respects target_device argument and removes hardcoded device assumptions.
    """
    with open(path, "r") as f:
        meta = json.load(f)

    if target_device is None:
        try:
            target_device = cp.cuda.runtime.getDevice()
        except Exception:
            target_device = 0
    
    # Ensure target_device is always an int after the check above
    assert isinstance(target_device, int), f"target_device must be int, got {type(target_device)}"
    
    # Use device context for all IPC operations
    with cp.cuda.Device(target_device):
        flags = getattr(cp.cuda.runtime, "cudaIpcMemLazyEnablePeerAccess", 0)

        if "buffers" in meta:  # bundle path
            buffers_cfg = meta["buffers"]
            bufs = {}
            for k, cfg in buffers_cfg.items():
                handle_bytes = base64.b64decode(cfg["handle_b64"])
                remote_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes, flags)
                bufs[k] = RemoteMemory(
                    ptr=remote_ptr,
                    nbytes=int(cfg["nbytes"]),
                    owner_device=int(cfg.get("device_owner", 0)),
                    opened_on_device=target_device,
                    shape=tuple(cfg.get("shape", [])),
                    dtype=getattr(torch, cfg.get("dtype", "float32")),
                    serialization_meta=None,
                )
            logger.info(f"Opened bundle with {len(bufs)} dtype buffers on device {target_device}: {list(bufs.keys())}")
            return RemoteBundle(buffers=bufs, serialization_meta=meta.get("serialization_meta", []))

        # legacy single-buffer
        handle_bytes, nbytes, owner_dev, shape, torch_dtype, serialization_meta = _open_ipc_handle(path)
        remote_ptr = cp.cuda.runtime.ipcOpenMemHandle(handle_bytes, flags)
        logger.info(f"Opened legacy single buffer on device {target_device}")
        return RemoteMemory(
            ptr=remote_ptr,
            nbytes=nbytes,
            owner_device=owner_dev,
            opened_on_device=target_device,
            shape=tuple(shape),
            dtype=torch_dtype,
            serialization_meta=serialization_meta,
        )
