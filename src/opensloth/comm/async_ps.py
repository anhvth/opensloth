"""Asynchronous Parameter-Server backend using PyTorch RPC.

This module provides an optional alternative to synchronous NCCL all-reduce.
It focuses on synchronizing only *trainable* (e.g., LoRA) parameters to reduce
communication volume. Workers push flattened gradients to a rank 0 parameter
server which applies updates with its own optimizer. Workers periodically pull
fresh parameters based on a configurable interval and staleness threshold.

Design notes:
- We avoid transmitting CUDA tensors (RPC CPU-only guarantee) by flattening on CPU.
- Only parameters with requires_grad=True and with gradients are synced.
- Worker optimizers can have learning rate zeroed (drop_local_lr=True) so only server updates apply.
- Backpressure is enforced with max_inflight_rpcs to avoid unbounded memory.
- A simple SGD optimizer is used server-side (could be extended later).

Limitations / Future work:
- No fault tolerance / retry logic yet.
- No adaptive staleness; simple max_staleness check.
- Not integrated with distributed autograd (manual grad push model).
"""
from __future__ import annotations

import os
import threading
import time
from typing import List, Optional

import torch
import torch.distributed.rpc as rpc
from torch import nn
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

from opensloth.opensloth_config import AsyncPSConfig
from opensloth.logging_config import get_opensloth_logger

logger = get_opensloth_logger()

# --------------------------- Parameter Server ---------------------------

class ParameterServer:
    """Holds master copy of trainable parameter vector and applies updates."""
    def __init__(self, param_shapes: List[torch.Size], lr: float):
        self.param_shapes = param_shapes
        self.lr = lr
        self.lock = threading.Lock()
        self.version = 0
        self._last_update_ts = time.time()
        # Allocate flat parameter storage on CPU
        total_elems = sum(int(torch.tensor(1).new_empty(s).numel()) for s in param_shapes)
        self.flat_params = torch.zeros(total_elems, dtype=torch.float32)
        logger.info(f"[AsyncPS] Server initialized with flat size {total_elems}")

    # RPC-exposed methods
    def get_params(self) -> tuple[torch.Tensor, int]:  # returns (flat_params, version)
        with self.lock:
            return self.flat_params.clone(), self.version

    def apply_gradients(self, flat_grads: torch.Tensor, worker_version: int, worker_rank: int) -> int:
        # Simple SGD update: p -= lr * g
        if flat_grads.dtype != self.flat_params.dtype:
            flat_grads = flat_grads.to(self.flat_params.dtype)
        with self.lock:
            self.flat_params -= self.lr * flat_grads
            self.version += 1
            self._last_update_ts = time.time()
            v = self.version
        # Lightweight logging (avoid spam)
        if v % 50 == 0:
            logger.debug(f"[AsyncPS] Applied gradients -> new version {v} (from worker {worker_rank}, worker_version={worker_version})")
        return v

# --------------------------- RPC Setup / Teardown ---------------------------

_GLOBAL_SERVER_RREF = None
_RPC_INITIALIZED = False


def _init_parameter_server_if_needed(rank: int, model: nn.Module, lr: float) -> None:
    global _GLOBAL_SERVER_RREF
    if rank == 0:
        # Extract trainable param shapes from model (LoRA or full)
        param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
        _GLOBAL_SERVER_RREF = rpc.remote(  # type: ignore
            to=0, func=ParameterServer, args=(param_shapes, lr)
        )
        logger.info("[AsyncPS] ParameterServer RRef created on rank 0")
    else:
        # Wait until server RRef is available via rpc to rank 0
        # Poll by querying an RPC that returns a bool until success
        # For simplicity, we just sleep; server will service get_params calls
        timeout_s = 30
        start = time.time()
        while _GLOBAL_SERVER_RREF is None:
            if time.time() - start > timeout_s:
                raise RuntimeError("Timeout waiting for server RRef initialization")
            time.sleep(0.1)


def setup_rpc(rank: int, world_size: int, master_addr: str = "127.0.0.1", master_port: str = "29512") -> None:
    global _RPC_INITIALIZED
    if _RPC_INITIALIZED:
        return
    os.environ.setdefault("MASTER_ADDR", master_addr)
    os.environ.setdefault("MASTER_PORT", master_port)
    rpc.init_rpc(name=f"worker{rank}", rank=rank, world_size=world_size)
    _RPC_INITIALIZED = True
    logger.info(f"[AsyncPS] RPC initialized on rank {rank}/{world_size}")


def shutdown_rpc() -> None:
    global _RPC_INITIALIZED
    if _RPC_INITIALIZED:
        try:
            rpc.shutdown()
            logger.info("[AsyncPS] RPC shutdown complete")
        finally:
            _RPC_INITIALIZED = False

# --------------------------- Trainer Callback ---------------------------

class AsyncPSCallback(TrainerCallback):
    def __init__(self, model: nn.Module, rank: int, world_size: int, config: AsyncPSConfig):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.cfg = config
        self.param_refs: List[nn.Parameter] = [p for p in model.parameters() if p.requires_grad]
        self.shapes = [p.shape for p in self.param_refs]
        self.numel_cumsum = self._compute_cumsum(self.shapes)
        self.local_version = 0
        self.last_pull_version = 0
        self.inflight: List[torch.futures.Future] = []
        # Optionally zero out local optimizer LR via Trainer hook later
        logger.info(f"[AsyncPS] Callback initialized on rank {rank} with {len(self.param_refs)} trainable params")

    @staticmethod
    def _compute_cumsum(shapes: List[torch.Size]) -> List[int]:
        cumsum = [0]
        total = 0
        for s in shapes:
            numel = 1
            for d in s:
                numel *= d
            total += numel
            cumsum.append(total)
        return cumsum

    def _flatten_grads(self) -> Optional[torch.Tensor]:
        grads = []
        for p in self.param_refs:
            if p.grad is None:
                return None  # skip push if any grad missing (e.g., unused)
            grads.append(p.grad.detach().float().cpu().view(-1))
        if not grads:
            return None
        return torch.cat(grads, dim=0)

    def _apply_flat_params(self, flat_params: torch.Tensor):
        # Copy into existing parameter tensors in-place (on model device)
        for i, p in enumerate(self.param_refs):
            start = self.numel_cumsum[i]
            end = self.numel_cumsum[i+1]
            slice_flat = flat_params[start:end]
            # reshape and copy to param device
            reshaped = slice_flat.view(p.shape).to(p.data.device)
            p.data.copy_(reshaped)

    # Hook before optimizer step (gradients ready)
    def on_pre_optimizer_step(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore
        if self.rank == 0:
            # Server does not push gradients upstream; local optimizer may be disabled
            return control
        flat = self._flatten_grads()
        if flat is None:
            return control
        # Backpressure
        self.inflight = [f for f in self.inflight if not f.done()]
        if len(self.inflight) >= self.cfg.max_inflight_rpcs:
            # Wait oldest
            self.inflight[0].wait()
            self.inflight = [f for f in self.inflight if not f.done()]
        # Send gradients to server rank 0
        fut = rpc.rpc_async(to=0, func=_ps_apply_grads, args=(flat, self.local_version, self.rank))
        self.inflight.append(fut)
        self.local_version += 1
        return control

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):  # type: ignore
        if self.rank == 0:
            return control
        # Periodic pull OR forced if stale
        need_pull = False
        if state.global_step % self.cfg.pull_every_n_steps == 0:
            need_pull = True
        if (self.local_version - self.last_pull_version) > self.cfg.max_staleness:
            need_pull = True
        if need_pull:
            flat_params, server_version = rpc.rpc_sync(to=0, func=_ps_get_params, args=())
            self._apply_flat_params(flat_params)
            self.last_pull_version = self.local_version
            if server_version % 50 == 0:
                logger.debug(f"[AsyncPS] Pulled params at server_version={server_version} local_version={self.local_version}")
        return control

# --------------------------- RPC Helper Functions ---------------------------

# These top-level functions are required because functions passed to rpc.{rpc_async,rpc_sync}
# must be picklable; bound instance methods would require RRef-based indirection.

_SERVER_INSTANCE: Optional[ParameterServer] = None


def _ensure_server_instance() -> ParameterServer:
    global _SERVER_INSTANCE
    if _SERVER_INSTANCE is None:
        raise RuntimeError("ParameterServer instance not initialized on this rank")
    return _SERVER_INSTANCE

def _ps_get_params():  # executed on server rank (0)
    return _ensure_server_instance().get_params()

def _ps_apply_grads(flat_grads: torch.Tensor, worker_version: int, worker_rank: int):  # executed on server rank (0)
    return _ensure_server_instance().apply_gradients(flat_grads, worker_version, worker_rank)

# Monkey-patch server creation once RPC init finished on rank 0
# This helper should be invoked by training setup after RPC init and model creation.

def initialize_server_if_rank0(model: nn.Module, rank: int, async_cfg: AsyncPSConfig):
    global _SERVER_INSTANCE
    if rank == 0:
        if _SERVER_INSTANCE is None:
            param_shapes = [p.shape for p in model.parameters() if p.requires_grad]
            _SERVER_INSTANCE = ParameterServer(param_shapes, async_cfg.server_lr)
            logger.info("[AsyncPS] In-process ParameterServer instantiated on rank 0")

__all__ = [
    "setup_rpc",
    "shutdown_rpc",
    "AsyncPSCallback",
    "initialize_server_if_rank0",
]
