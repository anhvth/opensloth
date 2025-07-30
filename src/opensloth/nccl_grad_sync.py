def get_callback_and_setup_method():

    import os
    import socket
    import torch
    import torch.distributed as dist

    from typing import List
    from transformers.trainer_callback import TrainerCallback
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

    class NCCLGradSyncCallback(TrainerCallback):
        """NCCL-based gradient synchronization callback for Transformers trainer.

        This callback provides the same interface as MmapGradSyncCallback but uses
        NCCL for gradient synchronization instead of memory-mapped files.
        """

        def __init__(
            self,
            model,
            gpu: int,
            gpus: List[int],
        ):
            self.model = model
            self.gpu = gpu
            self.gpus = gpus
            self.local_rank = gpus.index(gpu)
            self.world_size = len(gpus)
            self.gradient_accumulation_steps = 1
            self.current_step = 0

            # Ensure distributed is initialized
            if not dist.is_initialized():
                raise RuntimeError(
                    "NCCL distributed training not initialized. "
                    "Call torch.distributed.init_process_group() first."
                )

            # Determine the reduction operation to use Prefer AVG if available (PyTorch >= 1.8.0)
            self.reduce_op = getattr(dist.ReduceOp, 'AVG', dist.ReduceOp.SUM)
            self.use_avg_op = (self.reduce_op == dist.ReduceOp.AVG)
            if not self.use_avg_op:
                print(f"[GPU {self.gpu}] dist.ReduceOp.AVG not available.Falling back to SUM and manual division.")

        def on_init_end(self, args, state, control, **kwargs):
            """Store gradient accumulation steps from training arguments."""
            self.gradient_accumulation_steps = args.gradient_accumulation_steps
            print(f"[GPU {self.gpu}] Gradient accumulation steps set to: {self.gradient_accumulation_steps}")


        def _flatten_dense_tensors(tensors):
            return torch.cat([t.contiguous().view(-1) for t in tensors], dim=0)

        def _unflatten_dense_tensors(flat_tensor, tensors):
            outputs = []
            offset = 0
            for tensor in tensors:
                numel = tensor.numel()
                outputs.append(flat_tensor.narrow(0, offset, numel).view_as(tensor))
                offset += numel
            return outputs

        def _sync_gradients(self, model: torch.nn.Module) -> None:
            """Synchronize gradients across all ranks using NCCL all-reduce."""

            grads = [p.grad for p in model.parameters() if p.grad is not None]

            if not grads:
                return

            flat_grads = _flatten_dense_tensors(grads)

            dist.all_reduce(flat_grads, op=self.reduce_op)

            if not self.use_avg_op:
                flat_grads.div_(self.world_size)

            for old_tensor, new_tensor in zip(grads, _unflatten_dense_tensors(flat_grads, grads)):
                old_tensor.copy_(new_tensor)

        def on_step_end(self, args, state, control, **kwargs) -> None:
            """Track steps for gradient accumulation."""
            self.current_step = state.global_step + 1

        def on_pre_optimizer_step(self, args, state, control, **kwargs) -> None:
            """Called before optimizer step - synchronize gradients and only synchronize gradients after completing the accumulation cycle"""
            if (self.current_step % self.gradient_accumulation_steps) == 0:
                self._sync_gradients(self.model)

        def on_train_end(self, args, state, control, **kwargs) -> None:
            """Called at the end of training to properly clean up distributed process group."""
            if dist.is_initialized():
                dist.destroy_process_group()
                print(f"[GPU {self.gpu}] Distributed process group destroyed.")

    def setup_nccl_for_opensloth(rank: int, gpus: list) -> None:
        """Setup NCCL environment variables for opensloth integration."""
        if len(gpus) <= 1:
            return

        world_size = len(gpus)

        # Force NCCL to use IPv4 sockets only if no IPv6 support
        if not socket.has_ipv6:
            os.environ.setdefault('NCCL_SOCKET_FAMILY', 'AF_INET')

        # Set required NCCL environment variables for single machine
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
        os.environ["MASTER_ADDR"] = "127.0.0.1"

        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29501"  # Use fixed port

        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

        # Reduce to current os variables
        relevant_vars = ['NCCL_SOCKET_FAMILY', 'NCCL_SOCKET_IFNAME', 'MASTER_ADDR', 'MASTER_PORT',
                         'NCCL_DEBUG', 'NCCL_MIN_NRINGS', 'NCCL_ALG', 'NCCL_NVLS_ENABLE']
        env_info = {var: os.environ.get(var) for var in relevant_vars if var in os.environ}
        print(f"[RANK={rank}] NCCL Environment setup: {env_info}")

    return NCCLGradSyncCallback, setup_nccl_for_opensloth