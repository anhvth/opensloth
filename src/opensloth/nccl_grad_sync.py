def get_callback_and_setup_method():

    import torch
    import torch.distributed as dist
    from transformers.trainer_callback import TrainerCallback

    class NCCLGradSyncCallback(TrainerCallback):
        """NCCL-based gradient synchronization callback for Transformers trainer.

        This callback provides the same interface as MmapGradSyncCallback but uses
        NCCL for gradient synchronization instead of memory-mapped files.
        """

        def __init__(
            self,
            model,
            gpu: int,
            gpus: list[int],
        ):
            self.model = model
            self.gpu = gpu
            self.gpus = gpus
            self.local_rank = gpus.index(gpu)
            self.world_size = len(gpus)

            # Ensure distributed is initialized
            if not dist.is_initialized():
                raise RuntimeError(
                    "NCCL distributed training not initialized. "
                    "Call torch.distributed.init_process_group() first."
                )

        def _sync_gradients(self, model: torch.nn.Module) -> None:
            """Synchronize gradients across all ranks using NCCL all-reduce."""

            for _, param in model.named_parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)

    def on_pre_optimizer_step(self, *_, **__):
            """Called before optimizer step - synchronize gradients."""
            # Synchronize gradients across all ranks
            self._sync_gradients(self.model)

    # Add this integration code for opensloth at the end of the file

    def setup_nccl_for_opensloth(rank: int, gpus: list) -> None:
        """Setup NCCL environment variables for opensloth integration."""
        if len(gpus) <= 1:
            return
        import os

        import torch.distributed as dist

        world_size = len(gpus)

        # Set required NCCL environment variables
        os.environ["MASTER_ADDR"] = "127.0.0.1"  # Localhost for single machine
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29501"  # Use fixed port
        print(f"[RANK={rank}] {os.environ}")
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )

    return NCCLGradSyncCallback, setup_nccl_for_opensloth
