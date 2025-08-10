def get_callback_and_setup_method():
    # Localized imports (project rule: avoid global Unsloth / heavy lib import side-effects)
    import os

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
                    "NCCL distributed training not initialized. Call torch.distributed.init_process_group() first."
                )

        def _sync_gradients(self, model: torch.nn.Module) -> None:
            """Synchronize gradients across all ranks using NCCL all-reduce."""

            name_value = {}
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(self.world_size)
                name_value[name] = param.grad.mean().item()

            # print(f"[RANK={self.local_rank}] Gradient sync complete: {name_value}")
            # import ipdb; ipdb.set_trace()

        def on_pre_optimizer_step(self, *_, **__):
            """Called before optimizer step - synchronize gradients."""
            # Synchronize gradients across all ranks
            self._sync_gradients(self.model)

    # ============== END OF CLASS

    def setup_nccl_for_opensloth(rank: int, gpus: list) -> None:
        """Setup NCCL process group for OpenSloth.

        Safe against double-initialization (some backends like Unsloth / vLLM
        may already have initialized torch.distributed). If already initialized
        we validate rank & world size match expectations and skip.
        """
        if len(gpus) <= 1:
            return

        expected_world_size = len(gpus)

        # If already initialized, verify compatibility and bail out early (or repair if safe).
        if dist.is_available() and dist.is_initialized():
            try:
                current_world_size = dist.get_world_size()
                current_rank = dist.get_rank()
            except Exception as exc:
                raise RuntimeError("Distributed backend initialized but rank/world size unknown.") from exc

            # Repair scenario: a local (world_size=1) group was auto-created (eg by Unsloth)
            # before spawning multi-GPU ranks. We can safely tear it down and re-init with
            # the correct multi-rank configuration.
            if current_world_size == 1 and expected_world_size > 1:
                if rank == 0:
                    print(
                        "[OpenSloth NCCL] Detected pre-initialized single-rank process group; "
                        f"reinitializing for world_size={expected_world_size}."
                    )
                import contextlib
                with contextlib.suppress(Exception):
                    dist.destroy_process_group()
            else:
                # Normal validation path
                if current_world_size != expected_world_size or current_rank != rank:
                    raise RuntimeError(
                        "Existing torch.distributed process group has mismatched configuration: "
                        f"(rank={current_rank}, world_size={current_world_size}) vs expected "
                        f"(rank={rank}, world_size={expected_world_size})."
                    )
                print(f"[RANK={rank}] torch.distributed already initialized; skipping NCCL init.")
                return

        # Set required NCCL environment variables (idempotent)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")  # Localhost for single machine
        os.environ.setdefault("MASTER_PORT", "29501")

        print(f"[RANK={rank}] Initializing NCCL with world_size={expected_world_size}")
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=expected_world_size)

    def destroy_nccl_if_initialized():
        """Gracefully destroy the NCCL process group if it was created.

        Avoids PyTorch warnings about leaked resources.
        """
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            import contextlib

            with contextlib.suppress(Exception):
                dist.destroy_process_group()

    return NCCLGradSyncCallback, setup_nccl_for_opensloth, destroy_nccl_if_initialized
