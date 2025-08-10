"""Trainer factory utilities for creating trainers (SFT, DPO, GRPO, etc.)."""
# ruff: noqa: I001

from __future__ import annotations
import os


from collections.abc import Iterable
from .logging_config import get_opensloth_logger
from .opensloth_config import OpenSlothConfig, TrainingArguments


# --------------------------- Helpers ---------------------------

def _ensure_data_correct_for_training_type(dataset, training_type: str) -> None:
    logger = get_opensloth_logger()
    if training_type == "sft":
        feats = getattr(dataset, "features", None)
        if feats is None or "input_ids" not in feats:
            raise ValueError("SFT dataset must contain 'input_ids'.")
        if "labels" not in feats:
            logger.warning("SFT dataset missing 'labels' column (may be OK if trainer builds it).")
    elif training_type == "dpo":
        required = ["prompt", "chosen", "rejected"]
        feats = getattr(dataset, "features", None)
        if feats is None:
            raise ValueError("DPO dataset must expose features.")
        missing = [c for c in required if c not in feats]
        if missing:
            raise ValueError(f"DPO dataset missing required columns {missing}; has {list(feats.keys())}")
    else:
        raise NotImplementedError(f"Validation for training_type={training_type} not implemented")


# --------------------------- Trainer constructors ---------------------------

def _create_sft_trainer(
    model,
    tokenizer,
    train_dataset,
    _cfg: OpenSlothConfig,  # kept for uniform signature
    hf_train_args: TrainingArguments,
):
    from transformers import DataCollatorForSeq2Seq
    from trl import SFTTrainer

    _ensure_data_correct_for_training_type(train_dataset, "sft")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    hf_train_args.skip_prepare_dataset = True  # type: ignore[attr-defined]
    return SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=hf_train_args,          # type: ignore[arg-type]
        tokenizer=tokenizer,         # type: ignore[arg-type]
        data_collator=data_collator,
    )


def _create_dpo_trainer(
    model,
    tokenizer,
    train_dataset,
    cfg: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    from trl import DPOTrainer, DPOConfig

    _ensure_data_correct_for_training_type(train_dataset, "dpo")
    dpo_args = cfg.dpo_args
    if dpo_args is None:
        raise ValueError("dpo_args must be provided for DPO training")
    d = hf_train_args.to_dict()
    d.update(
        {
            "beta": dpo_args.beta,
            "max_length": dpo_args.max_length,
            "max_prompt_length": dpo_args.max_prompt_length,
        }
    )
    dpo_config = DPOConfig(**d)
    return DPOTrainer(
        model=model,
        args=dpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )


def _create_kto_trainer(model, tokenizer, train_dataset, _cfg: OpenSlothConfig, _args: TrainingArguments):
    raise NotImplementedError("KTO training not implemented yet")


def _create_orpo_trainer(model, tokenizer, train_dataset, _cfg: OpenSlothConfig, _args: TrainingArguments):
    raise NotImplementedError("ORPO training not implemented yet")


# ---- GRPO ----

def _validate_grpo_dataset(ds) -> Iterable[str]:
    feats = getattr(ds, "features", None)
    if feats is None:
        raise ValueError("GRPO dataset must have features")
    keys = list(feats.keys())
    if "prompt" not in keys and "input_ids" not in keys:
        raise ValueError(f"GRPO dataset must have 'prompt' or 'input_ids'; has {keys}")
    return keys


def _create_grpo_trainer(
    model,
    tokenizer,
    train_dataset,
    cfg: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    grpo_args = cfg.grpo_args
    if grpo_args is None:
        raise ValueError("grpo_args must be provided for GRPO training")

    from trl import GRPOTrainer, GRPOConfig
    try:
        from .grpo_rewards import (
            get_reward_functions,
            create_reward_preset,
        )
    except Exception:
        # Fallback dummy reward
        def _dummy_reward(*args, **_kw):
            if len(args) > 1 and isinstance(args[1], list | tuple):
                return [0.0] * len(args[1])
            return [0.0]

        def get_reward_functions(_names):  # type: ignore
            class _RF:
                name = "dummy"
                def __call__(self, prompts, completions, **_):
                    return _dummy_reward(prompts, completions)
            return [_RF()]

        def create_reward_preset(_task_type):  # type: ignore
            return ["dummy"]

    _validate_grpo_dataset(train_dataset)

    cfg_dict = hf_train_args.to_dict()
    cfg_dict.pop("dataset_num_proc", None)
    cfg_dict.update(
        {
            "num_generations": grpo_args.group_size,
            "max_completion_length": grpo_args.max_new_tokens,
            "temperature": grpo_args.temperature,
            "top_p": grpo_args.top_p,
            "beta": grpo_args.kl_coef,
            "max_prompt_length": grpo_args.max_prompt_length,
        }
    )
    if grpo_args.top_k is not None:
        cfg_dict["top_k"] = grpo_args.top_k

    stop_sequences = list(grpo_args.stop_sequences)
    eos = getattr(tokenizer, "eos_token", None)
    if eos and eos not in stop_sequences:
        stop_sequences.append(eos)
    generation_kwargs = {"do_sample": True, "pad_token_id": eos}
    if stop_sequences:
        generation_kwargs["stop_strings"] = stop_sequences
    if grpo_args.min_p is not None:
        cfg_dict["min_p"] = grpo_args.min_p
    cfg_dict["generation_kwargs"] = generation_kwargs

    grpo_config = GRPOConfig(**cfg_dict)

    reward_names = grpo_args.reward_functions or create_reward_preset(grpo_args.task_type)
    if not grpo_args.reward_functions:
        grpo_args.reward_functions = list(reward_names)  # type: ignore
    reward_fns = get_reward_functions(reward_names)

    def _wrap(rf):
        def inner(prompts, completions, **kw):
            return rf(prompts, completions, **kw)
        return inner

    reward_funcs = [_wrap(r) for r in reward_fns]
    try:
        return GRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
        )
    except TypeError as exc:
        if "reward_funcs" in str(exc):
            return GRPOTrainer(
                model=model,
                args=grpo_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
            )
        raise


# --------------------------- Factory dispatcher ---------------------------

def _create_trainer_by_type(model, tokenizer, train_dataset, cfg: OpenSlothConfig, hf_train_args: TrainingArguments):
    factories = {
        "sft": _create_sft_trainer,
        "dpo": _create_dpo_trainer,
        "kto": _create_kto_trainer,
        "orpo": _create_orpo_trainer,
        "grpo": _create_grpo_trainer,
    }
    factory = factories.get(cfg.training_type)
    if factory is None:
        raise ValueError(f"Unsupported training_type {cfg.training_type}")
    return factory(model, tokenizer, train_dataset, cfg, hf_train_args)


# --------------------------- Dataset/path validation ---------------------------

def _validate_dataset_compatibility(dataset_path: str, model_max_seq_length: int) -> None:
    import json
    from pathlib import Path
    logger = get_opensloth_logger()
    cfg_path = Path(dataset_path) / "dataset_config.json"
    if not cfg_path.exists():
        logger.warning("Dataset missing dataset_config.json; length mismatch checks skipped.")
        return
    with open(cfg_path) as f:
        data = json.load(f)
    ds_len = data.get("max_seq_length")
    if ds_len is None:
        logger.warning("dataset_config.json missing max_seq_length key")
        return
    if model_max_seq_length < ds_len:
        raise ValueError(
            "Training max_seq_length is smaller than dataset max_seq_length: "
            f"{model_max_seq_length} < {ds_len}. Rebuild dataset or increase training setting."
        )
    if model_max_seq_length > ds_len:
        logger.warning(
            "Training max_seq_length (%s) > dataset max_seq_length (%s); may waste padding.",
            model_max_seq_length,
            ds_len,
        )


def _maybe_hot_fix_gemma(cfg: OpenSlothConfig, logger, tokenizer) -> None:
    if "gemma-3" in cfg.fast_model_args.model_name and cfg.sequence_packing:
        from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing
        logger.info("Applying Gemma-3 sequence packing patch.")
        patch_gemma3_unsloth_for_sequence_packing()
    if not hasattr(tokenizer, "pad") and cfg.sequence_packing:
        from transformers import AutoTokenizer
        logger.info("Tokenizer lacks pad(); patching from AutoTokenizer.")
        tk2 = AutoTokenizer.from_pretrained(cfg.fast_model_args.model_name)
        tokenizer.pad = tk2.pad  # type: ignore[attr-defined]


def _apply_grpo_model_args(cfg: OpenSlothConfig, args: dict, logger) -> None:
    if cfg.training_type != "grpo":
        return
    args["fast_inference"] = True
    if cfg.lora_args and args.get("max_lora_rank") is None:
        args["max_lora_rank"] = cfg.lora_args.r
    if args.get("load_in_4bit"):
        name = args["model_name"]
        if not name.lower().endswith("-bnb-4bit"):
            logger.warning("4-bit GRPO model name should end with -bnb-4bit for vLLM best support: %s", name)
    args.setdefault("gpu_memory_utilization", 0.6)


def _setup_comm_backend(cfg: OpenSlothConfig, _logger) -> None:
    if len(cfg.devices) <= 1:
        return
    backend = getattr(cfg, "comm_backend", "allreduce")
    if backend == "async-ps":
        from opensloth.comm.async_ps import setup_rpc
        rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
        world = len(cfg.devices)
        rpc_cfg = getattr(cfg, "async_ps_args", None)
        addr = getattr(rpc_cfg, "master_addr", "127.0.0.1") if rpc_cfg else "127.0.0.1"
        port = getattr(rpc_cfg, "master_port", "29512") if rpc_cfg else "29512"
        os.environ.setdefault("MASTER_ADDR", addr)
        os.environ.setdefault("MASTER_PORT", port)
        setup_rpc(rank=rank, world_size=world, master_addr=addr, master_port=port)
    else:
        from opensloth.nccl_grad_sync import get_callback_and_setup_method
        _cb, setup_nccl, _destroy = get_callback_and_setup_method()  # local names unused
        setup_nccl(rank=int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0")), gpus=cfg.devices)


def _init_model_and_tokenizer(cfg: OpenSlothConfig, unsloth_modules: dict[str, object] | None = None):
    logger = get_opensloth_logger()
    if cfg.pretrained_lora:
        cfg.fast_model_args.model_name = cfg.pretrained_lora
    model_args = cfg.fast_model_args.model_dump()
    _apply_grpo_model_args(cfg, model_args, logger)
    if unsloth_modules is not None:
        FastLanguageModel = unsloth_modules["FastLanguageModel"]  # noqa: N806
    else:
        from unsloth import FastLanguageModel  # type: ignore
    model, tokenizer = FastLanguageModel.from_pretrained(**model_args)
    _maybe_hot_fix_gemma(cfg, logger, tokenizer)
    _setup_comm_backend(cfg, logger)
    if (
        not cfg.fast_model_args.full_finetuning
        and not cfg.pretrained_lora
        and cfg.lora_args is not None
    ):
        if unsloth_modules is not None:
            FastModel = unsloth_modules["FastModel"]  # noqa: N806
        else:
            from unsloth import FastModel  # type: ignore
        model = FastModel.get_peft_model(model, **cfg.lora_args.model_dump())  # type: ignore[attr-defined]
    return model, tokenizer


# --------------------------- Dataset loading & trainer wrapper ---------------------------

def _get_trainer(model, tokenizer, cfg: OpenSlothConfig, hf_train_args: TrainingArguments):
    from datasets import load_from_disk
    logger = get_opensloth_logger()
    base = cfg.data_cache_path
    rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    shard_path = os.path.join(base, f"shard_{rank}")
    path = shard_path if os.path.isdir(shard_path) else base
    train_ds = load_from_disk(path)
    _validate_dataset_compatibility(base, cfg.fast_model_args.max_seq_length)
    if cfg.training_type == "sft" and cfg.filter_overlength_samples:
        max_len = int(cfg.fast_model_args.max_seq_length)
        def _ok(ex):
            ids = ex.get("input_ids")
            return True if ids is None else len(ids) <= max_len
        before = len(train_ds)
        train_ds = train_ds.filter(_ok, num_proc=getattr(hf_train_args, "dataset_num_proc", 1))
        drop = before - len(train_ds)
        if drop:
            logger.warning("Filtered %d/%d samples (%.2f%%) > max_len=%d", drop, before, 100*drop/max(1,before), max_len)
    return _create_trainer_by_type(model, tokenizer, train_ds, cfg, hf_train_args)


def _configure_batch_size(hf_train_args: TrainingArguments) -> None:
    rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    if rank != 0:
        hf_train_args.report_to = "none"


def _create_trainer(model, tokenizer, cfg: OpenSlothConfig, hf_train_args: TrainingArguments):
    trainer = _get_trainer(model, tokenizer, cfg, hf_train_args)
    _configure_batch_size(hf_train_args)
    if cfg.training_type == "sft":
        from opensloth.patching.inner_training_loop import patch_inner_training_loop_for_sft
        patch_inner_training_loop_for_sft(trainer, cfg.sequence_packing)
    if len(cfg.devices) > 1 and os.getenv("OPENSLOTH_LOCAL_RANK", "0") != "0":
        def _no_op(*_a, **_k):
            pass
        trainer._save = _no_op  # type: ignore[attr-defined]
    return trainer


def setup_model_and_training(cfg: OpenSlothConfig, hf_train_args: TrainingArguments, unsloth_modules: dict[str, object] | None = None):
    logger = get_opensloth_logger()
    logger.start_timing("total_setup")
    logger.start_timing("model_init")
    model, tokenizer = _init_model_and_tokenizer(cfg, unsloth_modules)
    logger.finish_timing("model_init")
    logger.start_timing("trainer_creation")
    trainer = _create_trainer(model, tokenizer, cfg, hf_train_args)
    logger.finish_timing("trainer_creation")
    logger.finish_timing("total_setup")
    return trainer, model, tokenizer


def _create_trainer_by_type(
    model,
    tokenizer,
    train_dataset,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Factory function to create the appropriate trainer based on training_type.
    """
    logger = get_opensloth_logger()
    training_type = opensloth_config.training_type
    logger.info(f"Creating trainer for training type: {training_type}")

    factory = {
        "sft": _create_sft_trainer,
        "dpo": _create_dpo_trainer,
        "kto": _create_kto_trainer,
        "orpo": _create_orpo_trainer,
        "grpo": _create_grpo_trainer,
    }.get(training_type)

    if factory is None:
        supported = ["sft", "dpo", "kto", "orpo", "grpo"]
        raise ValueError(f"Unsupported training type: {training_type}. Supported types: {supported}")

    try:
        return factory(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            opensloth_config=opensloth_config,
            hf_train_args=hf_train_args,
        )
    except Exception as e:
        logger.error(f"Failed to create {training_type.upper()} trainer: {e}")
        raise


# --------------------------- Model/tokenizer init & patches ---------------------------

def _validate_dataset_compatibility(dataset_path: str, model_max_seq_length: int) -> None:
    """Validate that the dataset is compatible with the training configuration."""
    import json
    from pathlib import Path

    dataset_config_path = Path(dataset_path) / "dataset_config.json"
    if not dataset_config_path.exists():
        logger = get_opensloth_logger()
        logger.warning(
            f"Dataset at {dataset_path} was prepared without max_seq_length specification. "
            f"This may cause issues if sequences exceed training max_seq_length ({model_max_seq_length}). "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length}."
        )
        return

    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    dataset_max_seq_length = dataset_config.get("max_seq_length")
    if dataset_max_seq_length is None:
        logger = get_opensloth_logger()
        logger.warning(
            f"Dataset at {dataset_path} lacks 'max_seq_length'. "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length}."
        )
        return

    if model_max_seq_length < dataset_max_seq_length:
        raise ValueError(
            "Training configuration mismatch!\n"
            f"Dataset was prepared with max_seq_length={dataset_max_seq_length}, "
            f"but training config specifies max_seq_length={model_max_seq_length}.\n"
            "The training max_seq_length must be >= dataset max_seq_length.\n"
            "Either:\n"
            f"  1. Increase training max_seq_length to {dataset_max_seq_length} or higher\n"
            f"  2. Re-prepare the dataset with --max-seq-length {model_max_seq_length}"
        )

    if model_max_seq_length > dataset_max_seq_length:
        logger = get_opensloth_logger()
        logger.warning(
            f"Training max_seq_length ({model_max_seq_length}) is larger than "
            f"dataset max_seq_length ({dataset_max_seq_length}). This is OK but may be inefficient. "
            f"Consider re-preparing the dataset with --max-seq-length {model_max_seq_length} for optimal performance."
        )


def _maybe_hot_fix_gemma(opensloth_config: OpenSlothConfig, logger, tokenizer) -> None:
    if "gemma-3" in opensloth_config.fast_model_args.model_name and opensloth_config.sequence_packing:
        from opensloth.patching.gemma import patch_gemma3_unsloth_for_sequence_packing

        logger.info("Detected Gemma-3 model, applying Unsloth patch for sequence packing.")
        patch_gemma3_unsloth_for_sequence_packing()

    if not hasattr(tokenizer, "pad") and opensloth_config.sequence_packing:
        logger.info(
            "Tokenizer missing 'pad' method; attempting to patch using transformers.AutoTokenizer. "
            "This may indicate an Unsloth issue."
        )
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(
            opensloth_config.fast_model_args.model_name,
        )
        tokenizer.pad = hf_tokenizer.pad  # type: ignore[attr-defined]


def _apply_grpo_model_args(opensloth_config: OpenSlothConfig, model_args: dict, logger):
    # Only adjust if GRPO
    if opensloth_config.training_type != "grpo":
        return
    
    # Always enable vLLM for GRPO (both new LoRA and pretrained LoRA)
    model_args["fast_inference"] = True
    
    # Set max_lora_rank for vLLM memory calculation
    if model_args.get("max_lora_rank") is None:
        if opensloth_config.lora_args:
            # New LoRA training
            model_args["max_lora_rank"] = opensloth_config.lora_args.r
            logger.info("Set max_lora_rank=%s for GRPO (new LoRA)", model_args["max_lora_rank"])
        elif opensloth_config.pretrained_lora:
            # Pretrained LoRA - extract rank from adapter config
            try:
                import json
                from pathlib import Path
                adapter_path = Path(opensloth_config.pretrained_lora) / "adapter_config.json"
                if adapter_path.exists():
                    with open(adapter_path) as f:
                        adapter_config = json.load(f)
                    lora_rank = adapter_config.get("r", 8)  # default rank 8
                    model_args["max_lora_rank"] = lora_rank
                    logger.info("Set max_lora_rank=%s for GRPO (pretrained LoRA)", model_args["max_lora_rank"])
                else:
                    logger.warning("adapter_config.json not found, using default max_lora_rank=8")
                    model_args["max_lora_rank"] = 8
            except Exception as e:
                logger.warning(f"Error reading LoRA rank from adapter config: {e}, using default max_lora_rank=8")
                model_args["max_lora_rank"] = 8
    
    # Check for 4-bit model naming convention for vLLM compatibility
    if model_args.get("load_in_4bit", False):
        model_name = model_args["model_name"]
        if not model_name.lower().endswith("-bnb-4bit"):
            logger.warning(
                "Model name '%s' does not end with '-bnb-4bit'; may impact vLLM 4-bit loading.",
                model_name,
            )
    
    model_args.setdefault("gpu_memory_utilization", 0.6)


def _setup_comm_backend(opensloth_config: OpenSlothConfig, logger):
    if len(opensloth_config.devices) <= 1:
        logger.info("Single GPU detected; skipping distributed comm setup.")
        return
    backend = getattr(opensloth_config, "comm_backend", "allreduce")
    if backend == "async-ps":
        logger.start_timing("rpc_setup")
        from opensloth.comm.async_ps import setup_rpc
        rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
        world_size = len(opensloth_config.devices)
        cfg = getattr(opensloth_config, "async_ps_args", None)
        master_addr = getattr(cfg, "master_addr", "127.0.0.1") if cfg else "127.0.0.1"
        master_port = getattr(cfg, "master_port", "29512") if cfg else "29512"
        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", master_port)
        setup_rpc(rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port)
        logger.finish_timing("rpc_setup")
        logger.info("Initialized Async Parameter Server RPC backend")
    else:
        logger.start_timing("nccl_setup")
        from opensloth.nccl_grad_sync import get_callback_and_setup_method
        _nccl_cb_cls, setup_nccl_for_opensloth, _destroy_nccl = get_callback_and_setup_method()
        setup_nccl_for_opensloth(
            rank=int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0")),
            gpus=opensloth_config.devices,
        )
        logger.finish_timing("nccl_setup")


def _init_model_and_tokenizer(
    opensloth_config: OpenSlothConfig,
    unsloth_modules: dict[str, object] | None = None,
) -> tuple[object, object]:
    """Initialize base/LoRA model and tokenizer; set up comm backend & LoRA."""
    logger = get_opensloth_logger()
    if unsloth_modules is not None:
        FastLanguageModel = unsloth_modules["FastLanguageModel"]  # noqa: N806 (external class)
    else:  # Lazy import per project policy
        from unsloth import FastLanguageModel  # type: ignore
    fast_language_model = FastLanguageModel  # local alias (keeps original class name intact)

    logger.start_timing("model_loading")
    if opensloth_config.pretrained_lora:
        logger.info(f"Loading base model with pretrained LoRA from {opensloth_config.pretrained_lora}")
        # For pretrained LoRA, we need to extract the base model name to enable vLLM
        # The LoRA will be loaded separately after model initialization
        try:
            import json
            from pathlib import Path
            adapter_path = Path(opensloth_config.pretrained_lora)
            if adapter_path.is_dir() and (adapter_path / "adapter_config.json").exists():
                with open(adapter_path / "adapter_config.json") as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get("base_model_name_or_path")
                if base_model_name:
                    opensloth_config.fast_model_args.model_name = base_model_name
                    logger.info(f"Using base model: {base_model_name}")
                else:
                    logger.warning("Could not find base_model_name_or_path in adapter config, using pretrained_lora path")
                    opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora
            else:
                logger.warning("adapter_config.json not found, using pretrained_lora path as model name")
                opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora
        except Exception as e:
            logger.warning(f"Error reading adapter config: {e}, using pretrained_lora path as model name")
            opensloth_config.fast_model_args.model_name = opensloth_config.pretrained_lora

    model_args = opensloth_config.fast_model_args.model_dump()
    _apply_grpo_model_args(opensloth_config, model_args, logger)

    model, tokenizer = fast_language_model.from_pretrained(**model_args)
    _maybe_hot_fix_gemma(opensloth_config, logger, tokenizer)
    logger.finish_timing("model_loading")

    _setup_comm_backend(opensloth_config, logger)

    logger.info(f"Model device: {model.device} | Tokenizer: {tokenizer.__class__.__name__}")

    # Setup LoRA if requested and not loading a pretrained LoRA
    if (
        not opensloth_config.fast_model_args.full_finetuning
        and not opensloth_config.pretrained_lora
        and opensloth_config.lora_args is not None
    ):
        logger.start_timing("lora_setup")
        if unsloth_modules is not None:
            FastModel = unsloth_modules["FastModel"]  # noqa: N806
        else:  # lazy import
            from unsloth import FastModel  # type: ignore
        model = FastModel.get_peft_model(model, **opensloth_config.lora_args.model_dump())  # type: ignore[attr-defined]
        logger.finish_timing("lora_setup")
    elif opensloth_config.pretrained_lora:
        # For pretrained LoRA, load the adapters from the saved checkpoint
        logger.start_timing("pretrained_lora_setup")
        try:
            if unsloth_modules is not None:
                FastModel = unsloth_modules["FastModel"]  # noqa: N806
            else:  # lazy import
                from unsloth import FastModel  # type: ignore
            # Load the pretrained LoRA adapters
            model = FastModel.from_pretrained(
                model=model,
                model_name=opensloth_config.pretrained_lora,
            )
            logger.info(f"Successfully loaded pretrained LoRA from {opensloth_config.pretrained_lora}")
        except Exception as e:
            logger.warning(f"Failed to load pretrained LoRA: {e}. Continuing without LoRA.")
        logger.finish_timing("pretrained_lora_setup")

    return model, tokenizer


# --------------------------- Trainer creation & global patches ---------------------------

def _get_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """
    Returns an appropriate trainer instance (SFT, DPO, GRPO, etc.) with a dataset loaded from disk.
    """
    from datasets import load_from_disk

    logger = get_opensloth_logger()
    base_path = opensloth_config.data_cache_path
    local_rank = int(os.environ.get("OPENSLOTH_LOCAL_RANK"))
    shard_path = os.path.join(base_path, f"shard_{local_rank}")
    if os.path.isdir(shard_path):
        logger.info(f"Loading rank-specific shard dataset from {shard_path}")
        dataset_load_path = shard_path
    else:
        logger.info(f"Shard path {shard_path} not found. Falling back to base dataset path {base_path}")
        dataset_load_path = base_path

    try:
        train_dataset = load_from_disk(dataset_load_path)

        # Dataset compatibility check (best-effort)
        _validate_dataset_compatibility(
            base_path,
            opensloth_config.fast_model_args.max_seq_length,
        )

        # Optional filtering for SFT only (DPO datasets lack input_ids)
        if opensloth_config.training_type == "sft" and opensloth_config.filter_overlength_samples:
            max_len = int(opensloth_config.fast_model_args.max_seq_length)
            before_n = len(train_dataset)

            def _len_ok(ex):
                ids = ex.get("input_ids")
                if ids is None:
                    return True
                try:
                    return len(ids) <= max_len
                except Exception:
                    return True

            train_dataset = train_dataset.filter(
                _len_ok,
                num_proc=getattr(hf_train_args, "dataset_num_proc", 1),
            )
            dropped = before_n - len(train_dataset)
            if dropped > 0:
                logger.warning(
                    f"Filtered {dropped}/{before_n} samples "
                    f"({dropped / max(1, before_n):.2%}) > max_len={max_len}."
                )
        elif opensloth_config.training_type != "sft":
            logger.info("Skipping over-length filtering (not applicable to non-SFT datasets).")
    except Exception as e:
        logger.error(f"Failed to load dataset. Ensure it was prepared correctly. Error: {e}")
        raise




    trainer = _create_trainer_by_type(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        opensloth_config=opensloth_config,
        hf_train_args=hf_train_args,
    )
    logger.info(f"Trainer created for training_type='{opensloth_config.training_type.upper()}'")
    return trainer

def _configure_batch_size(hf_train_args: TrainingArguments) -> None:
    """
    Adjust per-device batch size settings for multi-GPU and silence logging on non-zero ranks.

    Note: We keep per-device batch size as specified by the user to avoid surprising scaling.
    """
    gpu_ith = int(os.environ.get("OPENSLOTH_LOCAL_RANK", "0"))
    num_gpus = int(os.environ.get("OPENSLOTH_WORLD_SIZE", "1"))
    if num_gpus != 1:
        # Intentionally a no-op for now; place-holder for future logic.
        hf_train_args.per_device_train_batch_size *= 1
    if gpu_ith != 0:
        hf_train_args.report_to = "none"


def _create_trainer(
    model,
    tokenizer,
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
):
    """Create trainer (SFT/DPO/GRPO/etc.) and apply universal performance patches."""
    logger = get_opensloth_logger()
    logger.start_timing("trainer_setup")
    trainer = _get_trainer(model, tokenizer, opensloth_config, hf_train_args)
    logger.finish_timing("trainer_setup")

    # Apply UNIVERSAL patches that all trainers need for multi-GPU
    logger.start_timing("universal_patch")
    _configure_batch_size(hf_train_args)
    # Removed custom sampler & batch patching: pre-sharded datasets handle per-rank isolation.
    logger.finish_timing("universal_patch")

    # Apply UNIVERSAL multi-GPU log patch for all trainer types
    # if len(opensloth_config.devices) > 1:
    #     # Check if we should apply the IPC log patch
        # use_tmux = os.environ.get("USE_TMUX") == "1"
        # if not use_tmux:
        #     # Only apply IPC log patch in multiprocessing mode
        #     logger.start_timing("multi_gpu_log_patch")
        #     from opensloth.patching.patch_log_for_multi_gpu import patch_log_for_multi_gpu
        #     patch_log_for_multi_gpu(trainer)
        #     logger.info("Applied IPC log aggregation patch for multiprocessing mode.")
        #     logger.finish_timing("multi_gpu_log_patch")
        # else:
        #     logger.info("Skipping IPC log patch for tmux mode (each process has its own terminal).")
    # else:
        # logger.info("Single GPU detected; skipping multi-GPU log patch.")

    # Apply SFT-specific patches (inner loop, etc.)
    if opensloth_config.training_type == "sft":
        logger.start_timing("sft_specific_patch")
        from opensloth.patching.inner_training_loop import patch_inner_training_loop_for_sft

        patch_inner_training_loop_for_sft(trainer, opensloth_config.sequence_packing)
        logger.finish_timing("sft_specific_patch")
    else:
        logger.info(f"Skipping SFT-specific patches for training_type='{opensloth_config.training_type}'.")

    # Disable save operations on non-master ranks (multi-GPU only)
    if len(opensloth_config.devices) > 1 and os.getenv("OPENSLOTH_LOCAL_RANK", "0") != "0":
        def _no_op(*_args, **_kwargs):
            pass

        trainer._save = _no_op  # type: ignore[attr-defined]
        logger.info("Patched _save to no-op on non-master rank.")


    return trainer

    return trainer, model, tokenizer
