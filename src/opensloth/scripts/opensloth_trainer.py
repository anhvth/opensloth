"""
Multi-GPU training script for OpenSloth.
Supports SFT (Supervised Fine-Tuning) only.
Handles weight synchronization, model setup, and distributed training coordination.
"""

# ruff: noqa: I001,PLR0912,PLR0915

import importlib.util
import os
import sys
import time
import warnings
from typing import Any

from opensloth.logging_config import OpenslothLogger
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

warnings.filterwarnings("ignore")


def setup_python_env():
    import subprocess
    import sys
    try:
        python_path = subprocess.check_output([sys.executable, "-c", "import sys; print(sys.executable)"], text=True).strip()
        return python_path
    except subprocess.CalledProcessError as e:
        from opensloth.logging_config import get_opensloth_logger
        logger = get_opensloth_logger(allow_unknown_gpu=True)
        logger.error(f"Error getting Python path: {e}")
        return sys.executable


def _import_unsloth(gpu: int) -> dict[str, Any]:
    os.makedirs(".cache/", exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = f".cache/UNSLOTH_CACHE_DIR_{gpu}"

    # Enforce rule: unsloth & trl not imported globally yet
    assert "unsloth" not in sys.modules
    assert "trl" not in sys.modules

    # Dynamic imports (order matters)
    import unsloth  # type: ignore
    from trl import SFTConfig, SFTTrainer  # type: ignore
    from opensloth.logging_config import get_opensloth_logger

    logger = get_opensloth_logger(allow_unknown_gpu=True)
    logger.info(f"Unsloth version: {unsloth.__version__}")

    return {
        "unsloth": unsloth,
        "FastLanguageModel": unsloth.FastLanguageModel,
        "FastModel": unsloth.FastModel,
        "SFTConfig": SFTConfig,
        "SFTTrainer": SFTTrainer,
    }

MAGIC_TWO = 2
MAGIC_THREE = 3
IGNORE_INDEX = -100


def train_on_single_gpu(
    gpu: int, opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments
):
    # Set rank/env identifiers BEFORE any logging interception so logger picks them up
    os.environ["OPENSLOTH_LOCAL_RANK"] = str(opensloth_config.devices.index(gpu))
    os.environ["OPENSLOTH_WORLD_SIZE"] = str(len(opensloth_config.devices))
    # Also set standard torch.distributed expected env vars (helps if any library
    # attempts implicit initialization). We rely on spawn, not torchrun, so we
    # must populate these explicitly.
    os.environ.setdefault("RANK", os.environ["OPENSLOTH_LOCAL_RANK"])  # single-node so rank==local_rank
    os.environ.setdefault("LOCAL_RANK", os.environ["OPENSLOTH_LOCAL_RANK"])  # for completeness
    os.environ.setdefault("WORLD_SIZE", os.environ["OPENSLOTH_WORLD_SIZE"])  # matches device count
    # os.environ["OPENSLOTH_TRAINING_ACTIVE"] = "1"
    # Now install interception (logger will show proper GPU id instead of GPUUNSET)
    # setup_stdout_interception_for_training()
    unsloth_modules = _import_unsloth(gpu)

    from opensloth.trainer_factory import setup_model_and_training

    # Setup enhanced logger
    logger = OpenslothLogger()

    logger.info(f"Training on GPU {gpu} with output_dir {hf_train_args.output_dir}")

    # Start total training timer
    logger.start_total_training_timer()

    # setup_nccl_for_opensloth(gpu, opensloth_config.training.gpus)

    logger.start_timing("model_and_training_setup")
    trainer, model, tokenizer = setup_model_and_training(
        opensloth_config=opensloth_config,
        hf_train_args=hf_train_args,
        unsloth_modules=unsloth_modules,
    )
    logger.finish_timing("model_and_training_setup")

    def _register_batch_shape_assertion(_model, _logger: OpenslothLogger) -> None:
        """Attach a forward pre-hook performing light shape assertions."""

        def _pre_hook(_mod, _args, _kwargs):
            input_ids = _kwargs.get("input_ids")
            labels = _kwargs.get("labels")
            attention_mask = _kwargs.get("attention_mask")
            if input_ids is None or labels is None:
                return
            try:
                ids_shape = tuple(input_ids.size())
                lbl_shape = tuple(labels.size())
            except Exception:
                return
            msg_prefix = "[OpenSloth Debug] Batch shape check failed: "
            if labels.dim() != input_ids.dim():
                raise AssertionError(
                    f"{msg_prefix}ndim mismatch -> input_ids {ids_shape} vs labels {lbl_shape}"
                )
            if ids_shape != lbl_shape:
                advice = ""
                if (
                    len(ids_shape) == MAGIC_TWO
                    and len(lbl_shape) == MAGIC_TWO
                    and (
                        lbl_shape[-1] + 1 == ids_shape[-1]
                        or lbl_shape[-1] == ids_shape[-1] + 1
                    )
                ):
                    advice = (
                        "Possible off-by-one from shifting labels/padding. Ensure labels align to input_ids."
                    )
                try:
                    non_ignored = int((labels != IGNORE_INDEX).sum().item())
                except Exception:
                    non_ignored = -1
                raise AssertionError(
                    f"{msg_prefix}input_ids {ids_shape} != labels {lbl_shape}. {advice} "
                    f"non_ignored_labels={non_ignored} total_labels={getattr(labels, 'numel', lambda: 'n/a')()}"
                )
            if attention_mask is not None:
                am_shape = tuple(attention_mask.size())
                if len(am_shape) == MAGIC_TWO:
                    if am_shape != ids_shape:
                        raise AssertionError(
                            f"{msg_prefix}attention_mask {am_shape} != input_ids {ids_shape}"
                        )
                elif len(am_shape) == MAGIC_THREE:
                    b, seq_len = ids_shape
                    if not (
                        am_shape[0] == b
                        and am_shape[1] == seq_len
                        and am_shape[2] == seq_len
                    ):
                        raise AssertionError(
                            f"{msg_prefix}3D attention_mask {am_shape} expected (B, L, L) matching input_ids {ids_shape}"
                        )
                else:
                    raise AssertionError(
                        f"{msg_prefix}unsupported attention_mask ndim={len(am_shape)} shape={am_shape}"
                    )
            try:
                non_ignored = int((labels != IGNORE_INDEX).sum().item())
                if non_ignored == 0:
                    raise AssertionError(
                        f"{msg_prefix}all labels are -100 (ignored). Check response-only masking and packing."
                    )
            except Exception:
                pass

        reg = _model.register_forward_pre_hook
        try:
            handle = reg(_pre_hook, with_kwargs=True)  # type: ignore[arg-type]
        except TypeError:
            handle = reg(_pre_hook)  # type: ignore[arg-type]
        _model._opensloth_debug_shape_hook = handle  # type: ignore[attr-defined]
        _logger.info("Registered batch shape assertion hook")

    _register_batch_shape_assertion(trainer.model, logger)

    assert trainer.model is not None, "Trainer model is None"

    # Only use comm backend for multi-GPU training
    if len(opensloth_config.devices) > 1:
        from opensloth.nccl_grad_sync import get_callback_and_setup_method
        nccl_grad_sync_callback, _setup_nccl, destroy_nccl = get_callback_and_setup_method()
        grad_sync_cb = nccl_grad_sync_callback(
            model=trainer.model,
            gpu=gpu,
            gpus=opensloth_config.devices,
        )
        logger.info(f"Using NCCL gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)
    else:
        logger.info("Single GPU training detected, skipping distributed gradient sync")

    logger.start_timing("actual_training")
    
    logger.debug(f"Environment: {os.environ}")
    
    # Patch: Only resume from checkpoint if a valid path is provided
    try:
        resume_from_checkpoint = getattr(hf_train_args, 'resume_from_checkpoint', None)
        if resume_from_checkpoint and hasattr(trainer, "train") and "resume_from_checkpoint" in trainer.train.__code__.co_varnames:  # type: ignore[attr-defined]
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
        else:
            trainer.train()
    finally:
        pass
    logger.finish_timing("actual_training")

    # Save once from rank=0
    if gpu == opensloth_config.devices[0]:
        if hf_train_args.save_only_model:
            logger.start_timing("model_saving")
            logger.info(f"Save model to {hf_train_args.output_dir}")
            # Ensure output directory exists before saving model/tokenizer
            output_dir = hf_train_args.output_dir
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
            model.save_pretrained(hf_train_args.output_dir)
            tokenizer.save_pretrained(hf_train_args.output_dir)
            logger.finish_timing("model_saving")
        else:
            # user save_state
            trainer.save_model()
            trainer.save_state()

        # Log training summary
        logger.log_training_summary()

    if 'destroy_nccl' in locals():  # type: ignore[truthy-function]
        destroy_nccl()  # type: ignore[misc]


def load_config_from_path(
    config_path: str,
) -> tuple[OpenSlothConfig, TrainingArguments]:
    """Load configuration from Python file path."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)  # type: ignore
    spec.loader.exec_module(config_module)  # type: ignore
    # return config_module
    # Retrieve configs from the module
    if hasattr(config_module, "opensloth_config"):
        opensloth_config = config_module.opensloth_config
    elif hasattr(config_module, "opensloth_config"):
        opensloth_config = OpenSlothConfig(**config_module.opensloth_config)
    else:
        raise ValueError("No OpenSloth configuration found")
    # return opensloth_config,
    if hasattr(config_module, "training_config"):
        training_config = config_module.training_config
    elif hasattr(config_module, "training_config"):
        training_config = TrainingArguments(**config_module.training_config)
    else:
        raise ValueError("No training configuration found")
    return opensloth_config, training_config


def build_tmux_script(
    session_name: str,
    script_path: str,
    config_file: str,
    gpus: list,
    auto_kill: bool = False,
):
    """
    Build a script that:
    1. Kills any existing tmux session with `session_name`
    2. Creates a new session for the first GPU
    3. Creates new windows for the remaining GPUs
    4. Sends the appropriate commands to each window
    Saves the final script to `script_path`.
    """
    lines = []
    lines.append("#!/usr/bin/env bash")
    # remove grad_dir
    # lines.append(f"rm -rf {_get_hp_grad_dir(output_dir)}")
    lines.append(
        f"""# Create a new session with first GPU = 0
tmux new-session -d -s {session_name} -n MAIN"""
    )


    for local_rank, gpu_index in enumerate(gpus):
        # Use the os-tmux-worker entry point for cleaner command generation
        cmd = (
            f"USE_TMUX=1 "
            f"os-tmux-worker "
            f"{config_file} "
            f"--rank {local_rank} "
            f"--world_size {len(gpus)}"
        )
        lines.append(f"tmux new-window -t {session_name} -n gpu_{gpu_index}")
        lines.append(f"tmux send-keys -t {session_name}:gpu_{gpu_index} '{cmd}' Enter")
        lines.append("")

    lines.append(f'echo "Automatically attaching to session {session_name}..."')
    lines.append(f"tmux attach -t {session_name}")

    # Write out the script
    script_body = "\n".join(lines)
    with open(script_path, "w") as f:
        f.write(script_body)
    os.chmod(script_path, 0o755)

    is_session_exists = os.system(f"tmux has-session -t {session_name}")
    if is_session_exists == 0:
        if auto_kill:
            print(f"Auto-killing existing session {session_name}")
            os.system(f"tmux kill-session -t {session_name}")
        else:
            # ask user if they want to kill the session
            user_input = input(
                f"Session {session_name} exists, do you want to kill it? (y/n): "
            )
            if user_input.lower() == "y":
                os.system(f"tmux kill-session -t {session_name}")
                print(f"Session {session_name} killed")
            else:
                return
    os.system(f"bash {script_path}")
    print(f"Training sessions started and attached to session {session_name}")


def run_tmux_training(
    session_name: str,
    config_file: str,
    gpus: list,
    auto_kill: bool = False,
):
    """Handle multi-GPU training using tmux sessions."""
    script_path = "/tmp/hp_train.sh"
    build_tmux_script(
        session_name,
        script_path,
        config_file,
        gpus,
        auto_kill=auto_kill,
    )


def run_mp_training(
    gpus: list,
    opensloth_config: OpenSlothConfig,
    training_config: TrainingArguments,
):
    """Handle multi-GPU training using multi-processing."""
    if len(gpus) == 1:
        print("Only one GPU detected, running single GPU training")
        train_on_single_gpu(
            gpu=gpus[0],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
        return
    import multiprocessing as mp

    # Set spawn method for CUDA compatibility
    mp.set_start_method("spawn", force=True)

    print(f"[MP] Running on {len(gpus)} GPUs")
    processes = []
    for gpu_index in gpus:
        p = mp.Process(
            target=train_on_single_gpu,
            args=(gpu_index,),
            kwargs={
                "opensloth_config": opensloth_config,
                "hf_train_args": training_config,
            },
        )
        p.start()
        processes.append(p)

    # Wait for processes; if one errors, kill them all
    while processes:
        for i, proc in enumerate(processes):
            if not proc.is_alive():
                if proc.exitcode != 0:
                    for p in processes:
                        p.terminate()
                    if i == 0:
                        raise Exception("Error in training")
                else:
                    processes.remove(proc)
                    break
        time.sleep(1)
    print("All processes finished")


def initialize_training_config(config_file):
    # global USE_TMUX
    # USE_TMUX = USE_TMUX or use_tmux
    """Train entry-point. If rank/world_size are provided, we assume this is
    a child process that trains on a single GPU. Otherwise,
    we spawn multi-gpu runs either by generating a tmux script or by multi-process.
    """

    config_file = os.path.abspath(config_file)
    assert os.path.exists(config_file), f"Config file {config_file} not found"

    opensloth_config, training_config = load_config_from_path(config_file)
    print(
        f"Overriding max_seq_len to {opensloth_config.fast_model_args.max_seq_length} for data processing"
    )

    setup_envs(opensloth_config, training_config)
    return opensloth_config, training_config


def setup_envs(opensloth_config: OpenSlothConfig, training_config: TrainingArguments):
    world_size = len(opensloth_config.devices)
    os.environ["OPENSLOTH_WORLD_SIZE"] = str(world_size)
    
    # For single GPU training, explicitly disable distributed training
    if world_size == 1:
        os.environ["LOCAL_RANK"] = "-1"
        os.environ["RANK"] = "-1" 
        os.environ["WORLD_SIZE"] = "1"
        # Remove any existing distributed training env vars that might interfere
        for key in ["MASTER_ADDR", "MASTER_PORT"]:
            if key in os.environ:
                del os.environ[key]
    
    os.environ["OPENSLOTH_FORWARD_BZ"] = str(
        training_config.per_device_train_batch_size
        # * training_config.gradient_accumulation_steps
        * world_size
    )
    os.environ["OPENSLOTH_GLOBAL_BZ"] = str(
        training_config.per_device_train_batch_size
        * training_config.gradient_accumulation_steps
        * world_size
    )

    print(f"Global batch size: {os.environ['OPENSLOTH_GLOBAL_BZ']}")
    os.environ["OPENSLOTH_ACCUMULATION_STEPS"] = str(
        training_config.gradient_accumulation_steps
    )
    os.environ["OPENSLOTH_PER_DEVICE_TRAIN_BZ"] = str(
        training_config.per_device_train_batch_size
    )
    # output dir
    os.environ["OPENSLOTH_OUTPUT_DIR"] = training_config.output_dir
    os.environ["OPENSLOTH_LOG_LEVEL"] = opensloth_config.log_level
