"""
Multi-GPU training script for OpenSloth.
Supports various training types: SFT, DPO, KTO, ORPO, GRPO.
Handles weight synchronization, model setup, and distributed training coordination.
"""

import importlib.util
import os
import sys
import time
from typing import Any, List
import warnings

import argparse

from opensloth.logging_config import OpenslothLogger
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

warnings.filterwarnings("ignore")


def get_current_python_path():
    """
    Return output of which python
    """
    import subprocess

    try:
        result = subprocess.run(
            ["which", "python"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error getting Python path: {e}")
        return None


def _import_unsloth(gpu: int) -> None|List[Any]:
    import os
    os.makedirs(f".cache/", exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["UNSLOTH_COMPILE_LOCATION"] = f".cache/UNSLOTH_CACHE_DIR_{gpu}"

    try:
        unsloth_is_not_imported = "unsloth" not in ''.join(sys.modules.keys())
        trl_is_not_imported = "trl" not in ''.join(sys.modules.keys())
        assert unsloth_is_not_imported 
        assert trl_is_not_imported
        #====
        import unsloth # unsloth must be import before trl
        from trl import SFTConfig, SFTTrainer
        print("Unsloth version:", unsloth.__version__)
        return unsloth, SFTConfig, SFTTrainer
    except AttributeError as e:
        import warnings
        if "Unsloth" in str(e) and "has no attribute" in str(e):
            warnings.warn(
                f"[OpenSloth] Unsloth RL patching error detected: {e}\n"
                "This is likely due to a corrupted or stale compiled cache. "
                "consider `rm -r unsloth_compiled_cache` and try again"
            )
        raise

def train_on_single_gpu_grpo(
    gpu: int, opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments
):
    """
    GRPO training on a single GPU using native TRL GRPOTrainer + Unsloth.
    This is a thin wrapper around the Unsloth tutorial pattern.
    """
    
    os.environ["OPENSLOTH_LOCAL_RANK"] = str(opensloth_config.devices.index(gpu))
    logger = OpenslothLogger()
    
    logger.info(f"GRPO training on GPU {gpu} with output_dir {hf_train_args.output_dir}")
    logger.start_total_training_timer()
    
    try:
        # Setup GRPO training (follows Unsloth tutorial exactly)
        from opensloth.unsloth_grpo_trainer import setup_grpo_training, run_grpo_training
        
        trainer, model, tokenizer = setup_grpo_training(
            opensloth_config=opensloth_config,
            hf_train_args=hf_train_args,
            logger=logger,
            gpu=gpu
        )
        
        # Run GRPO training with multi-GPU support
        run_grpo_training(
            trainer=trainer,
            model=model,
            tokenizer=tokenizer,
            logger=logger,
            gpu=gpu,
            opensloth_config=opensloth_config
        )
        
    except Exception as e:
        logger.error(f"GRPO training failed: {e}")
        raise
    
    # Log training summary
    logger.log_training_summary()


def train_on_single_gpu(
    gpu: int, opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments
):
    
    # Route to GRPO-specific implementation if training_type is grpo
    if opensloth_config.training_type == "grpo":
        return train_on_single_gpu_grpo(gpu, opensloth_config, hf_train_args)
    
    _import_unsloth(gpu)
    from opensloth.opensloth_trainer_setup import setup_model_and_training

    os.environ["OPENSLOTH_LOCAL_RANK"] = str(opensloth_config.devices.index(gpu))
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
    )
    logger.finish_timing("model_and_training_setup")

    # Register a debug hook to assert batch shape consistency early with rich logs.
    def _register_batch_shape_assertion(_model, _logger: OpenslothLogger):
        """
        Asserts that input_ids, labels, and attention_mask (if present) have
        compatible shapes before the model forward. If a mismatch is detected,
        raises AssertionError with detailed context so issues are obvious
        instead of failing deep inside fused CE.
        """

        def _pre_hook(_mod, _args, _kwargs):
            try:
                input_ids = _kwargs.get("input_ids")
                labels = _kwargs.get("labels")
                attention_mask = _kwargs.get("attention_mask")

                # Only validate when training with labels
                if input_ids is None or labels is None:
                    return

                # Shapes
                try:
                    ids_shape = tuple(input_ids.size())
                    lbl_shape = tuple(labels.size())
                except Exception:
                    # If objects don't have size (unlikely), skip
                    return

                msg_prefix = "[OpenSloth Debug] Batch shape check failed: "

                # Basic dimension count check
                if labels.dim() != input_ids.dim():
                    raise AssertionError(
                        f"{msg_prefix}ndim mismatch -> input_ids {ids_shape} vs labels {lbl_shape}"
                    )

                # Typical LM training expects equal (B, L). If not, raise with advice.
                if ids_shape != lbl_shape:
                    advice = ""
                    if len(ids_shape) == 2 and len(lbl_shape) == 2:
                        if lbl_shape[-1] + 1 == ids_shape[-1] or lbl_shape[-1] == ids_shape[-1] + 1:
                            advice = "Possible off-by-one from shifting labels/padding. Ensure labels align to input_ids."
                    # Non-ignored label count can reveal masking issues
                    try:
                        non_ignored = int((labels != -100).sum().item())
                    except Exception:
                        non_ignored = -1
                    raise AssertionError(
                        f"{msg_prefix}input_ids {ids_shape} != labels {lbl_shape}. {advice} "
                        f"non_ignored_labels={non_ignored} total_labels={getattr(labels, 'numel', lambda: 'n/a')()}"
                    )

                # attention_mask, if provided, is valid when:
                # - 2D and equals (B, L)
                # - 3D causal mask (B, L, L)
                if attention_mask is not None:
                    am_shape = tuple(attention_mask.size())
                    if len(am_shape) == 2:
                        if am_shape != ids_shape:
                            raise AssertionError(
                                f"{msg_prefix}attention_mask {am_shape} != input_ids {ids_shape}"
                            )
                    elif len(am_shape) == 3:
                        b, l = ids_shape
                        if not (am_shape[0] == b and am_shape[1] == l and am_shape[2] == l):
                            raise AssertionError(
                                f"{msg_prefix}3D attention_mask {am_shape} expected (B, L, L) matching input_ids {ids_shape}"
                            )
                    else:
                        raise AssertionError(
                            f"{msg_prefix}unsupported attention_mask ndim={len(am_shape)} shape={am_shape}"
                        )

                # Sanity: make sure we have some tokens to learn from
                try:
                    non_ignored = int((labels != -100).sum().item())
                    if non_ignored == 0:
                        raise AssertionError(
                            f"{msg_prefix}all labels are -100 (ignored). Check response-only masking and packing."
                        )
                except Exception:
                    pass

            except AssertionError as e:
                gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                rank_env = os.environ.get("OPENSLOTH_LOCAL_RANK", "?")
                _logger.error(f"[GPU {gpu_env} | local_rank {rank_env}] {e}")
                # Extra stats (best-effort)
                try:
                    ids_dtype = getattr(locals().get("input_ids", None), "dtype", None)
                    ids_device = getattr(locals().get("input_ids", None), "device", None)
                    lbl_dtype = getattr(locals().get("labels", None), "dtype", None)
                    lbl_device = getattr(locals().get("labels", None), "device", None)
                    _logger.error(
                        f"Input stats: ids dtype={ids_dtype} device={ids_device} | labels dtype={lbl_dtype} device={lbl_device}"
                    )
                except Exception:
                    pass
                raise

        try:
            handle = _model.register_forward_pre_hook(_pre_hook, with_kwargs=True)  # type: ignore[arg-type]
            setattr(_model, "_opensloth_debug_shape_hook", handle)  # keep it alive
            _logger.info("Registered batch shape assertion hook")
        except TypeError:
            # Older PyTorch may not support with_kwargs; fall back to positional
            def _pre_hook_no_kwargs(_mod, _args):
                return _pre_hook(_mod, _args, {})

            handle = _model.register_forward_pre_hook(_pre_hook_no_kwargs)  # type: ignore[arg-type]
            setattr(_model, "_opensloth_debug_shape_hook", handle)
            _logger.info("Registered batch shape assertion hook (positional)")

    _register_batch_shape_assertion(trainer.model, logger)

    assert trainer.model is not None, "Trainer model is None"

    # Only use NCCL gradient sync for multi-GPU training
    if len(opensloth_config.devices) > 1:
        from opensloth.nccl_grad_sync import get_callback_and_setup_method

        nccl_grad_sync_callback, setup_nccl_for_opensloth = get_callback_and_setup_method()

        grad_sync_cb = nccl_grad_sync_callback(
            model=trainer.model,
            gpu=gpu,
            gpus=opensloth_config.devices,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)
    else:
        logger.info("Single GPU training detected, skipping NCCL gradient sync")

    logger.start_timing("actual_training")
    logger.debug(f"Environment: {os.environ}")
    # Patch: Only resume from checkpoint if a valid path is provided
    resume_from_checkpoint = getattr(hf_train_args, 'resume_from_checkpoint', None)
    if resume_from_checkpoint and hasattr(trainer, "train") and "resume_from_checkpoint" in trainer.train.__code__.co_varnames:  # type: ignore[attr-defined]
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore
    else:
        # GRPO custom path: UnslothGRPOTrainer doesn't accept max_steps in train()
        if opensloth_config.training_type == "grpo" and hasattr(trainer, "train"):
            # For GRPO, max_steps should be in the config, not passed to train()
            try:
                trainer.train()  # type: ignore
            except Exception as e:
                logger.error(f"GRPO training failed: {e}")
                # Try fallback with max_steps if the trainer supports it
                try:
                    max_steps = getattr(hf_train_args, "max_steps", None)
                    if max_steps is None or max_steps <= 0:
                        # fallback derive from epochs * dataset size / batch
                        try:
                            ds_size = len(getattr(trainer, "train_dataset", []))
                            per_dev = hf_train_args.per_device_train_batch_size
                            gas = hf_train_args.gradient_accumulation_steps
                            max_steps = max(1, int(ds_size / max(1, per_dev * gas)))
                        except Exception:
                            max_steps = 100
                    trainer.train(max_steps=max_steps)  # type: ignore
                except Exception as e2:
                    logger.error(f"GRPO training fallback also failed: {e2}")
                    raise e  # Re-raise original error
        else:
            trainer.train()
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
        opensloth_config = OpenSlothConfig()
    # return opensloth_config,
    if hasattr(config_module, "training_config"):
        training_config = config_module.training_config
    elif hasattr(config_module, "training_config"):
        training_config = TrainingArguments(**config_module.training_config)
    else:
        raise ValueError("No training configuration found")
    return opensloth_config, training_config


# We'll just detect if the user wants a tmux script:


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

    # First GPU
    # check tmux session command, if yes, ask user enter "y" to kill the session
    # check_if_session_exists_then_ask_to_kill = f"tmux has-session -t {session_name}
    # && read -p 'Session exists, kill it? (y/n): ' kill_session &&
    #  [ $kill_session == 'y' ] && tmux kill-session -t {session_name}"
    # lines.append(check_if_session_exists_then_ask_to_kill)
    # Remaining GPUs
    for local_rank, gpu_index in enumerate(gpus):
        cmd = (
            f"USE_TMUX=0 "
            f"{get_current_python_path()} {sys.argv[0]} "
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
            (f"Session {session_name} exists, please kill it before running the script")
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
    training_config: TrainingArguments,
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
    os.environ["OPENSLOTH_WORLD_SIZE"] = str(len(opensloth_config.devices))
    os.environ["OPENSLOTH_FORWARD_BZ"] = str(
        training_config.per_device_train_batch_size
        # * training_config.gradient_accumulation_steps
        * len(opensloth_config.devices)
    )
    os.environ["OPENSLOTH_GLOBAL_BZ"] = str(
        training_config.per_device_train_batch_size
        * training_config.gradient_accumulation_steps
        * len(opensloth_config.devices)
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



def main():
    parser = argparse.ArgumentParser(description="OpenSloth SFT Trainer")
    parser.add_argument("config_file", type=str, help="Path to config file")
    parser.add_argument("--rank", type=int, default=None, help="Local rank for distributed training")
    parser.add_argument("--world_size", type=int, default=None, help="World size for distributed training")
    parser.add_argument("--tmux", type=str, default=None, help="tmux session name")
    parser.add_argument("-y", action="store_true", help="Auto-kill existing tmux session")
    args = parser.parse_args()

    opensloth_config, training_config = initialize_training_config(args.config_file)

    # CASE 1: Child process => single GPU
    if args.rank is not None and args.world_size is not None:
        print(f"[CASE 1] Running on rank {args.rank} with world size {args.world_size}")
        train_on_single_gpu(
            gpu=opensloth_config.devices[args.rank],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
        return

    # CASE 2: Top-level process => spawn multi-GPU or single GPU
    if len(opensloth_config.devices) > 1:
        if os.environ.get("USE_TMUX", "0") == "1" or args.tmux is not None:
            session_name = args.tmux if args.tmux is not None else "train_hp"
            run_tmux_training(
                session_name=session_name,
                config_file=args.config_file,
                training_config=training_config,
                gpus=opensloth_config.devices,
                auto_kill=args.y,
            )
        else:
            run_mp_training(
                gpus=opensloth_config.devices,
                opensloth_config=opensloth_config,
                training_config=training_config,
            )
    else:
        # Single GPU
        assert args.tmux is None, "Cannot use tmux with a single GPU"
        train_on_single_gpu(
            gpu=opensloth_config.devices[0],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )


if __name__ == "__main__":
    main()
