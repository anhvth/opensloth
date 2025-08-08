import importlib.util
import os
import sys
import time
import warnings

from fastcore.all import call_parse

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


def train_on_single_gpu(
    gpu: int, opensloth_config: OpenSlothConfig, hf_train_args: TrainingArguments
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
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

    assert trainer.model is not None, "Trainer model is None"

    # Only use NCCL gradient sync for multi-GPU training
    if len(opensloth_config.devices) > 1:
        from opensloth.nccl_grad_sync import get_callback_and_setup_method

        NCCLGradSyncCallback, setup_nccl_for_opensloth = get_callback_and_setup_method()

        grad_sync_cb = NCCLGradSyncCallback(
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
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
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
    output_dir: str,
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
        training_config.output_dir,
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


@call_parse
def train(
    config_file: str,
    rank: int = None,
    world_size: int = None,
    tmux: str = None,
    y: bool = False,
):
    opensloth_config, training_config = initialize_training_config(config_file)

    # CASE 1: Child process => single GPU
    if rank is not None and world_size is not None:
        print(f"[CASE 1] Running on rank {rank} with world size {world_size}")
        train_on_single_gpu(
            gpu=opensloth_config.devices[rank],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
        return

    # CASE 2: Top-level process => spawn multi-GPU or single GPU

    # If multiple GPUs:
    if len(opensloth_config.devices) > 1:
        if os.environ.get("USE_TMUX", "0") == "1" or tmux is not None:
            session_name = tmux if tmux is not None else "train_hp"
            run_tmux_training(
                session_name=session_name,
                config_file=config_file,
                training_config=training_config,
                gpus=opensloth_config.devices,
                auto_kill=y,
            )
        else:
            run_mp_training(
                gpus=opensloth_config.devices,
                opensloth_config=opensloth_config,
                training_config=training_config,
            )
    else:
        # Single GPU
        assert tmux is None, "Cannot use tmux with a single GPU"
        train_on_single_gpu(
            gpu=opensloth_config.devices[0],
            opensloth_config=opensloth_config,
            hf_train_args=training_config,
        )
