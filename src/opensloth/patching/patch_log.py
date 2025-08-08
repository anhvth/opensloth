import os
import time

import numpy as np
from fastcore.all import patch
from filelock import BaseFileLock, FileLock
from transformers.trainer_utils import speed_metrics

TIME_OUT = 300
SLEEP_TIME = 0.01
WAIT_WARNING_THRESHOLD = 2


class Flag:
    def __init__(self, world_size: int, file_path: str, is_master: bool = False):
        self.world_size = world_size
        self.file_path = file_path
        self.is_master = is_master
        self.lock_path = self.file_path + ".lock"
        self.lock = FileLock(self.lock_path)

        if self.is_master:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with self.lock, open(self.file_path, "wb") as f:
                f.truncate(world_size * 4)
        else:
            self._wait_for_file()

        self._create_memmap()
        if self.is_master:
            self.reset()

    def _wait_for_file(self) -> None:
        t0 = time.time()
        while not os.path.exists(self.file_path):
            time.sleep(SLEEP_TIME)
            if time.time() - t0 > TIME_OUT:
                raise TimeoutError(f"Worker timed out waiting for {self.file_path}")

    def _create_memmap(self) -> None:
        t0 = time.time()
        while True:
            try:
                if os.path.getsize(self.file_path) == self.world_size * 4:
                    self.mem = np.memmap(
                        self.file_path,
                        dtype="float32",
                        mode="r+",
                        shape=(self.world_size,),
                    )
                    return
                time.sleep(SLEEP_TIME)
            except (FileNotFoundError, Exception):
                time.sleep(SLEEP_TIME)

            if time.time() - t0 > TIME_OUT:
                raise TimeoutError(f"Timeout creating memmap for {self.file_path}")

    def update(self, rank: int) -> None:
        with self.lock:
            self.mem[rank] = 1.0
            self.mem.flush()

    def wait_for_all(self, step: int = -1, timeout: float = TIME_OUT) -> None:
        t0 = time.time()
        has_logged = False

        while True:
            with self.lock:
                if np.all(self.mem == 1.0):
                    return

            elapsed = time.time() - t0
            if elapsed > WAIT_WARNING_THRESHOLD and not has_logged:
                print(f"[Flag] waiting {elapsed:.1f}s at step={step}")
                has_logged = True

            if elapsed > timeout:
                raise RuntimeError(f"Timeout after {elapsed:.1f}s at step={step}")

            time.sleep(SLEEP_TIME)

    def wait_for_reset(
        self, rank: int, step: int = -1, timeout: float = TIME_OUT
    ) -> None:
        t0 = time.time()
        has_logged = False

        while True:
            with self.lock:
                if self.mem[rank] == 0.0:
                    return

            elapsed = time.time() - t0
            if elapsed > WAIT_WARNING_THRESHOLD and not has_logged:
                print(f"[Flag] rank={rank} waiting reset {elapsed:.1f}s at step={step}")
                has_logged = True

            if elapsed > timeout:
                raise RuntimeError(f"Timeout waiting reset at step={step}")

            time.sleep(SLEEP_TIME)

    def reset(self) -> None:
        if not self.is_master:
            raise RuntimeError("Only master can reset")
        with self.lock:
            self.mem[:] = 0.0
            self.mem.flush()


def patch_log(trainer):
    support_keys = [
        "loss",
        "grad_norm",
    ]
    log_mmap: dict[str, np.memmap] = {}
    log_locks: dict[str, BaseFileLock] = {}

    try:
        local_rank = int(os.environ["OPENSLOTH_LOCAL_RANK"])
        world_size = int(os.environ["OPENSLOTH_WORLD_SIZE"])
        log_cache_dir = os.environ["OPENSLOTH_OUTPUT_DIR"]
        is_main = local_rank == 0

        print(f"[{local_rank=}] Patching log. Dir: {log_cache_dir}, GPUs: {world_size}")

        if is_main:
            os.makedirs(log_cache_dir, exist_ok=True)
        else:
            _wait_for_directory(log_cache_dir, local_rank)

        _initialize_mmaps(
            support_keys,
            log_cache_dir,
            world_size,
            local_rank,
            is_main,
            log_mmap,
            log_locks,
        )

        log_sync_flag = Flag(
            world_size=world_size,
            file_path=f"{log_cache_dir}/log_sync_flag.dat",
            is_master=is_main,
        )
        print(f"[{local_rank=}] Log patch initialization complete.")

    except Exception as e:
        print(f"[{local_rank=}] CRITICAL ERROR during initialization: {e}")
        raise e

    @patch
    def log(
        self:type(trainer), logs: dict[str, float], start_time: float | None = None
    ) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None and is_main:
                speed_metrics(
                    "train", start_time, num_tokens=self.state.num_input_tokens_seen
                )

        # opensloth specific fields
        for attr in [
            "trained_token_ratio",
            "non_padding_ratio_before",
            "non_padding_ratio_after",
        ]:
            if hasattr(self.state, attr):
                logs[attr] = getattr(self.state, attr)

        output = {**logs, **{"step": self.state.global_step}}
        current_step = self.state.global_step
        self.state.log_history.append(output)

        if "loss" in logs:
            try:
                _write_logs_to_mmap(logs, support_keys, log_mmap, log_locks, local_rank)
                log_sync_flag.update(local_rank)

                if is_main:
                    _handle_master_logging(
                        log_sync_flag,
                        support_keys,
                        log_mmap,
                        log_locks,
                        logs,
                        current_step,
                        self,
                    )
                else:
                    log_sync_flag.wait_for_reset(rank=local_rank, step=current_step)

            except Exception as e:
                print(f"Rank {local_rank} logging error at step {current_step}: {e}")

        elif is_main:
            self.control = self.callback_handler.on_log(
                self.args, self.state, self.control, logs
            )
    return trainer


def _wait_for_directory(cache_dir: str, rank: int) -> None:
    t0 = time.time()
    while not os.path.exists(cache_dir):
        time.sleep(SLEEP_TIME)
        if time.time() - t0 > 60:
            raise TimeoutError(f"Worker {rank} timed out waiting for {cache_dir}")


def _initialize_mmaps(
    support_keys: list,
    cache_dir: str,
    world_size: int,
    rank: int,
    is_main: bool,
    log_mmap: dict,
    log_locks: dict,
) -> None:
    for key in support_keys:
        mmap_path = f"{cache_dir}/log_{key}.mmap"
        log_locks[key] = FileLock(mmap_path + ".lock")

        if is_main:
            with log_locks[key], open(mmap_path, "wb") as f:
                f.truncate(world_size * 4)
        else:
            t0 = time.time()
            while not os.path.exists(mmap_path):
                time.sleep(SLEEP_TIME)
                if time.time() - t0 > TIME_OUT:
                    raise TimeoutError(
                        f"Worker {rank} timed out waiting for {mmap_path}"
                    )

        _create_mmap(mmap_path, world_size, rank, log_mmap, key)

        if is_main:
            with log_locks[key]:
                log_mmap[key][:] = 0.0
                log_mmap[key].flush()


def _create_mmap(
    mmap_path: str, world_size: int, rank: int, log_mmap: dict, key: str
) -> None:
    t0 = time.time()
    expected_size = world_size * 4

    while True:
        try:
            if (
                os.path.exists(mmap_path)
                and os.path.getsize(mmap_path) == expected_size
            ):
                log_mmap[key] = np.memmap(
                    mmap_path, dtype="float32", mode="r+", shape=(world_size,)
                )
                return
            time.sleep(SLEEP_TIME)
        except (FileNotFoundError, Exception):
            time.sleep(SLEEP_TIME)

        if time.time() - t0 > TIME_OUT:
            raise TimeoutError(f"Rank {rank} timeout creating mmap {mmap_path}")


def _write_logs_to_mmap(
    logs: dict[str, float],
    support_keys: list,
    log_mmap: dict,
    log_locks: dict,
    rank: int,
) -> None:
    for key in support_keys:
        if key in logs:
            with log_locks[key]:
                log_mmap[key][rank] = logs[key]
                log_mmap[key].flush()


def _aggregate_logs(
    logs: dict[str, float], support_keys: list, log_mmap: dict, log_locks: dict
) -> dict[str, float]:
    aggregated = logs.copy()

    for key in support_keys:
        with log_locks[key]:
            all_vals = log_mmap[key].copy()

        if key in ["num_input_tokens_seen"]:
            aggregated[key] = int(all_vals.sum())
        elif key in ["trained_token_ratio"]:
            valid_vals = all_vals[all_vals != 0.0]
            aggregated[key] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0
        else:
            aggregated[key] = float(all_vals.sum())

    return aggregated


def _reset_mmaps(support_keys: list, log_mmap: dict, log_locks: dict) -> None:
    for key in support_keys:
        with log_locks[key]:
            log_mmap[key][:] = 0.0
            log_mmap[key].flush()


def _handle_master_logging(
    flag,
    support_keys: list,
    log_mmap: dict,
    log_locks: dict,
    logs: dict,
    step: int,
    trainer,
) -> None:
    flag.wait_for_all(step=step)
    aggregated_logs = _aggregate_logs(logs, support_keys, log_mmap, log_locks)

    trainer.control = trainer.callback_handler.on_log(
        trainer.args, trainer.state, trainer.control, aggregated_logs
    )

    _reset_mmaps(support_keys, log_mmap, log_locks)
    flag.reset()


__all__ = ["patch_log"]
