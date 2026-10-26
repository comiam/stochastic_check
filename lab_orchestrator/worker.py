"""Execute scheduler tasks in a child process assigned to one GPU slot."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import time

from lab_orchestrator.task import Task

logger = logging.getLogger(__name__)


def _physical_gpu_id(gpu_id: int, visible_devices: str | None) -> str:
    """Resolve a logical index within the parent's CUDA visibility mask."""
    if visible_devices is None or visible_devices == "":
        return str(gpu_id)

    visible = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if 0 <= gpu_id < len(visible):
        return visible[gpu_id]

    return str(gpu_id)


def _environment(gpu_id: int) -> dict[str, str]:
    """Limit child visibility to one physical GPU and retain deterministic cuBLAS."""
    physical_id = _physical_gpu_id(gpu_id, os.environ.get("CUDA_VISIBLE_DEVICES"))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = physical_id
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return environment


def _execute(
    task: Task,
    gpu_id: int,
    environment: dict[str, str],
    log_dir: str,
    cwd: str | None,
    timeout: float | None,
) -> tuple[int, float]:
    """Run one subprocess and translate timeouts/OS errors into worker results."""
    log_name = f"gpu{gpu_id}_{task.name}_seed{task.seed}.log"
    log_path = os.path.join(log_dir, log_name)
    started = time.monotonic()

    try:
        with open(log_path, "w") as log_file:
            process = subprocess.run(
                task.cmd,
                env=environment,
                cwd=cwd,
                stdout=log_file,
                stderr=log_file,
                timeout=timeout,
            )
        code = process.returncode

    except subprocess.TimeoutExpired:
        code = -1
        logger.warning("GPU %d  TIMEOUT after %.0fs  %s", gpu_id, timeout, task.label)

    except OSError as exc:
        code = -2
        logger.error("GPU %d  OS error for %s: %s", gpu_id, task.label, exc)

    return code, time.monotonic() - started


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    log_dir: str,
    *,
    cwd: str | None = None,
    timeout: float | None = None,
) -> None:
    """Process tasks with READY/START/DONE events until receiving None.

    CUDA visibility uses the parent's logical-to-physical mapping, avoiding
    accidental assignment to another physical GPU when a mask is present.
    """
    environment = _environment(gpu_id)
    os.makedirs(log_dir, exist_ok=True)

    while True:
        result_queue.put(("READY", gpu_id))
        task: Task | None = task_queue.get()
        if task is None:
            break

        result_queue.put(("START", gpu_id, task.label, task.weight, os.getpid()))
        code, elapsed = _execute(task, gpu_id, environment, log_dir, cwd, timeout)
        result_queue.put(
            ("DONE", gpu_id, task.label, code, elapsed, task.weight, os.getpid())
        )
