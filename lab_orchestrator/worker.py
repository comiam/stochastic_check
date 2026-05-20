"""Subprocess worker executed in a child process (one per GPU slot)."""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import subprocess
import time

from lab_orchestrator.task import Task

logger = logging.getLogger(__name__)


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,  # type: ignore[type-arg]
    result_queue: mp.Queue,  # type: ignore[type-arg]
    log_dir: str,
    *,
    cwd: str | None = None,
    timeout: float | None = None,
) -> None:
    """Pull tasks from *task_queue*, run each on *gpu_id*.

    Protocol:
    - Sends ``("READY", gpu_id)`` when idle.
    - Sends ``("START", gpu_id, label, weight, pid)`` before execution.
    - Sends ``("DONE", gpu_id, label, return_code, elapsed, weight, pid)``
      after completion.
    - Exits on ``None`` (poison pill).

    Each task is run as a subprocess with ``CUDA_VISIBLE_DEVICES`` set.
    Stdout/stderr are captured to a per-task log file in *log_dir*.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    os.makedirs(log_dir, exist_ok=True)

    while True:
        result_queue.put(("READY", gpu_id))
        task: Task | None = task_queue.get()
        if task is None:
            break

        result_queue.put(("START", gpu_id, task.label, task.weight, os.getpid()))

        log_name = f"gpu{gpu_id}_{task.name}_seed{task.seed}.log"
        log_path = os.path.join(log_dir, log_name)

        t0 = time.monotonic()
        try:
            with open(log_path, "w") as log_file:
                proc = subprocess.run(
                    task.cmd,
                    env=env,
                    cwd=cwd,
                    stdout=log_file,
                    stderr=log_file,
                    timeout=timeout,
                )
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = -1
            logger.warning(
                "GPU %d  TIMEOUT after %.0fs  %s", gpu_id, timeout, task.label
            )
        except OSError as exc:
            rc = -2
            logger.error("GPU %d  OS error for %s: %s", gpu_id, task.label, exc)
        elapsed = time.monotonic() - t0

        result_queue.put(
            ("DONE", gpu_id, task.label, rc, elapsed, task.weight, os.getpid())
        )
