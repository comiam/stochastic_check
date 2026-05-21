"""Load-aware LPT scheduler with per-GPU demand-driven dispatch.

Dispatches :class:`Task` objects across multiple GPUs.  A centralised
dispatcher in the main process assigns each new task to the GPU whose
active workload (sum of ``Task.weight``) is smallest - load-aware
scheduling.

Tasks should be pre-sorted by weight descending (LPT) so heavy tasks
start early and short ones fill gaps at the end.
"""

from __future__ import annotations

import heapq
import logging
import multiprocessing as mp
import os
import queue
import time

from lab_orchestrator.progress import (
    ProgressFormatter,
    set_done_tag,
    update_progress,
)
from lab_orchestrator.task import Task
from lab_orchestrator.worker import gpu_worker

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5.0  # seconds between worker liveness checks


# ---------------------------------------------------------------------------
#  Internal dispatch
# ---------------------------------------------------------------------------


def _dispatch_pending(
    task_pool: list[Task],
    pending_ready: dict[int, int],
    gpu_active_weight: dict[int, float],
    gpu_queues: dict[int, mp.Queue],  # type: ignore[type-arg]
    done_names: set[str],
) -> int:
    """Send queued tasks to ready workers on the least-loaded GPU.

    Tasks with unsatisfied :attr:`Task.depends_on` are skipped over until
    their dependencies appear in *done_names*.  Returns the number of
    poison pills sent (when *task_pool* is exhausted).
    """
    pills = 0
    while task_pool:
        candidates = [g for g, n in pending_ready.items() if n > 0]
        if not candidates:
            break
        idx = next(
            (i for i, t in enumerate(task_pool) if all(d in done_names for d in t.depends_on)),
            None,
        )
        if idx is None:
            break  # only dep-blocked tasks remain; wait for in-flight to finish
        task = task_pool.pop(idx)
        best = min(candidates, key=lambda g: (gpu_active_weight[g], g))
        gpu_queues[best].put(task)
        gpu_active_weight[best] += task.weight
        pending_ready[best] -= 1

    if not task_pool:
        for gpu_id in list(pending_ready):
            while pending_ready[gpu_id] > 0:
                gpu_queues[gpu_id].put(None)
                pending_ready[gpu_id] -= 1
                pills += 1

    return pills


# ---------------------------------------------------------------------------
#  Dry-run simulation
# ---------------------------------------------------------------------------


def _dry_run(
    gpu_ids: list[int],
    tasks: list[Task],
    workers_per_gpu: int,
) -> None:
    """Simulate scheduling and print estimated plan.

    Mirrors the real dispatcher: tasks are assigned to the GPU with the
    lowest ``gpu_active_weight`` among GPUs that have an idle worker.
    A heap-based timeline drives worker completion events so that
    ``--dry-run`` output matches the actual execution order.
    """
    # Completion events: (finish_time, gpu_id, task_weight, task_name).
    event_heap: list[tuple[float, int, float, str]] = []
    pending_ready: dict[int, int] = {g: workers_per_gpu for g in gpu_ids}
    gpu_active_weight: dict[int, float] = {g: 0.0 for g in gpu_ids}
    gpu_assignments: dict[int, list[str]] = {g: [] for g in gpu_ids}
    pending: list[Task] = list(tasks)
    done_names: set[str] = set()
    sim_time = 0.0

    while pending:
        # Find next dispatchable task: deps satisfied and a worker free.
        ready_idx = next(
            (i for i, t in enumerate(pending) if all(d in done_names for d in t.depends_on)),
            None,
        )
        if ready_idx is None or not any(n > 0 for n in pending_ready.values()):
            if not event_heap:
                logger.error("dry-run: deadlock detected (unsatisfiable dependencies)")
                break
            finish_time, g, w, fname = heapq.heappop(event_heap)
            sim_time = finish_time
            gpu_active_weight[g] -= w
            pending_ready[g] += 1
            done_names.add(fname)
            continue

        task = pending.pop(ready_idx)
        candidates = [g for g, n in pending_ready.items() if n > 0]
        best = min(candidates, key=lambda g: (gpu_active_weight[g], g))

        gpu_assignments[best].append(task.label)
        gpu_active_weight[best] += task.weight
        pending_ready[best] -= 1
        heapq.heappush(event_heap, (sim_time + task.weight, best, task.weight, task.name))

    # Drain remaining events to find per-GPU peak finish times.
    gpu_peak: dict[int, float] = {g: 0.0 for g in gpu_ids}
    while event_heap:
        finish_time, g, _w, _fname = heapq.heappop(event_heap)
        gpu_peak[g] = max(gpu_peak[g], finish_time)

    for g in gpu_ids:
        logger.info("GPU %d (%d workers, est %.1f h):", g, workers_per_gpu, gpu_peak[g])
        for label in gpu_assignments[g]:
            logger.info("    %s", label)

    bottleneck = max(gpu_peak.values()) if gpu_peak else 0.0
    logger.info(
        "Estimated wall-clock: %.1f h (bottleneck GPU, %d workers/gpu)",
        bottleneck,
        workers_per_gpu,
    )


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------


def run_schedule(
    gpu_ids: list[int],
    tasks: list[Task],
    dry_run: bool = False,
    *,
    cwd: str | None = None,
    log_dir: str | None = None,
    max_failures: int | None = None,
    pre_done: int = 0,
    pre_done_weight: float = 0.0,
    stall_warning: float | None = None,
    task_timeout: float | None = None,
    workers_per_gpu: int = 1,
) -> list[tuple[int, str, int, float]]:
    """Dispatch *tasks* across GPUs with load-aware scheduling.

    Parameters
    ----------
    gpu_ids : list[int]
        CUDA device indices to use.
    tasks : list[Task]
        Work units (should be sorted by weight descending - LPT).
    dry_run : bool
        Print estimated schedule without executing.
    cwd : str | None
        Working directory for subprocess workers.
    log_dir : str | None
        Base directory for per-task log files.  Defaults to ``./logs``.
    max_failures : int | None
        Stop scheduling new tasks after this many failures.
        *None* means no limit.
    pre_done, pre_done_weight : int, float
        Account for already-completed tasks (``--resume``) in progress.
    stall_warning : float | None
        Log a warning when a GPU has had no activity for this many
        seconds.  Useful for detecting stuck subprocesses when
        *task_timeout* is not set.  *None* disables the check.
    task_timeout : float | None
        Per-task timeout in seconds.  *None* means no limit.
    workers_per_gpu : int
        Concurrent workers per GPU.  Raise when tasks under-utilise GPU
        (e.g. small model + CPU-bound HVP): workers sharing a GPU
        overlap CPU autograd work with GPU kernels from the other worker.

    Returns
    -------
    list[tuple[int, str, int, float]]
        ``(gpu_id, label, return_code, elapsed_sec)`` per completed task.
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")

    if not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list of GPU indices")

    n_gpus = len(gpu_ids)
    n_tasks = len(tasks)
    total_weight = sum(t.weight for t in tasks)
    n_tasks_global = n_tasks + pre_done
    total_weight_global = total_weight + pre_done_weight

    logger.info(
        "GPUs: %s (%d)  |  Tasks: %d  |  Weight: %.1f h",
        gpu_ids,
        n_gpus,
        n_tasks,
        total_weight,
    )

    if dry_run:
        _dry_run(gpu_ids, tasks, workers_per_gpu)
        return []

    # Install progress-aware formatter on all handlers.
    fmt = ProgressFormatter("%(asctime)s [%(levelname)s] %(message)s")
    for h in logging.root.handlers:
        h.setFormatter(fmt)

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    run_log_dir = os.path.join(log_dir, run_ts)
    os.makedirs(run_log_dir, exist_ok=True)
    logger.info("Task logs: %s/gpu_*.log", run_log_dir)

    result_queue: mp.Queue = mp.Queue()  # type: ignore[type-arg]
    gpu_queues: dict[int, mp.Queue] = {g: mp.Queue() for g in gpu_ids}

    n_total_workers = len(gpu_ids) * workers_per_gpu
    task_pool: list[Task] = list(tasks)
    pending_ready: dict[int, int] = {g: 0 for g in gpu_ids}
    gpu_active_weight: dict[int, float] = {g: 0.0 for g in gpu_ids}
    pills_sent = 0
    done_names: set[str] = set()
    failed_names: set[str] = set()
    # pid -> (gpu_id, label, weight) for in-flight task tracking.
    active_by_pid: dict[int, tuple[int, str, float]] = {}
    worker_gpu: dict[int, int] = {}  # mp.Process.pid -> gpu_id
    # Stall detection: last time any message arrived from each GPU.
    last_activity: dict[int, float] = {}
    _warned_stall: set[int] = set()

    wall_t0 = time.monotonic()
    update_progress(
        pre_done,
        n_tasks_global,
        pre_done_weight,
        total_weight_global,
        wall_t0,
        0,
        pre_done_weight=pre_done_weight,
    )

    workers = []
    for g in gpu_ids:
        for wi in range(workers_per_gpu):
            p = mp.Process(
                target=gpu_worker,
                args=(g, gpu_queues[g], result_queue, run_log_dir),
                kwargs={"cwd": cwd, "timeout": task_timeout},
                daemon=True,
            )
            p.start()
            workers.append(p)
            assert p.pid is not None  # guaranteed after start()
            worker_gpu[p.pid] = g
            if workers_per_gpu > 1:
                logger.info("GPU %d  worker %d/%d started", g, wi + 1, workers_per_gpu)
            else:
                logger.info("GPU %d  worker started", g)

    # ---- Main loop ----
    results: list[tuple[int, str, int, float]] = []
    done_count = 0
    done_weight = 0.0
    fail_count = 0

    while done_count < n_tasks:
        try:
            batch: list[tuple] = [result_queue.get(timeout=_POLL_INTERVAL)]
        except queue.Empty:
            # Check for individually dead workers with in-flight tasks.
            for p in workers:
                if p.is_alive() or p.pid not in active_by_pid:
                    continue
                gpu_id, label, weight = active_by_pid.pop(p.pid)
                gpu_active_weight[gpu_id] -= weight
                done_count += 1
                done_weight += weight
                fail_count += 1
                failed_names.add(label.rsplit("[seed=", 1)[0])
                results.append((gpu_id, label, -9, 0.0))
                logger.error(
                    "GPU %d  worker (pid %d) died while running %s",
                    gpu_id,
                    p.pid,
                    label,
                )
            if not any(p.is_alive() for p in workers):
                remaining = n_tasks - done_count
                if remaining > 0:
                    logger.error("All workers died with %d tasks remaining", remaining)
                break
            # Stall detection: warn if a GPU has no activity for too long.
            if stall_warning is not None:
                now = time.monotonic()
                for g in gpu_ids:
                    idle = now - last_activity.get(g, wall_t0)
                    if idle >= stall_warning and g not in _warned_stall:
                        _warned_stall.add(g)
                        labels = [info[1] for info in active_by_pid.values() if info[0] == g]
                        logger.warning(
                            "GPU %d  no activity for %.0fs  (in-flight: %s)",
                            g,
                            idle,
                            ", ".join(labels) or "none",
                        )
            continue
        try:
            while True:
                batch.append(result_queue.get_nowait())
        except queue.Empty:
            pass

        for msg in batch:
            if msg[0] == "READY":
                last_activity[msg[1]] = time.monotonic()
                _warned_stall.discard(msg[1])
                pending_ready[msg[1]] += 1
                continue

            if msg[0] == "START":
                _, gpu_id, label, _weight, pid = msg
                last_activity[gpu_id] = time.monotonic()
                _warned_stall.discard(gpu_id)
                active_by_pid[pid] = (gpu_id, label, _weight)
                logger.info("GPU %d  >> %s", gpu_id, label)
                continue

            # msg[0] == "DONE"
            _, gpu_id, label, rc, elapsed, weight, done_pid = msg
            last_activity[gpu_id] = time.monotonic()
            _warned_stall.discard(gpu_id)
            active_by_pid.pop(done_pid, None)
            gpu_active_weight[gpu_id] -= weight
            done_count += 1
            done_weight += weight
            # ``label`` is ``Task.label`` == "<name>[seed=<seed>]"; dependency
            # ids are bare ``Task.name`` values, so strip the seed suffix.
            task_name = label.rsplit("[seed=", 1)[0]
            if rc == 0:
                done_names.add(task_name)
            else:
                failed_names.add(task_name)
                fail_count += 1
            results.append((gpu_id, label, rc, elapsed))

            update_progress(
                pre_done + done_count,
                n_tasks_global,
                pre_done_weight + done_weight,
                total_weight_global,
                wall_t0,
                fail_count,
                pre_done_weight=pre_done_weight,
            )

            status = "OK" if rc == 0 else f"FAIL(rc={rc})"
            logger.info("GPU %d  %s  %s  (%.1f min)", gpu_id, status, label, elapsed / 60)

        # Stop dispatching new tasks when failure threshold is reached.
        if max_failures is not None and fail_count >= max_failures:
            if task_pool:
                skipped = len(task_pool)
                task_pool.clear()
                logger.error(
                    "max_failures=%d reached (%d failures) - dropping %d remaining tasks",
                    max_failures,
                    fail_count,
                    skipped,
                )

        # Cascade-skip tasks whose dependencies have failed.  Keep
        # iterating: a skipped task's name goes into ``failed_names`` and
        # may transitively block further tasks.
        if failed_names:
            changed = True
            while changed:
                changed = False
                remaining: list[Task] = []
                for t in task_pool:
                    blocked_by = [d for d in t.depends_on if d in failed_names]
                    if blocked_by:
                        failed_names.add(t.name)
                        fail_count += 1
                        done_count += 1
                        done_weight += t.weight
                        results.append((-1, t.label, -2, 0.0))
                        logger.error(
                            "SKIP  %s  (dependency failed: %s)",
                            t.label,
                            ", ".join(blocked_by),
                        )
                        changed = True
                    else:
                        remaining.append(t)
                task_pool = remaining

        pills_sent += _dispatch_pending(
            task_pool, pending_ready, gpu_active_weight, gpu_queues, done_names
        )

    # Retire workers that became ready after last task finished.
    while pills_sent < n_total_workers:
        try:
            msg = result_queue.get(timeout=_POLL_INTERVAL)
        except queue.Empty:
            if not any(p.is_alive() for p in workers):
                break
            continue
        if msg[0] == "READY":
            gpu_queues[msg[1]].put(None)
            pills_sent += 1

    for p in workers:
        p.join(timeout=10)

    # ---- Summary ----
    set_done_tag()

    failed = [r for r in results if r[2] != 0]
    if failed:
        logger.error("FAILED tasks (%d):", len(failed))
        for gpu_id, label, rc, _elapsed in failed:
            logger.error("  GPU %d  %s  rc=%d", gpu_id, label, rc)
    else:
        logger.info("All %d tasks completed successfully.", len(results))

    total_gpu_h = sum(r[3] for r in results) / 3600
    wall_h = (time.monotonic() - wall_t0) / 3600
    logger.info("Total GPU-hours: %.1f  |  Wall-clock: %.1f h", total_gpu_h, wall_h)
    logger.info("Detailed logs: %s/", run_log_dir)

    return results
