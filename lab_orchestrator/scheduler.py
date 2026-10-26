"""Dispatch tasks to ready workers on the least-loaded GPU.

Callers supply tasks in their desired order, normally descending weight
for LPT scheduling. Dependencies become ready only after successful
completion; workers communicate through READY, START, and DONE events.
"""

from __future__ import annotations

import heapq
import logging
import multiprocessing as mp
import os
import queue
import time
from typing import Any

from lab_orchestrator.progress import ProgressFormatter, set_done_tag, update_progress
from lab_orchestrator.task import Task
from lab_orchestrator.worker import gpu_worker

logger = logging.getLogger(__name__)
_POLL_INTERVAL = 5.0


def _ready_index(tasks: list[Task], done: set[str]) -> int | None:
    """Find the first task whose dependencies have succeeded."""
    return next(
        (
            index
            for index, task in enumerate(tasks)
            if all(dep in done for dep in task.depends_on)
        ),
        None,
    )


def _retire_idle(pending_ready, gpu_queues) -> int:
    """Send one termination marker to each currently idle worker."""
    pills = 0

    for gpu_id in list(pending_ready):
        while pending_ready[gpu_id] > 0:
            gpu_queues[gpu_id].put(None)
            pending_ready[gpu_id] -= 1
            pills += 1

    return pills


def _dispatch_pending(
    task_pool: list[Task],
    pending_ready: dict[int, int],
    gpu_active_weight: dict[int, float],
    gpu_queues: dict,
    done_names: set[str],
) -> int:
    """Send ready tasks to the least-loaded GPU; retire workers when exhausted."""
    while task_pool:
        candidates = [gpu for gpu, count in pending_ready.items() if count > 0]
        if not candidates:
            break

        index = _ready_index(task_pool, done_names)
        if index is None:
            break

        task = task_pool.pop(index)
        best = min(candidates, key=lambda gpu: (gpu_active_weight[gpu], gpu))
        gpu_queues[best].put(task)
        gpu_active_weight[best] += task.weight
        pending_ready[best] -= 1

    return _retire_idle(pending_ready, gpu_queues) if not task_pool else 0


def _report_simulation(gpu_ids, workers_per_gpu, assignments, event_heap) -> None:
    """Drain simulated completions and report the original estimated finish times."""
    peak = dict.fromkeys(gpu_ids, 0.0)
    while event_heap:
        finish, gpu, _, _ = heapq.heappop(event_heap)
        peak[gpu] = max(peak[gpu], finish)

    for gpu in gpu_ids:
        logger.info("GPU %d (%d workers, est %.1f h):", gpu, workers_per_gpu, peak[gpu])
        for label in assignments[gpu]:
            logger.info("    %s", label)

    bottleneck = max(peak.values()) if peak else 0.0
    logger.info(
        "Estimated wall-clock: %.1f h (bottleneck GPU, %d workers/gpu)",
        bottleneck,
        workers_per_gpu,
    )


def _dry_run(gpu_ids: list[int], tasks: list[Task], workers_per_gpu: int) -> None:
    """Simulate dependency-aware dispatch with a heap of completion events."""
    events: list[tuple[float, int, float, str]] = []
    ready = dict.fromkeys(gpu_ids, workers_per_gpu)
    active = dict.fromkeys(gpu_ids, 0.0)
    assignments: dict[int, list[str]] = {gpu: [] for gpu in gpu_ids}

    pending = list(tasks)
    done: set[str] = set()
    now = 0.0

    while pending:
        index = _ready_index(pending, done)

        if index is None or not any(count > 0 for count in ready.values()):
            if not events:
                logger.error("dry-run: deadlock detected (unsatisfiable dependencies)")
                break

            finish, gpu, weight, name = heapq.heappop(events)
            now = finish
            active[gpu] -= weight
            ready[gpu] += 1
            done.add(name)

            continue

        task = pending.pop(index)
        candidates = [gpu for gpu, count in ready.items() if count > 0]
        best = min(candidates, key=lambda gpu: (active[gpu], gpu))

        assignments[best].append(task.label)
        active[best] += task.weight
        ready[best] -= 1

        heapq.heappush(events, (now + task.weight, best, task.weight, task.name))

    _report_simulation(gpu_ids, workers_per_gpu, assignments, events)


def _prepare_logs(log_dir: str) -> str:
    """Install progress formatting and create this schedule's log directory."""
    formatter = ProgressFormatter("%(asctime)s [%(levelname)s] %(message)s")
    for handler in logging.root.handlers:
        handler.setFormatter(formatter)

    destination = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(destination, exist_ok=True)
    logger.info("Task logs: %s/gpu_*.log", destination)
    return destination


class _Schedule:
    """Own the mutable state of a single worker schedule."""

    def __init__(
        self,
        gpu_ids,
        tasks,
        workers_per_gpu,
        pre_done,
        pre_done_weight,
        max_failures,
        stall_warning,
    ) -> None:
        """Initialize queues and counters before starting any processes."""
        self.gpu_ids = gpu_ids
        self.tasks = list(tasks)
        self.workers_per_gpu = workers_per_gpu
        self.pre_done = pre_done
        self.pre_done_weight = pre_done_weight

        self.max_failures = max_failures
        self.stall_warning = stall_warning
        self.n_tasks = len(tasks)
        self.total_weight = sum(task.weight for task in tasks)

        self.result_queue: Any = mp.Queue()
        self.gpu_queues: dict[int, Any] = {gpu: mp.Queue() for gpu in gpu_ids}

        self.pending_ready = dict.fromkeys(gpu_ids, 0)
        self.active_weight = dict.fromkeys(gpu_ids, 0.0)
        self.done_names: set[str] = set()
        self.failed_names: set[str] = set()
        self.active_by_pid: dict[int, tuple[int, str, float]] = {}

        self.last_activity: dict[int, float] = {}
        self.warned_stall: set[int] = set()
        self.workers: list[Any] = []
        self.results: list[tuple[int, str, int, float]] = []
        self.pills_sent = 0

        self.done_count = 0
        self.done_weight = 0.0
        self.fail_count = 0
        self.wall_started = time.monotonic()
        self._progress()

    def _progress(self) -> None:
        """Publish completion counts using the existing resume offsets."""
        update_progress(
            self.pre_done + self.done_count,
            self.n_tasks + self.pre_done,
            self.pre_done_weight + self.done_weight,
            self.total_weight + self.pre_done_weight,
            self.wall_started,
            self.fail_count,
            pre_done_weight=self.pre_done_weight,
        )

    def start(self, log_dir: str, cwd: str | None, timeout: float | None) -> None:
        """Start the configured workers with unchanged subprocess settings."""
        for gpu in self.gpu_ids:
            for index in range(self.workers_per_gpu):
                worker = mp.Process(
                    target=gpu_worker,
                    args=(gpu, self.gpu_queues[gpu], self.result_queue, log_dir),
                    kwargs={"cwd": cwd, "timeout": timeout},
                    daemon=True,
                )

                worker.start()
                self.workers.append(worker)
                assert worker.pid is not None

                if self.workers_per_gpu > 1:
                    logger.info(
                        "GPU %d  worker %d/%d started",
                        gpu,
                        index + 1,
                        self.workers_per_gpu,
                    )
                else:
                    logger.info("GPU %d  worker started", gpu)

    def _touch(self, gpu: int) -> None:
        """Record activity and clear any previous stall warning."""
        self.last_activity[gpu] = time.monotonic()
        self.warned_stall.discard(gpu)

    def _record_done(self, message: tuple) -> None:
        """Account for a completed task and its dependency outcome."""
        _, gpu, label, code, elapsed, weight, pid = message
        self._touch(gpu)
        self.active_by_pid.pop(pid, None)
        self.active_weight[gpu] -= weight

        self.done_count += 1
        self.done_weight += weight

        # Dependency IDs omit the seed suffix included in worker log labels.
        task_name = label.rsplit("[seed=", 1)[0]

        if code == 0:
            self.done_names.add(task_name)
        else:
            self.failed_names.add(task_name)
            self.fail_count += 1

        self.results.append((gpu, label, code, elapsed))
        self._progress()
        status = "OK" if code == 0 else f"FAIL(rc={code})"
        logger.info("GPU %d  %s  %s  (%.1f min)", gpu, status, label, elapsed / 60)

    def _handle_message(self, message: tuple) -> None:
        """Apply one READY, START, or DONE event to schedule state."""
        if message[0] == "READY":
            self._touch(message[1])
            self.pending_ready[message[1]] += 1
            return

        if message[0] == "START":
            _, gpu, label, weight, pid = message
            self._touch(gpu)
            self.active_by_pid[pid] = (gpu, label, weight)
            logger.info("GPU %d  >> %s", gpu, label)

            return

        self._record_done(message)

    def _dead_workers(self) -> None:
        """Record vanished workers' in-flight tasks with the existing failure code."""
        for worker in self.workers:
            if worker.is_alive() or worker.pid not in self.active_by_pid:
                continue

            gpu, label, weight = self.active_by_pid.pop(worker.pid)
            self.active_weight[gpu] -= weight
            self.done_count += 1
            self.done_weight += weight
            self.fail_count += 1

            self.failed_names.add(label.rsplit("[seed=", 1)[0])
            self.results.append((gpu, label, -9, 0.0))
            logger.error(
                "GPU %d  worker (pid %d) died while running %s", gpu, worker.pid, label
            )

    def _warn_stalls(self) -> None:
        """Log one warning per inactive GPU until its next event."""
        if self.stall_warning is None:
            return

        now = time.monotonic()
        for gpu in self.gpu_ids:
            idle = now - self.last_activity.get(gpu, self.wall_started)
            if not (idle >= self.stall_warning and gpu not in self.warned_stall):
                continue

            self.warned_stall.add(gpu)
            labels = [info[1] for info in self.active_by_pid.values() if info[0] == gpu]
            logger.warning(
                "GPU %d  no activity for %.0fs  (in-flight: %s)",
                gpu,
                idle,
                ", ".join(labels) or "none",
            )

    def _timeout(self) -> bool:
        """Check worker liveness without changing the existing timeout policy."""
        self._dead_workers()
        if not any(worker.is_alive() for worker in self.workers):
            remaining = self.n_tasks - self.done_count
            if remaining > 0:
                logger.error("All workers died with %d tasks remaining", remaining)

            return False

        self._warn_stalls()
        return True

    def _limit_failures(self) -> None:
        """Stop dispatching new tasks after the configured failure limit."""
        if self.max_failures is None or self.fail_count < self.max_failures:
            return
        if not self.tasks:
            return

        skipped = len(self.tasks)
        self.tasks.clear()
        logger.error(
            "max_failures=%d reached (%d failures) - dropping %d remaining tasks",
            self.max_failures,
            self.fail_count,
            skipped,
        )

    def _skip_failed_dependencies(self) -> None:
        """Cascade existing dependency failures through queued tasks."""
        if not self.failed_names:
            return

        changed = True
        while changed:
            changed = False
            remaining = []

            for task in self.tasks:
                blocked_by = [
                    dep for dep in task.depends_on if dep in self.failed_names
                ]

                if blocked_by:
                    self._skip_task(task, blocked_by)
                    changed = True
                else:
                    remaining.append(task)

            self.tasks = remaining

    def _skip_task(self, task: Task, blocked_by: list[str]) -> None:
        """Record a queued task skipped because one of its dependencies failed."""
        self.failed_names.add(task.name)
        self.fail_count += 1
        self.done_count += 1
        self.done_weight += task.weight
        self.results.append((-1, task.label, -2, 0.0))

        logger.error(
            "SKIP  %s  (dependency failed: %s)", task.label, ", ".join(blocked_by)
        )

    def run(self) -> None:
        """Consume event batches and dispatch only after applying the whole batch."""
        while self.done_count < self.n_tasks:
            try:
                batch = [self.result_queue.get(timeout=_POLL_INTERVAL)]

            except queue.Empty:
                if not self._timeout():
                    break
                continue

            try:
                while True:
                    batch.append(self.result_queue.get_nowait())
            except queue.Empty:
                pass

            for message in batch:
                self._handle_message(message)

            self._limit_failures()
            self._skip_failed_dependencies()
            self.pills_sent += _dispatch_pending(
                self.tasks,
                self.pending_ready,
                self.active_weight,
                self.gpu_queues,
                self.done_names,
            )

    def retire(self) -> None:
        """Send remaining termination markers and join all worker processes."""
        while self.pills_sent < len(self.gpu_ids) * self.workers_per_gpu:
            try:
                message = self.result_queue.get(timeout=_POLL_INTERVAL)

            except queue.Empty:
                if not any(worker.is_alive() for worker in self.workers):
                    break
                continue

            if message[0] == "READY":
                self.gpu_queues[message[1]].put(None)
                self.pills_sent += 1

        for worker in self.workers:
            worker.join(timeout=10)

    def report(self, log_dir: str) -> None:
        """Summarize failures and elapsed GPU/wall-clock hours."""
        set_done_tag()
        failed = [result for result in self.results if result[2] != 0]

        if failed:
            logger.error("FAILED tasks (%d):", len(failed))
            for gpu, label, code, _ in failed:
                logger.error("  GPU %d  %s  rc=%d", gpu, label, code)

        else:
            logger.info("All %d tasks completed successfully.", len(self.results))

        gpu_hours = sum(result[3] for result in self.results) / 3600
        wall_hours = (time.monotonic() - self.wall_started) / 3600
        logger.info(
            "Total GPU-hours: %.1f  |  Wall-clock: %.1f h", gpu_hours, wall_hours
        )
        logger.info("Detailed logs: %s/", log_dir)


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
    """Dispatch tasks and return (GPU, label, return code, seconds) results.

    Resume offsets affect progress only. Optional failure and task-timeout
    limits retain their original behavior; multiple workers can share a GPU.
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    if not gpu_ids:
        raise ValueError("gpu_ids must be a non-empty list of GPU indices")

    logger.info(
        "GPUs: %s (%d)  |  Tasks: %d  |  Weight: %.1f h",
        gpu_ids,
        len(gpu_ids),
        len(tasks),
        sum(task.weight for task in tasks),
    )
    if dry_run:
        _dry_run(gpu_ids, tasks, workers_per_gpu)
        return []

    destination = _prepare_logs(log_dir)
    schedule = _Schedule(
        gpu_ids,
        tasks,
        workers_per_gpu,
        pre_done,
        pre_done_weight,
        max_failures,
        stall_warning,
    )
    schedule.start(destination, cwd, task_timeout)
    schedule.run()
    schedule.retire()

    schedule.report(destination)
    return schedule.results
