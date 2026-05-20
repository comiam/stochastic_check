"""lab-orchestrator - lightweight single-node GPU experiment scheduler.

Zero-dependency (stdlib + optional torch for GPU detection) load-aware
scheduler for ``task * seed`` experiment matrices.  Uses LPT ordering,
per-GPU work-stealing queues, and subprocess isolation.

Quick start::

    from lab_orchestrator import Task, run_schedule, detect_gpus

    tasks = [
        Task(weight=2.0, name="train_big", seed=42,
             cmd=["python", "train.py", "--size=big"]),
        Task(weight=0.5, name="train_small", seed=43,
             cmd=["python", "train.py", "--size=small"]),
    ]
    run_schedule(detect_gpus(), tasks)
"""

from lab_orchestrator.gpu import detect_gpus, parse_gpu_ids
from lab_orchestrator.progress import ProgressFormatter
from lab_orchestrator.resume import scan_completed
from lab_orchestrator.scheduler import run_schedule
from lab_orchestrator.seed import set_seed
from lab_orchestrator.task import Task, build_task_matrix
from lab_orchestrator.tmux import generate_tmux_script
from lab_orchestrator.worker import gpu_worker

__all__ = [
    "Task",
    "build_task_matrix",
    "detect_gpus",
    "generate_tmux_script",
    "gpu_worker",
    "parse_gpu_ids",
    "ProgressFormatter",
    "run_schedule",
    "scan_completed",
    "set_seed",
]
