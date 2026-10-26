"""Construct the experiment task matrix without executing it."""

from __future__ import annotations

import sys
from collections import Counter
from itertools import product
from pathlib import Path

from lab_orchestrator import Task

from .config import SweepConfig
from .utils import run_slug

# Estimated hours on the original GPU tiers; preserve the scheduling weights.
_ARCH_HOURS = {"resnet18": 1.5, "vgg11": 1.0, "resnet34": 2.5}
Cell = tuple[str, str, float, int]


def _cell_command(stage: str, cell: Cell) -> list[str]:
    """Build the shared, ordered command prefix for one experimental cell."""
    arch, dataset, noise, seed = cell
    return [
        sys.executable,
        "-m",
        "main",
        stage,
        "--arch",
        arch,
        "--dataset",
        dataset,
        "--noise",
        str(noise),
        "--seed",
        str(seed),
    ]


def _training_command(
    cell: Cell, sweep: SweepConfig, checkpoints: list[int]
) -> list[str]:
    """Append the training budget, snapshot epochs, and output path."""
    return _cell_command("train", cell) + [
        "--epochs",
        str(sweep.epochs),
        "--checkpoint-epochs",
        *[str(epoch) for epoch in checkpoints],
        "--output-dir",
        str(Path(sweep.output_root) / "runs"),
    ]


def _comparison_command(cell: Cell, sweep: SweepConfig) -> list[str]:
    """Point the comparison at the existing per-cell snapshots."""
    return _cell_command("compare", cell) + [
        "--runs-root",
        str(Path(sweep.output_root) / "runs"),
        "--output-dir",
        "results/exp_a/runs",
    ]


def _cell_tasks(
    cell: Cell, sweep: SweepConfig, checkpoints: list[int], train: bool, compare: bool
) -> list[Task]:
    """Create a cell's training task and its dependent comparison task."""
    arch, dataset, noise, seed = cell
    name = run_slug(arch=arch, dataset=dataset, noise=noise, seed=seed)
    weight = _ARCH_HOURS.get(arch, 1.0)
    train_name = f"train:{name}"
    tasks = []

    if train:
        tasks.append(
            Task(
                weight=weight,
                name=train_name,
                seed=seed,
                cmd=_training_command(cell, sweep, checkpoints),
            )
        )

    if compare:
        tasks.append(
            Task(
                weight=weight * 0.5,
                name=f"compare:{name}",
                seed=seed,
                cmd=_comparison_command(cell, sweep),
                depends_on=[train_name] if train else [],
            )
        )

    return tasks


def build_sweep_tasks(
    sweep: SweepConfig,
    checkpoint_epochs: list[int],
    only_stage: str,
    also_compare: bool,
) -> list[Task]:
    """Expand cells and stages in the existing order with unchanged weights."""
    train = only_stage in ("train", "both")
    compare = only_stage in ("compare", "both") or also_compare
    cells = product(sweep.archs, sweep.datasets, sweep.noise_rates, sweep.seeds)
    tasks = []

    for cell in cells:
        tasks.extend(_cell_tasks(cell, sweep, checkpoint_epochs, train, compare))

    return tasks


def completion_status(
    tasks: list[Task], results: list[tuple[int, str, int, float]]
) -> int:
    """Reject failed, missing, or duplicated completion reports."""
    expected = Counter(task.label for task in tasks)
    completed = Counter(label for _, label, _, _ in results)
    failed = any(code != 0 for _, _, code, _ in results)
    return int(failed or completed != expected)
