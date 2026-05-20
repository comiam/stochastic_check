"""Task definition and matrix builder."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    """A single schedulable unit dispatched to one GPU.

    Parameters
    ----------
    weight : float
        Estimated GPU-hours for this task (used for LPT ordering and
        load-aware dispatch).
    name : str
        Human-readable experiment/regime identifier.
    seed : int
        Random seed for this task.
    cmd : list[str]
        Full command to execute as a subprocess.  The scheduler sets
        ``CUDA_VISIBLE_DEVICES`` in the environment; the command itself
        should NOT include GPU selection flags.
    """

    weight: float = field(compare=True)
    name: str = field(compare=True)
    seed: int = field(compare=True)
    cmd: list[str] = field(default_factory=list, compare=False)

    @property
    def label(self) -> str:
        return f"{self.name}[seed={self.seed}]"


def build_task_matrix(
    names: list[str],
    seeds: list[int],
    weights: dict[str, float],
    cmd_factory: Callable[[str, int], list[str]],
    *,
    available_names: list[str] | None = None,
) -> list[Task]:
    """Build an LPT-sorted task list from a ``name * seed`` matrix.

    Parameters
    ----------
    names : list[str]
        Experiment/regime names to schedule.
    seeds : list[int]
        Seed values; one task per ``(name, seed)`` pair.
    weights : dict[str, float]
        Estimated GPU-hours per seed for each name.
    cmd_factory : callable
        ``(name, seed) -> list[str]`` - builds the subprocess command.
    available_names : list[str] | None
        If given, validates *names* against this whitelist.

    Returns
    -------
    list[Task]
        De-duplicated, sorted descending by weight (LPT).

    Raises
    ------
    ValueError
        On unknown names.
    """
    if available_names is not None:
        for n in names:
            if n not in available_names:
                raise ValueError(
                    f"Unknown name {n!r}. Available: {', '.join(available_names)}"
                )

    seen: set[tuple[str, int]] = set()
    tasks: list[Task] = []

    for name in names:
        w = weights.get(name, 1.0)
        for seed in seeds:
            if (name, seed) in seen:
                continue
            seen.add((name, seed))
            tasks.append(
                Task(
                    weight=w,
                    name=name,
                    seed=seed,
                    cmd=cmd_factory(name, seed),
                )
            )

    tasks.sort(reverse=True)
    return tasks
