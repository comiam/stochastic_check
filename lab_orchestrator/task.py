"""Represent scheduled tasks and expand deduplicated experiment matrices."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(order=True)
class Task:
    """Describe one subprocess, ordered by estimated GPU hours/name/seed.

    CUDA selection belongs to the worker environment, not cmd. Dependencies
    name tasks that must succeed; a failed dependency yields a skipped task
    with return code -2. Command and dependency lists do not affect sorting.
    """

    weight: float = field(compare=True)
    name: str = field(compare=True)
    seed: int = field(compare=True)
    cmd: list[str] = field(default_factory=list, compare=False)
    depends_on: list[str] = field(default_factory=list, compare=False)

    @property
    def label(self) -> str:
        """Return the worker/log identifier including the seed."""
        return f"{self.name}[seed={self.seed}]"


def build_task_matrix(
    names: list[str],
    seeds: list[int],
    weights: dict[str, float],
    cmd_factory: Callable[[str, int], list[str]],
    *,
    available_names: list[str] | None = None,
) -> list[Task]:
    """Expand unique name/seed pairs and sort them in descending LPT order.

    An optional whitelist rejects unknown names before constructing commands.
    Missing weights use one GPU-hour, preserving the existing default.
    """
    if available_names is not None:
        for name in names:
            if name not in available_names:
                raise ValueError(
                    f"Unknown name {name!r}. Available: {', '.join(available_names)}"
                )

    seen: set[tuple[str, int]] = set()
    tasks: list[Task] = []

    for name in names:
        weight = weights.get(name, 1.0)

        for seed in seeds:
            if (name, seed) in seen:
                continue

            seen.add((name, seed))
            tasks.append(
                Task(weight=weight, name=name, seed=seed, cmd=cmd_factory(name, seed))
            )

    tasks.sort(reverse=True)
    return tasks
