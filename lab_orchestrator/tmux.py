"""Generate tmux session scripts for experiment dispatch.

Produces a bash script that creates a tmux session with one window per
task, sources the virtualenv, and sends the experiment command.  Useful
as an alternative to the programmatic scheduler when you want manual
control or need to attach/detach from running tasks.

Zero dependencies - uses a plain Python renderer by default.
Pass a custom ``renderer`` callable to fully control the output.

Usage::

    from lab_orchestrator.tmux import generate_tmux_script

    script = generate_tmux_script(
        session="my_exp",
        tasks=tasks,
        venv_activate="source /path/to/.venv/bin/activate",
        cwd="/path/to/experiment",
        gpu_ids=[0, 1, 2, 3],
    )
    # Write to file and run: bash run_tmux.sh

Custom renderer::

    def my_renderer(ctx: dict) -> str:
        lines = [f"# {ctx['session']}"]
        for w in ctx["windows"]:
            lines.append(w["full_cmd"])
        return "\\n".join(lines)

    script = generate_tmux_script(
        session="my_exp",
        tasks=tasks,
        renderer=my_renderer,
    )
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Any

from lab_orchestrator.task import Task


def _default_renderer(ctx: dict[str, Any]) -> str:
    """Render a tmux session script from *ctx* using plain Python."""
    session = ctx["session"]
    windows: list[dict[str, Any]] = ctx["windows"]

    lines: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'SESSION="{session}"',
        "",
        "# Kill previous session if exists",
        'tmux kill-session -t "$SESSION" 2>/dev/null || true',
        "",
    ]

    for i, win in enumerate(windows):
        name = win["name"]
        if i == 0:
            lines.append(f'tmux new-session -d -s "$SESSION" -n {name}')
        else:
            lines.append(f'tmux new-window -t "$SESSION" -n {name}')
        lines.append(
            f'tmux send-keys -t "$SESSION:{name}" \\\n  {shlex.quote(win["full_cmd"])} Enter'
        )
        lines.append("")

    lines.append(
        f"echo \"tmux session '$SESSION' started with {len(windows)} windows.\""
    )
    lines.append('echo "Attach: tmux attach -t $SESSION"')
    lines.append("")

    return "\n".join(lines)


def _build_windows(
    tasks: list[Task],
    *,
    venv_activate: str,
    cwd: str,
    gpu_ids: list[int] | None,
) -> list[dict[str, Any]]:
    """Prepare per-window context dicts for the renderer."""
    preamble_parts: list[str] = []
    if venv_activate:
        preamble_parts.append(venv_activate)
    if cwd:
        preamble_parts.append(f"cd {cwd}")
    preamble = " && ".join(preamble_parts) if preamble_parts else ""

    windows: list[dict[str, Any]] = []
    gpu_load: dict[int, float] = {g: 0.0 for g in gpu_ids} if gpu_ids else {}
    for _i, task in enumerate(tasks):
        cmd_str = shlex.join(task.cmd)
        win_name = f"{task.name}_s{task.seed}"

        parts: list[str] = []
        gpu: int | None = None
        if gpu_ids:
            gpu = min(gpu_ids, key=lambda g: (gpu_load[g], g))
            gpu_load[gpu] += task.weight
            parts.append(f"export CUDA_VISIBLE_DEVICES={gpu}")
        if preamble:
            parts.append(preamble)
        parts.append(cmd_str)

        windows.append(
            {
                "name": win_name,
                "gpu": gpu,
                "task": task,
                "cmd_str": cmd_str,
                "full_cmd": " && ".join(parts),
            }
        )
    return windows


def generate_tmux_script(
    session: str,
    tasks: list[Task],
    *,
    venv_activate: str = "",
    cwd: str = "",
    gpu_ids: list[int] | None = None,
    renderer: Callable[[dict[str, Any]], str] | None = None,
    extra_context: dict[str, Any] | None = None,
) -> str:
    """Build a bash script that launches tasks in a tmux session.

    Parameters
    ----------
    session : str
        tmux session name.
    tasks : list[Task]
        Tasks to dispatch; each gets its own tmux window.
    venv_activate : str
        Shell command to activate the virtualenv (e.g.
        ``"source .venv/bin/activate"``).
    cwd : str
        Working directory for each window.
    gpu_ids : list[int] | None
        Available GPUs.  When given, tasks are round-robin assigned to
        GPUs and ``CUDA_VISIBLE_DEVICES`` is exported in each window.
    renderer : callable | None
        Custom ``(ctx: dict) -> str`` function that produces the script
        content.  Receives a dict with keys ``session``, ``windows``,
        ``tasks``, ``gpu_ids``, ``venv_activate``, ``cwd``, and any
        *extra_context* entries.  When *None*, the built-in default
        renderer is used.
    extra_context : dict | None
        Additional variables merged into the context dict.

    Returns
    -------
    str
        Complete bash script content.
    """
    windows = _build_windows(
        tasks,
        venv_activate=venv_activate,
        cwd=cwd,
        gpu_ids=gpu_ids,
    )

    ctx: dict[str, Any] = {
        "session": session,
        "windows": windows,
        "tasks": tasks,
        "gpu_ids": gpu_ids or [],
        "venv_activate": venv_activate,
        "cwd": cwd,
        **(extra_context or {}),
    }

    render = renderer or _default_renderer
    return render(ctx)
