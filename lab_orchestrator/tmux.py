"""Render tmux launch scripts without executing them.

Each task gets a window with optional environment and directory setup.
A custom renderer may use the prepared context instead of the built-in
shell template. The built-in script replaces a session of the same name.
"""

from __future__ import annotations

import shlex
from collections.abc import Callable
from typing import Any

from lab_orchestrator.task import Task


def _default_renderer(ctx: dict[str, Any]) -> str:
    """Render session creation and one send-keys command per window."""
    session = ctx["session"]
    windows: list[dict[str, Any]] = ctx["windows"]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        f'SESSION="{session}"',
        "",
        "# Kill previous session if exists",
        'tmux kill-session -t "$SESSION" 2>/dev/null || true',
        "",
    ]

    for index, window in enumerate(windows):
        name = window["name"]

        if index == 0:
            lines.append(f'tmux new-session -d -s "$SESSION" -n {name}')
        else:
            lines.append(f'tmux new-window -t "$SESSION" -n {name}')

        lines.append(
            f'tmux send-keys -t "$SESSION:{name}" \\\n  {shlex.quote(window["full_cmd"])} Enter'
        )
        lines.append("")

    lines.append(
        f"echo \"tmux session '$SESSION' started with {len(windows)} windows.\""
    )
    lines.append('echo "Attach: tmux attach -t $SESSION"')
    lines.append("")
    return "\n".join(lines)


def _build_windows(
    tasks: list[Task], *, venv_activate: str, cwd: str, gpu_ids: list[int] | None
) -> list[dict[str, Any]]:
    """Prepare window commands with least-loaded GPU assignment."""
    preamble_parts: list[str] = []
    if venv_activate:
        preamble_parts.append(venv_activate)

    if cwd:
        preamble_parts.append(f"cd {cwd}")
    preamble = " && ".join(preamble_parts) if preamble_parts else ""

    windows: list[dict[str, Any]] = []
    gpu_load = {gpu: 0.0 for gpu in gpu_ids} if gpu_ids else {}

    for task in tasks:
        command = shlex.join(task.cmd)
        name = f"{task.name}_s{task.seed}"
        parts: list[str] = []
        gpu: int | None = None

        if gpu_ids:
            gpu = min(
                gpu_ids, key=lambda identifier: (gpu_load[identifier], identifier)
            )
            gpu_load[gpu] += task.weight
            parts.append(f"export CUDA_VISIBLE_DEVICES={gpu}")

        if preamble:
            parts.append(preamble)
        parts.append(command)

        windows.append(
            {
                "name": name,
                "gpu": gpu,
                "task": task,
                "cmd_str": command,
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
    """Build a launch script, optionally using a custom context renderer.

    venv_activate is a shell command, e.g. "source /path/to/.venv/bin/activate",
    not a directory. It runs before changing to cwd, the task working directory.
    Save the returned script and run it with Bash to launch the session.

    Context contains session, windows, tasks, gpu_ids, venv_activate, and
    cwd. Extra context is merged last, retaining the original override order.
    """
    windows = _build_windows(
        tasks, venv_activate=venv_activate, cwd=cwd, gpu_ids=gpu_ids
    )
    context: dict[str, Any] = {
        "session": session,
        "windows": windows,
        "tasks": tasks,
        "gpu_ids": gpu_ids or [],
        "venv_activate": venv_activate,
        "cwd": cwd,
        **(extra_context or {}),
    }
    render = renderer or _default_renderer
    return render(context)
