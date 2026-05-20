"""CLI entry point for lab-orchestrator.

Reads a declarative YAML/Python config and dispatches tasks.  Also
supports ``--tmux`` mode to generate a tmux script instead of running.

Usage::

    python -m lab_orchestrator config.yaml
    python -m lab_orchestrator config.yaml --dry-run
    python -m lab_orchestrator config.yaml --tmux > run_tmux.sh
    python -m lab_orchestrator config.yaml --resume --gpus 0,1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shlex
import sys
from typing import Any

try:
    import yaml as _yaml
except ImportError:
    _yaml = None  # type: ignore[assignment]

from lab_orchestrator.gpu import parse_gpu_ids
from lab_orchestrator.resume import scan_completed
from lab_orchestrator.scheduler import run_schedule
from lab_orchestrator.task import Task, build_task_matrix
from lab_orchestrator.tmux import generate_tmux_script

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    """Load YAML config (PyYAML soft-dependency)."""
    if _yaml is None:
        raise ImportError(
            "pyyaml is required for YAML configs.  Install with: pip install lab-orchestrator[yaml]"
        )
    with open(path) as f:
        result: dict[str, Any] = _yaml.safe_load(f)
        return result


def _load_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        result: dict[str, Any] = json.load(f)
        return result


def _load_python(path: str) -> dict[str, Any]:
    """Import a Python file and return its ``CONFIG`` dict."""
    spec = importlib.util.spec_from_file_location("_user_config", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, "CONFIG"):
        raise AttributeError(f"{path} must define a CONFIG dict")
    return mod.CONFIG  # type: ignore[no-any-return]


def load_config(path: str) -> dict[str, Any]:
    """Load config from YAML, JSON, or Python file."""
    if path.endswith((".yaml", ".yml")):
        return _load_yaml(path)
    if path.endswith(".json"):
        return _load_json(path)
    if path.endswith(".py"):
        return _load_python(path)
    raise ValueError(f"Unsupported config format: {path}")


# ---------------------------------------------------------------------------
#  Config -> Task list
# ---------------------------------------------------------------------------


def _build_tasks_from_config(cfg: dict[str, Any]) -> list[Task]:
    """Convert a declarative config dict into an LPT-sorted task list.

    Config schema::

        names: ["train_big", "train_small"]      # experiment names
        seeds: [42, 43, 44]                       # seed values
        weights:                                  # GPU-hours per seed
          train_big: 2.0
          train_small: 0.5
        cmd_template: "python train.py {name} --seed {seed}"

    Or with ``cmd_parts`` for structured commands.  The scheduler inserts
    ``name`` and ``--seed`` after ``base`` automatically, producing e.g.
    ``python train.py train_big --seed 42 --size=big --epochs 50``::

        cmd_parts:
          base: ["python", "train.py"]
          per_name:
            train_big: ["--size=big"]
            train_small: ["--size=small"]
          common: ["--epochs", "50"]
    """
    names = cfg["names"]
    seeds = cfg["seeds"]
    weights = cfg.get("weights", {})

    # Build cmd_factory from either template or parts.
    if "cmd_template" in cfg:
        template = cfg["cmd_template"]

        def cmd_factory(name: str, seed: int) -> list[str]:
            rendered: str = template.format(name=name, seed=seed)
            return shlex.split(rendered)

    elif "cmd_parts" in cfg:
        parts = cfg["cmd_parts"]
        base = parts["base"]
        per_name = parts.get("per_name", {})
        common = parts.get("common", [])

        def cmd_factory(name: str, seed: int) -> list[str]:
            return [
                *base,
                name,
                "--seed",
                str(seed),
                *per_name.get(name, []),
                *common,
            ]

    else:
        raise ValueError("Config must define 'cmd_template' or 'cmd_parts'")

    return build_task_matrix(
        names=names,
        seeds=seeds,
        weights=weights,
        cmd_factory=cmd_factory,
    )


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="lab-orchestrator",
        description="Lightweight single-node GPU experiment scheduler",
    )
    parser.add_argument("config", help="Config file (YAML, JSON, or Python)")
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids (default: auto-detect)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print schedule only")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks whose results already exist",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory to scan for completed seeds on --resume",
    )
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent workers per GPU (default: 1)",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Generate tmux launch script instead of running",
    )
    parser.add_argument(
        "--tmux-session",
        type=str,
        default="lab",
        help="tmux session name (default: lab)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for worker subprocesses",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help="Virtualenv activate command for tmux mode",
    )
    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help="Per-task timeout in seconds (default: no timeout)",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Stop scheduling new tasks after this many failures (default: no limit)",
    )

    args = parser.parse_args()

    # Load config.
    cfg = load_config(args.config)

    # Build tasks.
    tasks = _build_tasks_from_config(cfg)
    if not tasks:
        logger.info("No tasks to schedule.")
        return

    # GPU detection.
    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    # tmux mode: generate script and exit.
    if args.tmux:
        script = generate_tmux_script(
            session=args.tmux_session,
            tasks=tasks,
            venv_activate=args.venv or "",
            cwd=args.cwd or "",
            gpu_ids=gpu_ids or None,
        )
        print(script)
        return

    if not gpu_ids:
        logger.error("No GPUs detected. Use --gpus to specify manually.")
        sys.exit(1)

    # Resume filtering.
    pre_done = 0
    pre_done_weight = 0.0

    if args.resume:
        results_dir = args.results_dir or cfg.get("results_dir", "results")
        done = scan_completed(results_dir, validate=True)
        remaining = [t for t in tasks if (t.name, t.seed) not in done]
        skipped = len(tasks) - len(remaining)
        pre_done_weight = sum(t.weight for t in tasks if (t.name, t.seed) in done)
        if skipped:
            logger.info("--resume: skipping %d already-completed tasks", skipped)
        if not remaining:
            logger.info("All tasks already completed.")
            return
        pre_done = skipped
        tasks = remaining

    # Dispatch.
    run_schedule(
        gpu_ids,
        tasks,
        dry_run=args.dry_run,
        cwd=args.cwd,
        max_failures=args.max_failures,
        pre_done=pre_done,
        pre_done_weight=pre_done_weight,
        task_timeout=args.task_timeout,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()
