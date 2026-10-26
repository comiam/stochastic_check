"""Dispatch declarative experiment configs or emit their tmux launch script.

Configs may be JSON, YAML with optional PyYAML, or Python files defining
CONFIG. Commands support dry runs, resume filtering, and worker limits.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import shlex
import sys
from collections.abc import Callable
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
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    """Load YAML when the optional PyYAML dependency is installed."""
    if _yaml is None:
        raise ImportError(
            "pyyaml is required for YAML configs.  Install with: pip install lab-orchestrator[yaml]"
        )

    with open(path) as stream:
        result: dict[str, Any] = _yaml.safe_load(stream)
        return result


def _load_json(path: str) -> dict[str, Any]:
    """Load a declarative JSON config."""
    with open(path) as stream:
        result: dict[str, Any] = json.load(stream)
        return result


def _load_python(path: str) -> dict[str, Any]:
    """Import the configured Python file and read its CONFIG mapping."""
    spec = importlib.util.spec_from_file_location("_user_config", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"{path} must define a CONFIG dict")

    return module.CONFIG


def load_config(path: str) -> dict[str, Any]:
    """Choose the existing config loader by filename extension."""
    if path.endswith((".yaml", ".yml")):
        return _load_yaml(path)

    if path.endswith(".json"):
        return _load_json(path)

    if path.endswith(".py"):
        return _load_python(path)

    raise ValueError(f"Unsupported config format: {path}")


def _command_factory(config: dict[str, Any]) -> Callable[[str, int], list[str]]:
    """Build commands from either a shell template or structured argument parts."""
    if "cmd_template" in config:
        template = config["cmd_template"]

        def template_command(name: str, seed: int) -> list[str]:
            """Render and split a configured command template."""
            rendered: str = template.format(name=name, seed=seed)
            return shlex.split(rendered)

        return template_command

    if "cmd_parts" in config:
        parts = config["cmd_parts"]
        base = parts["base"]
        per_name = parts.get("per_name", {})
        common = parts.get("common", [])

        def structured_command(name: str, seed: int) -> list[str]:
            """Preserve base, name, seed, per-name, and common argument order."""
            return [*base, name, "--seed", str(seed), *per_name.get(name, []), *common]

        return structured_command

    raise ValueError("Config must define 'cmd_template' or 'cmd_parts'")


def _build_tasks_from_config(cfg: dict[str, Any]) -> list[Task]:
    """Expand names/seeds with weights, preserving the LPT task ordering.

    Configs require names, seeds, and one command form. Optional weights map
    names to relative scheduling costs. A template config looks like::

        names: [train_big, train_small]
        seeds: [42, 43, 44]
        weights:
          train_big: 2.0
          train_small: 0.5
        cmd_template: "python train.py {name} --seed {seed}"

    Alternatively, replace cmd_template with structured arguments::

        cmd_parts:
          base: [python, train.py]
          per_name:
            train_big: [--size=big]
            train_small: [--size=small]
          common: [--epochs, "50"]

    Templates format name and seed before shell splitting. Structured commands
    insert name and --seed after base, before per_name and common arguments:
    python train.py train_big --seed 42 --size=big --epochs 50.
    """
    names = cfg["names"]
    seeds = cfg["seeds"]
    weights = cfg.get("weights", {})
    factory = _command_factory(cfg)
    return build_task_matrix(
        names=names, seeds=seeds, weights=weights, cmd_factory=factory
    )


def _add_execution_options(parser: argparse.ArgumentParser) -> None:
    """Configure resource selection, execution limits, and working directory."""
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="Comma-separated GPU ids (default: auto-detect).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print schedule only.")
    parser.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent workers per GPU (default: 1).",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory for worker subprocesses.",
    )

    parser.add_argument(
        "--task-timeout",
        type=float,
        default=None,
        help="Per-task timeout in seconds (default: no timeout).",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=None,
        help="Stop scheduling after this many failures (default: no limit).",
    )


def _parser() -> argparse.ArgumentParser:
    """Describe config, resume, tmux, and execution options."""
    parser = argparse.ArgumentParser(
        prog="lab-orchestrator",
        description="Lightweight single-node GPU experiment scheduler",
    )
    parser.add_argument("config", help="Config file (YAML, JSON, or Python)")
    _add_execution_options(parser)

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks with validated completed results.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory scanned for completed seeds with --resume.",
    )
    parser.add_argument(
        "--tmux",
        action="store_true",
        help="Print a tmux launch script instead of running tasks.",
    )
    parser.add_argument(
        "--tmux-session",
        type=str,
        default="lab",
        help="Session name for --tmux (default: lab).",
    )
    parser.add_argument(
        "--venv",
        type=str,
        default=None,
        help="Shell activation command for --tmux, e.g. 'source /path/to/.venv/bin/activate'.",
    )

    return parser


def _resume_tasks(args, config, tasks):
    """Exclude validated completed runs and compute progress offsets."""
    if not args.resume:
        return tasks, 0, 0.0

    results_dir = args.results_dir or config.get("results_dir", "results")
    completed = scan_completed(results_dir, validate=True)
    remaining = [task for task in tasks if (task.name, task.seed) not in completed]
    skipped = len(tasks) - len(remaining)
    weight = sum(task.weight for task in tasks if (task.name, task.seed) in completed)

    if skipped:
        logger.info("--resume: skipping %d already-completed tasks", skipped)
    if not remaining:
        logger.info("All tasks already completed.")

    return remaining, skipped, weight


def main() -> None:
    """Load, optionally filter, and dispatch the configured experiment tasks."""
    args = _parser().parse_args()
    config = load_config(args.config)
    tasks = _build_tasks_from_config(config)

    if not tasks:
        logger.info("No tasks to schedule.")
        return

    try:
        gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    if args.tmux:
        print(
            generate_tmux_script(
                session=args.tmux_session,
                tasks=tasks,
                venv_activate=args.venv or "",
                cwd=args.cwd or "",
                gpu_ids=gpu_ids or None,
            )
        )
        return

    if not gpu_ids:
        logger.error("No GPUs detected. Use --gpus to specify manually.")
        sys.exit(1)

    tasks, pre_done, pre_done_weight = _resume_tasks(args, config, tasks)
    if not tasks:
        return

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
