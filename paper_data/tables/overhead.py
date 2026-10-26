"""Summarize stored trace timings without rerunning a GPU benchmark."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import DataPaths
from ..loader import iter_run_finals, load_jsonl

LABELS = {"resnet18": "ResNet-18", "resnet34": "ResNet-34", "vgg11": "VGG-11"}


@dataclass(frozen=True)
class TimingSummary:
    """Mean per-step and per-snapshot costs for one architecture."""

    n_runs: int
    training_ms: float
    trace_seconds: float
    overhead_percent: float


def _run_timings(final: dict, records: list[dict]) -> dict[str, float]:
    """Collect the same timing components as the original overhead script."""
    return {
        "total": final["wallclock_s"],
        "steps": final["train_summary"]["global_steps"],
        "trace": sum(record.get("wallclock_s", 0) for record in records),
        "validation": sum(record.get("val_wallclock_s", 0) for record in records),
        "snapshots": len(records),
    }


def _summarize(runs: list[dict[str, float]]) -> TimingSummary:
    """Subtract recorded diagnostics from total time, then amortize."""
    means = {key: float(np.mean([run[key] for run in runs])) for key in runs[0]}
    training = means["total"] - means["trace"] - means["validation"]

    return TimingSummary(
        len(runs),
        training / means["steps"] * 1000.0,
        means["trace"] / means["snapshots"],
        means["trace"] / training * 100.0,
    )


def evaluate(runs_root: Path) -> dict[str, TimingSummary]:
    """Pool the recorded timings by architecture in original path order."""
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for run_dir, final in iter_run_finals(runs_root):
        records = load_jsonl(run_dir / "steps.jsonl")
        arch = final["config"]["arch"]
        groups[arch].append(_run_timings(final, records))

    if not groups:
        raise FileNotFoundError(f"No timing records under {runs_root}")

    return {arch: _summarize(runs) for arch, runs in sorted(groups.items())}


def _table_row(arch: str, timing: TimingSummary) -> str:
    """Render one row with the article's units and precision."""
    training = f"{timing.training_ms:.0f}"
    if len(training) < 3:
        training = r"\phantom{0}" + training

    return (
        f"{LABELS[arch]:9s} & ${training}$ & ${timing.trace_seconds:.1f}$ & "
        f"${timing.overhead_percent:.0f}\\%$ " + r"\\"
    )


def generate(paths: DataPaths, out_dir: Path) -> Path:
    """Write the overhead table using immutable rerun timing records.

    Training time is the residual after recorded trace and validation work;
    it is not a separately measured run with diagnostics disabled.
    """
    summaries = evaluate(paths.rerun_runs_root)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Architecture & Training (ms/step) & Trace estimate (s) & Overhead \\",
        r"\midrule",
    ]

    lines.extend(_table_row(arch, timing) for arch, timing in summaries.items())
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "tab_overhead.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
