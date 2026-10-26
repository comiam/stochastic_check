"""Summarize estimator errors and cost relative to Hutchinson K=10."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from ..config import DataPaths
from ..formatting import fmt_mean_std
from ..loader import iter_run_comparisons

_ESTIMATORS = ["hutchinson_k5", "hutchinson_k10", "hutch_plus_plus_hvp13", "kfac"]
_DISPLAY_NAMES = {
    "hutchinson_k5": r"Hutchinson ($K=5$, $5$ HVP)",
    "hutchinson_k10": r"Hutchinson ($K=10$, $10$ HVP)",
    "hutch_plus_plus_hvp13": r"Hutch++ ($n_{\mathrm{total}}=10$, $13$ HVP)",
    "kfac": r"K-FAC (Fisher, $1$ pass)",
}


def _collect_record(record, reference_time, errors, times, relative_times) -> None:
    """Collect one estimator's available timing and non-NaN layer errors."""
    wallclock = record.get("wallclock_s")
    if wallclock is not None:
        times.append(float(wallclock))
        if reference_time:
            relative_times.append(float(wallclock) / float(reference_time))

    for value in record.get("relative_error", {}).values():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        errors.append(float(value))


def _collect_results(runs_root: Path):
    """Pool checkpoint results in their original run/estimator/layer order."""
    errors: dict[str, list[float]] = defaultdict(list)
    times: dict[str, list[float]] = defaultdict(list)
    relative_times: dict[str, list[float]] = defaultdict(list)

    for _, comparison in iter_run_comparisons(runs_root):
        for checkpoint in comparison.get("checkpoints", {}).values():
            estimators = checkpoint.get("estimators", {})
            reference_time = estimators.get("hutchinson_k10", {}).get("wallclock_s")

            for name in _ESTIMATORS:
                record = estimators.get(name)
                if record is None:
                    continue

                _collect_record(
                    record,
                    reference_time,
                    errors[name],
                    times[name],
                    relative_times[name],
                )

    return errors, times, relative_times


def generate(paths: DataPaths, out_dir: Path) -> Path:
    """Write mean errors and raw/normalized timings for each estimator."""
    errors, times, relative_times = _collect_results(paths.exp_a_runs_root)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Estimator & Rel.\ error & Wallclock (s) & Wallclock / Hutchinson$_{K=10}$ \\",
        r"\midrule",
    ]
    for name in _ESTIMATORS:
        cells = [
            _DISPLAY_NAMES.get(name, name.replace("_", r"\_")),
            fmt_mean_std(errors[name], digits=4),
            fmt_mean_std(times[name], digits=3),
            fmt_mean_std(relative_times[name], digits=2),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output = out_dir / "tab_comparison.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
