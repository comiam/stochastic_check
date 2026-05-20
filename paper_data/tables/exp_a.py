"""Table ``tab:comparison`` -- estimator-by-estimator relative error and cost.

Aggregates the per-checkpoint comparison files under
``results/exp_a/runs/`` across all ``(arch, dataset, noise, seed)``
cells and produces a LaTeX table with rows = estimator, columns =
``rel. error (mean +/- std)``, ``normalised wallclock``.

The normalisation reference is the Hutchinson-K=10 wallclock at each
checkpoint, so the table reads "X times slower than the cheap default".
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

from ..config import DataPaths
from ..formatting import fmt_mean_std
from ..loader import iter_run_comparisons

_ESTIMATORS = [
    "hutchinson_k10",
    "hutchinson_k50",
    "hutch_plus_plus_k10",
    "bl_diag_k200",
    "kfac",
]


def generate(paths: DataPaths, out_dir: Path) -> Path:
    rel_errors: dict[str, list[float]] = defaultdict(list)
    wallclocks: dict[str, list[float]] = defaultdict(list)
    norm_wallclocks: dict[str, list[float]] = defaultdict(list)

    for _run_dir, comp in iter_run_comparisons(paths.exp_a_runs_root):
        for _label, cp in comp.get("checkpoints", {}).items():
            ests = cp.get("estimators", {})
            ref_wc = ests.get("hutchinson_k10", {}).get("wallclock_s")
            for name in _ESTIMATORS:
                rec = ests.get(name)
                if rec is None:
                    continue
                wc = rec.get("wallclock_s")
                if wc is not None:
                    wallclocks[name].append(float(wc))
                    if ref_wc:
                        norm_wallclocks[name].append(float(wc) / float(ref_wc))
                rel = rec.get("relative_error", {})
                for v in rel.values():
                    if v is None or (isinstance(v, float) and math.isnan(v)):
                        continue
                    rel_errors[name].append(float(v))

    lines: list[str] = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Estimator & Rel.\ error & Wallclock (s) & "
        r"Wallclock / Hutchinson$_{K=10}$ \\"
    )
    lines.append(r"\midrule")
    for name in _ESTIMATORS:
        lines.append(
            " & ".join(
                [
                    name.replace("_", r"\_"),
                    fmt_mean_std(rel_errors[name], digits=4),
                    fmt_mean_std(wallclocks[name], digits=3),
                    fmt_mean_std(norm_wallclocks[name], digits=2),
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = out_dir / "tab_comparison.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
