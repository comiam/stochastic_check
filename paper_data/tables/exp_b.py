"""Table ``tab:cohens-d`` -- effect size of label noise on layer-wise trace.

For each (arch, dataset, layer) cell we read the per-step trace
trajectory of every seed, average :math:`\\hat T_\\ell` over a fixed
**epoch window** ``[epoch_lo, epoch_hi]`` (closed interval, default
``[10, 20]``, matching section 7.2 of the paper), and compute Cohen's
d between the clean baseline (``noise = 0``) and each non-zero noise
level.  Bootstrap 95% CIs are obtained over 10 000 resamples.

The output table has columns = ``arch / dataset / layer``, one
sub-column per non-zero noise level, cells = ``d [lo, hi]``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import DataPaths
from ..formatting import fmt_ci
from ..loader import iter_run_finals, load_jsonl

# Default averaging window (inclusive in epoch index), aligned with
# section 7.2 of the paper.  Kept module-level so the value is visible
# in a single place; can be overridden by callers via :func:`generate`.
DEFAULT_EPOCH_WINDOW: tuple[int, int] = (10, 20)


def _windowed_trace_per_layer(
    run_dir: Path, window: tuple[int, int]
) -> dict[str, float]:
    """Average the per-step trace over a closed epoch window.

    Reads ``steps.jsonl`` and, for each layer, averages every record
    whose ``epoch`` field lies in ``[window[0], window[1]]``.  Layers
    whose trace is non-finite at every in-window step are dropped.
    Missing file or empty window -> empty dict.
    """
    jsonl = run_dir / "steps.jsonl"
    if not jsonl.exists():
        return {}
    lo, hi = window
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for rec in load_jsonl(jsonl):
        epoch = rec.get("epoch")
        if epoch is None or not (lo <= int(epoch) <= hi):
            continue
        for k, v in (rec.get("trace") or {}).items():
            if isinstance(v, (int, float)) and not math.isnan(float(v)):
                sums[k] += float(v)
                counts[k] += 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def generate(
    paths: DataPaths,
    out_dir: Path,
    epoch_window: tuple[int, int] = DEFAULT_EPOCH_WINDOW,
) -> Path:
    # Index runs by (arch, dataset, noise, layer) -> list of values across seeds.
    buckets: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for run_dir, final in iter_run_finals(paths.exp_b_runs_root):
        cfg = final.get("config", {})
        arch = cfg.get("arch")
        dataset = cfg.get("dataset")
        noise = cfg.get("training", {}).get("noise_rate")
        if None in (arch, dataset, noise):
            continue
        traces = _windowed_trace_per_layer(run_dir, epoch_window)
        for layer_name, value in traces.items():
            buckets[(arch, dataset, float(noise), layer_name)].append(value)

    # Lazy import: heavy stats only when at least one bucket exists.
    from stochastic_check.analysis import bootstrap_cohens_d  # noqa: WPS433

    # Per (arch, dataset, layer) and per noise>0, compute d vs noise=0.
    rows: list[dict[str, Any]] = []
    keys = {(a, d, l) for (a, d, _n, l) in buckets}
    noises_nonzero = sorted({n for (_a, _d, n, _l) in buckets if n > 0.0})

    for arch, dataset, layer in sorted(keys):
        clean = buckets.get((arch, dataset, 0.0, layer), [])
        if len(clean) < 2:
            continue
        row: dict[str, Any] = {"arch": arch, "dataset": dataset, "layer": layer}
        for eta in noises_nonzero:
            noisy = buckets.get((arch, dataset, eta, layer), [])
            if len(noisy) < 2:
                row[f"eta_{eta}"] = (math.nan, math.nan, math.nan)
                continue
            es = bootstrap_cohens_d(clean, noisy, n_resamples=10_000, seed=0)
            row[f"eta_{eta}"] = (es.d, es.ci_low, es.ci_high)
        rows.append(row)

    # ---- LaTeX rendering ----
    lines: list[str] = []
    n_eta = len(noises_nonzero)
    cols = "lll" + "c" * n_eta
    lines.append(rf"\begin{{tabular}}{{{cols}}}")
    lines.append(r"\toprule")
    header = ["Arch", "Dataset", "Layer"] + [
        rf"$\eta = {eta:g}$" for eta in noises_nonzero
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        cells = [
            row["arch"].replace("_", r"\_"),
            row["dataset"].replace("_", r"\_"),
            row["layer"].replace("_", r"\_"),
        ]
        for eta in noises_nonzero:
            d, lo, hi = row[f"eta_{eta}"]
            cells.append(fmt_ci(d, lo, hi, digits=2))
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = out_dir / "tab_cohens_d.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
