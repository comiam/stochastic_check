"""Table ``tab:cohens-d`` -- effect size of label noise on layer-wise trace.

For each (arch, dataset, layer) cell we read the per-step trace
trajectory of every seed, take the **final-epoch mean** of the trace
for the layer, and compute Cohen's d between the clean baseline
(``noise = 0``) and each non-zero noise level.  Bootstrap 95% CIs
are obtained over 10 000 resamples.

The output table has columns = ``arch / layer``, rows = noise levels,
cells = ``d [lo, hi]``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import DataPaths
from ..formatting import fmt_ci
from ..loader import iter_run_finals, load_jsonl


def _final_trace_per_layer(run_dir: Path) -> dict[str, float]:
    """Read ``steps.jsonl`` and take the **last** trace per layer.

    Missing file -> empty dict.
    """
    jsonl = run_dir / "steps.jsonl"
    if not jsonl.exists():
        return {}
    records = load_jsonl(jsonl)
    if not records:
        return {}
    last = records[-1]
    return {
        k: float(v)
        for k, v in last.get("trace", {}).items()
        if not (isinstance(v, float) and math.isnan(v))
    }


def generate(paths: DataPaths, out_dir: Path) -> Path:
    # Index runs by (arch, dataset, noise, layer) -> list of values across seeds.
    buckets: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)
    for run_dir, final in iter_run_finals(paths.exp_b_runs_root):
        cfg = final.get("config", {})
        arch = cfg.get("arch")
        dataset = cfg.get("dataset")
        noise = cfg.get("training", {}).get("noise_rate")
        if None in (arch, dataset, noise):
            continue
        traces = _final_trace_per_layer(run_dir)
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
