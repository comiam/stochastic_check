"""Compare clean/noisy per-layer trace means over a closed epoch window.

The default [10,20] window follows the paper. Each seed contributes one
mean per layer; Cohen's d and 10000 independent bootstrap resamples then
compare conditions without treating snapshots as independent runs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import DataPaths
from ..formatting import fmt_ci
from ..loader import iter_run_finals, load_jsonl

DEFAULT_EPOCH_WINDOW: tuple[int, int] = (10, 20)


def _windowed_trace_per_layer(
    run_dir: Path, window: tuple[int, int]
) -> dict[str, float]:
    """Average recorded values per layer, skipping missing files and NaN."""
    stream = run_dir / "steps.jsonl"
    if not stream.exists():
        return {}

    low, high = window
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for record in load_jsonl(stream):
        epoch = record.get("epoch")
        if epoch is None or not (low <= int(epoch) <= high):
            continue

        for name, value in (record.get("trace") or {}).items():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                sums[name] += float(value)
                counts[name] += 1

    return {name: sums[name] / counts[name] for name in sums if counts[name] > 0}


def _collect_buckets(runs_root: Path, window: tuple[int, int]):
    """Group seed means by architecture, dataset, noise, and layer."""
    buckets: dict[tuple[str, str, float, str], list[float]] = defaultdict(list)

    for run_dir, final in iter_run_finals(runs_root):
        config = final.get("config", {})
        arch = config.get("arch")
        dataset = config.get("dataset")
        noise = config.get("training", {}).get("noise_rate")

        if None in (arch, dataset, noise):
            continue

        for name, value in _windowed_trace_per_layer(run_dir, window).items():
            buckets[(arch, dataset, float(noise), name)].append(value)

    return buckets


def _effect_rows(buckets):
    """Compare each noisy condition with its matching clean-seed means."""
    from stochastic_check.analysis import bootstrap_cohens_d

    rows: list[dict[str, Any]] = []
    keys = {(arch, dataset, layer) for arch, dataset, _, layer in buckets}
    noises = sorted({noise for _, _, noise, _ in buckets if noise > 0.0})

    for arch, dataset, layer in sorted(keys):
        clean = buckets.get((arch, dataset, 0.0, layer), [])
        if len(clean) < 2:
            continue

        row: dict[str, Any] = {"arch": arch, "dataset": dataset, "layer": layer}

        for noise in noises:
            noisy = buckets.get((arch, dataset, noise, layer), [])

            if len(noisy) < 2:
                row[f"eta_{noise}"] = (math.nan, math.nan, math.nan)
                continue

            effect = bootstrap_cohens_d(clean, noisy, n_resamples=10_000, seed=0)
            row[f"eta_{noise}"] = (effect.d, effect.ci_low, effect.ci_high)

        rows.append(row)

    return rows, noises


def _render_rows(rows, noises) -> list[str]:
    """Render effect sizes with the original column order and precision."""
    columns = "lll" + "c" * len(noises)
    header = ["Arch", "Dataset", "Layer"] + [rf"$\eta = {noise:g}$" for noise in noises]
    lines = [
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]

    for row in rows:
        cells = [row[key].replace("_", r"\_") for key in ("arch", "dataset", "layer")]

        for noise in noises:
            value, low, high = row[f"eta_{noise}"]
            cells.append(fmt_ci(value, low, high, digits=2))
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return lines


def generate(
    paths: DataPaths,
    out_dir: Path,
    epoch_window: tuple[int, int] = DEFAULT_EPOCH_WINDOW,
) -> Path:
    """Write windowed layer effects without changing the bootstrap settings."""
    buckets = _collect_buckets(paths.exp_b_runs_root, epoch_window)
    rows, noises = _effect_rows(buckets)
    lines = _render_rows(rows, noises)

    output = out_dir / "tab_cohens_d.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
