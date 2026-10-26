"""Generate the head-layer CUSUM alert table from original trajectories.

Each architecture/dataset gets a clean Phase-I baseline and an ARL0-calibrated
threshold. Clean rows use leave-one-out baselines; noisy rows use the full
clean ensemble. Detection epochs are summarized only over alerted seeds.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from stochastic_check.decision import (
    EnsembleBaseline,
    calibrate_threshold,
    online_cusum_alert,
)

from ..config import DataPaths
from ..formatting import fmt_mean_std
from ..loader import RunIndex, index_runs, load_jsonl

# A half-sigma reference targets one-sigma shifts; ARL0 is in snapshot steps.
K: float = 0.5
TARGET_ARL0: float = 1000.0
_HEAD_LAYER = {"resnet18": "fc", "resnet34": "fc", "vgg11": "classifier"}


def _head_of(arch: str) -> str | None:
    """Select the classifier layer monitored in the paper."""
    return _HEAD_LAYER.get(arch)


def _index_runs(runs_root: Path) -> RunIndex:
    """Keep the legacy alias for the shared cell/seed index."""
    return index_runs(runs_root)


def _alert_epoch_under_baseline(
    records: list[dict[str, Any]], baseline: EnsembleBaseline, h: float
) -> int | None:
    """Return the first alert epoch under the configured reference value."""
    alert = online_cusum_alert(records, baseline=baseline, k=K, h=h)
    return alert.alert_epoch


def _clean_alert_epochs(clean_records: dict, head: str, threshold: float) -> list[int]:
    """Score each clean seed against the other clean seeds."""
    epochs = []
    for seed, records in clean_records.items():
        rest = [run for other, run in clean_records.items() if other != seed]
        baseline = EnsembleBaseline.from_runs(rest, head)
        epoch = _alert_epoch_under_baseline(records, baseline, threshold)

        if epoch is not None:
            epochs.append(epoch)

    return epochs


def _noisy_alert_epochs(
    paths: dict, baseline: EnsembleBaseline, threshold: float
) -> list[int]:
    """Score noisy seeds against the full clean ensemble."""
    epochs = []
    for path in paths.values():
        epoch = _alert_epoch_under_baseline(load_jsonl(path), baseline, threshold)
        if epoch is not None:
            epochs.append(epoch)

    return epochs


def _evaluate_group(
    arch: str,
    dataset: str,
    clean_paths: dict[int, Path],
    noisy_index: dict[float, dict[int, Path]],
) -> dict[float, dict[str, Any]]:
    """Calibrate one group and return detected epochs, seed counts, and threshold."""
    head = _head_of(arch)
    if head is None or len(clean_paths) < 3:
        return {}

    clean = {seed: load_jsonl(path) for seed, path in clean_paths.items()}
    calibration_runs = list(clean.values())
    baseline = EnsembleBaseline.from_runs(calibration_runs, head)
    threshold = calibrate_threshold(
        calibration_runs,
        layer=head,
        k=K,
        target_arl0=TARGET_ARL0,
        n_replicates=400,
        rng=np.random.default_rng(0),
    )

    result = {
        0.0: {
            "alert_epochs": _clean_alert_epochs(clean, head, threshold),
            "n_seeds": len(clean),
            "h": threshold,
        }
    }
    for noise, paths in noisy_index.items():
        result[noise] = {
            "alert_epochs": _noisy_alert_epochs(paths, baseline, threshold),
            "n_seeds": len(paths),
            "h": threshold,
        }

    return result


def _table_rows(index: RunIndex) -> list[tuple]:
    """Group runs and preserve the published architecture/dataset/noise ordering."""
    groups: dict = defaultdict(dict)
    for (arch, dataset, noise), seeds in index.items():
        groups[(arch, dataset)][noise] = seeds

    rows = []
    for (arch, dataset), by_noise in sorted(groups.items()):
        clean = by_noise.get(0.0, {})
        noisy = {noise: seeds for noise, seeds in by_noise.items() if noise > 0.0}
        result = _evaluate_group(arch, dataset, clean, noisy)

        for noise in sorted(result):
            cell = result[noise]
            rows.append(
                (arch, dataset, noise, cell["alert_epochs"], cell["n_seeds"], cell["h"])
            )

    return rows


def _format_row(row: tuple) -> str:
    """Render a row using the original conditional epoch and rate conventions."""
    arch, dataset, noise, alerted, n_seeds, threshold = row
    rate = len(alerted) / n_seeds if n_seeds > 0 else math.nan
    return (
        " & ".join(
            [
                arch.replace("_", r"\_"),
                dataset.replace("_", r"\_"),
                f"${noise:.2f}$",
                fmt_mean_std(np.asarray(alerted, dtype=float), digits=1)
                if alerted
                else "---",
                f"{rate:.2f}" if not math.isnan(rate) else "---",
                f"{threshold:.2f}",
            ]
        )
        + r" \\"
    )


def generate(paths: DataPaths, out_dir: Path) -> Path:
    """Write the original-data alert table without modifying its source records."""
    rows = _table_rows(_index_runs(paths.exp_b_runs_root))
    lines = [
        r"\begin{tabular}{lllccc}",
        r"\toprule",
        r"Arch & Dataset & $\eta$ & Det.\ epoch (mean$\pm$std) & Alert rate & $h$ \\",
        r"\midrule",
    ]
    lines.extend(_format_row(row) for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    output = out_dir / "tab_alerts.tex"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
