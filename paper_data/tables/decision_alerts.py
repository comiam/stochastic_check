"""Table ``tab:alerts`` -- online CUSUM detection rule (paper sec. 7.2).

For every ``(arch, dataset)`` group we

  1. collect the per-step trace stream of all ``noise = 0`` runs to
     build the Phase-I ensemble baseline
     :class:`stochastic_check.decision.EnsembleBaseline` on the
     monitored head layer;
  2. calibrate the CUSUM threshold ``h`` so that the empirical
     :math:`\\mathrm{ARL}_0` matches :data:`TARGET_ARL0`, using
     :func:`stochastic_check.decision.calibrate_threshold`;
  3. for every noisy run of the group, replay the trace stream
     through :func:`stochastic_check.decision.online_cusum_alert` and
     record the alert epoch.

Clean-noise rows are kept as a control: we re-evaluate each clean
seed with the leave-one-out baseline (the same that drives the
calibration), so the table also publishes the realised false-alarm
rate of the detector.

The monitored layer is the head (last classifier layer): paper
sec. 7.3 phrases the discriminability claim on this layer
specifically and the detector inherits the same scope.
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
from ..loader import iter_run_finals, load_jsonl

# Detector hyper-parameters (paper sec. 7.2):
#   K               -- Page-Hinkley reference value; 0.5 targets a
#                       1-sigma shift, the canonical setting.
#   TARGET_ARL0     -- desired average run length under H_0, in
#                       trace-snapshot steps.  With ~117 snapshots per
#                       run the per-experiment false-alarm probability
#                       sits below ~12 %.
K: float = 0.5
TARGET_ARL0: float = 1000.0

# Head layer monitored for each architecture (paper sec. 7.3).
_HEAD_LAYER: dict[str, str] = {
    "resnet18": "fc",
    "resnet34": "fc",
    "vgg11": "classifier",
}


def _head_of(arch: str) -> str | None:
    return _HEAD_LAYER.get(arch)


def _index_runs(
    runs_root: Path,
) -> dict[tuple[str, str, float], dict[int, Path]]:
    """Map ``(arch, dataset, noise) -> {seed: steps.jsonl path}``."""
    index: dict[tuple[str, str, float], dict[int, Path]] = defaultdict(dict)
    for run_dir, final in iter_run_finals(runs_root):
        cfg = final.get("config", {})
        arch = cfg.get("arch")
        dataset = cfg.get("dataset")
        noise = cfg.get("training", {}).get("noise_rate")
        seed = cfg.get("training", {}).get("seed")
        if None in (arch, dataset, noise, seed):
            continue
        jsonl = run_dir / "steps.jsonl"
        if not jsonl.exists():
            continue
        index[(str(arch), str(dataset), float(noise))][int(seed)] = jsonl
    return index


def _alert_epoch_under_baseline(
    records: list[dict[str, Any]], baseline: EnsembleBaseline, h: float
) -> int | None:
    alert = online_cusum_alert(records, baseline=baseline, k=K, h=h)
    return alert.alert_epoch


def _evaluate_group(
    arch: str,
    dataset: str,
    clean_paths: dict[int, Path],
    noisy_index: dict[float, dict[int, Path]],
) -> dict[float, dict[str, Any]]:
    """Calibrate ``h`` on the clean ensemble and score every noise level.

    Returns ``{noise: {alert_epochs, alert_rate, n_seeds, h}}``.
    ``alert_epochs`` lists detection epochs of seeds that actually
    fired; ``alert_rate`` is over all seeds of the cell.
    """
    head = _head_of(arch)
    if head is None or len(clean_paths) < 3:
        return {}

    clean_records: dict[int, list[dict[str, Any]]] = {
        seed: load_jsonl(p) for seed, p in clean_paths.items()
    }
    calibration_runs = list(clean_records.values())
    baseline = EnsembleBaseline.from_runs(calibration_runs, head)
    h = calibrate_threshold(
        calibration_runs,
        layer=head,
        k=K,
        target_arl0=TARGET_ARL0,
        n_replicates=400,
        rng=np.random.default_rng(0),
    )

    out: dict[float, dict[str, Any]] = {}

    # Clean control row: leave-one-out evaluation.
    clean_epochs: list[int] = []
    for seed, recs in clean_records.items():
        rest = [r for s, r in clean_records.items() if s != seed]
        loo_baseline = EnsembleBaseline.from_runs(rest, head)
        ep = _alert_epoch_under_baseline(recs, loo_baseline, h)
        if ep is not None:
            clean_epochs.append(ep)
    out[0.0] = {
        "alert_epochs": clean_epochs,
        "n_seeds": len(clean_records),
        "h": h,
    }

    # Noisy cells: each noisy run scored against the full clean ensemble.
    for noise, seed_paths in noisy_index.items():
        epochs: list[int] = []
        for path in seed_paths.values():
            ep = _alert_epoch_under_baseline(load_jsonl(path), baseline, h)
            if ep is not None:
                epochs.append(ep)
        out[noise] = {
            "alert_epochs": epochs,
            "n_seeds": len(seed_paths),
            "h": h,
        }
    return out


def generate(paths: DataPaths, out_dir: Path) -> Path:
    index = _index_runs(paths.exp_b_runs_root)

    # Reshape: (arch, dataset) -> {noise: {seed: jsonl}}
    groups: dict[tuple[str, str], dict[float, dict[int, Path]]] = defaultdict(dict)
    for (arch, dataset, noise), seeds in index.items():
        groups[(arch, dataset)][noise] = seeds

    rows: list[tuple[str, str, float, list[int], int, float]] = []
    for (arch, dataset), by_noise in sorted(groups.items()):
        clean = by_noise.get(0.0, {})
        noisy = {eta: s for eta, s in by_noise.items() if eta > 0.0}
        result = _evaluate_group(arch, dataset, clean, noisy)
        if not result:
            continue
        for noise in sorted(result):
            cell = result[noise]
            rows.append(
                (
                    arch,
                    dataset,
                    noise,
                    cell["alert_epochs"],
                    cell["n_seeds"],
                    cell["h"],
                )
            )

    # ---- LaTeX rendering ----
    lines: list[str] = []
    lines.append(r"\begin{tabular}{lllccc}")
    lines.append(r"\toprule")
    lines.append(
        r"Arch & Dataset & $\eta$ & "
        r"Det.\ epoch (mean$\pm$std) & Alert rate & $h$ \\"
    )
    lines.append(r"\midrule")
    for arch, dataset, noise, alerted, n_seeds, h in rows:
        rate = (len(alerted) / n_seeds) if n_seeds > 0 else math.nan
        lines.append(
            " & ".join(
                [
                    arch.replace("_", r"\_"),
                    dataset.replace("_", r"\_"),
                    f"${noise:g}$",
                    fmt_mean_std(np.asarray(alerted, dtype=float), digits=1)
                    if alerted
                    else "---",
                    f"{rate:.2f}" if not math.isnan(rate) else "---",
                    f"{h:.2f}",
                ]
            )
            + r" \\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_path = out_dir / "tab_alerts.tex"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
