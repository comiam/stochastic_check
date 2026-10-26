"""Compare detector signals at the same empirical false-alarm rate."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stochastic_check.decision import (
    EnsembleBaseline,
    Signal,
    cusum_statistic,
    scalar,
    trace_layer,
)

from ..config import DataPaths
from ..loader import index_runs, load_jsonl
from .decision_alerts import _HEAD_LAYER, K

SIGNALS = ("trace", "grad_norm", "train_loss", "val_loss")
LABELS = {
    "trace": r"Layer-wise trace $\hat T_\ell$ (head)",
    "grad_norm": r"Gradient norm $\|\nabla_\theta\mathcal{L}\|$",
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
}


@dataclass
class SignalTrajectories:
    """Clean maxima and noisy CUSUM paths for one monitored signal."""

    clean: list[float]
    noisy: list[tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class OperatingPoint:
    """Pooled detector outcomes at one threshold."""

    threshold: float
    false_alarm_rate: float
    power: float
    mean_epoch: float


def _signal(name: str, arch: str) -> Signal:
    """Select the architecture's head trace or a recorded scalar."""
    return trace_layer(_HEAD_LAYER[arch]) if name == "trace" else scalar(name)


def _cusum_path(records, baseline: EnsembleBaseline) -> tuple[np.ndarray, np.ndarray]:
    """Standardize one stream and retain its two-sided CUSUM path."""
    _, epochs, values = baseline.standardize(records)
    positive, negative = cusum_statistic(values, k=K)
    return epochs, np.maximum(positive, negative)


def _clean_maxima(clean: dict, signal: Signal) -> list[float]:
    """Score clean streams against leave-one-out baselines."""
    maxima = []
    for seed, records in clean.items():
        other = [run for key, run in clean.items() if key != seed]
        baseline = EnsembleBaseline.from_runs(other, signal)
        _, trajectory = _cusum_path(records, baseline)

        maxima.append(float(trajectory.max()) if trajectory.size else 0.0)

    return maxima


def collect_trajectories(runs_root: Path) -> dict[str, SignalTrajectories]:
    """Pool group-wise standardized trajectories without mixing baselines."""
    groups: dict = defaultdict(dict)
    for (arch, dataset, noise), seeds in index_runs(runs_root).items():
        groups[(arch, dataset)][noise] = seeds

    pooled = {name: SignalTrajectories([], []) for name in SIGNALS}
    for (arch, _), by_noise in sorted(groups.items()):
        _collect_group(pooled, arch, by_noise)

    if not pooled["trace"].clean:
        raise FileNotFoundError(f"No usable detector runs under {runs_root}")

    return pooled


def _collect_group(pooled: dict, arch: str, by_noise: dict) -> None:
    """Add one architecture/dataset group to each signal's pooled results."""
    clean_paths = by_noise.get(0.0, {})
    if arch not in _HEAD_LAYER or len(clean_paths) < 3:
        return

    clean = {seed: load_jsonl(path) for seed, path in clean_paths.items()}
    noisy = [
        load_jsonl(path)
        for noise, paths in by_noise.items()
        if noise > 0.0
        for path in paths.values()
    ]

    for name in SIGNALS:
        signal = _signal(name, arch)
        baseline = EnsembleBaseline.from_runs(list(clean.values()), signal)
        pooled[name].clean.extend(_clean_maxima(clean, signal))
        pooled[name].noisy.extend(_cusum_path(run, baseline) for run in noisy)


def at_threshold(data: SignalTrajectories, threshold: float) -> OperatingPoint:
    """Measure power and detection epoch conditional on a threshold crossing."""
    false_alarm = float(np.mean(np.asarray(data.clean) > threshold))
    delays = []

    for epochs, trajectory in data.noisy:
        crossed = trajectory > threshold
        if crossed.any():
            delays.append(int(epochs[int(np.argmax(crossed))]))

    power = len(delays) / len(data.noisy)
    delay = float(np.mean(delays)) if delays else float("nan")
    return OperatingPoint(threshold, false_alarm, power, delay)


def evaluate(runs_root: Path, target_far: float = 0.10) -> dict[str, OperatingPoint]:
    """Choose the first point on the original 400-threshold matched-FAR grid."""
    trajectories = collect_trajectories(runs_root)
    points = {}

    for name, data in trajectories.items():
        for threshold in np.linspace(0.5, 40.0, 400):
            point = at_threshold(data, float(threshold))

            if point.false_alarm_rate <= target_far:
                points[name] = point
                break

    return points


def generate(paths: DataPaths, out_dir: Path) -> Path:
    """Write the article's matched-FAR baseline table from rerun records."""
    points = evaluate(paths.rerun_runs_root)
    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Monitored signal & Detection power & Detection epoch \\",
        r"\midrule",
    ]

    for name in SIGNALS:
        point = points[name]
        lines.append(
            f"{LABELS[name]} & ${point.power:.2f}$ & ${point.mean_epoch:.1f}$ " + r"\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "tab_baselines.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output
