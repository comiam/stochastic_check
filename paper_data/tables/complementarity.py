"""Complementarity of the layer-wise Hessian trace and first-order signals.

The per-layer trace is not a faster stand-alone detector of the label-memorisation
regime than the gradient norm (a matched-false-alarm comparison shows first-order
signals detect at least as early).  Its value is *complementary*: the false alarms
of a curvature detector and of a gradient-norm detector are largely independent,
so requiring **both** to fire (a logical AND) suppresses the joint false-alarm rate
while retaining detection power.

Each signal is calibrated exactly as in the paper's rule (Page-Hinkley CUSUM,
threshold set to :data:`TARGET_ARL0` via leave-one-out Monte-Carlo).  The AND
false-alarm rate is then an *emergent* quantity -- we do not re-calibrate the
fused detector -- which keeps the original per-signal claim intact.

Run ``python -m paper_data.tables.complementarity`` for the per-architecture and
pooled table; add ``--observe-layers ARCH DATASET`` for the (observation-only)
per-layer sensitivity readout.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from stochastic_check.decision import (
    EnsembleBaseline,
    Signal,
    calibrate_threshold,
    online_cusum_alert,
    scalar,
    trace_layer,
)

from ..config import DataPaths
from ..loader import index_runs, load_jsonl
from .decision_alerts import _HEAD_LAYER, TARGET_ARL0, K

_CALIB_PROBES = 400
_BOOTSTRAP = 5000


@dataclass(frozen=True)
class SignalEval:
    """Per-seed alert outcomes of one calibrated detector over one group."""

    name: str
    h: float
    clean_alert: dict[int, bool]  # clean seed -> alerted (leave-one-out)
    noisy_alert: dict[tuple[float, int], int | None]  # (noise, seed) -> alert epoch


def evaluate_signal(
    clean_paths: Mapping[int, Path],
    noisy_paths: Mapping[tuple[float, int], Path],
    signal: Signal,
) -> SignalEval:
    """Calibrate ``signal`` at ``TARGET_ARL0`` and score clean + noisy runs."""
    clean_recs = {s: load_jsonl(p) for s, p in clean_paths.items()}
    runs = list(clean_recs.values())
    h = calibrate_threshold(
        runs,
        signal,
        k=K,
        target_arl0=TARGET_ARL0,
        n_replicates=_CALIB_PROBES,
        rng=np.random.default_rng(0),
    )

    clean_alert: dict[int, bool] = {}
    for seed, recs in clean_recs.items():
        rest = [r for s, r in clean_recs.items() if s != seed]
        base = EnsembleBaseline.from_runs(rest, signal)
        clean_alert[seed] = (
            online_cusum_alert(recs, baseline=base, k=K, h=h).alert_epoch is not None
        )

    base_full = EnsembleBaseline.from_runs(runs, signal)
    noisy_alert: dict[tuple[float, int], int | None] = {}

    for key, p in noisy_paths.items():
        noisy_alert[key] = online_cusum_alert(
            load_jsonl(p), baseline=base_full, k=K, h=h
        ).alert_epoch

    return SignalEval(signal.name, h, clean_alert, noisy_alert)


def _far(values) -> float:
    """Compute the observed alert fraction over clean seeds."""
    vals = [bool(v) for v in values]
    return float(np.mean(vals)) if vals else float("nan")


def _power(epochs) -> float:
    """Compute the fraction of noisy runs that produced an alert."""
    eps = list(epochs)
    return float(np.mean([e is not None for e in eps])) if eps else float("nan")


def _mean_delay(epochs) -> float:
    """Average detection epochs only over runs that triggered."""
    eps = [e for e in epochs if e is not None]
    return float(np.mean(eps)) if eps else float("nan")


def _bootstrap_far_ci(values, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a false-alarm rate over the clean seeds."""
    arr = np.asarray([1.0 if v else 0.0 for v in values], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(seed)
    boot = np.array(
        [rng.choice(arr, arr.size, replace=True).mean() for _ in range(_BOOTSTRAP)]
    )

    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def and_fuse(
    a: SignalEval, b: SignalEval
) -> tuple[dict[int, bool], dict[tuple[float, int], int | None]]:
    """AND two calibrated detectors: alert only when *both* fire.

    The fused detection epoch is the later of the two crossings (the point at
    which the second detector confirms the first).
    """
    clean = {
        s: a.clean_alert[s] and b.clean_alert[s]
        for s in a.clean_alert.keys() & b.clean_alert.keys()
    }
    noisy: dict[tuple[float, int], int | None] = {}
    for k in a.noisy_alert.keys() & b.noisy_alert.keys():
        ea, eb = a.noisy_alert[k], b.noisy_alert[k]
        noisy[k] = max(ea, eb) if (ea is not None and eb is not None) else None

    return clean, noisy


def _group_paths(index, arch, dataset):
    """Select clean and noisy trajectories for one architecture/dataset."""
    clean = index.get((arch, dataset, 0.0), {})
    noisy = {
        (noise, seed): p
        for (a, d, noise), seeds in index.items()
        if a == arch and d == dataset and noise > 0.0
        for seed, p in seeds.items()
    }

    return clean, noisy


def complementarity_report(runs_root: Path) -> None:
    """Print per-group and pooled outcomes of the two detectors and their AND."""
    index = index_runs(runs_root)
    cells = sorted({(a, d) for (a, d, _) in index})

    p_tr: list[bool] = []
    p_gr: list[bool] = []
    p_and_clean: list[bool] = []
    p_and_noisy: list[int | None] = []

    hdr = (
        f"{'arch':9s} {'dataset':9s} | {'trFAR':>6s} {'grFAR':>6s} {'andFAR':>6s} "
        f"{'and CI95':>15s} {'and pow':>7s} {'and del':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))

    for arch, dataset in cells:
        head = _HEAD_LAYER.get(arch)
        clean_paths, noisy_paths = _group_paths(index, arch, dataset)
        if head is None or len(clean_paths) < 3 or not noisy_paths:
            continue

        ev_tr = evaluate_signal(clean_paths, noisy_paths, trace_layer(head))
        ev_gr = evaluate_signal(clean_paths, noisy_paths, scalar("grad_norm"))
        and_clean, and_noisy = and_fuse(ev_tr, ev_gr)
        lo, hi = _bootstrap_far_ci(and_clean.values())

        print(
            f"{arch:9s} {dataset:9s} | {_far(ev_tr.clean_alert.values()):6.3f} "
            f"{_far(ev_gr.clean_alert.values()):6.3f} {_far(and_clean.values()):6.3f} "
            f"{f'({lo:.3f},{hi:.3f})':>15s} {_power(and_noisy.values()):7.3f} "
            f"{_mean_delay(and_noisy.values()):7.1f}"
        )

        p_tr += list(ev_tr.clean_alert.values())
        p_gr += list(ev_gr.clean_alert.values())
        p_and_clean += list(and_clean.values())
        p_and_noisy += list(and_noisy.values())

    lo, hi = _bootstrap_far_ci(p_and_clean)
    print("-" * len(hdr))
    print(
        f"{'POOLED':9s} {'':9s} | {_far(p_tr):6.3f} {_far(p_gr):6.3f} {_far(p_and_clean):6.3f} "
        f"{f'({lo:.3f},{hi:.3f})':>15s} {_power(p_and_noisy):7.3f} {_mean_delay(p_and_noisy):7.1f}"
    )


def layer_sensitivity_report(runs_root: Path, arch: str, dataset: str) -> None:
    """Observation only: per-layer FAR/power/delay at the paper's calibration.

    There is no universal best layer; this is a probe to watch which layers
    respond and how the head compares -- not a claim.
    """
    index = index_runs(runs_root)
    clean_paths, noisy_paths = _group_paths(index, arch, dataset)
    sample = load_jsonl(next(iter(clean_paths.values())))
    layers = list(sample[0]["trace"].keys())
    head = _HEAD_LAYER.get(arch)

    rows = []

    for layer in layers:
        ev = evaluate_signal(clean_paths, noisy_paths, trace_layer(layer))
        rows.append(
            (
                layer,
                _far(ev.clean_alert.values()),
                _power(ev.noisy_alert.values()),
                _mean_delay(ev.noisy_alert.values()),
            )
        )

    rows.sort(key=lambda r: float("inf") if r[3] != r[3] else r[3])
    print(f"[observer] {arch}/{dataset}: per-layer detection (ARL0={TARGET_ARL0:.0f})")
    print(f"  {'layer':24s} {'FAR':>6s} {'power':>6s} {'delay':>6s}")

    for layer, f, p, d in rows:
        mark = "  <- head" if layer == head else ""
        print(f"  {layer:24s} {f:6.2f} {p:6.2f} {d:6.1f}{mark}")


def main() -> None:
    """Report complementarity or layer sensitivity from the rerun profile."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="runs root (default: DataPaths().rerun_runs_root)",
    )
    ap.add_argument(
        "--observe-layers", nargs=2, metavar=("ARCH", "DATASET"), default=None
    )

    args = ap.parse_args()
    runs_root = args.runs_root or DataPaths().rerun_runs_root

    if args.observe_layers:
        layer_sensitivity_report(runs_root, *args.observe_layers)
    else:
        complementarity_report(runs_root)


if __name__ == "__main__":
    main()
