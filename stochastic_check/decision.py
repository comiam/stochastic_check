"""Drift-standardized, two-sided CUSUM monitoring of recorded signals.

The clean Phase-I ensemble defines mu_0(t) and the unbiased sigma_0(t).
Standardization z_t = (T(t) - mu_0(t)) / sigma_0(t) removes learning-rate
and BatchNorm drift that makes a fixed baseline window unreliable.

Page's recursions are S+ = max(0, S+ + z - k) and
S- = max(0, S- - z - k); an alert occurs when max(S+, S-) > h.
The default k=0.5 targets a one-sigma shift. Monte Carlo calibration
chooses h for the requested null average run length (ARL0).

The asymptotic null is approximately normal. IID resampling assumes
that per-step centering has removed the dominant temporal correlation.
This module consumes JSONL records and has no PyTorch dependency.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Signal:
    """Name a scalar extractor; missing observations return None."""

    name: str
    extract: Callable[[dict], float | None]


def trace_layer(layer: str) -> Signal:
    """Read a layer trace from a record's trace mapping."""
    return Signal(layer, lambda rec: (rec.get("trace") or {}).get(layer))


def scalar(key: str) -> Signal:
    """Read a top-level scalar such as grad_norm or val_loss."""
    return Signal(key, lambda rec: rec.get(key))


def _as_signal(signal: str | Signal) -> Signal:
    """Preserve the legacy API accepting a layer name."""
    return signal if isinstance(signal, Signal) else trace_layer(signal)


def _finite_value(value) -> float | None:
    """Convert a usable observation, ignoring missing or non-finite values."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _series_arrays(steps, epochs, values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build aligned arrays with the detector's original dtypes."""
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(epochs, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def _signal_series(
    records: Iterable[dict], signal: Signal
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract finite observations in input order; do not sort the stream."""
    steps: list[int] = []
    epochs: list[int] = []
    values: list[float] = []

    for record in records:
        value = _finite_value(signal.extract(record))
        if value is None:
            continue

        steps.append(int(record["step"]))
        epochs.append(int(record["epoch"]))
        values.append(value)

    return _series_arrays(steps, epochs, values)


def _layer_series(
    records: Iterable[dict], layer: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a layer series through the legacy helper interface."""
    return _signal_series(records, trace_layer(layer))


def _common_series(calibration_runs, signal: Signal):
    """Collect nonempty series and their common step grid."""
    series: list[tuple[np.ndarray, np.ndarray]] = []
    common: set[int] | None = None

    for records in calibration_runs:
        steps, _, values = _signal_series(records, signal)
        if steps.size == 0:
            continue

        series.append((steps, values))
        common = set(steps.tolist()) if common is None else common & set(steps.tolist())

    if not series or not common:
        raise ValueError(
            f"No common step indices found for signal {signal.name!r}"
            " across the calibration runs."
        )

    return series, np.asarray(sorted(common), dtype=np.int64)


def _aligned_matrix(series, grid: np.ndarray) -> np.ndarray:
    """Index every calibration series onto the common step grid."""
    matrix = np.empty((len(series), grid.size), dtype=np.float64)
    for index, (steps, values) in enumerate(series):
        positions = {int(step): offset for offset, step in enumerate(steps)}
        matrix[index] = np.asarray(
            [values[positions[int(step)]] for step in grid], dtype=np.float64
        )

    return matrix


@dataclass(frozen=True)
class EnsembleBaseline:
    """Per-step clean mean and sample dispersion on the common grid."""

    signal: Signal
    steps: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    n: int

    @property
    def layer(self) -> str:
        """Expose the monitored name through the legacy layer property."""
        return self.signal.name

    @classmethod
    def from_runs(
        cls, calibration_runs: Sequence[Iterable[dict]], layer: str | Signal
    ) -> EnsembleBaseline:
        """Align clean runs and compute the mean and unbiased dispersion.

        Only shared steps are retained, keeping the ensemble balanced.
        At least two input runs are required for the sample dispersion.
        """
        if len(calibration_runs) < 2:
            raise ValueError(
                "EnsembleBaseline requires at least 2 calibration runs"
                " for an unbiased dispersion estimate."
            )

        signal = _as_signal(layer)
        series, grid = _common_series(calibration_runs, signal)
        matrix = _aligned_matrix(series, grid)
        mu = matrix.mean(axis=0)
        sigma = matrix.std(axis=0, ddof=1)

        return cls(signal=signal, steps=grid, mu=mu, sigma=sigma, n=len(series))

    def standardize(
        self, records: Iterable[dict]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standardize observations with finite, positive baseline dispersion."""
        steps, epochs, values = _signal_series(records, self.signal)
        if steps.size == 0:
            return steps, epochs, np.empty(0, dtype=np.float64)

        index = {int(step): offset for offset, step in enumerate(self.steps)}
        keep_steps: list[int] = []
        keep_epochs: list[int] = []
        keep_values: list[float] = []

        for step, epoch, value in zip(steps, epochs, values, strict=True):
            position = index.get(int(step))
            if position is None:
                continue

            dispersion = float(self.sigma[position])
            if not math.isfinite(dispersion) or dispersion <= 0.0:
                continue

            keep_steps.append(int(step))
            keep_epochs.append(int(epoch))
            keep_values.append((value - float(self.mu[position])) / dispersion)

        return _series_arrays(keep_steps, keep_epochs, keep_values)


@dataclass(frozen=True)
class CUSUMAlert:
    """Record the first crossing, or the maximum when no crossing occurs."""

    layer: str
    alert_step: int | None
    alert_epoch: int | None
    s_max: float
    k: float
    h: float
    n_observations: int


def cusum_statistic(z: np.ndarray, *, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the two-sided Page-Hinkley recursions from a zero state."""
    if k < 0:
        raise ValueError("CUSUM reference value k must be non-negative.")

    n = int(z.size)
    s_pos = np.zeros(n, dtype=np.float64)
    s_neg = np.zeros(n, dtype=np.float64)
    if n == 0:
        return s_pos, s_neg

    prev_p = 0.0
    prev_n = 0.0

    for t in range(n):
        prev_p = max(0.0, prev_p + float(z[t]) - k)
        prev_n = max(0.0, prev_n - float(z[t]) - k)
        s_pos[t] = prev_p
        s_neg[t] = prev_n

    return s_pos, s_neg


def online_cusum_alert(
    records: Iterable[dict],
    baseline: EnsembleBaseline,
    *,
    k: float = 0.5,
    h: float,
) -> CUSUMAlert:
    """Find the first strict crossing of h on a standardized stream."""
    steps, epochs, z = baseline.standardize(records)
    s_pos, s_neg = cusum_statistic(z, k=k)
    s_max_traj = np.maximum(s_pos, s_neg)

    if s_max_traj.size == 0:
        return CUSUMAlert(baseline.layer, None, None, math.nan, k, h, 0)

    crossed = s_max_traj > h
    if not crossed.any():
        return CUSUMAlert(
            baseline.layer, None, None, float(s_max_traj.max()), k, h, int(z.size)
        )

    index = int(np.argmax(crossed))
    return CUSUMAlert(
        baseline.layer,
        int(steps[index]),
        int(epochs[index]),
        float(s_max_traj[index]),
        k,
        h,
        int(z.size),
    )


def _null_pool(calibration_runs, layer: str | Signal) -> np.ndarray:
    """Concatenate leave-one-out null scores in calibration-run order."""
    cached = [list(run) for run in calibration_runs]
    chunks: list[np.ndarray] = []

    for index, holdout in enumerate(cached):
        rest = [run for offset, run in enumerate(cached) if offset != index]
        baseline = EnsembleBaseline.from_runs(rest, layer)
        _, _, values = baseline.standardize(holdout)

        if values.size > 0:
            chunks.append(values)

    if not chunks:
        raise ValueError(f"Leave-one-out null sample is empty for layer {layer!r}.")

    return np.concatenate(chunks)


def _sample_cusum_blocks(pool, n_replicates, block_length, rng, k) -> np.ndarray:
    """Draw the original IID null blocks and cache their CUSUM paths."""
    blocks = rng.choice(pool, size=(n_replicates, block_length), replace=True)
    maxima = np.empty_like(blocks)
    for index in range(n_replicates):
        positive, negative = cusum_statistic(blocks[index], k=k)
        np.maximum(positive, negative, out=maxima[index])

    return maxima


def _mean_run_length(blocks: np.ndarray, threshold: float, censored: float) -> float:
    """Average first crossings, censoring runs at the block length."""
    lengths = np.empty(len(blocks), dtype=np.float64)
    for index in range(len(blocks)):
        crossing = np.argmax(blocks[index] > threshold)

        if blocks[index, crossing] > threshold:
            lengths[index] = float(crossing + 1)
        else:
            lengths[index] = censored

    return float(lengths.mean())


def _bisect_threshold(blocks, target_arl0: float, censored: float) -> float:
    """Use the established bracket, iteration count, and stopping tolerance."""
    lo, hi = 0.1, 100.0

    for _ in range(64):
        mid = 0.5 * (lo + hi)

        if _mean_run_length(blocks, mid, censored) < target_arl0:
            lo = mid
        else:
            hi = mid

        if hi - lo < 1e-3:
            break

    return 0.5 * (lo + hi)


def calibrate_threshold(
    calibration_runs: Sequence[Iterable[dict]],
    layer: str | Signal,
    *,
    k: float = 0.5,
    target_arl0: float,
    n_replicates: int = 200,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Match empirical ARL0 by bisection on resampled null CUSUM paths.

    Leave-one-out scores supply the IID null pool. Each replicate has
    block_length observations (default: max(8 * ARL0, 1024)); no temporal
    ordering is retained after sampling. Per-step centering must therefore
    remove the dominant learning-rate-driven autocorrelation first.
    """
    if len(calibration_runs) < 3:
        raise ValueError(
            "calibrate_threshold needs at least 3 calibration runs"
            " for a leave-one-out null sample."
        )

    rng = rng if rng is not None else np.random.default_rng(0)
    pool = _null_pool(calibration_runs, layer)
    length = (
        block_length if block_length is not None else max(int(8 * target_arl0), 1024)
    )
    maxima = _sample_cusum_blocks(pool, n_replicates, length, rng, k)

    return _bisect_threshold(maxima, target_arl0, float(length))
