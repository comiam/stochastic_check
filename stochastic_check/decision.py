"""Online decision rule on the trace trajectory.

Section 7.2 of the paper monitors the layer-wise trace
:math:`\\hat T_\\ell(t)` against a calibration ensemble of clean-label
runs.  The trace is a *non-stationary* signal -- under :math:`H_0`
(clean labels) it drifts with the LR schedule -- so a naive z-test
against a fixed baseline window is ill-posed: the empirical
:math:`\\sigma_0` underestimates the post-window dispersion and the
false-positive rate saturates at one.

The detector implemented here removes the drift component
explicitly.  Given a calibration ensemble of :math:`S\\geq 2`
clean-label runs we form the per-step Phase-I statistics

    mu_0(t)     = (1/S)  sum_s  hat T_\\ell^{(s)}(t),
    sigma_0(t)  = unbiased std over s,

and standardise the test trajectory as

    z_t  =  ( hat T_\\ell(t)  -  mu_0(t) )  /  sigma_0(t),

which is :math:`\\mathcal{N}(0,1)` asymptotically under :math:`H_0`.
A two-sided cumulative-sum (CUSUM, Page 1954) detector is then
applied to :math:`z_t`:

    S^{+}_t = max( 0,  S^{+}_{t-1} + z_t - k ),
    S^{-}_t = max( 0,  S^{-}_{t-1} - z_t - k ),
    alert   <=>  max(S^{+}_t, S^{-}_t)  >  h.

The reference value ``k`` is fixed to half the smallest shift the
detector should react to (in units of :math:`\\sigma_0`); the
default ``k = 0.5`` targets a :math:`1\\sigma` shift, which is the
canonical Page-Hinkley setting.  The threshold ``h`` is calibrated
empirically so that the average run length under :math:`H_0`
(:math:`\\mathrm{ARL}_0`) attains a prescribed value.

The module is intentionally free of PyTorch and works on the raw
JSONL records produced by :class:`stochastic_check.training.Trainer`,
so it can be reused both inside the training process and post-hoc
by the paper-table generators.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
#  Low-level helpers
# ---------------------------------------------------------------------------
def _layer_series(records: Iterable[dict], layer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned ``(steps, epochs, traces)`` arrays for one layer.

    Non-finite trace values are dropped.  The arrays are ordered by the
    order of ``records``; callers that need monotonic ``step`` should
    pre-sort.
    """
    steps: list[int] = []
    epochs: list[int] = []
    values: list[float] = []
    for rec in records:
        v = (rec.get("trace") or {}).get(layer)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(fv):
            continue
        steps.append(int(rec["step"]))
        epochs.append(int(rec["epoch"]))
        values.append(fv)
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(epochs, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
#  Phase-I baseline
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnsembleBaseline:
    """Per-step null statistics of layer ``layer`` over a calibration ensemble.

    ``steps`` is the common grid (intersection of step indices observed
    in every run); ``mu`` and ``sigma`` are the across-seed mean and
    unbiased standard deviation evaluated on that grid; ``n`` is the
    number of calibration runs.
    """

    layer: str
    steps: np.ndarray
    mu: np.ndarray
    sigma: np.ndarray
    n: int

    @classmethod
    def from_runs(cls, calibration_runs: Sequence[Iterable[dict]], layer: str) -> EnsembleBaseline:
        """Build the baseline from ``S`` calibration record streams.

        Only step indices that appear in **every** calibration run are
        retained, which keeps ``mu(t), sigma(t)`` balanced across seeds.
        Requires ``S >= 2`` runs so that the unbiased standard deviation
        is well defined.
        """
        if len(calibration_runs) < 2:
            raise ValueError(
                "EnsembleBaseline requires at least 2 calibration runs"
                " for an unbiased dispersion estimate."
            )
        series: list[tuple[np.ndarray, np.ndarray]] = []
        common: set[int] | None = None
        for recs in calibration_runs:
            s, _, v = _layer_series(recs, layer)
            if s.size == 0:
                continue
            series.append((s, v))
            common = set(s.tolist()) if common is None else common & set(s.tolist())
        if not series or not common:
            raise ValueError(
                f"No common step indices found for layer {layer!r}" " across the calibration runs."
            )
        grid = np.asarray(sorted(common), dtype=np.int64)
        # Index every series onto the common grid.
        matrix = np.empty((len(series), grid.size), dtype=np.float64)
        for i, (s, v) in enumerate(series):
            pos = {int(ss): j for j, ss in enumerate(s)}
            matrix[i] = np.asarray([v[pos[int(t)]] for t in grid], dtype=np.float64)
        mu = matrix.mean(axis=0)
        sigma = matrix.std(axis=0, ddof=1)
        return cls(layer=layer, steps=grid, mu=mu, sigma=sigma, n=len(series))

    def standardize(self, records: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(steps, epochs, z)`` for a test record stream.

        Steps not present in :attr:`steps` and steps at which the
        baseline ``sigma`` is non-positive are dropped: a finite
        baseline dispersion is mandatory for a well-defined z-score.
        """
        s, e, v = _layer_series(records, self.layer)
        if s.size == 0:
            return s, e, np.empty(0, dtype=np.float64)
        index = {int(t): j for j, t in enumerate(self.steps)}
        keep_s: list[int] = []
        keep_e: list[int] = []
        keep_z: list[float] = []
        for ss, ee, vv in zip(s, e, v, strict=True):
            j = index.get(int(ss))
            if j is None:
                continue
            sd = float(self.sigma[j])
            if not math.isfinite(sd) or sd <= 0.0:
                continue
            keep_s.append(int(ss))
            keep_e.append(int(ee))
            keep_z.append((vv - float(self.mu[j])) / sd)
        return (
            np.asarray(keep_s, dtype=np.int64),
            np.asarray(keep_e, dtype=np.int64),
            np.asarray(keep_z, dtype=np.float64),
        )


# ---------------------------------------------------------------------------
#  CUSUM detector
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CUSUMAlert:
    """Outcome of one CUSUM monitoring pass on a standardised stream."""

    layer: str
    alert_step: int | None
    alert_epoch: int | None
    s_max: float
    k: float
    h: float
    n_observations: int


def cusum_statistic(z: np.ndarray, *, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided Page-Hinkley CUSUM transform of a standardised signal.

    Returns ``(S_plus, S_minus)`` with ``S_plus[t] = max(0, S_plus[t-1]
    + z[t] - k)`` and ``S_minus[t] = max(0, S_minus[t-1] - z[t] - k)``,
    both initialised at zero.
    """
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
    """First-passage CUSUM detector for one layer of a test run.

    Parameters
    ----------
    records
        Per-step trace stream of the run under test.
    baseline
        Phase-I ensemble baseline produced by
        :meth:`EnsembleBaseline.from_runs`.
    k
        Reference value (slack) of the CUSUM; default ``0.5``.
    h
        Decision threshold; calibrated via :func:`calibrate_threshold`.
    """
    steps, epochs, z = baseline.standardize(records)
    s_pos, s_neg = cusum_statistic(z, k=k)
    s_max_traj = np.maximum(s_pos, s_neg)
    if s_max_traj.size == 0:
        return CUSUMAlert(baseline.layer, None, None, math.nan, k, h, 0)
    crossed = s_max_traj > h
    if not crossed.any():
        return CUSUMAlert(
            baseline.layer,
            None,
            None,
            float(s_max_traj.max()),
            k,
            h,
            int(z.size),
        )
    idx = int(np.argmax(crossed))
    return CUSUMAlert(
        baseline.layer,
        int(steps[idx]),
        int(epochs[idx]),
        float(s_max_traj[idx]),
        k,
        h,
        int(z.size),
    )


# ---------------------------------------------------------------------------
#  Threshold calibration via empirical ARL_0
# ---------------------------------------------------------------------------
def calibrate_threshold(
    calibration_runs: Sequence[Iterable[dict]],
    layer: str,
    *,
    k: float = 0.5,
    target_arl0: float,
    n_replicates: int = 200,
    block_length: int | None = None,
    rng: np.random.Generator | None = None,
) -> float:
    """Bisection on ``h`` matching empirical ARL\\(_0\\) to ``target_arl0``.

    The null sample is generated by leave-one-out standardisation on
    the calibration ensemble: each run is z-scored against the baseline
    built from the remaining :math:`S-1` runs, and the concatenated
    z-scores form the pool of null observations.  ARL\\(_0\\) is then
    estimated by ``n_replicates`` Monte-Carlo blocks of length
    ``block_length`` (default: ``8 * target_arl0``) drawn with
    replacement from that pool.

    Resampling without preserving temporal order is justified because
    the per-step centring already removes the dominant LR-driven
    autocorrelation; the residual z-stream is approximately iid under
    :math:`H_0`.
    """
    if len(calibration_runs) < 3:
        raise ValueError(
            "calibrate_threshold needs at least 3 calibration runs"
            " for a leave-one-out null sample."
        )
    rng = rng if rng is not None else np.random.default_rng(0)
    cached = [list(r) for r in calibration_runs]
    pool_chunks: list[np.ndarray] = []
    for i in range(len(cached)):
        holdout = cached[i]
        rest = [r for j, r in enumerate(cached) if j != i]
        loo_baseline = EnsembleBaseline.from_runs(rest, layer)
        _, _, z = loo_baseline.standardize(holdout)
        if z.size > 0:
            pool_chunks.append(z)
    if not pool_chunks:
        raise ValueError(f"Leave-one-out null sample is empty for layer {layer!r}.")
    pool = np.concatenate(pool_chunks)
    block_length = block_length if block_length is not None else max(int(8 * target_arl0), 1024)

    # Precompute Monte-Carlo blocks and their CUSUM transforms once;
    # the bisection then only re-evaluates the threshold crossing,
    # which is an O(block_length) numpy operation per ``h``.
    blocks = rng.choice(pool, size=(n_replicates, block_length), replace=True)
    s_max_blocks = np.empty_like(blocks)
    for r in range(n_replicates):
        sp, sn = cusum_statistic(blocks[r], k=k)
        np.maximum(sp, sn, out=s_max_blocks[r])

    censored = float(block_length)

    def arl0_at(h: float) -> float:
        run_lengths = np.empty(n_replicates, dtype=np.float64)
        for r in range(n_replicates):
            cross = np.argmax(s_max_blocks[r] > h)
            if s_max_blocks[r, cross] > h:
                run_lengths[r] = float(cross + 1)
            else:
                run_lengths[r] = censored
        return float(run_lengths.mean())

    # Bisection bracket.
    lo, hi = 0.1, 100.0
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if arl0_at(mid) < target_arl0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-3:
            break
    return 0.5 * (lo + hi)
