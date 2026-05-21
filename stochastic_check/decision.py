"""Online :math:`3\\sigma` decision rule for the trace trajectory.

Section 7.2 of the paper specifies the on-line detector

    z_\\ell(t)  =  ( hat T_\\ell(t)  -  mu_0(\\ell) )  /  s_0(\\ell),

where ``mu_0`` and ``s_0`` are the mean and the unbiased standard
deviation of the **clean-label** trace trajectory of layer ``ell``,
estimated on the first ``baseline_epochs`` of training.  An alert is
fired at the first step ``t`` at which ``|z_\\ell(t)| >= threshold``;
the conventional threshold is ``3``.

The function returns the alert step (or ``None`` if no alert), the
alert epoch and the realised z-score, so that downstream tabulators
can express the detection delay either in steps or in epochs.

The module is deliberately free of PyTorch and only takes the raw
JSONL records produced by :class:`stochastic_check.training.Trainer`,
so it can be reused both inside the training process (for live
diagnostics) and post-hoc by the paper-table generators.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AlertResult:
    layer: str
    alert_step: int | None
    alert_epoch: int | None
    z_score: float
    baseline_mean: float
    baseline_std: float
    n_baseline: int


def _layer_series(
    records: Iterable[dict], layer: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract aligned ``(steps, epochs, traces)`` arrays for one layer."""
    steps: list[int] = []
    epochs: list[int] = []
    values: list[float] = []
    for rec in records:
        v = (rec.get("trace") or {}).get(layer)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        steps.append(int(rec["step"]))
        epochs.append(int(rec["epoch"]))
        values.append(float(v))
    return (
        np.asarray(steps, dtype=np.int64),
        np.asarray(epochs, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
    )


def online_3sigma_alert(
    records: Iterable[dict],
    layer: str,
    *,
    baseline_epochs: int = 5,
    threshold: float = 3.0,
) -> AlertResult:
    """First-passage detector on the clean-label baseline statistics.

    Parameters
    ----------
    records
        Iterable of JSONL records ``{step, epoch, trace: {layer: ...}}``
        from the per-step trace stream of a single (clean or noisy)
        run.  The baseline is computed from the **same** stream on the
        first ``baseline_epochs``; this matches the paper's stipulation
        that the detector is calibrated on the clean prefix of the
        training trajectory.
    layer
        Name of the layer to monitor.
    baseline_epochs
        Number of leading epochs used to estimate ``mu_0`` and ``s_0``.
    threshold
        Critical absolute z-score; ``3`` reproduces the paper's
        :math:`3\\sigma` rule.

    Returns
    -------
    :class:`AlertResult`
        ``alert_step`` and ``alert_epoch`` are ``None`` when the trace
        never crosses the threshold; ``z_score`` is then the maximum
        absolute z-score actually observed.
    """
    steps, epochs, traces = _layer_series(list(records), layer)
    if traces.size == 0:
        return AlertResult(layer, None, None, math.nan, math.nan, math.nan, 0)

    in_baseline = epochs <= baseline_epochs
    baseline = traces[in_baseline]
    if baseline.size < 2:
        # Not enough samples to estimate dispersion; declare no alert.
        return AlertResult(layer, None, None, math.nan, math.nan, math.nan, int(baseline.size))

    mu0 = float(baseline.mean())
    s0 = float(baseline.std(ddof=1))
    if s0 == 0.0:
        return AlertResult(layer, None, None, math.nan, mu0, 0.0, int(baseline.size))

    z = (traces - mu0) / s0
    crossed = np.abs(z) >= threshold
    crossed_after_baseline = crossed & ~in_baseline
    if not crossed_after_baseline.any():
        # No alert; report the largest excursion observed so far for
        # diagnostics.
        idx_max = int(np.argmax(np.abs(z)))
        return AlertResult(
            layer,
            None,
            None,
            float(z[idx_max]),
            mu0,
            s0,
            int(baseline.size),
        )

    idx = int(np.argmax(crossed_after_baseline))
    return AlertResult(
        layer,
        int(steps[idx]),
        int(epochs[idx]),
        float(z[idx]),
        mu0,
        s0,
        int(baseline.size),
    )
