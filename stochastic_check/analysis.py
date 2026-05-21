"""Post-hoc analysis: Cohen's d on per-layer trace trajectories.

Compares the trace trajectory of layer ``ell`` between the clean
baseline (``noise=0``) and a noisy condition (``noise=eta``) over the
training epochs.  For each (arch, dataset, layer, epoch, eta) cell the
effect size is, following equation (eq:cohens-d) of the paper,

    d  =  ( mean(T_clean) - mean(T_noisy) ) / s_pooled,

with ``s_pooled`` the pooled within-condition standard deviation
across seeds.  The sign convention matches the paper: a positive
``d`` means the clean regime exhibits the larger trace, which is the
direction predicted for the head layer under symmetric label noise.
Confidence intervals are obtained by 10 000-resample bootstrap.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EffectSize:
    d: float
    ci_low: float
    ci_high: float
    n_clean: int
    n_noisy: int


def cohens_d(clean: Sequence[float], noisy: Sequence[float]) -> float:
    """Pooled-variance Cohen's d.

    Returns 0 when both groups are constant; NaN when fewer than 2
    samples in either group.
    """
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(noisy, dtype=np.float64)
    if a.size < 2 or b.size < 2:
        return math.nan
    s2 = ((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (
        a.size + b.size - 2
    )
    s = math.sqrt(max(s2, 1e-30))
    if s == 0.0:
        return 0.0
    return float((a.mean() - b.mean()) / s)


def bootstrap_cohens_d(
    clean: Sequence[float],
    noisy: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> EffectSize:
    """Percentile-bootstrap confidence interval for Cohen's d.

    Independent (unpaired) resampling within each group -- this is
    valid even when the two arms have different seed counts.
    """
    a = np.asarray(clean, dtype=np.float64)
    b = np.asarray(noisy, dtype=np.float64)
    rng = np.random.default_rng(seed)

    point = cohens_d(a, b)
    if math.isnan(point) or a.size < 2 or b.size < 2:
        return EffectSize(point, math.nan, math.nan, a.size, b.size)

    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        ar = rng.choice(a, size=a.size, replace=True)
        br = rng.choice(b, size=b.size, replace=True)
        samples[i] = cohens_d(ar, br)
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return EffectSize(point, float(lo), float(hi), a.size, b.size)
