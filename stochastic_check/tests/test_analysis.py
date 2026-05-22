"""Tests for :mod:`stochastic_check.analysis`."""

from __future__ import annotations

import math

import numpy as np

from stochastic_check.analysis import bootstrap_cohens_d, cohens_d


def test_cohens_d_returns_zero_for_constant_inputs() -> None:
    """Regression: when both groups have zero within-group variance,
    Cohen's d is undefined; we return ``0.0`` rather than dividing by a
    floor like ``1e-30`` (which silently blows up to ``~1e15``)."""
    assert cohens_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0
    assert cohens_d([3.0, 3.0], [3.0, 3.0]) == 0.0


def test_cohens_d_basic_separation() -> None:
    """Sanity: well-separated normal samples produce a large positive
    d.  The clean group has the *larger* mean so the sign follows the
    paper's convention (positive d = clean curvature dominates)."""
    rng = np.random.default_rng(0)
    clean = rng.normal(loc=1.0, scale=0.1, size=200).tolist()
    noisy = rng.normal(loc=0.0, scale=0.1, size=200).tolist()
    d = cohens_d(clean, noisy)
    assert d > 5.0


def test_cohens_d_nan_for_tiny_samples() -> None:
    assert math.isnan(cohens_d([1.0], [2.0, 3.0]))
    assert math.isnan(cohens_d([1.0, 2.0], [3.0]))


def test_bootstrap_cohens_d_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(1)
    clean = rng.normal(loc=1.0, scale=0.5, size=40).tolist()
    noisy = rng.normal(loc=0.0, scale=0.5, size=40).tolist()
    es = bootstrap_cohens_d(clean, noisy, n_resamples=500, seed=7)
    assert es.ci_low <= es.d <= es.ci_high
    assert es.n_clean == 40
    assert es.n_noisy == 40
