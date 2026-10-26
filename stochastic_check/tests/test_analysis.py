"""Check effect-size conventions and bootstrap summaries."""

from __future__ import annotations

import math

import numpy as np

from stochastic_check.analysis import bootstrap_cohens_d, cohens_d


def test_cohens_d_returns_zero_for_constant_inputs() -> None:
    """Keep undefined constant-group effects at zero rather than inflating them."""
    assert cohens_d([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]) == 0.0
    assert cohens_d([3.0, 3.0], [3.0, 3.0]) == 0.0


def test_cohens_d_basic_separation() -> None:
    """Give positive d when the clean sample has the larger mean."""
    rng = np.random.default_rng(0)
    clean = rng.normal(loc=1.0, scale=0.1, size=200).tolist()
    noisy = rng.normal(loc=0.0, scale=0.1, size=200).tolist()
    d = cohens_d(clean, noisy)
    assert d > 5.0


def test_cohens_d_nan_for_tiny_samples() -> None:
    """Reject a group too small to define sample variance."""
    assert math.isnan(cohens_d([1.0], [2.0, 3.0]))
    assert math.isnan(cohens_d([1.0, 2.0], [3.0]))


def test_bootstrap_cohens_d_ci_contains_point_estimate() -> None:
    """Retain the point estimate within the sampled interval and report sizes."""
    rng = np.random.default_rng(1)
    clean = rng.normal(loc=1.0, scale=0.5, size=40).tolist()
    noisy = rng.normal(loc=0.0, scale=0.5, size=40).tolist()
    es = bootstrap_cohens_d(clean, noisy, n_resamples=500, seed=7)

    assert es.ci_low <= es.d <= es.ci_high
    assert es.n_clean == 40
    assert es.n_noisy == 40
