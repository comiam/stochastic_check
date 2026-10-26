"""Tests for the CUSUM decision rule (section 7.2 of the paper)."""

from __future__ import annotations

import numpy as np

from stochastic_check.decision import (
    EnsembleBaseline,
    calibrate_threshold,
    cusum_statistic,
    online_cusum_alert,
)


def _records_from_trace(
    trace: np.ndarray, *, snapshots_per_epoch: int = 4
) -> list[dict]:
    """Wrap a per-step trace into the JSONL record schema."""
    return [
        {
            "step": i + 1,
            "epoch": (i // snapshots_per_epoch) + 1,
            "trace": {"L0": float(v)},
        }
        for i, v in enumerate(trace)
    ]


def _make_calibration_runs(
    n_runs: int, n_steps: int, *, mu_t: np.ndarray, sigma: float, seed: int
) -> list[list[dict]]:
    """Synthesise ``n_runs`` clean runs sharing the drift ``mu_t``."""
    rng = np.random.default_rng(seed)
    runs: list[list[dict]] = []
    for _ in range(n_runs):
        eps = rng.normal(0.0, sigma, size=n_steps)
        runs.append(_records_from_trace(mu_t + eps))

    return runs


# ---------------------------------------------------------------------------
#  EnsembleBaseline
# ---------------------------------------------------------------------------
def test_ensemble_baseline_recovers_drift_and_dispersion() -> None:
    """Recover the ensemble drift and dispersion within sampling uncertainty."""
    n_steps = 120
    mu_t = np.linspace(1.0, 3.0, n_steps)
    runs = _make_calibration_runs(20, n_steps, mu_t=mu_t, sigma=0.2, seed=0)
    base = EnsembleBaseline.from_runs(runs, layer="L0")

    assert base.steps.size == n_steps
    # Aggregate, distribution-level recovery: pointwise stderr of mu is
    # sigma / sqrt(S) ~ 0.045, so we verify the global drift profile.
    assert abs(float(np.mean(base.mu - mu_t))) < 0.02
    assert float(np.std(base.mu - mu_t)) < 0.1
    assert 0.15 < float(np.median(base.sigma)) < 0.25


def test_standardize_yields_unit_variance_under_null() -> None:
    """Keep held-out null scores centered with approximately unit variance."""
    n_steps = 200
    mu_t = np.linspace(0.0, 5.0, n_steps)
    runs = _make_calibration_runs(40, n_steps, mu_t=mu_t, sigma=0.3, seed=1)
    base = EnsembleBaseline.from_runs(runs[:-1], layer="L0")
    _, _, z = base.standardize(runs[-1])
    # Mean ~ 0, std ~ 1 with finite-ensemble inflation O(sqrt(1 + 1/(S-1))).

    assert abs(float(z.mean())) < 0.25
    assert 0.85 < float(z.std(ddof=1)) < 1.25


def test_from_runs_requires_two_calibration_runs() -> None:
    """Reject an ensemble that cannot define a sample dispersion."""
    runs = _make_calibration_runs(1, 10, mu_t=np.zeros(10), sigma=1.0, seed=2)

    try:
        EnsembleBaseline.from_runs(runs, layer="L0")
    except ValueError:
        return
    raise AssertionError("EnsembleBaseline.from_runs must reject a single run.")


# ---------------------------------------------------------------------------
#  CUSUM detector
# ---------------------------------------------------------------------------
def test_cusum_stays_at_zero_on_zero_input() -> None:
    """Keep both CUSUM arms clamped when standardized input is zero."""
    z = np.zeros(50)
    s_pos, s_neg = cusum_statistic(z, k=0.5)
    assert np.all(s_pos == 0.0)
    assert np.all(s_neg == 0.0)


def test_cusum_grows_linearly_on_constant_positive_drift() -> None:
    """Accumulate positive drift at the expected rate without negative drift."""
    z = np.full(40, 1.0)
    s_pos, s_neg = cusum_statistic(z, k=0.5)
    # Positive arm accumulates at rate (1 - k) = 0.5; negative arm stays clamped.
    assert np.isclose(s_pos[-1], 40 * 0.5)
    assert float(s_neg.max()) == 0.0


def test_cusum_alert_fires_on_shifted_test_run() -> None:
    """Detect the injected shift after its onset."""
    n_steps = 200
    mu_t = np.linspace(0.0, 4.0, n_steps)
    runs = _make_calibration_runs(20, n_steps, mu_t=mu_t, sigma=0.25, seed=3)
    base = EnsembleBaseline.from_runs(runs, layer="L0")
    rng = np.random.default_rng(4)
    # Test run: clean for first 80 steps, then shifted by +1.5 sigma.

    test_trace = mu_t + rng.normal(0.0, 0.25, size=n_steps)
    test_trace[80:] += 1.5 * 0.25
    alert = online_cusum_alert(
        _records_from_trace(test_trace),
        baseline=base,
        k=0.5,
        h=6.0,
    )
    assert alert.alert_step is not None
    assert alert.alert_step > 80


def test_cusum_alert_silent_under_null() -> None:
    """Avoid a false alarm on the held-out null trajectory."""
    n_steps = 200
    mu_t = np.linspace(0.0, 4.0, n_steps)
    runs = _make_calibration_runs(20, n_steps, mu_t=mu_t, sigma=0.25, seed=5)
    base = EnsembleBaseline.from_runs(runs[:-1], layer="L0")

    alert = online_cusum_alert(
        runs[-1],
        baseline=base,
        k=0.5,
        h=12.0,  # conservative threshold; expected ARL_0 well above n_steps
    )
    assert alert.alert_step is None


# ---------------------------------------------------------------------------
#  Threshold calibration
# ---------------------------------------------------------------------------
def test_calibrate_threshold_matches_target_arl0_within_tolerance() -> None:
    """Check calibrated run length using independent Monte Carlo samples."""
    n_steps = 300
    mu_t = np.linspace(0.0, 5.0, n_steps)
    runs = _make_calibration_runs(15, n_steps, mu_t=mu_t, sigma=0.4, seed=6)
    target = 500.0

    h = calibrate_threshold(
        runs,
        layer="L0",
        k=0.5,
        target_arl0=target,
        n_replicates=400,
        block_length=4000,
        rng=np.random.default_rng(7),
    )
    # Verify on an independent Monte-Carlo pass using the same null sample.

    pool_chunks = []

    for i in range(len(runs)):
        rest = [r for j, r in enumerate(runs) if j != i]
        loo = EnsembleBaseline.from_runs(rest, layer="L0")
        _, _, z = loo.standardize(runs[i])
        pool_chunks.append(z)

    pool = np.concatenate(pool_chunks)

    rng = np.random.default_rng(8)
    rl = []

    for _ in range(2000):
        block = rng.choice(pool, size=4000, replace=True)
        s_pos, s_neg = cusum_statistic(block, k=0.5)
        s_max = np.maximum(s_pos, s_neg)

        cross = np.flatnonzero(s_max > h)
        rl.append(int(cross[0] + 1) if cross.size > 0 else 4000)

    empirical = float(np.mean(rl))
    # Allow a generous 25 % tolerance: ARL is a long-tailed functional.
    assert 0.75 * target < empirical < 1.5 * target, (h, empirical, target)
