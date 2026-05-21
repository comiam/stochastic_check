"""Tests for the online 3-sigma decision rule (section 7.2 of the paper)."""

from __future__ import annotations

import math

import numpy as np

from stochastic_check.decision import online_3sigma_alert


def _make_records(epochs_per_record: int, traces: list[float]) -> list[dict]:
    return [
        {
            "step": i + 1,
            "epoch": (i // epochs_per_record) + 1,
            "trace": {"L0": v},
        }
        for i, v in enumerate(traces)
    ]


def test_no_alert_on_constant_trace() -> None:
    records = _make_records(epochs_per_record=2, traces=[1.0] * 20)
    res = online_3sigma_alert(records, layer="L0", baseline_epochs=5, threshold=3.0)
    assert res.alert_step is None
    assert res.baseline_std == 0.0


def test_alert_fires_on_clear_jump() -> None:
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(loc=1.0, scale=0.05, size=10))
    spike = [1.0 + 10.0] * 10
    records = _make_records(epochs_per_record=2, traces=baseline + spike)
    res = online_3sigma_alert(records, layer="L0", baseline_epochs=5, threshold=3.0)
    assert res.alert_step is not None
    assert res.alert_epoch is not None and res.alert_epoch > 5
    assert abs(res.z_score) >= 3.0


def test_baseline_only_returns_no_alert() -> None:
    rng = np.random.default_rng(1)
    records = _make_records(epochs_per_record=2, traces=list(rng.normal(0.0, 1.0, 8)))
    res = online_3sigma_alert(records, layer="L0", baseline_epochs=5, threshold=3.0)
    # All in-baseline samples; no post-baseline data => no alert.
    assert res.alert_step is None
    assert not math.isnan(res.baseline_mean)
