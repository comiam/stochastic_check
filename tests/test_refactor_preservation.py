"""Verify numerical values and RNG consumption against the pre-refactor code."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hessian.trace import (
    BeckerLeCunDiagonalEstimator,
    HutchinsonTraceEstimator,
    HutchPlusPlusTraceEstimator,
    collect_layers,
)
from stochastic_check import decision
from stochastic_check.training import Trainer

REFERENCE = Path(__file__).parent / "fixtures" / "organization_reference.json"


def _reference():
    """Load outputs captured from the original committed implementation."""
    return json.loads(REFERENCE.read_text())


def _rng_digest(generator):
    """Fingerprint the remaining probe stream without consuming it."""
    return hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()


def _estimate(estimator_type, options):
    """Evaluate the same small smooth model with one isolated probe stream."""
    torch.manual_seed(71)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Linear(4, 2)).double()
    inputs = torch.randn(5, 3, dtype=torch.float64)
    targets = torch.tensor([0, 1, 1, 0, 1])

    generator = torch.Generator().manual_seed(811)
    estimator = estimator_type(**options)
    estimator.set_generator(generator)
    values = estimator.estimate(
        model, collect_layers(model), inputs, targets, nn.CrossEntropyLoss()
    )
    return {"values": values, "rng": _rng_digest(generator)}


def _decision_records():
    """Build deterministic drift plus noise with stable run/step ordering."""
    rng = np.random.default_rng(91)
    return [
        [
            {
                "step": (step + 1) * 100,
                "epoch": step // 4 + 1,
                "trace": {"head": float(np.sin(step / 3) + rng.normal())},
            }
            for step in range(12)
        ]
        for _ in range(6)
    ]


def _decision_summary(module):
    """Measure baseline statistics and the seeded calibration result."""
    records = _decision_records()
    baseline = module.EnsembleBaseline.from_runs(records, "head")
    threshold = module.calibrate_threshold(
        records,
        "head",
        target_arl0=30,
        n_replicates=32,
        block_length=256,
        rng=np.random.default_rng(14),
    )
    return {
        "mu": baseline.mu.tolist(),
        "sigma": baseline.sigma.tolist(),
        "h": threshold,
    }


def _training_summary(trainer_type, directory):
    """Train a tiny CPU model while recording every step and isolated RNGs."""
    torch.manual_seed(113)
    inputs = torch.randn(8, 3)
    targets = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0])
    dataset = TensorDataset(inputs, targets)

    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, generator=generator)
    val_loader = DataLoader(dataset, batch_size=4)
    model = nn.Sequential(nn.Linear(3, 4), nn.Tanh(), nn.Dropout(0.1), nn.Linear(4, 2))

    trainer = trainer_type(
        model,
        train_loader,
        val_loader,
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
        epochs=2,
        metric_interval_steps=1,
        hessian_batch_size=4,
        n_probes=3,
        probe_seed=17,
        device=torch.device("cpu"),
    )
    stream = directory / "steps.jsonl"
    summary = trainer.train([], stream, directory / "snapshots")

    records = [json.loads(line) for line in stream.read_text().splitlines()]
    for record in records:
        record.pop("wallclock_s")
        record.pop("val_wallclock_s")

    return {
        "summary": summary,
        "records": records,
        "weights": {name: value.tolist() for name, value in model.state_dict().items()},
        "probe_rng": _rng_digest(trainer._probe_rng),
        "loader_rng": _rng_digest(generator),
    }


@pytest.mark.parametrize(
    "name,estimator,options",
    [
        ("hutchinson", HutchinsonTraceEstimator, {"n_probes": 7}),
        ("bl_diag", BeckerLeCunDiagonalEstimator, {"n_probes": 7}),
        ("hutchpp", HutchPlusPlusTraceEstimator, {"n_total": 9}),
    ],
)
def test_estimator_values_and_rng_are_preserved(name, estimator, options):
    """Reorganization must not change trace values or the probe draw count."""
    actual = _estimate(estimator, options)
    expected = _reference()["estimators"][name]
    assert actual["values"] == pytest.approx(expected["values"], rel=1e-12, abs=1e-12)
    assert actual["rng"] == expected["rng"]


def test_calibration_values_are_preserved():
    """Alignment, leave-one-out pooling, and bisection retain their outputs."""
    assert _decision_summary(decision) == _reference()["decision"]


def test_training_values_and_rng_are_preserved(tmp_path):
    """Step extraction preserves weights, metrics, and both RNG streams."""
    assert _training_summary(Trainer, tmp_path) == _reference()["training"]
