"""Regression tests for :class:`stochastic_check.training.Trainer`.

The fine-grained per-step trace stream powers experiment B and the
paper's Cohen's-d decision rule.  Both rely on the HVP batch being
sampled from the *training* distribution -- using the (clean,
unshuffled) val loader silently decouples the measurement from the
noise regime and makes Cohen's d an artefact of softmax curvature on
held-out clean labels rather than of the curvature seen during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hessian import LayerSpec, TraceEstimator
from hessian.trace import LossFn
from stochastic_check.training import Trainer

# ---------------------------------------------------------------------------
#  Test doubles
# ---------------------------------------------------------------------------


class _RecordingEstimator(TraceEstimator):
    """Captures the ``(x, y)`` it was last invoked with.

    Returns zeros for every layer so the surrounding plumbing in
    :meth:`Trainer.measure_trace` (zero-grad, train/eval toggle) is
    exercised end-to-end.
    """

    name = "recording"

    def __init__(self) -> None:
        """Initialize captured inputs without affecting estimator randomness."""
        self.last_x: torch.Tensor | None = None
        self.last_y: torch.Tensor | None = None

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: torch.Tensor,
        y: torch.Tensor,
        loss_fn: LossFn,  # noqa: ARG002 - probe sink does not need the loss
    ) -> dict[str, float]:
        """Capture the supplied batch and return zero traces for the test."""
        self.last_x = x.detach().cpu().clone()
        self.last_y = y.detach().cpu().clone()
        return {layer.name: 0.0 for layer in layers}


def _sentinel_loader(
    value: float,
    label: int,
    *,
    n: int = 32,
    batch_size: int = 16,
) -> DataLoader:
    """Build a constant-input loader whose source is visible in the captured batch."""
    x = torch.full((n, 3, 4, 4), value, dtype=torch.float32)
    y = torch.full((n,), label, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _build_trainer(train_loader: DataLoader, val_loader: DataLoader) -> Trainer:
    """Construct a minimal CPU trainer for measurement behavior checks."""
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 2))
    return Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=0.1,
        momentum=0.0,
        weight_decay=0.0,
        epochs=1,
        scheduler="none",
        metric_interval_steps=0,
        hessian_batch_size=16,
        n_probes=1,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
#  measure_trace must consume the training distribution
# ---------------------------------------------------------------------------


def test_measure_trace_samples_from_train_loader() -> None:
    """Keep HVP inputs on the noisy, augmented training distribution.

    Using held-out validation labels would disconnect the measured
    trajectory from the training noise regime.
    """
    train_loader = _sentinel_loader(value=1.0, label=0)
    val_loader = _sentinel_loader(value=0.0, label=1)

    trainer = _build_trainer(train_loader, val_loader)
    recorder = _RecordingEstimator()
    trainer.estimator = recorder

    trainer.measure_trace()

    assert recorder.last_x is not None
    assert recorder.last_y is not None
    assert torch.all(recorder.last_x == 1.0), (
        "HVP inputs must come from train_loader (value=1), got val (value=0)."
    )
    assert torch.all(recorder.last_y == 0), (
        "HVP labels must come from train_loader (label=0), got val (label=1)."
    )


def test_measure_trace_restores_train_mode() -> None:
    """``measure_trace`` must leave the model in the mode it found it.

    Trace measurements run under ``eval()`` so BN / Dropout stay frozen
    during the HVP, but the outer training loop relies on the model
    being back in ``train()`` mode on return.
    """
    train_loader = _sentinel_loader(value=1.0, label=0)
    val_loader = _sentinel_loader(value=0.0, label=1)

    trainer = _build_trainer(train_loader, val_loader)
    trainer.estimator = _RecordingEstimator()

    trainer.model.train()
    trainer.measure_trace()
    assert trainer.model.training is True

    trainer.model.eval()
    trainer.measure_trace()
    assert trainer.model.training is False


def test_measure_trace_preserves_train_loader_shuffling() -> None:
    """Preserve future training-batch order across a trace measurement.

    Advancing the training sampler during HVP measurement would make later
    epochs depend on the measurement interval.
    """
    n, batch_size = 64, 8

    def _shuffled_loader(seed: int) -> DataLoader:
        """Create observable sample identities with an independent shuffle RNG."""
        # Distinct (x, y) per sample so the batch order is observable.
        x = torch.arange(n, dtype=torch.float32).view(n, 1, 1, 1).expand(n, 3, 4, 4)
        y = torch.arange(n) % 2
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(
            TensorDataset(x.clone(), y.clone()),
            batch_size=batch_size,
            shuffle=True,
            generator=gen,
        )

    # Loader A: drained twice without any HVP measurement in between.
    loader_a = _shuffled_loader(seed=123)
    reference = [batch[0].clone() for batch in loader_a]

    # Loader B: identically seeded; interleave a measure_trace() call
    # between the two iterations.
    loader_b = _shuffled_loader(seed=123)
    val_loader = _sentinel_loader(value=0.0, label=1, n=16, batch_size=8)
    trainer = _build_trainer(loader_b, val_loader)
    trainer.estimator = _RecordingEstimator()

    observed_first_pass = [batch[0].clone() for batch in loader_b]
    trainer.measure_trace()  # must not perturb loader_b's RNG state
    observed_second_pass = [batch[0].clone() for batch in loader_b]

    # The first pass should match the reference's first epoch and the
    # second pass should match the reference's second epoch.  We can
    # only check the first pass directly against the reference (the
    # second epoch's batches differ because the generator advanced
    # during the first iter).  The strict assertion is: with vs without
    # an intervening measure_trace, the second-pass sequences agree.
    loader_c = _shuffled_loader(seed=123)
    _ = [batch[0].clone() for batch in loader_c]  # drain once
    second_pass_no_hvp = [batch[0].clone() for batch in loader_c]

    assert all(
        torch.equal(a, b) for a, b in zip(observed_first_pass, reference, strict=True)
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(observed_second_pass, second_pass_no_hvp, strict=True)
    ), (
        "measure_trace consumed train_loader's shuffling generator: "
        "subsequent epoch ordering depends on metric_interval_steps."
    )


def test_trainer_owns_isolated_hvp_loader() -> None:
    """Use the same training dataset through a separate Hessian-batch loader."""
    train_loader = _sentinel_loader(value=1.0, label=0)
    val_loader = _sentinel_loader(value=0.0, label=1)
    trainer = _build_trainer(train_loader, val_loader)

    assert trainer._hvp_loader is not train_loader
    assert trainer._hvp_loader.dataset is train_loader.dataset
