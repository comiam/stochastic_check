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
    """A loader whose every sample is the constant ``value`` with class
    ``label`` -- lets a test distinguish the train and val sources by
    inspection alone."""
    x = torch.full((n, 3, 4, 4), value, dtype=torch.float32)
    y = torch.full((n,), label, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _build_trainer(train_loader: DataLoader, val_loader: DataLoader) -> Trainer:
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
    """Regression: the HVP batch must come from ``train_loader`` (noisy
    labels, augmented inputs), not from ``val_loader``.

    A loader mix-up here silently invalidates every experiment-B claim:
    the per-step trace trajectory becomes a function of held-out clean
    labels instead of the training noise regime.
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
