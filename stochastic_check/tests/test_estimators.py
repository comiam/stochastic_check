"""Tests for the layer-wise trace estimators on synthetic, controllable losses.

We bypass the full ``HutchinsonTraceEstimator`` API on real models and
test the algebra directly: build a small synthetic loss whose Hessian
is known in closed form, then check that

  * Hutchinson is unbiased and its variance scales as 1/K.
  * Hutch++ matches the exact trace at lower probe budgets on low-rank H.
  * Becker-LeCun diagonal collapses to the same value as Hutchinson
    (they are algebraically identical).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hessian.trace import (
    BeckerLeCunDiagonalEstimator,
    HutchinsonTraceEstimator,
    HutchPlusPlusTraceEstimator,
    collect_layers,
)

# ---------------------------------------------------------------------------
#  Synthetic model:  L(theta) = 0.5 || A theta - b ||^2
#  Hessian = A^T A  (independent of theta), exact trace = ||A||_F^2.
# ---------------------------------------------------------------------------


class LinearLoss(nn.Module):
    """Wraps theta as a parameter so autograd HVPs work."""

    def __init__(self, P: int) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(P))

    def forward(self, A: torch.Tensor) -> torch.Tensor:  # noqa: N803
        # A is (N, P); we treat A as "input" so that the trainer-style API
        # of estimate(model, layers, x, y, loss_fn) maps cleanly.
        return A @ self.theta


def _make_problem(P: int = 50, rank: int | None = None, seed: int = 0):
    torch.manual_seed(seed)
    if rank is None:
        rank = P
    U = torch.randn(rank, P)
    A = U  # (rank, P), H = A^T A has rank <= rank
    b = torch.randn(rank)
    model = LinearLoss(P)
    layers = collect_layers(model)
    assert len(layers) == 1

    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * ((pred - target) ** 2).sum()

    exact_trace = float((A * A).sum())  # ||A||_F^2 = tr(A^T A)
    return model, layers, A, b, loss_fn, exact_trace


# ---------------------------------------------------------------------------
#  Hutchinson: unbiased + variance shrinks with K
# ---------------------------------------------------------------------------


def test_hutchinson_unbiased_full_rank():
    torch.manual_seed(0)
    model, layers, A, b, loss_fn, exact = _make_problem(P=40, rank=40)
    estimator = HutchinsonTraceEstimator(n_probes=500)
    out = estimator.estimate(model, layers, A, b, loss_fn)
    name = layers[0].name
    assert out[name] == pytest.approx(exact, rel=0.10)


def test_hutchinson_variance_decreases_with_K():
    torch.manual_seed(1)
    model, layers, A, b, loss_fn, _ = _make_problem(P=30, rank=30)
    name = layers[0].name

    def std_at(K: int, n_trials: int = 20) -> float:
        vals = []
        for _ in range(n_trials):
            est = HutchinsonTraceEstimator(n_probes=K)
            vals.append(est.estimate(model, layers, A, b, loss_fn)[name])
        return float(torch.tensor(vals).std().item())

    s_small = std_at(K=5)
    s_large = std_at(K=80)
    # 1/sqrt(K) scaling => std should drop by ~ sqrt(16) = 4x; require >= 2x.
    assert s_large < s_small / 2.0


# ---------------------------------------------------------------------------
#  Hutch++ wins on low-rank H
# ---------------------------------------------------------------------------


def test_hutch_plus_plus_low_rank_variance_reduction():
    torch.manual_seed(2)
    model, layers, A, b, loss_fn, exact = _make_problem(P=60, rank=5)
    name = layers[0].name

    hpp = HutchPlusPlusTraceEstimator(n_total=9)  # 3 orth + 6 hutch
    hp = HutchinsonTraceEstimator(n_probes=9)

    hpp_vals, hp_vals = [], []
    for _ in range(15):
        hpp_vals.append(hpp.estimate(model, layers, A, b, loss_fn)[name])
        hp_vals.append(hp.estimate(model, layers, A, b, loss_fn)[name])

    hpp_t = torch.tensor(hpp_vals)
    hp_t = torch.tensor(hp_vals)

    # Both should be near the exact trace on average.
    assert hpp_t.mean().item() == pytest.approx(exact, rel=0.15)
    assert hp_t.mean().item() == pytest.approx(exact, rel=0.30)
    # Hutch++ should have lower variance on this low-rank H.
    assert hpp_t.std().item() < hp_t.std().item()


# ---------------------------------------------------------------------------
#  BL-diag == Hutchinson algebraically
# ---------------------------------------------------------------------------


def test_bl_diag_matches_hutchinson_per_probe():
    torch.manual_seed(3)
    model, layers, A, b, loss_fn, _ = _make_problem(P=20, rank=20)
    name = layers[0].name
    # Same n_probes => identical sample (we re-seed before each estimator
    # so they consume the same Rademacher draws).
    K = 30

    torch.manual_seed(7)
    hp_val = HutchinsonTraceEstimator(n_probes=K).estimate(
        model, layers, A, b, loss_fn
    )[name]
    torch.manual_seed(7)
    bl_val = BeckerLeCunDiagonalEstimator(n_probes=K).estimate(
        model, layers, A, b, loss_fn
    )[name]
    assert bl_val == pytest.approx(hp_val, rel=1e-6, abs=1e-6)
