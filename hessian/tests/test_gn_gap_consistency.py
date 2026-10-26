"""Consistency test for exact and stochastic GN-Gap.

Checks:
  1. Exact vs stochastic GN-Gap estimates converge (for a smooth activation).
  2. ReLU: GN-Gap ~= 0 (exact and stochastic).
  3. _assert_mean_reduction catches incorrect reduction.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from hessian import ExactGNDecomposition, StochasticGNGapEstimator


class _Block(nn.Module):
    """Combine a linear map, optional normalization, and activation."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act: nn.Module,
        *,
        use_layernorm: bool,
    ) -> None:
        """Initialize the test block in its original parameter order."""
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to a batch of intermediate features."""
        return self.act(self.norm(self.linear(x)))


class PlainMLP(nn.Module):
    """Inline plain MLP fixture compatible with the ``hessian`` model API.

    Exposes :meth:`get_layer_names` (``N`` measurement points) and
    :meth:`get_segments` (``N + 1`` callables) as expected by
    :class:`ExactGNDecomposition` and :class:`StochasticGNGapEstimator`.
    """

    def __init__(
        self,
        input_dim: int = 32,
        width: int = 8,
        depth: int = 3,
        num_classes: int = 10,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        """Create the smooth or piecewise-linear segmented test network."""
        super().__init__()
        act_cls: type[nn.Module] = {"gelu": nn.GELU, "relu": nn.ReLU}[activation]

        self.projection = _Block(
            input_dim, width, act_cls(), use_layernorm=use_layernorm
        )
        self.blocks = nn.ModuleList(
            _Block(width, width, act_cls(), use_layernorm=use_layernorm)
            for _ in range(depth)
        )
        self.head = nn.Linear(width, num_classes)
        self._depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the projection, hidden blocks, and classification head."""
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)

    def get_layer_names(self) -> list[str]:
        """Name the hidden-block measurement points."""
        return [f"block_{i}" for i in range(self._depth)]

    def get_segments(self) -> list[Callable[[torch.Tensor], torch.Tensor]]:
        """Expose forward segments ending at each measurement point and logits."""
        segs: list[Callable[[torch.Tensor], torch.Tensor]] = []
        proj = self.projection
        first_block = self.blocks[0]

        def seg0(x: torch.Tensor) -> torch.Tensor:
            """Combine projection and the first hidden block into one segment."""
            return first_block(proj(x))

        segs.append(seg0)
        for i in range(1, self._depth):
            segs.append(self.blocks[i])
        segs.append(self.head)
        return segs


def _make_data(
    batch_size: int = 4,
    input_dim: int = 32,
    num_classes: int = 10,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random input + labels."""
    x = torch.randn(batch_size, input_dim, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, y


# ------------------------------------------------------------------
# 1. Exact vs stochastic (GELU - smooth, gap > 0)
# ------------------------------------------------------------------


def test_exact_vs_stochastic_gelu() -> None:
    """Compare exact and sampled curvature decomposition on a smooth model."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    model = PlainMLP(input_dim=32, width=8, depth=3, activation="gelu").to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    exact = ExactGNDecomposition(model, loss_fn)
    gap_exact = exact.compute_gn_gap(x, y, v, w)

    stoch = StochasticGNGapEstimator(model, loss_fn, n_probes=500)
    gap_stoch = stoch.estimate_gn_gap(x, y, v, w)

    tol = 0.15
    diff = abs(gap_exact.gap - gap_stoch.gap)
    assert diff < tol, (
        f"Exact vs stochastic GN-Gap diverge: "
        f"exact={gap_exact.gap:.4f}, stochastic={gap_stoch.gap:.4f}, diff={diff:.4f}"
    )

    # Frobenius norm should also be close
    frob_diff = abs(gap_exact.frob_gn - gap_stoch.frob_gn) / (gap_exact.frob_gn + 1e-12)
    assert frob_diff < 0.2, (
        f"||H^GN|| diverges: exact={gap_exact.frob_gn:.4f}, "
        f"stochastic={gap_stoch.frob_gn:.4f}"
    )
    print(f"  GELU gap exact={gap_exact.gap:.4f}, stochastic={gap_stoch.gap:.4f} - OK")


# ------------------------------------------------------------------
# 2. ReLU: gap ~= 0
# ------------------------------------------------------------------


def test_relu_gap_zero() -> None:
    """Keep the tensor-curvature gap negligible for a piecewise-linear model."""
    device = torch.device("cpu")
    torch.manual_seed(0)

    # use_layernorm=False: network is piecewise-linear => H^T = 0 a.e.
    model = PlainMLP(
        input_dim=32, width=8, depth=3, activation="relu", use_layernorm=False
    ).to(device)
    model.eval()

    loss_fn = nn.CrossEntropyLoss()
    x, y = _make_data(device=device)

    names = model.get_layer_names()
    v, w = names[0], names[1]

    exact = ExactGNDecomposition(model, loss_fn)
    gap_exact = exact.compute_gn_gap(x, y, v, w)
    assert gap_exact.gap < 1e-6, f"ReLU exact gap should be ~0, got {gap_exact.gap:.6f}"

    stoch = StochasticGNGapEstimator(model, loss_fn, n_probes=200)
    gap_stoch = stoch.estimate_gn_gap(x, y, v, w)
    assert gap_stoch.gap < 1e-3, (
        f"ReLU stochastic gap should be ~0, got {gap_stoch.gap:.6f}"
    )
    print(f"  ReLU gap exact={gap_exact.gap:.2e}, stochastic={gap_stoch.gap:.2e} - OK")


# ------------------------------------------------------------------
# 3. _assert_mean_reduction
# ------------------------------------------------------------------


def test_reduction_assertion() -> None:
    """Require the mean-loss normalization assumed by both decompositions."""
    model = PlainMLP(input_dim=32, width=8, depth=3)

    bad_loss = nn.CrossEntropyLoss(reduction="sum")

    try:
        ExactGNDecomposition(model, bad_loss)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    try:
        StochasticGNGapEstimator(model, bad_loss)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # reduction='mean' - OK
    good_loss = nn.CrossEntropyLoss()
    ExactGNDecomposition(model, good_loss)
    StochasticGNGapEstimator(model, good_loss)

    print("  Reduction assertion - OK")


# ------------------------------------------------------------------

if __name__ == "__main__":
    print("test_exact_vs_stochastic_gelu...")
    test_exact_vs_stochastic_gelu()

    print("test_relu_gap_zero...")
    test_relu_gap_zero()

    print("test_reduction_assertion...")
    test_reduction_assertion()

    print("\nAll tests passed.")
