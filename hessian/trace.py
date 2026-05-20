"""Layer-wise Hessian-trace estimators.

The estimators target the **parameter-space block-diagonal trace**

    T_ell  :=  tr( H_{ell,ell} ),    H_{ell,ell} = d^2 L / d theta_ell d theta_ell,

where theta_ell are the parameters of layer ell.  All HVP-based estimators
share the single-pass convention: one Hessian-vector product against a
global probe is computed, then sliced per layer.  Under the joint-unbiased
property (cross-layer cross terms vanish in expectation because z_ell and
z_m are independent for ell != m), this gives an unbiased estimate of T_ell
for every layer at the cost of a single HVP per probe instead of L HVPs.

Estimators
----------

  - :class:`HutchinsonTraceEstimator`     - single-pass Rademacher Hutchinson.
  - :class:`HutchPlusPlusTraceEstimator`  - Meyer 2021 variance-reduced trace.
  - :class:`BeckerLeCunDiagonalEstimator` - stochastic diagonal (Becker & LeCun 1989).
  - :class:`KFACTraceEstimator`           - K-FAC trace via the ASDL library.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Layer specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerSpec:
    """Description of a single layer for trace estimation."""

    name: str
    params: tuple[nn.Parameter, ...]

    @property
    def numel(self) -> int:
        return sum(p.numel() for p in self.params)

    @property
    def n_tensors(self) -> int:
        return len(self.params)


def collect_layers(model: nn.Module) -> list[LayerSpec]:
    """Default layer partition: one :class:`LayerSpec` per ``nn.Module`` leaf
    that owns at least one trainable parameter.

    A leaf is detected via ``module.children() == []``; this groups
    ``weight`` and ``bias`` of each conv/linear/BN into a single layer,
    matching the convention used in the paper.
    """
    layers: list[LayerSpec] = []
    for name, module in model.named_modules():
        if list(module.children()):
            continue
        params = tuple(p for p in module.parameters(recurse=False) if p.requires_grad)
        if not params:
            continue
        layers.append(LayerSpec(name=name or "<root>", params=params))
    return layers


# ---------------------------------------------------------------------------
#  Internal HVP primitives
# ---------------------------------------------------------------------------


LossFn = Callable[[Tensor, Tensor], Tensor]


def _stable_grad(loss: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    """First-order gradient with ``create_graph=True`` for HVP composition.

    Unused parameters get zero-tensors of matching shape so that downstream
    slicing is layout-agnostic.
    """
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    return [
        g if g is not None else torch.zeros_like(p)
        for p, g in zip(params, grads, strict=True)
    ]


def _hvp(
    grad_outputs: list[Tensor],
    params: list[nn.Parameter],
    vec: list[Tensor],
) -> list[Tensor]:
    """One Hessian-vector product  Hv = d/dtheta (grad . vec)."""
    dot = sum((g * v).sum() for g, v in zip(grad_outputs, vec, strict=True))
    hv = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
    return [
        h if h is not None else torch.zeros_like(p)
        for p, h in zip(params, hv, strict=True)
    ]


def _sample_rademacher_like(params: Iterable[nn.Parameter]) -> list[Tensor]:
    return [
        torch.randint(0, 2, p.shape, device=p.device, dtype=p.dtype) * 2 - 1
        for p in params
    ]


def _sample_gaussian_like(params: Iterable[nn.Parameter]) -> list[Tensor]:
    return [torch.randn_like(p) for p in params]


def _flatten(tensors: list[Tensor]) -> Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])


def _unflatten(flat: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    out: list[Tensor] = []
    cursor = 0
    for p in params:
        n = p.numel()
        out.append(flat[cursor : cursor + n].reshape(p.shape))
        cursor += n
    return out


def _layer_quadratic(
    layers: list[LayerSpec],
    a: list[Tensor],
    b: list[Tensor],
) -> dict[str, float]:
    """``{layer.name -> sum_{i in layer} a_i * b_i}`` given lists indexed by
    the global param order.
    """
    out: dict[str, float] = {}
    cursor = 0
    for layer in layers:
        n = layer.n_tensors
        a_l = a[cursor : cursor + n]
        b_l = b[cursor : cursor + n]
        cursor += n
        out[layer.name] = float(
            sum((ai * bi).sum().item() for ai, bi in zip(a_l, b_l, strict=True))
        )
    return out


def _flat_params(layers: list[LayerSpec]) -> list[nn.Parameter]:
    return [p for layer in layers for p in layer.params]


# ---------------------------------------------------------------------------
#  Base class
# ---------------------------------------------------------------------------


class TraceEstimator(ABC):
    """Common protocol for layer-wise trace estimators."""

    name: str = "<abstract>"

    @abstractmethod
    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        """Returns ``{layer_name -> T_hat_ell}``."""


# ---------------------------------------------------------------------------
#  Single-pass Hutchinson
# ---------------------------------------------------------------------------


class HutchinsonTraceEstimator(TraceEstimator):
    """T_ell  ~=  (1/K) sum_k  z_{k,ell}^T (H z_k)_ell,   z_k ~ Rademacher.

    A single HVP per probe: ``z_k`` is sampled jointly over every parameter
    of every layer; the resulting ``H z_k`` is sliced per layer and the
    per-layer quadratic form contributes one unbiased sample of T_ell.
    """

    name = "hutchinson"

    def __init__(self, n_probes: int = 10) -> None:
        if n_probes < 1:
            raise ValueError("n_probes must be >= 1")
        self._n_probes = n_probes

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        params = _flat_params(layers)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        grads = _stable_grad(loss, params)

        acc: dict[str, float] = {layer.name: 0.0 for layer in layers}
        for _ in range(self._n_probes):
            z = _sample_rademacher_like(params)
            hz = _hvp(grads, params, z)
            for k, v in _layer_quadratic(layers, z, hz).items():
                acc[k] += v

        return {k: v / self._n_probes for k, v in acc.items()}


# ---------------------------------------------------------------------------
#  Hutch++ (Meyer 2021)
# ---------------------------------------------------------------------------


class HutchPlusPlusTraceEstimator(TraceEstimator):
    """Variance-reduced trace estimator (Meyer et al. 2021).

    Pipeline in the flattened parameter space ``P = sum_ell P_ell``:

      1. Sample Gaussian  ``G in R^{P x n_orth}``.
      2. Compute  ``Y = H G``     (n_orth HVPs).
      3. ``Q, _ = qr(Y)``.
      4. Low-rank trace:  ``T_lowrank = tr(Q^T H Q)``   (n_orth HVPs).
      5. Sample Rademacher  ``S in R^{P x n_hutch}``.
      6. Project  ``S' = (I - Q Q^T) S``.
      7. Residual trace:   ``T_resid = (1/n_hutch) tr(S'^T H S')``.

      tr(H) ~= T_lowrank + T_resid.

    Layer-wise quantities follow from slicing every flat vector (Q columns,
    H Q columns, S' columns, H S' columns) by layer and accumulating
    per-layer scalars; the construction is block-diagonal in layer slices,
    so the per-layer estimator inherits unbiasedness.

    Cost: ``2 * n_orth + n_hutch`` HVPs per call.  Recommended split
    ``n_orth = n_total / 3`` gives provably ``O(1/n_total^2)`` variance
    on rank-deficient H (vs ``O(1/n_total)`` for vanilla Hutchinson).
    """

    name = "hutch_plus_plus"

    def __init__(self, n_total: int = 10, n_orth_frac: float = 1.0 / 3.0) -> None:
        if n_total < 3:
            raise ValueError("Hutch++ requires n_total >= 3")
        n_orth = max(1, int(round(n_total * n_orth_frac)))
        n_hutch = max(1, n_total - n_orth)
        self._n_orth = n_orth
        self._n_hutch = n_hutch

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        params = _flat_params(layers)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        grads = _stable_grad(loss, params)

        # --- 1-2:  Y = H G ---------------------------------------------------
        Y_cols: list[Tensor] = []
        for _ in range(self._n_orth):
            g_struct = _sample_gaussian_like(params)
            hg_struct = _hvp(grads, params, g_struct)
            Y_cols.append(_flatten(hg_struct))
        Y = torch.stack(Y_cols, dim=1)  # P x n_orth

        # --- 3:  Q = qr(Y) ---------------------------------------------------
        Q, _ = torch.linalg.qr(Y, mode="reduced")

        # --- 4:  T_lowrank = tr(Q^T H Q),  layer-wise ------------------------
        lowrank: dict[str, float] = {layer.name: 0.0 for layer in layers}
        for j in range(Q.shape[1]):
            q_struct = _unflatten(Q[:, j], params)
            hq_struct = _hvp(grads, params, q_struct)
            for k, v in _layer_quadratic(layers, q_struct, hq_struct).items():
                lowrank[k] += v

        # --- 5-7:  Hutchinson on residual subspace --------------------------
        resid: dict[str, float] = {layer.name: 0.0 for layer in layers}
        for _ in range(self._n_hutch):
            s_struct = _sample_rademacher_like(params)
            s_flat = _flatten(s_struct)
            s_proj_flat = s_flat - Q @ (Q.T @ s_flat)
            s_proj_struct = _unflatten(s_proj_flat, params)
            hsp_struct = _hvp(grads, params, s_proj_struct)
            for k, v in _layer_quadratic(layers, s_proj_struct, hsp_struct).items():
                resid[k] += v

        return {
            layer.name: lowrank[layer.name] + resid[layer.name] / self._n_hutch
            for layer in layers
        }


# ---------------------------------------------------------------------------
#  Becker-LeCun stochastic diagonal
# ---------------------------------------------------------------------------


class BeckerLeCunDiagonalEstimator(TraceEstimator):
    """Stochastic diagonal a la Becker & LeCun (1989), summed to T_ell.

      h_ii  ~=  (1/K) sum_k  z_{k,i} (H z_k)_i,    z_k ~ Rademacher,
      T_ell  =  sum_{i in layer ell}  h_ii.

    Algebraically  ``sum_i z_i (H z)_i = z^T H z``,  so the estimator is
    mathematically identical to :class:`HutchinsonTraceEstimator` at the
    same K.  We expose a dedicated class because the conventional use is
    a much larger ``K_diag`` (the diagonal converges slowly) and to keep
    the comparison table explicit.
    """

    name = "bl_diag"

    def __init__(self, n_probes: int = 200) -> None:
        self._impl = HutchinsonTraceEstimator(n_probes=n_probes)

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        return self._impl.estimate(model, layers, x, y, loss_fn)


# ---------------------------------------------------------------------------
#  K-FAC trace via ASDL
# ---------------------------------------------------------------------------


class KFACTraceEstimator(TraceEstimator):
    """K-FAC trace approximation of tr(F) (empirical Fisher) via ASDL.

    .. warning::
       K-FAC approximates the Gauss-Newton / Fisher matrix F, not the
       Hessian H.  At a minimum of cross-entropy they coincide, but during
       training the gap ``H - F`` (the H_tensor term) is not captured.
       Numbers from this estimator are comparable to T_ell only up to the
       GN-gap and should be interpreted with that caveat in the paper.

    Implementation: for each Kronecker-factored layer block
    ``F_ell ~= A_ell (x) G_ell`` the trace decomposes as
    ``tr(F_ell) = tr(A_ell) * tr(G_ell)``.  Layers unsupported by ASDL
    (BatchNorm, embeddings) get ``NaN``; runners filter them out.
    """

    name = "kfac"

    def __init__(self) -> None:
        try:
            from asdfghjkl import FISHER_EMP, KFAC  # type: ignore[import]
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "K-FAC estimator requires the `asdfghjkl` package; "
                "install with `uv add asdfghjkl`."
            ) from exc
        self._KFAC = KFAC
        self._FISHER_EMP = FISHER_EMP

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,  # noqa: ARG002 - ASDL uses its own loss internally
    ) -> dict[str, float]:
        kfac = self._KFAC(model, fisher_type=self._FISHER_EMP)
        kfac.update_curvature(inputs=x, targets=y)
        kfac.average_curvature(n=1)

        modules: dict[str, Any] = dict(model.named_modules())
        traces: dict[str, float] = {}
        for layer in layers:
            mod = modules.get(layer.name)
            block = getattr(mod, "fisher", None) if mod is not None else None
            kron = getattr(block, "kron", None)
            if kron is None:
                traces[layer.name] = math.nan
                continue
            tr_A = torch.trace(kron.A).item()
            tr_G = torch.trace(kron.G).item()
            traces[layer.name] = float(tr_A * tr_G)
        return traces
