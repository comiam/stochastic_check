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
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Module-level latch: warn at most once about silently zero-filled gradients.
# Such zero-fills are scientifically silent (they mask a missing autograd
# path) but happen legitimately for unused tensors (e.g. an unused bias);
# we surface the first occurrence and stay quiet thereafter.
_WARNED_UNUSED_GRAD: bool = False
_WARNED_UNUSED_HVP: bool = False


def _warn_unused_grad_once(missing: int, total: int) -> None:
    global _WARNED_UNUSED_GRAD
    if _WARNED_UNUSED_GRAD:
        return
    _WARNED_UNUSED_GRAD = True
    warnings.warn(
        f"_stable_grad: {missing}/{total} parameters had no gradient flow "
        "and were zero-filled; verify this is intentional (unused bias, "
        "frozen branch) and not a bug in the forward pass.",
        RuntimeWarning,
        stacklevel=3,
    )


def _warn_unused_hvp_once(missing: int, total: int) -> None:
    global _WARNED_UNUSED_HVP
    if _WARNED_UNUSED_HVP:
        return
    _WARNED_UNUSED_HVP = True
    warnings.warn(
        f"_hvp: {missing}/{total} parameters had no second-order graph and "
        "were zero-filled; this usually means a non-smooth op (e.g. ReLU "
        "on a degenerate batch) and biases the per-layer trace downward.",
        RuntimeWarning,
        stacklevel=3,
    )


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
    missing = sum(1 for g in grads if g is None)
    if missing:
        _warn_unused_grad_once(missing, len(params))
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
    missing = sum(1 for h in hv if h is None)
    if missing:
        _warn_unused_hvp_once(missing, len(params))
    return [
        h if h is not None else torch.zeros_like(p)
        for p, h in zip(params, hv, strict=True)
    ]


def _sample_rademacher_like(
    params: Iterable[nn.Parameter],
    generator: torch.Generator | None = None,
) -> list[Tensor]:
    """Per-parameter Rademacher draws.

    A ``torch.Generator`` may be supplied to obtain a stream independent
    of the global RNG; this is essential for reproducibility of probes
    inside a training loop that also seeds the global RNG.
    """
    out: list[Tensor] = []
    for p in params:
        if generator is not None and generator.device != p.device:
            # torch.randint requires the generator to live on the same
            # device as the output; sample on the generator device and
            # move, which keeps a single deterministic stream.
            raw = torch.randint(
                0, 2, p.shape, device=generator.device, generator=generator
            ).to(device=p.device, dtype=p.dtype)
        else:
            raw = torch.randint(
                0, 2, p.shape, device=p.device, dtype=p.dtype, generator=generator
            )
        out.append(raw * 2 - 1)
    return out


def _sample_gaussian_like(
    params: Iterable[nn.Parameter],
    generator: torch.Generator | None = None,
) -> list[Tensor]:
    out: list[Tensor] = []
    for p in params:
        if generator is not None and generator.device != p.device:
            raw = torch.randn(p.shape, device=generator.device, generator=generator).to(
                device=p.device, dtype=p.dtype
            )
        else:
            raw = torch.empty_like(p)
            raw.normal_(generator=generator)
        out.append(raw)
    return out


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
    """Common protocol for layer-wise trace estimators.

    Concrete estimators may carry a dedicated ``torch.Generator`` for
    probe sampling; this isolates Hessian-trace randomness from the
    global RNG (data loaders, dropout) and gives bit-reproducible
    estimates given a fixed estimator seed.  The default implementation
    is a no-op so K-FAC and other generator-agnostic estimators are
    transparent.
    """

    name: str = "<abstract>"

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Install a dedicated probe-RNG; default: no-op."""
        return

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
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> None:
        self._generator = generator

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
            z = _sample_rademacher_like(params, generator=self._generator)
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
        self._generator: torch.Generator | None = None

    @property
    def hvp_budget(self) -> int:
        """Number of HVPs spent per call: ``2*n_orth + n_hutch``."""
        return 2 * self._n_orth + self._n_hutch

    def set_generator(self, generator: torch.Generator | None) -> None:
        self._generator = generator

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
            g_struct = _sample_gaussian_like(params, generator=self._generator)
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
            s_struct = _sample_rademacher_like(params, generator=self._generator)
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

    def set_generator(self, generator: torch.Generator | None) -> None:
        self._impl.set_generator(generator)

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
       K-FAC approximates the empirical Fisher ``F``, not the Hessian
       ``H``.  At a minimum of the cross-entropy loss the two coincide,
       but during training the gap ``H - F`` (the ``H_tensor`` term of
       the Gauss-Newton decomposition) is not captured.  Numbers from
       this estimator are comparable to ``T_ell`` only up to that gap
       and are reported with that caveat in section 6 of the paper.

    Implementation: ASDL accumulates a Kronecker-factored Fisher block
    ``F_ell = A_ell (x) B_ell`` for every supported layer (``nn.Linear``,
    ``nn.Conv2d``).  The trace then decomposes as
    ``tr(F_ell) = tr(A_ell) * tr(B_ell)``.  Layers unsupported by ASDL
    (``BatchNorm``, ``Embedding``, ``LayerNorm``) carry no ``fisher``
    attribute after the curvature pass and are reported as ``NaN``;
    downstream aggregators (paper_data) filter them.
    """

    name = "kfac"

    def __init__(self) -> None:
        try:
            from asdl import (  # type: ignore[import]
                FISHER_EMP,
                LOSS_CROSS_ENTROPY,
                SHAPE_KRON,
                FisherConfig,
                get_fisher_maker,
            )
        except ImportError as exc:  # pragma: no cover - optional dep
            raise RuntimeError(
                "K-FAC estimator requires the `asdl` package "
                "(distributed on PyPI as `asdfghjkl`); install with "
                "`uv add asdfghjkl`."
            ) from exc
        self._FisherConfig = FisherConfig
        self._get_fisher_maker = get_fisher_maker
        self._FISHER_EMP = FISHER_EMP
        self._SHAPE_KRON = SHAPE_KRON
        self._LOSS_CROSS_ENTROPY = LOSS_CROSS_ENTROPY

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,  # noqa: ARG002 - ASDL drives its own loss closure
    ) -> dict[str, float]:
        import torch.nn.functional as F  # local: only needed for K-FAC

        cfg = self._FisherConfig(
            fisher_type=self._FISHER_EMP,
            fisher_shapes=[self._SHAPE_KRON],
            loss_type=self._LOSS_CROSS_ENTROPY,
            data_size=int(x.shape[0]),
            fisher_attr="fisher",
        )
        maker = self._get_fisher_maker(model, cfg)
        dummy_logits = maker.setup_model_call(model, x)
        maker.setup_loss_call(F.cross_entropy, dummy_logits, y, reduction="sum")
        maker.forward_and_backward()

        modules: dict[str, Any] = dict(model.named_modules())
        traces: dict[str, float] = {}
        unsupported: list[str] = []
        for layer in layers:
            mod = modules.get(layer.name)
            block = getattr(mod, "fisher", None) if mod is not None else None
            kron = getattr(block, "kron", None)
            if kron is None:
                traces[layer.name] = math.nan
                unsupported.append(layer.name)
                continue
            # ASDL stores the input-side factor in `.A` and the
            # output-grad-side factor in `.B`; the data-side scaling
            # constant is absorbed at curvature-accumulation time, so
            # the layer-block trace is simply tr(A) * tr(B).
            tr_A = float(torch.trace(kron.A).item())
            tr_B = float(torch.trace(kron.B).item())
            traces[layer.name] = tr_A * tr_B
        if unsupported:
            # K-FAC routinely skips BatchNorm and other unsupported
            # leaves.  Surfacing the names once per call keeps the
            # comparison table interpretable without flooding the log.
            logger.warning(
                "K-FAC: %d/%d layers without Kronecker factor (e.g. BN); first few: %s",
                len(unsupported),
                len(layers),
                unsupported[:5],
            )
        return traces
