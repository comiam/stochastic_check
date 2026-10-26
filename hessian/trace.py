"""Estimate parameter-space layer traces T_ell = tr(H_ell,ell).

A global probe needs one HVP, then layer slices supply their quadratic
forms. Independent cross-layer probes cancel cross terms in expectation,
giving an unbiased trace per layer without a separate HVP for each layer.

Hutchinson uses Rademacher probes; Hutch++ adds a low-rank projection;
Becker-LeCun sums the equivalent stochastic diagonal. K-FAC estimates
Fisher curvature rather than the complete Hessian.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)
_WARNED_UNUSED_GRAD = False
_WARNED_UNUSED_HVP = False
LossFn = Callable[[Tensor, Tensor], Tensor]


def _warn_unused_grad_once(missing: int, total: int) -> None:
    """Report zero-filled gradients once without hiding missing graph paths."""
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
    """Report missing second-order paths once, including degenerate ReLU paths."""
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


@dataclass(frozen=True)
class LayerSpec:
    """Name a leaf module and its ordered trainable parameter tensors."""

    name: str
    params: tuple[nn.Parameter, ...]

    @property
    def numel(self) -> int:
        """Return the number of scalar parameters in the layer."""
        return sum(parameter.numel() for parameter in self.params)

    @property
    def n_tensors(self) -> int:
        """Return the number of tensors in the global parameter slice."""
        return len(self.params)


def collect_layers(model: nn.Module) -> list[LayerSpec]:
    """Group trainable weights and biases by leaf module in traversal order."""
    layers: list[LayerSpec] = []
    for name, module in model.named_modules():
        if list(module.children()):
            continue

        params = tuple(p for p in module.parameters(recurse=False) if p.requires_grad)
        if not params:
            continue

        layers.append(LayerSpec(name=name or "<root>", params=params))

    return layers


def _stable_grad(loss: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    """Retain the gradient graph for HVPs and zero-fill unused parameters."""
    grads = torch.autograd.grad(
        loss, params, create_graph=True, retain_graph=True, allow_unused=True
    )
    missing = sum(1 for grad in grads if grad is None)
    if missing:
        _warn_unused_grad_once(missing, len(params))

    return [
        grad if grad is not None else torch.zeros_like(parameter)
        for parameter, grad in zip(params, grads, strict=True)
    ]


def _hvp(
    grad_outputs: list[Tensor], params: list[nn.Parameter], vec: list[Tensor]
) -> list[Tensor]:
    """Compute H v by differentiating grad dot v, preserving the graph."""
    dot = sum(
        (grad * value).sum() for grad, value in zip(grad_outputs, vec, strict=True)
    )
    products = torch.autograd.grad(
        cast(Tensor, dot), params, retain_graph=True, allow_unused=True
    )
    missing = sum(1 for product in products if product is None)
    if missing:
        _warn_unused_hvp_once(missing, len(params))

    return [
        product if product is not None else torch.zeros_like(parameter)
        for parameter, product in zip(params, products, strict=True)
    ]


def _sample_rademacher_like(
    params: Iterable[nn.Parameter], generator: torch.Generator | None = None
) -> list[Tensor]:
    """Draw one Rademacher tensor per parameter from the selected RNG stream."""
    result: list[Tensor] = []

    for parameter in params:
        if generator is not None and generator.device != parameter.device:
            raw = torch.randint(
                0, 2, parameter.shape, device=generator.device, generator=generator
            ).to(device=parameter.device, dtype=parameter.dtype)
        else:
            raw = torch.randint(
                0,
                2,
                parameter.shape,
                device=parameter.device,
                dtype=parameter.dtype,
                generator=generator,
            )

        result.append(raw * 2 - 1)

    return result


def _sample_gaussian_like(
    params: Iterable[nn.Parameter], generator: torch.Generator | None = None
) -> list[Tensor]:
    """Draw Gaussian tensors on the generator's device before any transfer."""
    result: list[Tensor] = []

    for parameter in params:
        if generator is not None and generator.device != parameter.device:
            raw = torch.randn(
                parameter.shape, device=generator.device, generator=generator
            ).to(device=parameter.device, dtype=parameter.dtype)

        else:
            raw = torch.empty_like(parameter)
            raw.normal_(generator=generator)

        result.append(raw)

    return result


def _flatten(tensors: list[Tensor]) -> Tensor:
    """Concatenate tensors in their existing parameter order."""
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _unflatten(flat: Tensor, params: list[nn.Parameter]) -> list[Tensor]:
    """Restore parameter shapes without changing the flattened order."""
    result: list[Tensor] = []
    cursor = 0

    for parameter in params:
        size = parameter.numel()
        result.append(flat[cursor : cursor + size].reshape(parameter.shape))
        cursor += size

    return result


def _layer_quadratic(
    layers: list[LayerSpec], a: list[Tensor], b: list[Tensor]
) -> dict[str, float]:
    """Sum tensor-wise inner products within each layer's parameter slice."""
    result: dict[str, float] = {}
    cursor = 0

    for layer in layers:
        size = layer.n_tensors
        left = a[cursor : cursor + size]
        right = b[cursor : cursor + size]
        cursor += size

        result[layer.name] = float(
            sum((ai * bi).sum().item() for ai, bi in zip(left, right, strict=True))
        )

    return result


def _flat_params(layers: list[LayerSpec]) -> list[nn.Parameter]:
    """Collect tensors in layer order without flattening their storage."""
    return [parameter for layer in layers for parameter in layer.params]


def _gradient_graph(model, params, x, y, loss_fn) -> list[Tensor]:
    """Clear old gradients and build one retained loss-gradient graph."""
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y)
    return _stable_grad(loss, params)


class TraceEstimator(ABC):
    """Define layer-wise trace estimation with optional isolated probe RNGs."""

    name: str = "<abstract>"

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Accept a probe generator; generator-independent estimators ignore it."""
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
        """Return one estimated trace per layer name."""


class HutchinsonTraceEstimator(TraceEstimator):
    """Average z_ell^T (H z)_ell using one global Rademacher HVP per probe."""

    name = "hutchinson"

    def __init__(self, n_probes: int = 10) -> None:
        """Set the positive probe budget without consuming random numbers."""
        if n_probes < 1:
            raise ValueError("n_probes must be >= 1")

        self._n_probes = n_probes
        self._generator: torch.Generator | None = None

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Select the stream used for all subsequent probe draws."""
        self._generator = generator

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        """Accumulate global-probe quadratic forms without changing model mode."""
        params = _flat_params(layers)
        grads = _gradient_graph(model, params, x, y, loss_fn)
        accumulated: dict[str, float] = {layer.name: 0.0 for layer in layers}

        for _ in range(self._n_probes):
            probe = _sample_rademacher_like(params, generator=self._generator)
            product = _hvp(grads, params, probe)
            for name, value in _layer_quadratic(layers, probe, product).items():
                accumulated[name] += value

        return {name: value / self._n_probes for name, value in accumulated.items()}


class HutchPlusPlusTraceEstimator(TraceEstimator):
    """Estimate the low-rank trace plus a projected Hutchinson residual.

    Meyer et al. (2021): draw Gaussian G, compute Y=HG and Q=qr(Y), then
    add tr(Q^T H Q) to Hutchinson estimates projected by I-QQ^T.
    Layer slices accumulate the corresponding quadratic forms.

    Cost is 2*n_orth+n_hutch HVPs. The recommended one-third split yields
    O(1/n_total^2) variance on rank-deficient H, versus Hutchinson's O(1/n_total).
    """

    name = "hutch_plus_plus"

    def __init__(self, n_total: int = 10, n_orth_frac: float = 1.0 / 3.0) -> None:
        """Split the requested budget into basis and residual probes."""
        if n_total < 3:
            raise ValueError("Hutch++ requires n_total >= 3")

        n_orth = max(1, int(round(n_total * n_orth_frac)))
        n_hutch = max(1, n_total - n_orth)
        self._n_orth = n_orth
        self._n_hutch = n_hutch
        self._generator: torch.Generator | None = None

    @property
    def hvp_budget(self) -> int:
        """Return the actual number of HVPs consumed by one estimate."""
        return 2 * self._n_orth + self._n_hutch

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Select the shared stream for Gaussian and Rademacher probes."""
        self._generator = generator

    def _range_basis(self, grads, params) -> Tensor:
        """Construct the reduced QR basis of the Gaussian Hessian sketch."""
        columns: list[Tensor] = []
        for _ in range(self._n_orth):
            probe = _sample_gaussian_like(params, generator=self._generator)
            product = _hvp(grads, params, probe)
            columns.append(_flatten(product))

        sketch = torch.stack(columns, dim=1)
        basis, _ = torch.linalg.qr(sketch, mode="reduced")
        return basis

    def _low_rank_trace(self, layers, params, grads, basis) -> dict[str, float]:
        """Accumulate the layer slices of tr(Q^T H Q)."""
        result: dict[str, float] = {layer.name: 0.0 for layer in layers}

        for column in range(basis.shape[1]):
            probe = _unflatten(basis[:, column], params)
            product = _hvp(grads, params, probe)
            for name, value in _layer_quadratic(layers, probe, product).items():
                result[name] += value

        return result

    def _residual_trace(self, layers, params, grads, basis) -> dict[str, float]:
        """Accumulate Rademacher probes projected into the basis complement."""
        result: dict[str, float] = {layer.name: 0.0 for layer in layers}

        for _ in range(self._n_hutch):
            probe = _sample_rademacher_like(params, generator=self._generator)
            flat = _flatten(probe)
            projected = flat - basis @ (basis.T @ flat)

            structured = _unflatten(projected, params)
            product = _hvp(grads, params, structured)
            for name, value in _layer_quadratic(layers, structured, product).items():
                result[name] += value

        return result

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        """Evaluate the basis contribution before drawing residual probes."""
        params = _flat_params(layers)
        grads = _gradient_graph(model, params, x, y, loss_fn)
        basis = self._range_basis(grads, params)
        low_rank = self._low_rank_trace(layers, params, grads, basis)
        residual = self._residual_trace(layers, params, grads, basis)

        return {
            layer.name: low_rank[layer.name] + residual[layer.name] / self._n_hutch
            for layer in layers
        }


class BeckerLeCunDiagonalEstimator(TraceEstimator):
    """Sum the Becker-LeCun (1989) stochastic diagonal within each layer.

    Since sum_i z_i (H z)_i = z^T H z, this is exactly Hutchinson at the
    same probe budget. A separate class retains the literature convention
    and its typically larger budget for diagonal estimation.
    """

    name = "bl_diag"

    def __init__(self, n_probes: int = 200) -> None:
        """Configure the equivalent Hutchinson estimator."""
        self._impl = HutchinsonTraceEstimator(n_probes=n_probes)

    def set_generator(self, generator: torch.Generator | None) -> None:
        """Pass the isolated probe stream to the underlying estimator."""
        self._impl.set_generator(generator)

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        """Return the summed stochastic diagonal through the identical formula."""
        return self._impl.estimate(model, layers, x, y, loss_fn)


class KFACTraceEstimator(TraceEstimator):
    """Approximate empirical Fisher traces with ASDL Kronecker factors.

    K-FAC omits the Hessian-Fisher gap, so these are not full Hessian
    traces during training. For supported Linear/Conv2d layers,
    tr(F_ell)=tr(A_ell)*tr(B_ell); ASDL absorbs data scaling into factors.
    Unsupported layers such as BatchNorm are reported as NaN.
    """

    name = "kfac"

    def __init__(self) -> None:
        """Load the optional ASDL backend and its Fisher configuration."""
        try:
            from asdl import (
                FISHER_EMP,
                LOSS_CROSS_ENTROPY,
                SHAPE_KRON,
                FisherConfig,
                get_fisher_maker,
            )
        except ImportError as exc:
            logger.error("K-FAC asdl import failed: %r", exc)
            raise RuntimeError(
                "K-FAC estimator requires the `asdl` package "
                "(distributed on PyPI as `asdfghjkl`); install with "
                f"`uv add asdfghjkl`. Underlying import error: {exc!r}"
            ) from exc

        self._FisherConfig = FisherConfig
        self._get_fisher_maker = get_fisher_maker
        self._FISHER_EMP = FISHER_EMP
        self._SHAPE_KRON = SHAPE_KRON
        self._LOSS_CROSS_ENTROPY = LOSS_CROSS_ENTROPY

    def _accumulate_fisher(self, model: nn.Module, x: Tensor, y: Tensor) -> None:
        """Populate model Fisher attributes with one summed cross-entropy pass."""
        import torch.nn.functional as functional

        config = self._FisherConfig(
            fisher_type=self._FISHER_EMP,
            fisher_shapes=[self._SHAPE_KRON],
            loss_type=self._LOSS_CROSS_ENTROPY,
            data_size=int(x.shape[0]),
            fisher_attr="fisher",
        )
        maker = self._get_fisher_maker(model, config)
        logits = maker.setup_model_call(model, x)
        maker.setup_loss_call(functional.cross_entropy, logits, y, reduction="sum")
        maker.forward_and_backward()

    def _factor_traces(
        self, model: nn.Module, layers: list[LayerSpec]
    ) -> dict[str, float]:
        """Read factor traces and identify unsupported layer blocks."""
        modules: dict[str, Any] = dict(model.named_modules())
        traces: dict[str, float] = {}
        unsupported: list[str] = []

        for layer in layers:
            module = modules.get(layer.name)
            block = getattr(module, "fisher", None) if module is not None else None
            kron = getattr(block, "kron", None)

            if kron is None:
                traces[layer.name] = math.nan
                unsupported.append(layer.name)
                continue

            trace_a = float(torch.trace(kron.A).item())
            trace_b = float(torch.trace(kron.B).item())
            traces[layer.name] = trace_a * trace_b

        if unsupported:
            logger.warning(
                "K-FAC: %d/%d layers without Kronecker factor (e.g. BN); first few: %s",
                len(unsupported),
                len(layers),
                unsupported[:5],
            )

        return traces

    def estimate(
        self,
        model: nn.Module,
        layers: list[LayerSpec],
        x: Tensor,
        y: Tensor,
        loss_fn: LossFn,
    ) -> dict[str, float]:
        """Estimate supported Fisher traces; ASDL owns the loss closure."""
        self._accumulate_fisher(model, x, y)
        return self._factor_traces(model, layers)
