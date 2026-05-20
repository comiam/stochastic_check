"""
Stochastic estimation of ||H^f_{v,w}||_F via Hutchinson + HVP.

Frobenius norm estimation without building the full matrix:
  ||H||_F^2 = tr(H^T H) ~= (1/m) sum_k z_k^T H^T H z_k = (1/m) sum_k ||H z_k||^2

where z_k are random Rademacher vectors of dimension d_w.

HVP H_{v,w} z is computed via double backward:
  H_{v,w} z = d/df_v [ (dL/df_w)^T z ]

Stable rank D = ||H||_F^2 / ||H||_2^2 estimation via power iteration:
  ||H||_2 is computed by the iteration v <- H^T(Hv) / ||H^T(Hv)|| (converges to sigma_1).
  Adjoint HVP: H^T u = d/df_w [ (dL/df_v)^T u ].

Complexity: O(m * P) in time, O(P) in memory (m is the number of probes).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor

from .exact import SegmentedModel
from .types import LayerID


class StochasticHessianEstimator:
    """Estimation of ||H^f_{v,w}||_F and stable rank via Hutchinson / HVP."""

    def __init__(
        self,
        model: SegmentedModel,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_probes: int = 30,
        n_power_iter: int = 20,
    ) -> None:
        self._model = model
        self._loss_fn = loss_fn
        self._n_probes = n_probes
        self._n_power_iter = n_power_iter

    # ------------------------------------------------------------------
    # Graph setup
    # ------------------------------------------------------------------

    def _setup_graph(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
        *,
        need_grad_fv: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Builds the forward graph for HVP computation.

        Returns:
            (f_v, f_w_graph, grad_fw, grad_fv | None)
        """
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v

        h = x
        with torch.no_grad():
            for i in range(idx_v + 1):
                h = segments[i](h)
        f_v = h.detach().requires_grad_(True)

        h = f_v
        for i in range(idx_v + 1, idx_w + 1):
            h = segments[i](h)
        f_w_graph = h

        h = f_w_graph
        for i in range(idx_w + 1, len(segments)):
            h = segments[i](h)
        loss = self._loss_fn(h, y)

        grad_fw = torch.autograd.grad(
            loss,
            f_w_graph,
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_fv: Tensor | None = None
        if need_grad_fv:
            grad_fv = torch.autograd.grad(
                loss,
                f_v,
                create_graph=True,
                retain_graph=True,
            )[0]

        return f_v, f_w_graph, grad_fw, grad_fv

    # ------------------------------------------------------------------
    # Frobenius norm (existing behaviour)
    # ------------------------------------------------------------------

    def estimate_block_frob_norm(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """Estimates ||H_avg^f_{v,w}||_F via Hutchinson with a shared probe vector.

        ||H_avg||_F^2 = E_z[||H_avg z||^2],  z ~ Rademacher(d_w).
        H_avg z is computed via HVP: expand z to the batch dimension, then average the result.
        """
        f_v, f_w_graph, grad_fw, _ = self._setup_graph(x, y, layer_v, layer_w)

        device = f_v.device
        B = f_v.shape[0]
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        frob_sq = self._hutchinson_shared(f_v, f_w_graph, grad_fw, B, d_w, device)
        return frob_sq**0.5

    # ------------------------------------------------------------------
    # Stable rank (dimension metric)
    # ------------------------------------------------------------------

    def estimate_stable_rank(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """D(v,w) = ||H||_F^2 / ||H||_2^2 via Hutchinson + power iteration.

        Both norms are computed on the batch-averaged Hessian H_avg
        using shared probe vectors.
        """
        f_v, f_w_graph, grad_fw, grad_fv = self._setup_graph(
            x,
            y,
            layer_v,
            layer_w,
            need_grad_fv=True,
        )

        device = f_v.device
        B = f_v.shape[0]
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        # --- ||H||_2 via power iteration ---
        spectral_sq = self._power_iteration(
            f_v,
            f_w_graph,
            grad_fw,
            grad_fv,
            B,
            d_w,
            device,
        )

        if spectral_sq < 1e-24:
            return math.nan
        frob_sq = self._hutchinson_shared(
            f_v,
            f_w_graph,
            grad_fw,
            B,
            d_w,
            device,
        )

        return frob_sq / spectral_sq

    def _power_iteration(
        self,
        f_v: Tensor,
        f_w_graph: Tensor,
        grad_fw: Tensor,
        grad_fv: Tensor | None,
        B: int,
        d_w: int,
        device: torch.device,
    ) -> float:
        """Estimates sigma_1^2 = ||H||_2^2 via power iteration on H^T H.

        Returns:
            sigma_1^2  (squared spectral norm).
        """
        assert grad_fv is not None

        v = torch.randn(d_w, device=device)
        v = v / v.norm()

        for _ in range(self._n_power_iter):
            # w = H_avg v  (v lives in the space of f_w)
            # hvp[b] = (1/B) H_b v  (due to reduction='mean'), sum -> H_avg v
            v_batch = v.reshape(1, *f_w_graph.shape[1:]).expand_as(f_w_graph)
            dot = (grad_fw * v_batch).sum()
            hvp = torch.autograd.grad(dot, f_v, retain_graph=True)[0]
            w = hvp.sum(0)

            # v_new = H_avg^T w  (w lives in the space of f_v)
            w_batch = w.reshape(1, *f_v.shape[1:]).expand_as(f_v)
            dot2 = (grad_fv * w_batch).sum()
            htvp = torch.autograd.grad(dot2, f_w_graph, retain_graph=True)[0]
            v_new = htvp.sum(0)

            v_norm = v_new.norm()
            if v_norm < 1e-12:
                return 0.0  # sigma_1^2 = 0 - zero matrix, D is handled above
            v = v_new / v_norm

        # Final estimate sigma_1 = ||H_avg v||
        v_batch = v.reshape(1, *f_w_graph.shape[1:]).expand_as(f_w_graph)
        dot = (grad_fw * v_batch).sum()
        hvp = torch.autograd.grad(dot, f_v, retain_graph=True)[0]
        Hv = hvp.sum(0)
        return (Hv**2).sum().item()

    def _hutchinson_shared(
        self,
        f_v: Tensor,
        f_w_graph: Tensor,
        grad_fw: Tensor,
        B: int,
        d_w: int,
        device: torch.device,
    ) -> float:
        """Hutchinson estimator with shared probe vectors.

        Returns:
            Estimate of ||H_avg||_F^2.
        """
        frob_sq = 0.0
        for _ in range(self._n_probes):
            z = torch.randint(0, 2, (d_w,), device=device).float() * 2 - 1
            z_batch = z.reshape(1, *f_w_graph.shape[1:]).expand_as(f_w_graph)
            dot = (grad_fw * z_batch).sum()
            hvp = torch.autograd.grad(dot, f_v, retain_graph=True)[0]
            # hvp[b] = (1/B) H_b z, sum -> (1/B) sum_b H_b z = H_avg z
            Hz = hvp.sum(0)
            frob_sq += (Hz**2).sum().item()
        return frob_sq / self._n_probes

    # ------------------------------------------------------------------
    # Combined estimator (avoids double graph setup / Hutchinson)
    # ------------------------------------------------------------------

    def estimate_frob_and_stable_rank(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> tuple[float, float]:
        """Returns (||H||_F, D) in a single graph pass: shared Hutchinson + power iter.

        Avoids rebuilding the graph and saves m additional HVP calls.
        """
        f_v, f_w_graph, grad_fw, grad_fv = self._setup_graph(
            x,
            y,
            layer_v,
            layer_w,
            need_grad_fv=True,
        )

        device = f_v.device
        B = f_v.shape[0]
        d_w = f_w_graph[0].numel()  # C*H*W for 4D, d for 2D

        frob_sq = self._hutchinson_shared(f_v, f_w_graph, grad_fw, B, d_w, device)
        frob_norm = frob_sq**0.5

        spectral_sq = self._power_iteration(
            f_v,
            f_w_graph,
            grad_fw,
            grad_fv,
            B,
            d_w,
            device,
        )
        stable_rank = frob_sq / spectral_sq if spectral_sq > 1e-24 else math.nan

        return frob_norm, stable_rank

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def estimate_resonance(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """R(v,w) = ||H^f_{v,w}||_F."""
        return self.estimate_block_frob_norm(x, y, layer_v, layer_w)

    def estimate_coupling(
        self,
        x: Tensor,
        y: Tensor,
        layer_v: LayerID,
        layer_w: LayerID,
    ) -> float:
        """C(v,w) = ||H^f_{v,w}||_F / sqrt(||H^f_{v,v}||_F * ||H^f_{w,w}||_F)."""
        r_vw = self.estimate_block_frob_norm(x, y, layer_v, layer_w)
        r_vv = self.estimate_block_frob_norm(x, y, layer_v, layer_v)
        r_ww = self.estimate_block_frob_norm(x, y, layer_w, layer_w)
        denom = (r_vv * r_ww) ** 0.5
        if denom < 1e-12:
            return math.nan
        # C > 1 is possible when H^T != 0 (non-PSD regime);
        # clamping with min(., 1) would hide information about the tensor component.
        return r_vw / denom
