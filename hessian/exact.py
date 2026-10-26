"""Compute exact inter-layer activation Hessian blocks H^f_(v,w).

For v <= w, cache f_v without a graph, recreate it as a leaf, then build
f_v -> f_w -> logits -> loss. Differentiate dL/df_w with respect to f_v;
reverse-order requests use the transposed block. Diagonal blocks use the
same activation twice. Segmented models expose N names and N+1 callables.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from .types import BlockHessianResult, HessianBlock, LayerID


class SegmentedModel(Protocol):
    """Expose activation names and the sequential segments between them."""

    def get_layer_names(self) -> list[str]:
        """Return measurement-point names in forward order."""
        ...

    def get_segments(self) -> list[Callable[[Tensor], Tensor]]:
        """Return N+1 functions mapping input through N measured activations."""
        ...


class ExactBlockHessian:
    """Compute exact activation Hessian blocks for a segmented model."""

    def __init__(
        self, model: SegmentedModel, loss_fn: Callable[[Tensor, Tensor], Tensor]
    ) -> None:
        """Keep the model segmentation and scalar-loss callable."""
        self._model = model
        self._loss_fn = loss_fn

    def _loss_graph(self, fv, segments, idx_v, idx_w, y):
        """Forward from the recreated leaf through f_w to the scalar loss."""
        with torch.enable_grad():
            fw = fv
            for index in range(idx_v + 1, idx_w + 1):
                fw = segments[index](fw)

            value = fw
            for index in range(idx_w + 1, len(segments)):
                value = segments[index](value)
            loss = self._loss_fn(value, y)

        return fw, loss

    def compute_block(
        self, x: Tensor, y: Tensor, layer_v: LayerID, layer_w: LayerID
    ) -> HessianBlock:
        """Compute one block, transposing requests given in reverse layer order."""
        names = self._model.get_layer_names()
        segments = self._model.get_segments()
        idx_v = names.index(layer_v)
        idx_w = names.index(layer_w)

        swapped = False
        if idx_v > idx_w:
            idx_v, idx_w = idx_w, idx_v
            layer_v, layer_w = layer_w, layer_v
            swapped = True

        cached = self._forward_no_grad(x, segments, up_to=idx_v)
        fv = cached.detach().requires_grad_(True)
        fw, loss = self._loss_graph(fv, segments, idx_v, idx_w, y)

        if idx_v == idx_w:
            matrix = self._hessian_of_loss_wrt_tensor(loss, fv)
        else:
            matrix = self._cross_hessian(loss, fv, fw)

        if swapped:
            matrix = matrix.t()

        return HessianBlock(layer_v=layer_v, layer_w=layer_w, matrix=matrix.detach())

    def _layer_dimensions(
        self, x: Tensor, names: list[str], segments
    ) -> dict[str, int]:
        """Measure per-example activation dimensions without retaining a graph."""
        dimensions: dict[LayerID, int] = {}
        value = x

        for index, segment in enumerate(segments):
            with torch.no_grad():
                value = segment(value)
            if index < len(names):
                dimensions[names[index]] = value.numel() // value.shape[0]

        return dimensions

    def compute_all_blocks(
        self, x: Tensor, y: Tensor, pairs: list[tuple[LayerID, LayerID]] | None = None
    ) -> BlockHessianResult:
        """Compute requested pairs, or all pairs when none are specified."""
        names = self._model.get_layer_names()
        if pairs is None:
            pairs = [(v, w) for v in names for w in names]

        blocks: dict[tuple[LayerID, LayerID], HessianBlock] = {}
        for v, w in pairs:
            blocks[(v, w)] = self.compute_block(x, y, v, w)

        segments = self._model.get_segments()
        dimensions = self._layer_dimensions(x, names, segments)
        return BlockHessianResult(blocks=blocks, layer_dims=dimensions)

    @staticmethod
    def _forward_no_grad(
        x: Tensor, segments: list[Callable[[Tensor], Tensor]], up_to: int
    ) -> Tensor:
        """Cache the activation at one measurement point without a graph."""
        value = x
        with torch.no_grad():
            for index in range(up_to + 1):
                value = segments[index](value)

        return value

    @staticmethod
    def _hessian_of_loss_wrt_tensor(loss: Tensor, fv: Tensor) -> Tensor:
        """Differentiate an activation gradient to form its diagonal block.

        A mean-reduced loss already supplies 1/B. Summing gradient rows
        over the batch produces the mean Hessian; do not divide again.
        """
        batch_size = fv.shape[0]
        flat_dim = fv[0].numel()
        gradient = torch.autograd.grad(loss, fv, create_graph=True)[0]
        flat_gradient = gradient.reshape(batch_size, flat_dim)

        matrix = torch.zeros(flat_dim, flat_dim, device=fv.device)
        for column in range(flat_dim):
            column_sum = flat_gradient[:, column].sum()
            row = torch.autograd.grad(column_sum, fv, retain_graph=True)[0]
            matrix[column, :] = row.reshape(batch_size, flat_dim).sum(0)

        return matrix

    @staticmethod
    def _cross_hessian(loss: Tensor, fv: Tensor, fw: Tensor) -> Tensor:
        """Differentiate dL/df_w with respect to f_v for a cross-layer block.

        H_avg[i,j] = (1/B) sum_b d^2 L_b / df_v[i] df_w[j]. The mean loss
        already contributes 1/B, so batch rows are summed rather than averaged.
        """
        batch_size = fv.shape[0]
        dim_v = fv[0].numel()
        dim_w = fw[0].numel()
        gradient = torch.autograd.grad(loss, fw, create_graph=True)[0]
        flat_gradient = gradient.reshape(batch_size, dim_w)

        matrix = torch.zeros(dim_v, dim_w, device=fv.device)
        for column in range(dim_w):
            column_sum = flat_gradient[:, column].sum()
            row = torch.autograd.grad(column_sum, fv, retain_graph=True)[0]
            matrix[:, column] = row.reshape(batch_size, dim_v).sum(0)

        return matrix
