"""Result containers for Hessian blocks and Gauss-Newton decompositions."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

LayerID = str


@dataclass
class HessianBlock:
    """Store a single inter-layer Hessian block with its layer identifiers."""

    layer_v: LayerID
    layer_w: LayerID
    matrix: Tensor

    @property
    def shape(self) -> tuple[int, int]:
        """Return the two block dimensions."""
        return tuple(self.matrix.shape)  # type: ignore[return-value]

    @property
    def frobenius_norm(self) -> float:
        """Return the Frobenius norm of the block."""
        return torch.linalg.norm(self.matrix, ord="fro").item()

    @property
    def spectral_norm(self) -> float:
        """Return the spectral norm of the block."""
        return torch.linalg.norm(self.matrix, ord=2).item()

    @property
    def stable_rank(self) -> float:
        """Return squared Frobenius/spectral norm ratio, or NaN for a zero block.

        This effective dimension lies between one and rank(H) for nonzero H.
        """
        spectral = self.spectral_norm
        if spectral < 1e-12:
            return math.nan

        frobenius = self.frobenius_norm
        return (frobenius / spectral) ** 2


@dataclass
class GNGapResult:
    """Store norms from H = H_GN + H_tensor and their relative gap."""

    gap: float
    frob_full: float
    frob_gn: float
    frob_tensor: float


@dataclass
class BlockHessianResult:
    """Collect inter-layer blocks and their feature dimensions."""

    blocks: dict[tuple[LayerID, LayerID], HessianBlock]
    layer_dims: dict[LayerID, int]

    def get_block(self, v: LayerID, w: LayerID) -> HessianBlock | None:
        """Return a stored layer pair, or None when it was not computed."""
        return self.blocks.get((v, w))
