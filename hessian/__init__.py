from .exact import ExactBlockHessian
from .gn_decomposition import ExactGNDecomposition, StochasticGNGapEstimator
from .metrics import (
    compute_coupling,
    compute_dimension,
    compute_resonance,
    coupling_from_norms,
)
from .stochastic import StochasticHessianEstimator
from .trace import (
    BeckerLeCunDiagonalEstimator,
    HutchinsonTraceEstimator,
    HutchPlusPlusTraceEstimator,
    KFACTraceEstimator,
    LayerSpec,
    TraceEstimator,
    collect_layers,
)
from .types import BlockHessianResult, GNGapResult, HessianBlock, LayerID

__all__ = [
    "HessianBlock",
    "BlockHessianResult",
    "GNGapResult",
    "LayerID",
    "ExactBlockHessian",
    "StochasticHessianEstimator",
    "ExactGNDecomposition",
    "StochasticGNGapEstimator",
    "compute_resonance",
    "compute_coupling",
    "compute_dimension",
    "coupling_from_norms",
    # Layer-wise trace estimators
    "LayerSpec",
    "TraceEstimator",
    "collect_layers",
    "HutchinsonTraceEstimator",
    "HutchPlusPlusTraceEstimator",
    "BeckerLeCunDiagonalEstimator",
    "KFACTraceEstimator",
]
