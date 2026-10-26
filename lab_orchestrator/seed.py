"""Reproducibility seed fixation for all major ML frameworks.

Auto-detects installed libraries and seeds only what is available.
Supports: stdlib random, numpy, PyTorch, TensorFlow, JAX.

Usage::

    from lab_orchestrator.seed import set_seed

    set_seed(42)                          # seed everything available
    set_seed(42, frameworks=["torch"])     # seed only torch + stdlib
    set_seed(42, deterministic=True)      # also enable CUDA determinism
"""

from __future__ import annotations

import logging
import os
import random
from collections.abc import Sequence

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import tensorflow as tf
except ImportError:
    tf = None  # type: ignore[assignment]

try:
    import jax
except ImportError:
    jax = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_ALL_FRAMEWORKS = ("numpy", "torch", "tensorflow", "jax")


def set_seed(
    seed: int = 42,
    *,
    deterministic: bool = False,
    frameworks: Sequence[str] | None = None,
    warn_missing: bool = False,
) -> list[str]:
    """Seed the selected available frameworks and return their names.

    Deterministic mode enables PyTorch/CUDA and TensorFlow controls, trading
    throughput for reproducibility. Missing frameworks are ignored unless
    warn_missing requests a log message; stdlib random is always seeded.
    """
    random.seed(seed)
    seeded: list[str] = ["random"]

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    targets = frameworks if frameworks is not None else _ALL_FRAMEWORKS

    for name in targets:
        fn = _SEEDERS.get(name)

        if fn is None:
            if warn_missing:
                logger.warning("Unknown framework %r - skipped", name)
            continue

        try:
            fn(seed, deterministic)
            seeded.append(name)

        except ImportError:
            if warn_missing:
                logger.warning("%s not installed - skipped", name)

        except Exception:
            logger.exception("Failed to seed %s", name)

    return seeded


# ---------------------------------------------------------------------------
#  Per-framework seeders
# ---------------------------------------------------------------------------


def _seed_numpy(seed: int, _deterministic: bool) -> None:
    """Seed NumPy when installed."""
    if np is None:
        raise ImportError("numpy")
    np.random.seed(seed)


def _seed_torch(seed: int, deterministic: bool) -> None:
    """Seed PyTorch CPU/CUDA and optionally enable deterministic kernels."""
    if torch is None:
        raise ImportError("torch")
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)


def _seed_tensorflow(seed: int, deterministic: bool) -> None:
    """Seed TensorFlow and preserve its optional determinism fallback."""
    if tf is None:
        raise ImportError("tensorflow")
    tf.random.set_seed(seed)

    if deterministic:
        # Env var is read at import time; set for child processes
        # that have not yet imported tensorflow.
        os.environ["TF_DETERMINISTIC_OPS"] = "1"

        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            logger.warning(
                "TF < 2.9: enable_op_determinism() unavailable. "
                "Set TF_DETERMINISTIC_OPS=1 before importing tensorflow."
            )


def _seed_jax(seed: int, _deterministic: bool) -> None:
    """Seed NumPy and create a JAX key that callers must split before use."""
    if jax is None:
        raise ImportError("jax")
    # JAX uses explicit PRNG keys; we store a default for user code.
    # Users MUST call jax.random.split(jax_key) before each use to
    # avoid duplicate draws.
    # Also seed numpy since JAX often delegates to it for data loading.
    _seed_numpy(seed, _deterministic)
    # Provide a default key accessible via lab_orchestrator.seed.jax_key
    global jax_key  # noqa: PLW0603
    jax_key = jax.random.PRNGKey(seed)


# Set by _seed_jax(); split before each use to avoid duplicate draws.
jax_key = None


_SEEDERS = {
    "numpy": _seed_numpy,
    "torch": _seed_torch,
    "tensorflow": _seed_tensorflow,
    "jax": _seed_jax,
}
