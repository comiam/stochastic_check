"""GPU detection and CLI parsing helpers."""

from __future__ import annotations

import logging
import multiprocessing as mp

try:
    import torch as _torch
except ImportError:
    _torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def detect_gpus() -> list[int]:
    """Return list of available CUDA device indices.

    Falls back to an empty list if ``torch`` is not installed or no
    CUDA devices are present.

    .. note::

       Calling this function initialises the CUDA runtime.  On Linux
       with the default ``"fork"`` start method, subsequent
       ``multiprocessing.Process`` children inherit the (now invalid)
       CUDA context.  Workers in this package communicate via
       ``subprocess`` and do not import ``torch``, so this is typically
       harmless.  If your workers use ``torch`` directly, either call
       ``mp.set_start_method("spawn")`` before ``detect_gpus()`` or
       use ``parse_gpu_ids("0,1,2")`` to avoid CUDA initialisation.
    """
    if _torch is None:
        return []
    try:
        count = _torch.cuda.device_count()
    except Exception:
        return []

    if count and mp.get_start_method(allow_none=True) == "fork":
        logger.warning(
            "detect_gpus() initialised CUDA under fork start method; "
            "pass --gpus explicitly or use mp.set_start_method('spawn') "
            "if workers import torch directly"
        )

    return list(range(count))


def parse_gpu_ids(raw: str | None) -> list[int]:
    """Parse a ``--gpus`` CLI value into a list of GPU indices.

    Accepts:
    - ``"0,1,2,3"`` - comma-separated GPU IDs.
    - ``"2"``        - single GPU ID.
    - ``None``       - auto-detect all GPUs.
    """
    if raw is None:
        return detect_gpus()

    raw = raw.strip()
    if not raw:
        return []
    if "," in raw:
        return [int(g) for g in raw.split(",") if g.strip()]

    return [int(raw)]
