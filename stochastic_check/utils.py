"""General-purpose utilities: seeding, JSON streaming, slug naming.

The :func:`set_seed` function is the same deterministic recipe used in
``dag_hesse`` -- kept here verbatim so that this project has zero
relative-import dependencies on external research repositories.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
#  Reproducibility
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set the random seed across Python / NumPy / PyTorch for reproducibility.

    Also enables CuDNN deterministic mode and sets ``CUBLAS_WORKSPACE_CONFIG``
    so that algorithms in :mod:`torch.use_deterministic_algorithms` succeed
    on GPU.  Note that this disables the cudnn benchmark and can slow down
    training; this is acceptable for research where reproducibility
    dominates throughput.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
#  JSON streaming
# ---------------------------------------------------------------------------


class JSONLWriter:
    """JSON-lines writer for per-step measurement checkpoints.

    The file is opened lazily so that empty runs do not create empty
    artefacts; each call to :meth:`write` flushes the record to disk so
    that a killed worker still leaves a consistent prefix of records on
    disk.

    Parameters
    ----------
    path
        Destination file.
    mode
        ``"w"`` (default) truncates any pre-existing file so that a
        re-run of the same cell produces a clean trace stream;
        ``"a"`` appends, which is intended only for explicit resume
        workflows where the caller has verified the prefix.  The
        downstream aggregator reads every record and averages over
        an epoch window, so silently appending to an old stream from
        a previous run would mix two distinct training trajectories.
    """

    def __init__(self, path: Path, mode: str = "w") -> None:
        if mode not in {"w", "a"}:
            raise ValueError(f"JSONLWriter: unsupported mode {mode!r}")
        self._path = path
        self._mode = mode
        self._fh: Any = None

    def __enter__(self) -> JSONLWriter:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, self._mode, buffering=1, encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JSONLWriter must be used inside a `with` block")
        self._fh.write(json.dumps(_make_serializable(record)) + "\n")
        self._fh.flush()


def dump_json(payload: Any, path: Path) -> None:
    """Atomically write ``payload`` as pretty-printed JSON.

    Writes to ``path.tmp`` and renames, so the destination is never
    observed half-written -- the aggregator can read partial result
    directories safely.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_make_serializable(payload), f, indent=2, sort_keys=False)
    tmp.replace(path)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _make_serializable(obj: Any) -> Any:
    """Convert NumPy / PyTorch scalars and tensors into JSON-native types."""
    if isinstance(obj, dict):
        return {str(k): _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and (obj != obj):  # NaN
        return None
    return obj


# ---------------------------------------------------------------------------
#  Run naming
# ---------------------------------------------------------------------------


def run_slug(
    *,
    arch: str,
    dataset: str,
    noise: float,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> str:
    """Deterministic, filesystem-safe identifier for a single training run.

    Used both as a directory name under ``results/.../runs/`` and as the
    ``Task.name`` field for the lab-orchestrator scheduler.
    """
    parts = [
        f"arch={arch}",
        f"dataset={dataset}",
        f"noise={noise:.2f}",
        f"seed={seed}",
    ]
    if extra:
        for k in sorted(extra):
            parts.append(f"{k}={extra[k]}")
    return "_".join(parts)
