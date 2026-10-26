"""Deterministic seeding, streamed results, and stable experiment names."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch with deterministic CUDA settings.

    This is the original dag_hesse recipe. Disabling CuDNN benchmarking
    trades throughput for reproducibility; cuBLAS needs the fixed workspace.
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


class JSONLWriter:
    """Flush each record so interrupted runs retain a complete prefix.

    Default mode w replaces an old trajectory. Append mode a is only for
    explicit resume after validating the prefix; mixing runs corrupts
    downstream epoch averages.
    """

    def __init__(self, path: Path, mode: str = "w") -> None:
        """Configure the path and permit only replace or append mode."""
        if mode not in {"w", "a"}:
            raise ValueError(f"JSONLWriter: unsupported mode {mode!r}")

        self._path = path
        self._mode = mode
        self._fh: Any = None

    def __enter__(self) -> JSONLWriter:
        """Open the output stream on context entry."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self._path, self._mode, buffering=1, encoding="utf-8")
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close the output stream even after an interrupted write."""
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: dict[str, Any]) -> None:
        """Serialize and flush one record inside the writer context."""
        if self._fh is None:
            raise RuntimeError("JSONLWriter must be used inside a `with` block")

        self._fh.write(json.dumps(_make_serializable(record)) + "\n")
        self._fh.flush()


def dump_json(payload: Any, path: Path) -> None:
    """Write JSON through a temporary file and atomic destination rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(_make_serializable(payload), stream, indent=2, sort_keys=False)

    temporary.replace(path)


def load_json(path: Path) -> Any:
    """Load one UTF-8 JSON document."""
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load nonempty JSONL records without reordering them."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line.strip()) for line in stream if line.strip()]


def _make_serializable(obj: Any) -> Any:
    """Convert tensor/NumPy values recursively, representing float NaN as null."""
    if isinstance(obj, dict):
        return {str(key): _make_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(value) for value in obj]

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and obj != obj:
        return None

    return obj


def run_slug(
    *,
    arch: str,
    dataset: str,
    noise: float,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> str:
    """Build the existing directory/task identifier with sorted extra fields."""
    parts = [
        f"arch={arch}",
        f"dataset={dataset}",
        f"noise={noise:.2f}",
        f"seed={seed}",
    ]
    if extra:
        for key in sorted(extra):
            parts.append(f"{key}={extra[key]}")

    return "_".join(parts)
