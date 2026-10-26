"""Read experiment records without changing their order or precision."""

from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

RunKey = tuple[str, str, float]
RunIndex = dict[RunKey, dict[int, Path]]


@lru_cache(maxsize=256)
def load_json(path: str | Path) -> Any:
    """Load a JSON document, caching repeated calls with the same path."""
    with Path(path).resolve().open(encoding="utf-8") as stream:
        return json.load(stream)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read nonempty JSONL lines in their original order."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line.strip()) for line in stream if line.strip()]


def iter_run_finals(runs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return run directories and final manifests in path order."""
    return [
        (path.parent, load_json(path))
        for path in sorted(runs_root.glob("*/final.json"))
    ]


def iter_run_comparisons(runs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return run directories and comparison records in path order."""
    return [
        (path.parent, load_json(path))
        for path in sorted(runs_root.glob("*/comparison.json"))
    ]


def _run_identity(final: dict[str, Any]) -> tuple[RunKey, int] | None:
    """Extract a complete experimental cell, skipping missing metadata."""
    config = final.get("config", {})
    training = config.get("training", {})
    arch, dataset = config.get("arch"), config.get("dataset")
    noise, seed = training.get("noise_rate"), training.get("seed")

    if None in (arch, dataset, noise, seed):
        return None

    return (str(arch), str(dataset), float(noise)), int(seed)


def index_runs(runs_root: Path) -> RunIndex:
    """Index complete manifests with an existing trajectory by cell and seed.

    Preserve path order: the calibration's seeded resampling depends on the
    order of the clean-run pool. Missing metadata or streams are skipped.
    """
    index: RunIndex = defaultdict(dict)
    for run_dir, final in iter_run_finals(runs_root):
        identity = _run_identity(final)
        stream = run_dir / "steps.jsonl"

        if identity is not None and stream.exists():
            key, seed = identity
            index[key][seed] = stream

    return index
