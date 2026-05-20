"""Cached JSON / JSONL loading for paper-table generation."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=256)
def load_json(path: str | Path) -> Any:
    """Load a JSON document, cached by absolute path."""
    p = Path(path).resolve()
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Streaming load of a JSONL file (not cached, can be large)."""
    out: list[dict[str, Any]] = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def iter_run_finals(runs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Return ``[(run_dir, final_json), ...]`` for every run under ``runs_root``."""
    items: list[tuple[Path, dict[str, Any]]] = []
    for final in sorted(runs_root.glob("*/final.json")):
        items.append((final.parent, load_json(final)))
    return items


def iter_run_comparisons(runs_root: Path) -> list[tuple[Path, dict[str, Any]]]:
    items: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(runs_root.glob("*/comparison.json")):
        items.append((path.parent, load_json(path)))
    return items
