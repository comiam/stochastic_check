"""Find completed name/seed pairs in timestamped result directories."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)


def _result_directories(results_dir: str) -> Iterator[tuple[str, str]]:
    """Yield experiment directories beneath timestamp directories."""
    for timestamp in os.listdir(results_dir):
        timestamp_path = os.path.join(results_dir, timestamp)
        if not os.path.isdir(timestamp_path):
            continue

        for name in os.listdir(timestamp_path):
            path = os.path.join(timestamp_path, name)
            if os.path.isdir(path):
                yield name, path


def _valid_result(path: str) -> bool:
    """Reject unreadable or malformed JSON without hiding valid completed files."""
    try:
        with open(path) as stream:
            json.load(stream)

    except (json.JSONDecodeError, OSError):
        logger.warning("Skipping corrupt/incomplete result: %s", path)
        return False

    return True


def _completed_seeds(
    path: str, prefix: str, suffix: str, validate: bool
) -> Iterator[int]:
    """Read seed values from matching filenames, optionally validating contents."""
    for filename in os.listdir(path):
        if not (filename.startswith(prefix) and filename.endswith(suffix)):
            continue

        try:
            seed = int(filename[len(prefix) : -len(suffix)])
        except ValueError:
            continue

        if validate and not _valid_result(os.path.join(path, filename)):
            continue

        yield seed


def scan_completed(
    results_dir: str,
    *,
    seed_prefix: str = "seed_",
    seed_suffix: str = ".json",
    validate: bool = False,
) -> set[tuple[str, int]]:
    """Scan results_dir/timestamp/name/seed_N.json for completed runs.

    Optional JSON validation excludes interrupted or corrupt result files.
    Existing directory and filename traversal order is retained.
    """
    done: set[tuple[str, int]] = set()
    if not os.path.isdir(results_dir):
        logger.warning("results_dir %r does not exist, nothing to skip", results_dir)
        return done

    for name, path in _result_directories(results_dir):
        for seed in _completed_seeds(path, seed_prefix, seed_suffix, validate):
            done.add((name, seed))

    return done
