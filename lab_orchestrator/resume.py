"""Scan result directories for completed (name, seed) pairs."""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)


def scan_completed(
    results_dir: str,
    *,
    seed_prefix: str = "seed_",
    seed_suffix: str = ".json",
    validate: bool = False,
) -> set[tuple[str, int]]:
    """Walk *results_dir* for completed ``(name, seed)`` pairs.

    Expected layout::

        results_dir/
          <timestamp>/
            <name>/
              seed_42.json
              seed_43.json

    A seed is considered done when ``{seed_prefix}{N}{seed_suffix}``
    exists under a name directory.

    Parameters
    ----------
    results_dir : str
        Root directory to scan.
    seed_prefix, seed_suffix : str
        Filename pattern for completed seed files.
    validate : bool
        When *True*, attempt to parse each seed file as JSON and skip
        files that are empty or contain invalid JSON.  Useful for
        detecting incomplete writes from crashed tasks.

    Returns
    -------
    set[tuple[str, int]]
        ``(name, seed_value)`` pairs.
    """
    done: set[tuple[str, int]] = set()
    if not os.path.isdir(results_dir):
        logger.warning("results_dir %r does not exist, nothing to skip", results_dir)
        return done

    for ts_dir in os.listdir(results_dir):
        ts_path = os.path.join(results_dir, ts_dir)
        if not os.path.isdir(ts_path):
            continue
        for name in os.listdir(ts_path):
            name_path = os.path.join(ts_path, name)
            if not os.path.isdir(name_path):
                continue
            for fname in os.listdir(name_path):
                if fname.startswith(seed_prefix) and fname.endswith(seed_suffix):
                    seed_str = fname[len(seed_prefix) : -len(seed_suffix)]
                    try:
                        seed_val = int(seed_str)
                    except ValueError:
                        continue
                    if validate:
                        fpath = os.path.join(name_path, fname)
                        try:
                            with open(fpath) as f:
                                json.load(f)
                        except (json.JSONDecodeError, OSError):
                            logger.warning(
                                "Skipping corrupt/incomplete result: %s", fpath
                            )
                            continue
                    done.add((name, seed_val))

    return done
