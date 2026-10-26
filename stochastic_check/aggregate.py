"""Aggregate per-run ``final.json`` artefacts into a single sweep JSON.

The aggregation is index-only: every per-run JSON keeps its full
content on disk, and the aggregate stores the cell identifier
(arch / dataset / noise / seed), the relative path to the run
directory, and a few summary statistics extracted from the run
(final accuracy, total trace per checkpoint).  Heavy data
(per-step trajectories, per-layer estimator comparisons) is kept
in the per-run files and is loaded on demand by ``paper_data``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from .utils import dump_json, load_json

logger = logging.getLogger(__name__)


def aggregate_runs(root: Path) -> dict[str, Any]:
    """Walk ``root`` for ``*/final.json`` files and build an index."""
    runs: list[dict[str, Any]] = []

    for final_path in sorted(root.glob("*/final.json")):
        try:
            data = load_json(final_path)

        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable run %s: %s", final_path, exc)
            continue

        cfg = data.get("config", {})
        runs.append(
            {
                "run_dir": str(final_path.parent.relative_to(root)),
                "arch": cfg.get("arch"),
                "dataset": cfg.get("dataset"),
                "noise_rate": cfg.get("training", {}).get("noise_rate"),
                "seed": cfg.get("training", {}).get("seed"),
                "final_eval": data.get("train_summary", {}).get("final_eval"),
                "wallclock_s": data.get("wallclock_s"),
                "n_layers": data.get("n_layers"),
            }
        )

    return {"n_runs": len(runs), "root": str(root), "runs": runs}


def aggregate_comparisons(root: Path) -> dict[str, Any]:
    """Walk ``root`` for ``*/comparison.json`` files and build an index."""
    runs: list[dict[str, Any]] = []

    for path in sorted(root.glob("*/comparison.json")):
        try:
            data = load_json(path)

        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable comparison %s: %s", path, exc)
            continue

        runs.append(
            {
                "run_dir": str(path.parent.relative_to(root)),
                "config": data.get("config", {}),
                "checkpoints": list(data.get("checkpoints", {}).keys()),
            }
        )

    return {"n_runs": len(runs), "root": str(root), "runs": runs}


def main(argv: list[str] | None = None) -> None:
    """Write a training or comparison index from the selected run directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["exp_b", "exp_a"],
        default="exp_b",
        help="exp_b -> walk for final.json; exp_a -> walk for comparison.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.kind == "exp_b":
        agg = aggregate_runs(args.root)
    else:
        agg = aggregate_comparisons(args.root)

    dump_json(agg, args.out)
    logger.info("Wrote aggregate index with %d runs to %s", agg["n_runs"], args.out)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    main()
