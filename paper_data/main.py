"""CLI for paper-table regeneration.

Each table is registered as ``label -> (generator_fn, required_paths)``;
``--all`` regenerates every registered table.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from .config import DataPaths
from .tables import exp_a as table_exp_a
from .tables import exp_b as table_exp_b

logger = logging.getLogger(__name__)


Generator = Callable[[DataPaths, Path], Path]


REGISTRY: dict[str, Generator] = {
    "tab:comparison": table_exp_a.generate,
    "tab:cohens-d": table_exp_b.generate,
}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        action="append",
        choices=list(REGISTRY),
        help="Generate only the listed tables; may be repeated.",
    )
    parser.add_argument("--all", action="store_true", help="Generate every table.")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--out-dir", type=Path, default=Path("paper_data/tables_out"))
    args = parser.parse_args(argv)

    if not args.all and not args.label:
        parser.error("Pass --all or one or more --label values.")

    paths = DataPaths(results_root=args.results_root)
    labels = list(REGISTRY) if args.all else args.label
    for label in labels:
        gen = REGISTRY[label]
        try:
            out = gen(paths, args.out_dir)
            logger.info("Generated %s -> %s", label, out)
        except FileNotFoundError as exc:
            logger.error("Skipping %s: %s", label, exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
