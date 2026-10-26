"""Regenerate selected tables from bundled or explicitly supplied results."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from pathlib import Path

from .config import ARTIFACTS_ROOT, DataPaths
from .tables import baselines, decision_alerts, exp_a, exp_b, overhead

logger = logging.getLogger(__name__)
Generator = Callable[[DataPaths, Path], Path]
REGISTRY: dict[str, Generator] = {
    "tab:comparison": exp_a.generate,
    "tab:cohens-d": exp_b.generate,
    "tab:alerts": decision_alerts.generate,
    "tab:baselines": baselines.generate,
    "tab:overhead": overhead.generate,
}


def _parser() -> argparse.ArgumentParser:
    """Build the table selection and path options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", action="append", choices=list(REGISTRY))
    parser.add_argument("--all", action="store_true", help="Generate every table.")

    parser.add_argument("--results-root", type=Path, default=ARTIFACTS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("build/paper_tables"))
    return parser


def _require_inputs(label: str, paths: DataPaths) -> None:
    """Reject absent inputs instead of generating an empty success table."""
    if label == "tab:comparison":
        root, pattern = paths.exp_a_runs_root, "*/comparison.json"

    elif label in {"tab:baselines", "tab:overhead"}:
        root, pattern = paths.rerun_runs_root, "*/final.json"

    else:
        root, pattern = paths.exp_b_runs_root, "*/final.json"

    if not any(root.glob(pattern)):
        raise FileNotFoundError(f"No inputs for {label} under {root}")


def generate_selected(paths: DataPaths, labels: list[str], out_dir: Path) -> int:
    """Generate requested outputs and report any failed table as an error."""
    failed = False

    for label in labels:
        try:
            _require_inputs(label, paths)
            output = REGISTRY[label](paths, out_dir)
            logger.info("Generated %s -> %s", label, output)

        except (FileNotFoundError, ValueError) as exc:
            logger.error("Cannot generate %s: %s", label, exc)
            failed = True

    return int(failed)


def main(argv: list[str] | None = None) -> int:
    """Parse a table selection and return its generation status."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = _parser()
    args = parser.parse_args(argv)

    if not args.all and not args.label:
        parser.error("Pass --all or one or more --label values.")

    labels = list(REGISTRY) if args.all else args.label
    return generate_selected(DataPaths(args.results_root), labels, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
