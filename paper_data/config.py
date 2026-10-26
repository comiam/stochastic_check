"""Paths to bundled or explicitly selected experiment results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ARTIFACTS_ROOT = Path(__file__).resolve().parent / "artifacts"


@dataclass(frozen=True)
class DataPaths:
    """Keep original, rerun, and comparison inputs separate."""

    results_root: Path = ARTIFACTS_ROOT

    @property
    def exp_b_aggregate(self) -> Path:
        """Return the original training index."""
        return self.results_root / "exp_b" / "aggregate.json"

    @property
    def exp_b_runs_root(self) -> Path:
        """Return original trajectories used by the alert table."""
        return self.results_root / "exp_b" / "runs"

    @property
    def rerun_runs_root(self) -> Path:
        """Return trajectories with first-order metrics and timings."""
        return self.results_root / "exp_b_rerun" / "runs"

    @property
    def exp_a_aggregate(self) -> Path:
        """Return the optional estimator-comparison index."""
        return self.results_root / "exp_a" / "aggregate.json"

    @property
    def exp_a_runs_root(self) -> Path:
        """Return optional post-hoc comparisons, not bundled checkpoints."""
        return self.results_root / "exp_a" / "runs"

    @property
    def tables_dir(self) -> Path:
        """Return the default output directory, separate from source data."""
        return Path("build/paper_tables")
