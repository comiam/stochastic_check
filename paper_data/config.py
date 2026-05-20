"""Input-data paths for the paper-table generators."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Canonical locations of the aggregate JSONs."""

    results_root: Path = Path("results")

    @property
    def exp_b_aggregate(self) -> Path:
        return self.results_root / "exp_b" / "aggregate.json"

    @property
    def exp_b_runs_root(self) -> Path:
        return self.results_root / "exp_b" / "runs"

    @property
    def exp_a_aggregate(self) -> Path:
        return self.results_root / "exp_a" / "aggregate.json"

    @property
    def exp_a_runs_root(self) -> Path:
        return self.results_root / "exp_a" / "runs"

    @property
    def tables_dir(self) -> Path:
        return Path("paper_data") / "tables_out"
