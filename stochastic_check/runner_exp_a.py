"""Compare trace estimators on saved checkpoints with one shared data batch.

The reference is Hutchinson K=200 by default. Candidate models are
recreated from the checkpoint and independently seeded. Becker-LeCun is
omitted because summing its diagonal is algebraically Hutchinson at the
same K. K-FAC failures produce NaN rows rather than aborting a snapshot.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hessian import (
    HutchinsonTraceEstimator,
    HutchPlusPlusTraceEstimator,
    KFACTraceEstimator,
    TraceEstimator,
    collect_layers,
)

from .config import RunConfig
from .data import get_loaders, sample_hessian_batch
from .models import build_model
from .training import load_checkpoint
from .utils import dump_json, run_slug, set_seed

logger = logging.getLogger(__name__)


@dataclass
class CompareSpec:
    """Select a cell, its snapshots, and the reference-estimator budget."""

    cfg: RunConfig
    snapshot_dir: Path
    output_dir: Path
    checkpoint_labels: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )
    hessian_batch_size: int = 256
    reference_probes: int = 200

    def estimator_factories(self) -> dict[str, Callable[[], TraceEstimator]]:
        """Construct optional backends lazily inside estimator error handling."""
        return {
            "hutchinson_k5": lambda: HutchinsonTraceEstimator(n_probes=5),
            "hutchinson_k10": lambda: HutchinsonTraceEstimator(n_probes=10),
            # n_total=10 spends 2*3+7=13 HVPs; name the actual comparison cost.
            "hutch_plus_plus_hvp13": lambda: HutchPlusPlusTraceEstimator(n_total=10),
            "kfac": KFACTraceEstimator,
        }


class ExpARunner:
    """Compare the configured estimators on each available checkpoint."""

    def __init__(self, spec: CompareSpec) -> None:
        """Keep the comparison specification and select the execution device."""
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _snapshot_results(self, val_loader, num_classes: int) -> dict[str, Any]:
        """Evaluate existing snapshots in configured label order."""
        results = {}

        for label in self.spec.checkpoint_labels:
            path = self.spec.snapshot_dir / f"{label}.pt"
            if not path.exists():
                logger.warning("Snapshot %s missing, skipping", path)
                continue

            results[label] = self._compare_on_snapshot(
                path, self.spec.cfg.arch, num_classes, val_loader
            )

        return results

    def run(self) -> dict[str, Any]:
        """Load the cell's data, compare snapshots, and write comparison.json."""
        config = self.spec.cfg
        set_seed(config.training.seed)
        _, val_loader, num_classes = get_loaders(
            config.dataset,
            batch_size=config.training.batch_size,
            noise_rate=config.training.noise_rate,
            noise_seed=config.training.seed,
        )

        results = self._snapshot_results(val_loader, num_classes)
        payload: dict[str, Any] = {
            "config": {
                "arch": config.arch,
                "dataset": config.dataset,
                "noise_rate": config.training.noise_rate,
                "seed": config.training.seed,
            },
            "checkpoints": results,
        }
        slug = run_slug(
            arch=config.arch,
            dataset=config.dataset,
            noise=config.training.noise_rate,
            seed=config.training.seed,
        )
        dump_json(payload, self.spec.output_dir / slug / "comparison.json")
        return payload

    def _snapshot_model(self, arch, num_classes, path):
        """Create an independent model without changing its default mode."""
        model = build_model(arch, num_classes).to(self.device)
        load_checkpoint(model, path)
        return model, collect_layers(model)

    def _reference(self, snap_path, arch, num_classes, x, y, loss_fn) -> dict[str, Any]:
        """Compute the high-budget reference once without reseeding it."""
        model, layers = self._snapshot_model(arch, num_classes, snap_path)
        estimator = HutchinsonTraceEstimator(n_probes=self.spec.reference_probes)
        started = time.perf_counter()
        traces = estimator.estimate(model, layers, x, y, loss_fn)
        wallclock = time.perf_counter() - started

        return {
            "name": f"hutchinson_k{self.spec.reference_probes}",
            "trace_per_layer": traces,
            "trace_total": _finite_sum(traces.values()),
            "wallclock_s": wallclock,
        }

    def _estimate(self, name, factory, model, layers, x, y, loss_fn):
        """Time one candidate while retaining the optional-backend fallback."""
        started = time.perf_counter()

        try:
            estimator = factory()
            traces = estimator.estimate(model, layers, x, y, loss_fn)

        except RuntimeError as exc:
            logger.warning("Estimator %s unavailable: %s", name, exc)
            traces = {layer.name: math.nan for layer in layers}

        return traces, time.perf_counter() - started

    def _compare_on_snapshot(
        self, snap_path: Path, arch: str, num_classes: int, val_loader: Any
    ) -> dict[str, Any]:
        """Compare independently seeded models on the same deterministic batch."""
        x, y = sample_hessian_batch(
            val_loader, self.spec.hessian_batch_size, self.device
        )
        loss_fn = nn.CrossEntropyLoss()
        reference = self._reference(snap_path, arch, num_classes, x, y, loss_fn)
        per_estimator: dict[str, Any] = {}
        seed_base = int(self.spec.cfg.training.seed)

        for name, factory in self.spec.estimator_factories().items():
            torch.manual_seed(_estimator_seed(seed_base, name))
            model, layers = self._snapshot_model(arch, num_classes, snap_path)
            traces, wallclock = self._estimate(
                name, factory, model, layers, x, y, loss_fn
            )

            per_estimator[name] = {
                "trace_per_layer": traces,
                "trace_total": _finite_sum(traces.values()),
                "wallclock_s": wallclock,
                "relative_error": _relative_error(traces, reference["trace_per_layer"]),
            }

        return {"reference": reference, "estimators": per_estimator}


def _estimator_seed(base: int, name: str) -> int:
    """Mix the cell seed with SHA1 rather than Python's salted process hash."""
    digest = hashlib.sha1(name.encode("utf-8")).digest()
    salt = int.from_bytes(digest[:4], "big")
    return (base + salt) & 0x7FFFFFFF


def _finite_sum(values: Any) -> float:
    """Sum values after applying the existing float-NaN filter."""
    return float(
        sum(
            value
            for value in values
            if not (isinstance(value, float) and math.isnan(value))
        )
    )


def _relative_error(
    pred: dict[str, float], ref: dict[str, float], eps: float = 1e-12
) -> dict[str, float]:
    """Compute absolute relative errors per layer, propagating NaN."""
    errors = {}

    for name in pred:
        value, reference = pred[name], ref.get(name, math.nan)
        if math.isnan(value) or math.isnan(reference):
            errors[name] = math.nan
            continue

        errors[name] = abs(value - reference) / max(abs(reference), eps)

    return errors
