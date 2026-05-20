"""Experiment A: post-hoc estimator comparison on saved checkpoints.

For a given ``(arch, dataset, noise, seed)`` cell already trained by
:mod:`runner_exp_b`, this runner loads every snapshot (init / mid /
final), reconstructs the validation loader, samples a deterministic
Hessian batch, and evaluates **all five estimators**:

  1. Hutchinson  (K = 10)
  2. Hutchinson  (K = 50)
  3. Hutch++     (K_total = 10)
  4. Becker-LeCun diagonal  (K_diag = 200)
  5. K-FAC  (single update)

For each estimator we record per-layer trace, total trace, and
wallclock (CUDA-event timed when possible).  The cell against which
relative error is computed is selected as Hutchinson K=200 (the
de-facto "exact" surrogate used in the literature for moderately
sized layers).

The result is written to ``{output_dir}/{slug}/comparison.json``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from hessian import (
    BeckerLeCunDiagonalEstimator,
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


# ---------------------------------------------------------------------------
#  Spec
# ---------------------------------------------------------------------------


@dataclass
class CompareSpec:
    cfg: RunConfig
    snapshot_dir: Path
    output_dir: Path
    checkpoint_labels: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )
    hessian_batch_size: int = 256
    reference_probes: int = 200  # Hutchinson-K=200 surrogate for "ground truth"

    def estimators(self) -> dict[str, TraceEstimator]:
        return {
            "hutchinson_k10": HutchinsonTraceEstimator(n_probes=10),
            "hutchinson_k50": HutchinsonTraceEstimator(n_probes=50),
            "hutch_plus_plus_k10": HutchPlusPlusTraceEstimator(n_total=10),
            "bl_diag_k200": BeckerLeCunDiagonalEstimator(n_probes=200),
            "kfac": KFACTraceEstimator(),
        }


# ---------------------------------------------------------------------------
#  Runner
# ---------------------------------------------------------------------------


class ExpARunner:
    """Computes the estimator-comparison table for a single trained cell."""

    def __init__(self, spec: CompareSpec) -> None:
        self.spec = spec
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    #  Entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        cfg = self.spec.cfg
        set_seed(cfg.training.seed)

        _, val_loader, num_classes = get_loaders(
            cfg.dataset,
            batch_size=cfg.training.batch_size,
            noise_rate=cfg.training.noise_rate,
            noise_seed=cfg.training.seed,
        )

        # One model architecture, but we reinstantiate from the snapshot
        # for every estimator to keep their states isolated.
        results_per_checkpoint: dict[str, Any] = {}
        for label in self.spec.checkpoint_labels:
            snap_path = self.spec.snapshot_dir / f"{label}.pt"
            if not snap_path.exists():
                logger.warning("Snapshot %s missing, skipping", snap_path)
                continue
            results_per_checkpoint[label] = self._compare_on_snapshot(
                snap_path=snap_path,
                arch=cfg.arch,
                num_classes=num_classes,
                val_loader=val_loader,
            )

        payload: dict[str, Any] = {
            "config": {
                "arch": cfg.arch,
                "dataset": cfg.dataset,
                "noise_rate": cfg.training.noise_rate,
                "seed": cfg.training.seed,
            },
            "checkpoints": results_per_checkpoint,
        }
        slug = run_slug(
            arch=cfg.arch,
            dataset=cfg.dataset,
            noise=cfg.training.noise_rate,
            seed=cfg.training.seed,
        )
        out_path = self.spec.output_dir / slug / "comparison.json"
        dump_json(payload, out_path)
        return payload

    # ------------------------------------------------------------------
    #  Per-snapshot comparison
    # ------------------------------------------------------------------

    def _compare_on_snapshot(
        self,
        snap_path: Path,
        arch: str,
        num_classes: int,
        val_loader: Any,
    ) -> dict[str, Any]:
        # Deterministic Hessian batch shared across all estimators.
        x, y = sample_hessian_batch(
            val_loader, self.spec.hessian_batch_size, self.device
        )
        loss_fn = nn.CrossEntropyLoss()

        # Reference: Hutchinson with a large K -- the de-facto ground truth
        # at this scale.  Computed on its own model instance to isolate.
        ref_model = build_model(arch, num_classes).to(self.device)
        load_checkpoint(ref_model, snap_path)
        ref_layers = collect_layers(ref_model)
        ref_estimator = HutchinsonTraceEstimator(n_probes=self.spec.reference_probes)
        ref_t0 = time.perf_counter()
        ref_traces = ref_estimator.estimate(ref_model, ref_layers, x, y, loss_fn)
        ref_wallclock = time.perf_counter() - ref_t0

        per_estimator: dict[str, Any] = {}
        for name, estimator in self.spec.estimators().items():
            model = build_model(arch, num_classes).to(self.device)
            load_checkpoint(model, snap_path)
            layers = collect_layers(model)
            t0 = time.perf_counter()
            try:
                traces = estimator.estimate(model, layers, x, y, loss_fn)
            except RuntimeError as exc:
                logger.warning("Estimator %s failed: %s", name, exc)
                traces = {layer.name: math.nan for layer in layers}
            wallclock = time.perf_counter() - t0
            per_estimator[name] = {
                "trace_per_layer": traces,
                "trace_total": _finite_sum(traces.values()),
                "wallclock_s": wallclock,
                "relative_error": _relative_error(traces, ref_traces),
            }

        return {
            "reference": {
                "name": f"hutchinson_k{self.spec.reference_probes}",
                "trace_per_layer": ref_traces,
                "trace_total": _finite_sum(ref_traces.values()),
                "wallclock_s": ref_wallclock,
            },
            "estimators": per_estimator,
        }


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _finite_sum(values: Any) -> float:
    return float(sum(v for v in values if not (isinstance(v, float) and math.isnan(v))))


def _relative_error(
    pred: dict[str, float], ref: dict[str, float], eps: float = 1e-12
) -> dict[str, float]:
    """``|pred - ref| / max(|ref|, eps)`` per layer; NaN propagates."""
    out: dict[str, float] = {}
    for name in pred:
        p, r = pred[name], ref.get(name, math.nan)
        if math.isnan(p) or math.isnan(r):
            out[name] = math.nan
            continue
        out[name] = abs(p - r) / max(abs(r), eps)
    return out
