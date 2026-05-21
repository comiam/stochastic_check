"""Experiment A: post-hoc estimator comparison on saved checkpoints.

For a given ``(arch, dataset, noise, seed)`` cell already trained by
:mod:`runner_exp_b`, this runner loads every snapshot, reconstructs
the validation loader, samples a deterministic Hessian batch, and
evaluates the estimators reported in section 6 of the paper:

  1. Hutchinson  (K = 5)
  2. Hutchinson  (K = 10)
  3. Hutch++     (K_total = 10)
  4. K-FAC       (single update; optional, falls back to a NaN
                  payload when the ASDL backend is unavailable)

The Becker-LeCun stochastic diagonal is omitted from the table
because its layer trace is, by construction, the algebraic identity
:math:`\\sum_i z_i (Hz)_i = z^\\top H z`, i.e. it coincides with
Hutchinson at the same probe budget.  This is stated in section 6
of the paper and the dedicated estimator class is kept in
:mod:`hessian.trace` for tests and external use.

For each estimator we record per-layer trace, total trace, and
wallclock.  The cell against which relative error is computed is
Hutchinson with ``K_ref = 200`` (the de-facto exact surrogate at
this scale; cf. section 5.3 of the paper).  The reference is
computed once per snapshot and is not re-seeded inside the loop.

The result is written to ``{output_dir}/{slug}/comparison.json``.
"""

from __future__ import annotations

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

    def estimator_factories(self) -> dict[str, Callable[[], TraceEstimator]]:
        """Lazy estimator factories.

        Construction is deferred to the comparison loop so that an
        unavailable backend (e.g. ASDL for K-FAC) raises only inside
        the per-estimator try/except and never aborts the whole
        snapshot.
        """
        return {
            "hutchinson_k5": lambda: HutchinsonTraceEstimator(n_probes=5),
            "hutchinson_k10": lambda: HutchinsonTraceEstimator(n_probes=10),
            # Hutch++(n_total=10, n_orth_frac=1/3) spends 2*3 + 7 = 13 HVPs
            # per call (Meyer 2021, eq. 4).  The key is named after the
            # actual HVP budget rather than ``n_total`` to keep the
            # paper table honest about cost.
            "hutch_plus_plus_hvp13": lambda: HutchPlusPlusTraceEstimator(n_total=10),
            "kfac": KFACTraceEstimator,
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
        seed_base = int(self.spec.cfg.training.seed)
        for name, factory in self.spec.estimator_factories().items():
            # Re-seed the global RNG before each estimator so that the
            # probe sequences of distinct estimators are independent of
            # the order of the loop and of the reference's probe count.
            # The seed mixes the run seed with the estimator name to
            # keep estimators decorrelated while staying reproducible.
            torch.manual_seed(_estimator_seed(seed_base, name))
            model = build_model(arch, num_classes).to(self.device)
            load_checkpoint(model, snap_path)
            layers = collect_layers(model)
            t0 = time.perf_counter()
            try:
                estimator = factory()
                traces = estimator.estimate(model, layers, x, y, loss_fn)
            except RuntimeError as exc:
                logger.warning("Estimator %s unavailable: %s", name, exc)
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


def _estimator_seed(base: int, name: str) -> int:
    """Deterministic per-estimator seed in the 32-bit unsigned range.

    A salted Python ``hash`` would be non-reproducible across
    interpreter launches; we mix ``base`` with the SHA1 digest of the
    estimator name instead so that the same ``(seed, name)`` always
    produces the same probes.
    """
    import hashlib

    digest = hashlib.sha1(name.encode("utf-8")).digest()
    salt = int.from_bytes(digest[:4], "big")
    return (base + salt) & 0x7FFFFFFF


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
