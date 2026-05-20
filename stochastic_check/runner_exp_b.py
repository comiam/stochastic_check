"""Experiment B: train a single cell + per-step trace trajectory.

A "cell" is one ``(arch, dataset, noise_rate, seed)`` configuration.
The runner:

  1. Seeds NumPy / PyTorch with ``training.seed`` for full reproducibility.
  2. Loads CIFAR-10/100 with symmetric label noise injected into the
     training labels.
  3. Builds the architecture and starts the :class:`Trainer`.
  4. The trainer streams per-step trace measurements to ``steps.jsonl``
     and saves init / mid / final snapshots to ``snapshots/{label}.pt``.
  5. A ``final.json`` summary is dumped at the end and is the canonical
     unit of aggregation.

This file contains only the orchestration; the numerical work happens
in :mod:`stochastic_check.training` and :mod:`hessian.trace`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .config import RunConfig
from .data import get_loaders
from .models import build_model
from .training import Trainer
from .utils import dump_json, run_slug, set_seed

logger = logging.getLogger(__name__)


@dataclass
class ExpBSpec:
    cfg: RunConfig


class ExpBRunner:
    def __init__(self, spec: ExpBSpec) -> None:
        self.spec = spec

    def run(self) -> dict[str, Any]:
        cfg = self.spec.cfg
        set_seed(cfg.training.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        train_loader, val_loader, num_classes = get_loaders(
            cfg.dataset,
            batch_size=cfg.training.batch_size,
            noise_rate=cfg.training.noise_rate,
            noise_seed=cfg.training.seed,
        )
        model = build_model(cfg.arch, num_classes).to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
            epochs=cfg.training.epochs,
            scheduler=cfg.training.scheduler,
            metric_interval_steps=cfg.training.metric_interval_steps,
            hessian_batch_size=cfg.estimator.hessian_batch_size,
            n_probes=cfg.estimator.n_probes,
            device=device,
        )

        run_dir = Path(cfg.output_dir) / run_slug(
            arch=cfg.arch,
            dataset=cfg.dataset,
            noise=cfg.training.noise_rate,
            seed=cfg.training.seed,
        )
        snapshot_dir = run_dir / "snapshots"
        jsonl_path = run_dir / "steps.jsonl"

        # The 'init' / 'mid' / 'final' label mapping is symmetric across
        # all runs: init = before epoch 1, mid = halfway, final = last.
        cps = _resolve_checkpoint_epochs(
            cfg.training.checkpoint_labels, cfg.training.epochs
        )

        t0 = time.perf_counter()
        train_summary = trainer.train(
            checkpoint_epochs=cps,
            jsonl_path=jsonl_path,
            snapshot_dir=snapshot_dir,
        )
        wallclock = time.perf_counter() - t0

        payload: dict[str, Any] = {
            "config": asdict(cfg),
            "checkpoint_epochs": cps,
            "train_summary": train_summary,
            "wallclock_s": wallclock,
            "num_classes": num_classes,
            "n_layers": len(trainer.layers),
        }
        dump_json(payload, run_dir / "final.json")
        return payload


def _resolve_checkpoint_epochs(
    labels: list[str], n_epochs: int
) -> list[tuple[str, int]]:
    """Map labels -> concrete epoch indices.

    ``init`` -> 0  (before any update)
    ``mid``  -> n_epochs // 2
    ``final``-> n_epochs
    Unknown labels are passed through as ``label -> n_epochs``.
    """
    mapping: dict[str, int] = {}
    for label in labels:
        if label == "init":
            mapping[label] = 0
        elif label == "mid":
            mapping[label] = max(1, n_epochs // 2)
        elif label == "final":
            mapping[label] = n_epochs
        else:
            mapping[label] = n_epochs
    return list(mapping.items())
