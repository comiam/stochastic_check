"""Run one training cell and persist its trace, snapshots, and final manifest."""

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
    """Carry the configuration for a single training run."""

    cfg: RunConfig


class ExpBRunner:
    """Prepare, execute, and serialize one experiment-B cell."""

    def __init__(self, spec: ExpBSpec) -> None:
        """Keep the requested run configuration."""
        self.spec = spec

    def _prepare_trainer(self, device: torch.device) -> tuple[Trainer, int]:
        """Create data, model, and trainer in their original initialization order."""
        config = self.spec.cfg
        train_loader, val_loader, num_classes = get_loaders(
            config.dataset,
            batch_size=config.training.batch_size,
            noise_rate=config.training.noise_rate,
            noise_seed=config.training.seed,
        )
        model = build_model(config.arch, num_classes).to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=config.training.lr,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            epochs=config.training.epochs,
            scheduler=config.training.scheduler,
            grad_clip=config.training.grad_clip,
            metric_interval_steps=config.training.metric_interval_steps,
            hessian_batch_size=config.estimator.hessian_batch_size,
            n_probes=config.estimator.n_probes,
            # Keep probe randomness separate from data loading and dropout.
            probe_seed=config.training.seed * 1_000_003 + 17,
            device=device,
        )
        return trainer, num_classes

    def _checkpoints(self) -> list[tuple[str, int]]:
        """Prefer explicit checkpoint epochs over the legacy symbolic labels."""
        training = self.spec.cfg.training
        if training.checkpoint_epochs is not None:
            return _resolve_explicit_checkpoint_epochs(
                training.checkpoint_epochs, training.epochs
            )

        return _resolve_checkpoint_epochs(training.checkpoint_labels, training.epochs)

    def run(self) -> dict[str, Any]:
        """Execute the cell and write its final manifest after training completes."""
        config = self.spec.cfg
        set_seed(config.training.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        trainer, num_classes = self._prepare_trainer(device)

        run_dir = Path(config.output_dir) / run_slug(
            arch=config.arch,
            dataset=config.dataset,
            noise=config.training.noise_rate,
            seed=config.training.seed,
        )
        checkpoints = self._checkpoints()
        started = time.perf_counter()
        summary = trainer.train(
            checkpoint_epochs=checkpoints,
            jsonl_path=run_dir / "steps.jsonl",
            snapshot_dir=run_dir / "snapshots",
        )
        wallclock = time.perf_counter() - started

        payload: dict[str, Any] = {
            "config": asdict(config),
            "checkpoint_epochs": checkpoints,
            "train_summary": summary,
            "wallclock_s": wallclock,
            "num_classes": num_classes,
            "n_layers": len(trainer.layers),
        }
        dump_json(payload, run_dir / "final.json")
        return payload


def _resolve_checkpoint_epochs(
    labels: list[str], n_epochs: int
) -> list[tuple[str, int]]:
    """Map legacy labels to epochs, assigning unknown labels to the final epoch."""
    known = {"init": 0, "mid": max(1, n_epochs // 2), "final": n_epochs}
    mapping: dict[str, int] = {}
    for label in labels:
        mapping[label] = known.get(label, n_epochs)

    return list(mapping.items())


def _checkpoint_label(index: int, count: int, epoch: int) -> str:
    """Name sorted explicit epochs using the original endpoint convention."""
    if index == 0:
        return "init"
    if index == count - 1:
        return "final"

    return f"ep{epoch:02d}"


def _resolve_explicit_checkpoint_epochs(
    epochs: list[int], n_epochs: int
) -> list[tuple[str, int]]:
    """Deduplicate explicit epochs and preserve init/epNN/final naming."""
    cleaned = sorted({int(epoch) for epoch in epochs})
    for epoch in cleaned:
        if not 0 <= epoch <= n_epochs:
            raise ValueError(f"checkpoint epoch {epoch} outside [0, {n_epochs}]")

    return [
        (_checkpoint_label(index, len(cleaned), epoch), epoch)
        for index, epoch in enumerate(cleaned)
    ]
