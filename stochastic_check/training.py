"""Train models, record trace trajectories, and save epoch snapshots.

Periodic Hessian measurements use noisy, augmented training inputs with
fixed BatchNorm statistics. Saved state dicts support the separate
post-hoc estimator comparison.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from hessian import HutchinsonTraceEstimator, LayerSpec, TraceEstimator, collect_layers

from .data import sample_hessian_batch
from .utils import JSONLWriter

logger = logging.getLogger(__name__)


class Trainer:
    """Run SGD with periodic Hutchinson measurements.

    The inexpensive training estimator is separate from the Hutch++,
    K-FAC, and BL-diag comparisons performed on saved checkpoints.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
        epochs: int,
        scheduler: str = "cosine",
        grad_clip: float | None = None,
        metric_interval_steps: int = 100,
        hessian_batch_size: int = 128,
        n_probes: int = 10,
        probe_seed: int | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Prepare training, measurement, and isolated sampling state."""
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.grad_clip = grad_clip
        self.metric_interval_steps = metric_interval_steps
        self.hessian_batch_size = hessian_batch_size

        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.scheduler = self._make_scheduler(scheduler)
        self._prepare_estimator(n_probes, probe_seed)
        self._hvp_loader = self._make_hvp_loader(probe_seed)
        self._global_step = 0

    def _make_scheduler(self, name: str) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Construct the configured per-optimization-step schedule."""
        if name == "none":
            return None
        if name != "cosine":
            raise ValueError(f"Unknown scheduler: {name}")

        steps_per_epoch = max(1, len(self.train_loader))
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs * steps_per_epoch
        )

    def _prepare_estimator(self, n_probes: int, probe_seed: int | None) -> None:
        """Keep probe draws separate from data-loader and dropout randomness."""
        self.loss_fn = nn.CrossEntropyLoss()
        self.layers: list[LayerSpec] = collect_layers(self.model)
        self.estimator: TraceEstimator = HutchinsonTraceEstimator(n_probes=n_probes)

        if probe_seed is not None:
            self._probe_rng: torch.Generator | None = torch.Generator(
                device=self.device
            ).manual_seed(int(probe_seed))
            self.estimator.set_generator(self._probe_rng)
        else:
            self._probe_rng = None

    def _make_hvp_loader(self, probe_seed: int | None) -> DataLoader:
        """Sample training inputs without advancing the training sampler.

        A shared sampler would make future batch order depend on how often
        curvature is measured. Keep noisy labels and augmentation, but use
        a separate generator and no worker processes for Hessian batches.
        """
        generator = torch.Generator().manual_seed(
            int(probe_seed) + 1 if probe_seed is not None else 0
        )
        return DataLoader(
            self.train_loader.dataset,
            batch_size=self.hessian_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self.train_loader.pin_memory,
            drop_last=True,
            generator=generator,
        )

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return a detached CPU copy of the complete model state."""
        return {
            key: value.detach().cpu().clone()
            for key, value in self.model.state_dict().items()
        }

    @staticmethod
    def save_snapshot(state: dict[str, torch.Tensor], path: Path) -> None:
        """Save a state dict after creating its output directory."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate mean cross-entropy and accuracy on clean validation data."""
        self.model.eval()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in self.val_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = self.model(x)

            loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return {"val_loss": loss_sum / total, "val_acc": correct / total}

    def _evaluate_noperturb(self) -> dict[str, float]:
        """Restore RNG state and training mode after periodic validation.

        Validation iterator construction consumes the global RNG to seed
        workers. Restoring CPU/CUDA states isolates this added diagnostic;
        eval-mode BatchNorm leaves its running statistics unchanged.
        """
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        was_training = self.model.training

        try:
            return self.evaluate()
        finally:
            torch.set_rng_state(cpu_rng_state)

            if cuda_rng_state is not None:
                torch.cuda.set_rng_state_all(cuda_rng_state)

            if was_training:
                self.model.train()

    def measure_trace(self) -> dict[str, float]:
        """Measure curvature with fixed BatchNorm statistics and fresh inputs.

        Use the noisy, augmented training distribution, not repeated clean
        validation samples. Clear HVP gradients and restore training mode
        afterward so the next optimizer step remains unaffected.
        """
        was_training = self.model.training
        self.model.eval()
        x, y = sample_hessian_batch(
            self._hvp_loader, self.hessian_batch_size, self.device
        )
        traces = self.estimator.estimate(self.model, self.layers, x, y, self.loss_fn)
        self.model.zero_grad(set_to_none=True)

        if was_training:
            self.model.train()

        return traces

    def _global_grad_norm(self) -> float:
        """Read the stacked L2 gradient norm before trace measurement clears it.

        This is the clip_grad_norm_ convention, without modifying gradients
        or parameters and without drawing random numbers.
        """
        norms = [
            parameter.grad.detach().norm(2)
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        if not norms:
            return 0.0

        return float(torch.norm(torch.stack(norms), 2).item())

    def _train_batch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply one optimizer update followed by the optional LR update."""
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(self.model(x), y)
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._global_step += 1
        return loss

    def _record_metrics(
        self, loss: torch.Tensor, epoch: int, stream: JSONLWriter
    ) -> None:
        """Write one measurement, keeping trace and validation timings separate."""
        grad_norm = self._global_grad_norm()
        started = time.perf_counter()
        traces = self.measure_trace()
        trace_wallclock = time.perf_counter() - started

        validation_started = time.perf_counter()
        validation = self._evaluate_noperturb()
        stream.write(
            {
                "step": self._global_step,
                "epoch": epoch,
                "train_loss": loss.item(),
                "grad_norm": grad_norm,
                "val_loss": validation["val_loss"],
                "val_acc": validation["val_acc"],
                "trace": traces,
                "wallclock_s": trace_wallclock,
                "val_wallclock_s": time.perf_counter() - validation_started,
            }
        )

    def _train_epoch(self, epoch: int, stream: JSONLWriter) -> None:
        """Train one epoch and record metrics at the configured step interval."""
        self.model.train()
        for x, y in self.train_loader:
            loss = self._train_batch(x, y)
            if (
                self.metric_interval_steps > 0
                and self._global_step % self.metric_interval_steps == 0
            ):
                self._record_metrics(loss, epoch, stream)

    def _record_due_checkpoints(
        self, epoch, checkpoints, snapshot_dir, snapshots
    ) -> None:
        """Save requested checkpoints in their original label order."""
        for label, requested_epoch in checkpoints.items():
            if requested_epoch == epoch:
                snapshots[label] = self._record_checkpoint(
                    label, requested_epoch, snapshot_dir
                )

    def train(
        self,
        checkpoint_epochs: Iterable[tuple[str, int]],
        jsonl_path: Path,
        snapshot_dir: Path,
    ) -> dict[str, Any]:
        """Train all epochs, saving epoch-zero and post-epoch checkpoints."""
        checkpoints = dict(checkpoint_epochs)
        snapshots: dict[str, dict[str, Any]] = {}

        with JSONLWriter(jsonl_path) as stream:
            self._record_due_checkpoints(0, checkpoints, snapshot_dir, snapshots)
            for epoch in range(1, self.epochs + 1):
                self._train_epoch(epoch, stream)
                self._record_due_checkpoints(
                    epoch, checkpoints, snapshot_dir, snapshots
                )

        return {
            "epochs": self.epochs,
            "global_steps": self._global_step,
            "snapshots": snapshots,
            "final_eval": self.evaluate(),
        }

    def _record_checkpoint(
        self, label: str, epoch: int, snapshot_dir: Path
    ) -> dict[str, Any]:
        """Save the model state before evaluating the checkpoint."""
        state = self.snapshot()
        path = snapshot_dir / f"{label}.pt"
        self.save_snapshot(state, path)
        evaluation = self.evaluate()

        return {
            "label": label,
            "epoch": epoch,
            "step": self._global_step,
            "path": str(path),
            **evaluation,
        }


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    """Load tensor weights without changing the model's training mode."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def clone_model_to(model: nn.Module, device: torch.device) -> nn.Module:
    """Copy a model to isolate post-hoc estimator state mutations."""
    new = deepcopy(model)
    return new.to(device)
