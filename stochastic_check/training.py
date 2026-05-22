"""Training loop with per-step trace-checkpointing and epoch snapshots.

Two distinct kinds of measurements are produced:

  * **Per-step trace stream**  -- every ``metric_interval_steps`` an HVP-based
    trace estimate is computed on a deterministic Hessian batch and
    appended (as a JSONL record) to ``steps.jsonl``.  This produces the
    fine-grained trajectory used in experiment B.

  * **Epoch snapshots**  -- at the labelled checkpoints (``init``,
    ``mid``, ``final``) the full model state dict is cloned to CPU and
    saved to disk for use by experiment A (post-hoc estimator
    comparison).
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

from hessian import (
    HutchinsonTraceEstimator,
    LayerSpec,
    TraceEstimator,
    collect_layers,
)

from .data import sample_hessian_batch
from .utils import JSONLWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """SGD trainer with built-in trace-trajectory collection.

    Notes
    -----
    The trace estimator instance is held by the trainer; only the
    cheap single-pass Hutchinson is used during training (K typically
    10) to keep the per-step cost ~ 2 * forward+backward.  The full
    estimator-comparison battery (Hutch++, K-FAC, BL-diag, ...) runs
    post-hoc on the saved checkpoints in ``runner_exp_a``.
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        self.metric_interval_steps = metric_interval_steps
        self.hessian_batch_size = hessian_batch_size

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        if scheduler == "cosine":
            steps_per_epoch = max(1, len(train_loader))
            self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = (
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=epochs * steps_per_epoch
                )
            )
        elif scheduler == "none":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")

        self.loss_fn = nn.CrossEntropyLoss()
        self.layers: list[LayerSpec] = collect_layers(self.model)
        self.estimator: TraceEstimator = HutchinsonTraceEstimator(n_probes=n_probes)

        # Dedicated probe RNG: isolates per-step trace stochasticity from
        # the global torch RNG (which is consumed by data loaders and
        # model dropout) so the trace stream is bit-reproducible given
        # ``probe_seed`` even if the training data order is identical.
        if probe_seed is not None:
            self._probe_rng: torch.Generator | None = torch.Generator(
                device=self.device
            ).manual_seed(int(probe_seed))
            self.estimator.set_generator(self._probe_rng)
        else:
            self._probe_rng = None

        # Dedicated HVP-batch loader: shares the training *dataset*
        # (noisy labels, augmented transforms) but has its own RNG, so
        # consuming a Hessian batch does *not* mutate the sampler state
        # of ``train_loader``.  Without this, every ``measure_trace``
        # call would advance ``train_loader``'s shuffling generator and
        # the order of training batches in subsequent epochs would
        # silently depend on ``metric_interval_steps`` -- breaking
        # bit-reproducibility w.r.t. that knob.
        hvp_generator = torch.Generator().manual_seed(
            int(probe_seed) + 1 if probe_seed is not None else 0
        )
        self._hvp_loader = DataLoader(
            train_loader.dataset,
            batch_size=hessian_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=train_loader.pin_memory,
            drop_last=True,
            generator=hvp_generator,
        )

        self._global_step = 0

    # ------------------------------------------------------------------
    #  Snapshotting
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, torch.Tensor]:
        """Return a CPU-cloned copy of the full state dict."""
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    @staticmethod
    def save_snapshot(state: dict[str, torch.Tensor], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)

    # ------------------------------------------------------------------
    #  Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
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

    # ------------------------------------------------------------------
    #  Trace measurement
    # ------------------------------------------------------------------

    def measure_trace(self) -> dict[str, float]:
        """Compute the layer-wise trace on a fresh Hessian batch.

        The HVP batch is drawn from a dedicated ``_hvp_loader`` that
        shares the training *dataset* (noisy labels, augmented
        transforms) but uses an independent RNG, so the measurement
        reflects the training distribution -- which is the quantity the
        paper's theory (sections 4-7) speaks about -- without mutating
        the shuffling state of ``train_loader``.  Sampling from
        ``val_loader`` would decouple the trace from the noise regime
        (clean labels, no augmentation, ``shuffle=False`` => the same
        128 examples on every call).

        The model is left in ``eval()`` mode to disable dropout / BN
        stats updates and the previous training gradients are cleared
        afterwards so that the optimiser step is not polluted.
        """
        was_training = self.model.training
        self.model.eval()
        x, y = sample_hessian_batch(self._hvp_loader, self.hessian_batch_size, self.device)
        traces = self.estimator.estimate(self.model, self.layers, x, y, self.loss_fn)
        self.model.zero_grad(set_to_none=True)
        if was_training:
            self.model.train()
        return traces

    # ------------------------------------------------------------------
    #  Training loop
    # ------------------------------------------------------------------

    def train(
        self,
        checkpoint_epochs: Iterable[tuple[str, int]],
        jsonl_path: Path,
        snapshot_dir: Path,
    ) -> dict[str, Any]:
        """Run ``self.epochs`` training epochs.

        Parameters
        ----------
        checkpoint_epochs
            Iterable of ``(label, epoch_index)`` -- a state dict is saved
            to ``snapshot_dir/{label}.pt`` after the named epoch completes.
            ``epoch_index = 0`` saves *before* training starts.
        jsonl_path
            Destination for the per-step JSONL trace stream.
        snapshot_dir
            Directory that will hold the ``{label}.pt`` checkpoints and
            their accompanying ``{label}.meta.json`` files.
        """
        cps = dict(checkpoint_epochs)
        snapshots: dict[str, dict[str, Any]] = {}

        with JSONLWriter(jsonl_path) as stream:
            # --- "init" checkpoint -----------------------------------
            for label, ep in cps.items():
                if ep == 0:
                    snapshots[label] = self._record_checkpoint(label, ep, snapshot_dir)

            for epoch in range(1, self.epochs + 1):
                self.model.train()
                for x, y in self.train_loader:
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

                    if (
                        self.metric_interval_steps > 0
                        and self._global_step % self.metric_interval_steps == 0
                    ):
                        t0 = time.perf_counter()
                        traces = self.measure_trace()
                        stream.write(
                            {
                                "step": self._global_step,
                                "epoch": epoch,
                                "train_loss": loss.item(),
                                "trace": traces,
                                "wallclock_s": time.perf_counter() - t0,
                            }
                        )

                for label, ep in cps.items():
                    if ep == epoch:
                        snapshots[label] = self._record_checkpoint(label, ep, snapshot_dir)

        return {
            "epochs": self.epochs,
            "global_steps": self._global_step,
            "snapshots": snapshots,
            "final_eval": self.evaluate(),
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _record_checkpoint(self, label: str, epoch: int, snapshot_dir: Path) -> dict[str, Any]:
        state = self.snapshot()
        path = snapshot_dir / f"{label}.pt"
        self.save_snapshot(state, path)
        eval_metrics = self.evaluate()
        return {
            "label": label,
            "epoch": epoch,
            "step": self._global_step,
            "path": str(path),
            **eval_metrics,
        }


# ---------------------------------------------------------------------------
#  Restore helper (used by runner_exp_a)
# ---------------------------------------------------------------------------


def load_checkpoint(model: nn.Module, path: Path) -> nn.Module:
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model


def clone_model_to(model: nn.Module, device: torch.device) -> nn.Module:
    """Deep-copy a model and move it to ``device``.

    Used by experiment A to evaluate each estimator on an independent
    model copy, so that any in-place mutations from a buggy estimator
    cannot contaminate the next.
    """
    new = deepcopy(model)
    return new.to(device)
