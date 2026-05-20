"""Experiment configuration dataclasses for stochastic-check."""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Hyperparameters for the training loop.

    Defaults follow the protocol in section 7.1 of the paper:
    SGD(momentum=0.9), lr=0.1, batch_size=128, weight_decay=5e-4,
    cosine schedule over the full epoch budget.
    """

    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 30
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"  # "none" | "cosine"

    # Steps between trace measurements (0 disables periodic collection).
    metric_interval_steps: int = 100

    # Per-run seed and label-noise level.
    seed: int = 42
    noise_rate: float = 0.0  # symmetric label-noise probability in [0, 1)

    # Checkpoint epochs requested by the runner (resolved against epochs).
    checkpoint_labels: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


# ---------------------------------------------------------------------------
#  Estimator
# ---------------------------------------------------------------------------


@dataclass
class EstimatorConfig:
    """Choice and hyperparameters of the trace estimator used during training.

    The training loop runs a single estimator (typically the cheap
    single-pass Hutchinson with K=10) at periodic intervals.  The full
    estimator comparison (Hutch++, K-FAC, ...) is done post-hoc by
    ``runner_exp_a`` on saved checkpoints.
    """

    name: str = "hutchinson"  # "hutchinson" | "hutch_plus_plus" | "bl_diag" | "kfac"
    n_probes: int = 10
    n_orth_frac: float = 1.0 / 3.0  # Hutch++ only

    # Number of validation samples used for the Hessian batch.
    hessian_batch_size: int = 128


# ---------------------------------------------------------------------------
#  Per-run experiment
# ---------------------------------------------------------------------------


@dataclass
class RunConfig:
    """A single training run = one ``(arch, dataset, noise, seed)`` cell.

    The orchestrator dispatches one such run per task; this dataclass is
    serialised into the per-run ``final.json`` for full reproducibility.
    """

    arch: str  # "resnet18" | "vgg11" | "resnet34"
    dataset: str  # "cifar10" | "cifar100"
    training: TrainingConfig = field(default_factory=TrainingConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    output_dir: str = "results/exp_b/runs"

    @property
    def num_classes(self) -> int:
        return {"cifar10": 10, "cifar100": 100}[self.dataset]


# ---------------------------------------------------------------------------
#  Sweep specification (matrix over arch x dataset x noise x seed)
# ---------------------------------------------------------------------------


@dataclass
class SweepConfig:
    """A full sweep that expands into one :class:`RunConfig` per cell.

    Used by ``run_all.sh`` and ``main.py sweep`` to materialise the
    matrix and submit it to ``lab_orchestrator``.
    """

    archs: list[str] = field(default_factory=lambda: ["resnet18", "vgg11", "resnet34"])
    datasets: list[str] = field(default_factory=lambda: ["cifar10", "cifar100"])
    noise_rates: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.40, 0.60])
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    epochs: int = 30
    output_root: str = "results/exp_b"
