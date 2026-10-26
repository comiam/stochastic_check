"""Keep the experiment protocol and per-run configuration in dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Training defaults from paper section 7.1.

    SGD uses momentum 0.9, lr 0.1, batch size 128, and weight decay 5e-4.
    The stated L2 convention adds the constant Hessian term 2*lambda*I.
    The cosine schedule advances per step; gradient clipping is disabled.
    """

    lr: float = 0.1
    batch_size: int = 128
    epochs: int = 30
    momentum: float = 0.9
    weight_decay: float = 5e-4

    scheduler: str = "cosine"
    grad_clip: float | None = None
    metric_interval_steps: int = 100
    seed: int = 42
    noise_rate: float = 0.0

    # Explicit epochs override the legacy symbolic labels.
    checkpoint_epochs: list[int] | None = None
    checkpoint_labels: list[str] = field(
        default_factory=lambda: ["init", "mid", "final"]
    )


@dataclass
class EstimatorConfig:
    """Configure the periodic training estimator and its Hessian batch.

    Hutchinson K=10 is used during training. The other estimator choices
    are compared separately on saved checkpoints.
    """

    name: str = "hutchinson"
    n_probes: int = 10
    n_orth_frac: float = 1.0 / 3.0
    hessian_batch_size: int = 128


@dataclass
class RunConfig:
    """Describe one architecture/dataset/noise/seed cell for serialization."""

    arch: str
    dataset: str
    training: TrainingConfig = field(default_factory=TrainingConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    output_dir: str = "results/exp_b/runs"

    @property
    def num_classes(self) -> int:
        """Return the class count of the selected CIFAR dataset."""
        return {"cifar10": 10, "cifar100": 100}[self.dataset]


@dataclass
class SweepConfig:
    """Describe the ordered product of architectures, datasets, noise, and seeds."""

    archs: list[str] = field(default_factory=lambda: ["resnet18", "vgg11", "resnet34"])
    datasets: list[str] = field(default_factory=lambda: ["cifar10", "cifar100"])
    noise_rates: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.40, 0.60])
    seeds: list[int] = field(default_factory=lambda: list(range(10)))
    epochs: int = 30
    output_root: str = "results/exp_b"
