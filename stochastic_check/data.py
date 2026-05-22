"""Data loading: CIFAR-10/100 with symmetric label noise.

The noise injection follows the standard symmetric-flip protocol used in
the label-noise literature: each training label is replaced, with
probability ``noise_rate``, by a uniformly random label drawn from the
remaining classes.  Clean labels are preserved as
``dataset.clean_targets`` for diagnostics.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
#  Normalisation statistics
# ---------------------------------------------------------------------------


_STATS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}


def _build_transforms(
    dataset: str,
) -> tuple[T.Compose, T.Compose]:
    mean, std = _STATS[dataset]
    train_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    val_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    return train_tf, val_tf


# ---------------------------------------------------------------------------
#  Symmetric label noise
# ---------------------------------------------------------------------------


def corrupt_labels(
    targets: list[int] | np.ndarray,
    noise_rate: float,
    num_classes: int,
    seed: int,
) -> np.ndarray:
    """Inject symmetric label noise.

    For each index ``i``, with probability ``noise_rate`` replace the
    original label by a label sampled uniformly from
    ``{0, ..., num_classes - 1} \\ {original}``.  With probability
    ``1 - noise_rate`` the label is kept.

    Returns
    -------
    np.ndarray of shape ``(N,)`` with the corrupted integer labels.
    """
    if not 0.0 <= noise_rate < 1.0:
        raise ValueError(f"noise_rate must be in [0, 1), got {noise_rate}")
    rng = np.random.default_rng(seed)
    arr = np.asarray(targets, dtype=np.int64).copy()
    if noise_rate == 0.0:
        return arr
    n = len(arr)
    flip_mask = rng.random(n) < noise_rate
    n_flip = int(flip_mask.sum())
    if n_flip == 0:
        return arr
    # For each flipped index pick a new label uniformly from the other classes.
    offsets = rng.integers(1, num_classes, size=n_flip)
    arr[flip_mask] = (arr[flip_mask] + offsets) % num_classes
    return arr


# ---------------------------------------------------------------------------
#  Loaders
# ---------------------------------------------------------------------------


def _get_dataset_cls(dataset: str) -> type[torchvision.datasets.VisionDataset]:
    return {
        "cifar10": torchvision.datasets.CIFAR10,
        "cifar100": torchvision.datasets.CIFAR100,
    }[dataset]


def ensure_datasets_downloaded(data_root: str = "./data") -> None:
    """Single-process pre-download of every dataset used by the sweep.

    Call this *once* before the parallel orchestrator dispatches worker
    processes; concurrent ``download=True`` calls from multiple workers
    race on torchvision's marker file and one of them ends up with a
    half-written archive (``RuntimeError: Dataset not found or
    corrupted``).  The download is a no-op when the archive is already
    present, so the function is safe to call unconditionally.
    """
    for ds_cls in (torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR100):
        ds_cls(data_root, train=True, download=True)
        ds_cls(data_root, train=False, download=True)


def get_loaders(
    dataset: str,
    batch_size: int,
    noise_rate: float,
    noise_seed: int,
    data_root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, int]:
    """Returns ``(train_loader, val_loader, num_classes)``.

    The training-set targets are replaced with ``corrupt_labels(...)``
    before construction; the original clean labels are preserved in
    ``train_loader.dataset.clean_targets`` for diagnostics.
    """
    train_tf, val_tf = _build_transforms(dataset)
    ds_cls = _get_dataset_cls(dataset)

    train_ds = ds_cls(data_root, train=True, download=True, transform=train_tf)
    val_ds = ds_cls(data_root, train=False, download=True, transform=val_tf)

    num_classes = 10 if dataset == "cifar10" else 100
    clean_targets = np.asarray(train_ds.targets, dtype=np.int64).copy()
    noisy_targets = corrupt_labels(
        clean_targets, noise_rate, num_classes, seed=noise_seed
    )
    train_ds.targets = noisy_targets.tolist()
    train_ds.clean_targets = clean_targets  # type: ignore[attr-defined]

    g = torch.Generator()
    g.manual_seed(noise_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
#  Hessian batch sampler
# ---------------------------------------------------------------------------


def sample_hessian_batch(
    loader: DataLoader, batch_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate batches from ``loader`` until ``batch_size`` samples
    are collected, then move to ``device``.

    Used to assemble a fixed-size deterministic batch for the trace
    estimator at each measurement checkpoint -- the batch is reused
    across estimators in the comparison so that variance comes solely
    from the probe distribution.
    """
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    n = 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        n += x.size(0)
        if n >= batch_size:
            break
    x_cat = torch.cat(xs)[:batch_size].to(device, non_blocking=True)
    y_cat = torch.cat(ys)[:batch_size].to(device, non_blocking=True)
    return x_cat, y_cat
