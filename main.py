"""CLI entry point for stochastic-check.

Four subcommands:

  * ``train``     -- run a single training cell (experiment B).
  * ``compare``   -- run the post-hoc estimator comparison (experiment A)
                     on a previously trained cell.
  * ``sweep``     -- enumerate the full matrix and dispatch via
                     :mod:`lab_orchestrator`; this is what ``run_all.sh``
                     ultimately invokes.
  * ``aggregate`` -- collapse per-run JSONs into a single index.

Every subcommand carries its own ``--help``; the top-level ``--help``
lists them with a one-line summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from stochastic_check.aggregate import aggregate_comparisons, aggregate_runs
from stochastic_check.config import (
    EstimatorConfig,
    RunConfig,
    SweepConfig,
    TrainingConfig,
)
from stochastic_check.runner_exp_a import CompareSpec, ExpARunner
from stochastic_check.runner_exp_b import ExpBRunner, ExpBSpec
from stochastic_check.utils import dump_json, run_slug

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  train
# ---------------------------------------------------------------------------


def _add_train_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "train",
        help="Run a single training cell (exp B).",
        description=(
            "Train one ``(arch, dataset, noise, seed)`` cell and stream the "
            "per-step trace trajectory to ``steps.jsonl``. State-dict "
            "snapshots are saved at the requested checkpoint epochs and "
            "a ``final.json`` summary is dumped at the end."
        ),
        epilog=(
            "Example:\n"
            "  python -m main train --arch resnet18 --dataset cifar10 "
            "--noise 0.40 --seed 0 --epochs 30"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arch",
        required=True,
        help="Model architecture; one of: resnet18, vgg11, resnet34.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100"],
        help="Training dataset.",
    )
    p.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Symmetric label-noise rate in [0, 1) (default: %(default)s).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed for data, model init and probe RNG (default: %(default)s).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: %(default)s).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training mini-batch size (default: %(default)s).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Initial SGD learning rate (default: %(default)s).",
    )
    p.add_argument(
        "--metric-interval",
        type=int,
        default=100,
        help=(
            "Optimisation steps between per-step trace measurements; "
            "0 disables the stream (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--n-probes",
        type=int,
        default=10,
        help=(
            "Probes per per-step Hutchinson estimate (default: %(default)s); "
            "the post-hoc Exp-A comparison uses its own probe budgets."
        ),
    )
    p.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=[0, 10, 30],
        metavar="EPOCH",
        help=(
            "Epoch indices at which to save state-dict snapshots; "
            "the smallest becomes the `init` checkpoint, the largest "
            "becomes `final`, and intermediates are named `epNN`. "
            "Paper protocol: 0 10 30 (default: %(default)s)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp_b/runs"),
        help="Root directory for per-run artefacts (default: %(default)s).",
    )


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = RunConfig(
        arch=args.arch,
        dataset=args.dataset,
        training=TrainingConfig(
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            seed=args.seed,
            noise_rate=args.noise,
            metric_interval_steps=args.metric_interval,
            checkpoint_epochs=list(args.checkpoint_epochs),
        ),
        estimator=EstimatorConfig(n_probes=args.n_probes),
        output_dir=str(args.output_dir),
    )
    runner = ExpBRunner(ExpBSpec(cfg=cfg))
    payload = runner.run()
    logger.info(
        "Done: val_acc=%.4f val_loss=%.4f wallclock=%.1fs",
        payload["train_summary"]["final_eval"]["val_acc"],
        payload["train_summary"]["final_eval"]["val_loss"],
        payload["wallclock_s"],
    )
    return 0


# ---------------------------------------------------------------------------
#  compare
# ---------------------------------------------------------------------------


def _add_compare_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compare",
        help="Run estimator comparison on snapshots (exp A).",
        description=(
            "Replay every estimator listed in section 6 of the paper "
            "(Hutchinson K=5/10, Hutch++, K-FAC) on the snapshots produced "
            "by a prior ``train`` run for the same cell. The reference is "
            "Hutchinson with --reference-probes probes; relative errors "
            "are computed per layer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arch",
        required=True,
        help="Architecture of the trained cell to compare on.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        choices=["cifar10", "cifar100"],
        help="Dataset of the trained cell to compare on.",
    )
    p.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Noise rate of the trained cell (default: %(default)s).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed of the trained cell (default: %(default)s).",
    )
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/exp_b/runs"),
        help=(
            "Root under which the prior ``train`` run wrote its snapshots "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/exp_a/runs"),
        help="Where to write ``comparison.json`` (default: %(default)s).",
    )
    p.add_argument(
        "--hessian-batch",
        type=int,
        default=256,
        help=(
            "Validation-set batch size used as the Hessian batch "
            "(shared across every estimator; default: %(default)s)."
        ),
    )
    p.add_argument(
        "--reference-probes",
        type=int,
        default=200,
        help=(
            "Probe count for the Hutchinson reference surrogate of "
            "the exact trace (default: %(default)s; cf. paper sec. 5.3)."
        ),
    )


def _cmd_compare(args: argparse.Namespace) -> int:
    slug = run_slug(arch=args.arch, dataset=args.dataset, noise=args.noise, seed=args.seed)
    snapshot_dir = args.runs_root / slug / "snapshots"
    if not snapshot_dir.exists():
        logger.error("Snapshot directory not found: %s", snapshot_dir)
        return 2
    cfg = RunConfig(
        arch=args.arch,
        dataset=args.dataset,
        training=TrainingConfig(seed=args.seed, noise_rate=args.noise),
    )
    spec = CompareSpec(
        cfg=cfg,
        snapshot_dir=snapshot_dir,
        output_dir=args.output_dir,
        hessian_batch_size=args.hessian_batch,
        reference_probes=args.reference_probes,
    )
    ExpARunner(spec).run()
    return 0


# ---------------------------------------------------------------------------
#  sweep
# ---------------------------------------------------------------------------


def _add_sweep_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "sweep",
        help="Dispatch full sweep via lab_orchestrator.",
        description=(
            "Enumerate the cartesian product archs x datasets x noises x "
            "seeds and schedule one task per cell across the detected "
            "GPUs via :mod:`lab_orchestrator`. Heaviest cells run first "
            "(LPT weights baked into the runner)."
        ),
        epilog=(
            "Examples:\n"
            "  # full publication sweep (paper defaults)\n"
            "  python -m main sweep --also-compare\n"
            "  # rerun only Exp-A on an already-trained matrix\n"
            "  python -m main sweep --only-stage compare"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--archs",
        nargs="+",
        default=["resnet18", "vgg11", "resnet34"],
        metavar="ARCH",
        help="Architectures to enumerate (default: %(default)s).",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["cifar10", "cifar100"],
        metavar="DATASET",
        help="Datasets to enumerate (default: %(default)s).",
    )
    p.add_argument(
        "--noises",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.40, 0.60],
        metavar="ETA",
        help="Symmetric label-noise rates to enumerate (default: %(default)s).",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(10)),
        metavar="SEED",
        help="Seeds to enumerate (default: 0..9).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Epoch budget passed to every train task (default: %(default)s).",
    )
    p.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="Explicit GPU ids; defaults to all detected via nvidia-smi.",
    )
    p.add_argument(
        "--workers-per-gpu",
        type=int,
        default=1,
        help="Concurrent tasks per GPU (default: %(default)s).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/exp_b"),
        help=(
            "Root under which per-cell ``runs/`` and snapshots are written "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--also-compare",
        action="store_true",
        help="After every train task, schedule an estimator-comparison task.",
    )
    p.add_argument(
        "--only-stage",
        choices=["train", "compare", "both"],
        default="both",
        help=(
            "Restrict scheduled tasks to one stage. ``compare`` requires "
            "snapshots from a prior ``train`` sweep on the same matrix; "
            "this replaces the legacy ``grep | awk`` filter in run_all.sh "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=[0, 10, 30],
        metavar="EPOCH",
        help=(
            "Forwarded to every per-cell train task; see ``train --help`` "
            "(default: %(default)s)."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task matrix without executing.",
    )


def _cmd_sweep(args: argparse.Namespace) -> int:
    from lab_orchestrator import (  # local import: not always installed at lint time
        Task,
        detect_gpus,
        run_schedule,
    )

    sweep = SweepConfig(
        archs=args.archs,
        datasets=args.datasets,
        noise_rates=args.noises,
        seeds=args.seeds,
        epochs=args.epochs,
        output_root=str(args.output_root),
    )

    cells = [
        (arch, dataset, noise, seed)
        for arch in sweep.archs
        for dataset in sweep.datasets
        for noise in sweep.noise_rates
        for seed in sweep.seeds
    ]

    def _train_cmd(arch: str, dataset: str, noise: float, seed: int) -> list[str]:
        return [
            sys.executable,
            "-m",
            "main",
            "train",
            "--arch",
            arch,
            "--dataset",
            dataset,
            "--noise",
            str(noise),
            "--seed",
            str(seed),
            "--epochs",
            str(sweep.epochs),
            "--checkpoint-epochs",
            *[str(e) for e in args.checkpoint_epochs],
            "--output-dir",
            str(args.output_root / "runs"),
        ]

    def _compare_cmd(arch: str, dataset: str, noise: float, seed: int) -> list[str]:
        return [
            sys.executable,
            "-m",
            "main",
            "compare",
            "--arch",
            arch,
            "--dataset",
            dataset,
            "--noise",
            str(noise),
            "--seed",
            str(seed),
            "--runs-root",
            str(args.output_root / "runs"),
            "--output-dir",
            "results/exp_a/runs",
        ]

    # LPT weight: estimated wallclock hours per cell on the target GPU
    # tier (4090 for ResNet-18 / VGG-11, A6000 for ResNet-34) at 30
    # epochs.  These are the figures used in the plan and let the
    # scheduler dispatch the heaviest cells first.
    _ARCH_HOURS: dict[str, float] = {
        "resnet18": 1.5,
        "vgg11": 1.0,
        "resnet34": 2.5,
    }

    tasks: list[Task] = []
    schedule_train = args.only_stage in ("train", "both")
    schedule_compare = args.only_stage in ("compare", "both") or args.also_compare
    for arch, dataset, noise, seed in cells:
        name = run_slug(arch=arch, dataset=dataset, noise=noise, seed=seed)
        weight = _ARCH_HOURS.get(arch, 1.0)
        train_name = f"train:{name}"
        if schedule_train:
            tasks.append(
                Task(
                    weight=weight,
                    name=train_name,
                    seed=seed,
                    cmd=_train_cmd(arch, dataset, noise, seed),
                )
            )
        if schedule_compare:
            tasks.append(
                Task(
                    weight=weight * 0.5,
                    name=f"compare:{name}",
                    seed=seed,
                    cmd=_compare_cmd(arch, dataset, noise, seed),
                    # When training and comparison are scheduled together,
                    # the comparison reads snapshots written by training and
                    # must wait for it to finish.
                    depends_on=[train_name] if schedule_train else [],
                )
            )

    if args.dry_run:
        for t in tasks:
            print(t.weight, t.name, " ".join(t.cmd))
        return 0

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else detect_gpus()
    if not gpu_ids:
        logger.error("No GPUs available; aborting sweep.")
        return 1
    logger.info("Dispatching %d tasks across GPUs %s", len(tasks), gpu_ids)
    run_schedule(
        gpu_ids=gpu_ids,
        tasks=tasks,
        workers_per_gpu=args.workers_per_gpu,
    )
    return 0


# ---------------------------------------------------------------------------
#  aggregate
# ---------------------------------------------------------------------------


def _add_aggregate_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "aggregate",
        help="Aggregate per-run JSONs into one index.",
        description=(
            "Walk ``--root`` for per-cell ``final.json`` (exp_b) or "
            "``comparison.json`` (exp_a) files and collapse them into a "
            "single aggregate index written to ``--out``."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory tree containing the per-run JSON artefacts.",
    )
    p.add_argument(
        "--kind",
        choices=["exp_a", "exp_b"],
        default="exp_b",
        help=(
            "Aggregation schema: ``exp_b`` (training runs) or "
            "``exp_a`` (estimator comparisons); default: %(default)s."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Destination JSON path for the aggregate index.",
    )


def _cmd_aggregate(args: argparse.Namespace) -> int:
    agg = aggregate_runs(args.root) if args.kind == "exp_b" else aggregate_comparisons(args.root)
    dump_json(agg, args.out)
    logger.info("Wrote %s (%d runs)", args.out, agg["n_runs"])
    return 0


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="stochastic-check",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(
        dest="cmd",
        required=True,
        metavar="{train,compare,sweep,aggregate}",
        title="subcommands",
        description="Use ``<subcommand> --help`` for per-stage flag documentation.",
    )
    _add_train_subparser(sub)
    _add_compare_subparser(sub)
    _add_sweep_subparser(sub)
    _add_aggregate_subparser(sub)
    args = parser.parse_args(argv)

    return {
        "train": _cmd_train,
        "compare": _cmd_compare,
        "sweep": _cmd_sweep,
        "aggregate": _cmd_aggregate,
    }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
