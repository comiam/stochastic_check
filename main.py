"""Run single training cells, post-hoc comparisons, sweeps, or aggregation."""

from __future__ import annotations

import argparse
import logging
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


def _add_cell_options(parser: argparse.ArgumentParser) -> None:
    """Describe the architecture, dataset, noise, and seed of a single cell."""
    parser.add_argument("--arch", required=True, help="resnet18, vgg11, or resnet34.")
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    parser.add_argument(
        "--noise", type=float, default=0.0, help="Symmetric label-noise rate."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Master seed for the cell."
    )


def _add_checkpoint_options(parser: argparse.ArgumentParser) -> None:
    """Configure explicit snapshot epochs without changing their labels."""
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        nargs="+",
        default=[0, 10, 30],
        metavar="EPOCH",
        help="Smallest epoch is init, largest is final, intermediate epochs are epNN.",
    )


def _add_training_options(parser: argparse.ArgumentParser) -> None:
    """Configure the optimizer budget and periodic measurements."""
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)

    parser.add_argument(
        "--metric-interval",
        type=int,
        default=100,
        help="Steps between measurements; 0 disables the stream.",
    )
    parser.add_argument("--n-probes", type=int, default=10)


def _add_train_subparser(sub: argparse._SubParsersAction) -> None:
    """Register single-cell training options."""
    parser = sub.add_parser("train", help="Run a single training cell (exp B).")
    _add_cell_options(parser)
    _add_training_options(parser)
    _add_checkpoint_options(parser)
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp_b/runs"))


def _training_config(args: argparse.Namespace) -> RunConfig:
    """Translate training arguments to the existing experiment configuration."""
    training = TrainingConfig(
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        noise_rate=args.noise,
        metric_interval_steps=args.metric_interval,
        checkpoint_epochs=list(args.checkpoint_epochs),
    )
    return RunConfig(
        arch=args.arch,
        dataset=args.dataset,
        training=training,
        estimator=EstimatorConfig(n_probes=args.n_probes),
        output_dir=str(args.output_dir),
    )


def _cmd_train(args: argparse.Namespace) -> int:
    """Train one cell and report its final validation metrics."""
    config = _training_config(args)
    payload = ExpBRunner(ExpBSpec(cfg=config)).run()
    evaluation = payload["train_summary"]["final_eval"]
    logger.info(
        "Done: val_acc=%.4f val_loss=%.4f wallclock=%.1fs",
        evaluation["val_acc"],
        evaluation["val_loss"],
        payload["wallclock_s"],
    )
    return 0


def _add_compare_subparser(sub: argparse._SubParsersAction) -> None:
    """Register estimator-comparison inputs and budgets."""
    parser = sub.add_parser(
        "compare", help="Run estimator comparison on snapshots (exp A)."
    )
    _add_cell_options(parser)
    parser.add_argument("--runs-root", type=Path, default=Path("results/exp_b/runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/exp_a/runs"))

    parser.add_argument("--hessian-batch", type=int, default=256)
    parser.add_argument("--reference-probes", type=int, default=200)


def _cmd_compare(args: argparse.Namespace) -> int:
    """Replay estimators on one cell's saved snapshots."""
    slug = run_slug(
        arch=args.arch, dataset=args.dataset, noise=args.noise, seed=args.seed
    )
    snapshot_dir = args.runs_root / slug / "snapshots"
    if not snapshot_dir.exists():
        logger.error("Snapshot directory not found: %s", snapshot_dir)
        return 2

    config = RunConfig(
        arch=args.arch,
        dataset=args.dataset,
        training=TrainingConfig(seed=args.seed, noise_rate=args.noise),
    )
    spec = CompareSpec(
        cfg=config,
        snapshot_dir=snapshot_dir,
        output_dir=args.output_dir,
        hessian_batch_size=args.hessian_batch,
        reference_probes=args.reference_probes,
    )
    ExpARunner(spec).run()
    return 0


def _add_sweep_matrix(parser: argparse.ArgumentParser) -> None:
    """Configure the four axes of the experiment matrix."""
    parser.add_argument(
        "--archs", nargs="+", default=["resnet18", "vgg11", "resnet34"], metavar="ARCH"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["cifar10", "cifar100"], metavar="DATASET"
    )
    parser.add_argument(
        "--noises",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.40, 0.60],
        metavar="ETA",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(10)), metavar="SEED"
    )


def _add_sweep_execution(parser: argparse.ArgumentParser) -> None:
    """Configure sweep resources, stages, and output paths."""
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=None, metavar="ID")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("results/exp_b"))

    parser.add_argument("--also-compare", action="store_true")
    parser.add_argument(
        "--only-stage", choices=["train", "compare", "both"], default="both"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print tasks without executing."
    )


def _add_sweep_subparser(sub: argparse._SubParsersAction) -> None:
    """Register multi-GPU sweep options."""
    parser = sub.add_parser("sweep", help="Dispatch full sweep via lab_orchestrator.")
    _add_sweep_matrix(parser)
    _add_sweep_execution(parser)
    _add_checkpoint_options(parser)


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Build the existing task matrix and propagate completion failures."""
    from lab_orchestrator import detect_gpus, run_schedule
    from stochastic_check.sweep import build_sweep_tasks, completion_status

    sweep = SweepConfig(
        archs=args.archs,
        datasets=args.datasets,
        noise_rates=args.noises,
        seeds=args.seeds,
        epochs=args.epochs,
        output_root=str(args.output_root),
    )
    tasks = build_sweep_tasks(
        sweep, args.checkpoint_epochs, args.only_stage, args.also_compare
    )

    if args.dry_run:
        for task in tasks:
            print(task.weight, task.name, " ".join(task.cmd))
        return 0

    gpu_ids = args.gpu_ids if args.gpu_ids is not None else detect_gpus()
    if not gpu_ids:
        logger.error("No GPUs available; aborting sweep.")
        return 1

    logger.info("Dispatching %d tasks across GPUs %s", len(tasks), gpu_ids)
    results = run_schedule(
        gpu_ids=gpu_ids, tasks=tasks, workers_per_gpu=args.workers_per_gpu
    )
    return completion_status(tasks, results)


def _add_aggregate_subparser(sub: argparse._SubParsersAction) -> None:
    """Register summary-index inputs and output path."""
    parser = sub.add_parser("aggregate", help="Aggregate per-run JSONs into one index.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=["exp_a", "exp_b"], default="exp_b")
    parser.add_argument("--out", type=Path, required=True)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    """Aggregate existing records without loading model checkpoints."""
    aggregate = (
        aggregate_runs(args.root)
        if args.kind == "exp_b"
        else aggregate_comparisons(args.root)
    )
    dump_json(aggregate, args.out)
    logger.info("Wrote %s (%d runs)", args.out, aggregate["n_runs"])
    return 0


def _parser() -> argparse.ArgumentParser:
    """Build the four-subcommand CLI."""
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
        description="Use <subcommand> --help for options.",
    )

    _add_train_subparser(sub)
    _add_compare_subparser(sub)
    _add_sweep_subparser(sub)
    _add_aggregate_subparser(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch a command, returning its process exit status."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = _parser().parse_args(argv)
    commands = {
        "train": _cmd_train,
        "compare": _cmd_compare,
        "sweep": _cmd_sweep,
        "aggregate": _cmd_aggregate,
    }
    return commands[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
