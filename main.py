"""CLI entry point for stochastic-check.

Three subcommands:

  * ``train``   -- run a single training cell (experiment B).
  * ``compare`` -- run the post-hoc estimator comparison (experiment A)
                   on a previously trained cell.
  * ``sweep``   -- enumerate the full matrix and dispatch via
                   :mod:`lab_orchestrator`; this is what ``run_all.sh``
                   ultimately invokes.
  * ``aggregate`` -- collapse per-run JSONs into a single index.
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
    p = sub.add_parser("train", help="Run a single training cell (exp B).")
    p.add_argument("--arch", required=True)
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--metric-interval", type=int, default=100)
    p.add_argument("--n-probes", type=int, default=10)
    p.add_argument("--output-dir", type=Path, default=Path("results/exp_b/runs"))


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
    p = sub.add_parser("compare", help="Run estimator comparison on snapshots (exp A).")
    p.add_argument("--arch", required=True)
    p.add_argument("--dataset", required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("results/exp_b/runs"),
        help="Where the exp-B snapshots were written.",
    )
    p.add_argument("--output-dir", type=Path, default=Path("results/exp_a/runs"))
    p.add_argument("--hessian-batch", type=int, default=256)
    p.add_argument("--reference-probes", type=int, default=200)


def _cmd_compare(args: argparse.Namespace) -> int:
    slug = run_slug(
        arch=args.arch, dataset=args.dataset, noise=args.noise, seed=args.seed
    )
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
    p = sub.add_parser("sweep", help="Dispatch full sweep via lab_orchestrator.")
    p.add_argument("--archs", nargs="+", default=["resnet18", "vgg11", "resnet34"])
    p.add_argument("--datasets", nargs="+", default=["cifar10", "cifar100"])
    p.add_argument("--noises", nargs="+", type=float, default=[0.0, 0.25, 0.40, 0.60])
    p.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        default=None,
        help="Explicit GPU ids; defaults to all detected.",
    )
    p.add_argument("--workers-per-gpu", type=int, default=1)
    p.add_argument("--output-root", type=Path, default=Path("results/exp_b"))
    p.add_argument(
        "--also-compare",
        action="store_true",
        help="After every train task, schedule an estimator-comparison task.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the task matrix without executing.",
    )


def _cmd_sweep(args: argparse.Namespace) -> int:
    from lab_orchestrator import (  # local import: not always installed at lint time
        Task,
        build_task_matrix,
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

    tasks: list[Task] = []
    for arch, dataset, noise, seed in cells:
        name = run_slug(arch=arch, dataset=dataset, noise=noise, seed=seed)
        # LPT weight: ResNet-34 ~ 2x ResNet-18 cost; cifar-100 ~ same; we
        # use param-count proxy (rough) so the scheduler dispatches the
        # heaviest first.
        weight = {"resnet18": 11.0, "vgg11": 28.0, "resnet34": 21.0}.get(arch, 1.0)
        tasks.append(
            Task(
                weight=weight,
                name=f"train:{name}",
                seed=seed,
                cmd=_train_cmd(arch, dataset, noise, seed),
            )
        )
        if args.also_compare:
            tasks.append(
                Task(
                    weight=weight * 0.5,
                    name=f"compare:{name}",
                    seed=seed,
                    cmd=_compare_cmd(arch, dataset, noise, seed),
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
    # Note: build_task_matrix is the canonical factory in lab_orchestrator; we
    # build tasks manually here because per-cell args are heterogeneous.
    _ = build_task_matrix  # keep import live for future refactor
    return 0


# ---------------------------------------------------------------------------
#  aggregate
# ---------------------------------------------------------------------------


def _add_aggregate_subparser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("aggregate", help="Aggregate per-run JSONs into one index.")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kind", choices=["exp_a", "exp_b"], default="exp_b")
    p.add_argument("--out", type=Path, required=True)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    agg = (
        aggregate_runs(args.root)
        if args.kind == "exp_b"
        else aggregate_comparisons(args.root)
    )
    dump_json(agg, args.out)
    logger.info("Wrote %s (%d runs)", args.out, agg["n_runs"])
    return 0


# ---------------------------------------------------------------------------
#  main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="stochastic-check")
    sub = parser.add_subparsers(dest="cmd", required=True)
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
