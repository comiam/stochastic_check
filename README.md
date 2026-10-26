# Stochastic Estimation of the Layer-wise Hessian Trace

[![arXiv](https://img.shields.io/badge/arXiv-2605.25674-b31b1b.svg)](https://arxiv.org/abs/2605.25674)

Research code and experimental records for **Stochastic Estimation of the
Layer-wise Hessian Trace for Monitoring Neural-network Training**.

**Maksim Bolshim, Alexander Kugaevskikh**

Accepted for publication in _Optical Memory and Neural Networks (Information
Optics)_ on **7 September 2026**.

[Preprint](https://arxiv.org/abs/2605.25674) |
[Paper PDF](https://arxiv.org/pdf/2605.25674) |
[Reproduce results](#reproduce-the-paper-tables) |
[Citation](#citation)

## Overview

The project studies layer-wise loss curvature as a signal for monitoring
neural-network training under symmetric label noise. It combines stochastic
Hessian-trace estimation with a calibrated cumulative-sum (CUSUM) decision
rule to detect the label-memorisation regime.

- **Estimator:** a global Hessian-vector product (HVP) gives estimates for
  every layer without constructing the full Hessian. Hutchinson estimates
  are unbiased for the parameter-space layer blocks, including shared weights.
- **Experiments:** CIFAR-adapted ResNet-18, ResNet-34, and VGG-11 on CIFAR-10
  and CIFAR-100, with label-noise rates of 0, 0.25, 0.40, and 0.60.
- **Comparisons:** Hutchinson, Hutch++, Becker-LeCun diagonal, and K-FAC;
  detection baselines based on gradient norm, training loss, and validation loss.
- **Recorded results:** JSON/JSONL trajectories and manifests sufficient to
  regenerate the alert, matched-false-alarm-rate, and timing-overhead tables.

The original alert experiment detects **179 of 180 noisy runs**, with false
alarms in **16 of 120 clean runs**. These are empirical results for the
reported experiment, not a general detection guarantee.

## Installation

Use Python 3.12, [uv](https://docs.astral.sh/uv/), GNU Make, and Bash.
From the repository root:

```bash
git clone https://github.com/comiam/stochastic_check.git
cd stochastic_check
make install
```

`make install` creates the environment from `uv.lock`. Python source supports
3.10 and newer; the locked development and reproduction environment uses 3.12.
A CUDA-capable GPU is recommended for training and required by the sweep
scheduler. Rebuilding tables and running the tests do not require a GPU.

## Reproduce the paper tables

After installation:

```bash
make data-check
make tables
make paper-check
```

No training run, CIFAR download, or model checkpoint is needed. `make data-check`
verifies the bundled SHA-256 manifest; `make paper-check` checks the recorded
values and generated tables against regression fixtures.

The generated LaTeX files are written to `build/paper_tables/`:

| Table               | Bundled inputs | Calculation                         |
| ------------------- | -------------- | ----------------------------------- |
| `tab_alerts.tex`    | `exp_b`        | `paper_data.tables.decision_alerts` |
| `tab_baselines.tex` | `exp_b_rerun`  | `paper_data.tables.baselines`       |
| `tab_overhead.tex`  | `exp_b_rerun`  | `paper_data.tables.overhead`        |

### Experimental records

[`paper_data/artifacts/`](paper_data/artifacts/README.md) contains two sets of
300 runs each. Each set covers six architecture/dataset pairs, with 20 clean
seeds and 10 seeds at each nonzero noise rate per pair. Each run contains
117 measurements.

**The original and rerun records are not interchangeable.** The alert table
uses the original `exp_b` trajectories; the baseline and overhead tables use
`exp_b_rerun`, which includes gradient norms, validation metrics, and timings.
The rerun is not claimed to be numerically identical to the original.

Checkpoints, raw CIFAR data, logs, and temporary outputs are not bundled.
Historical checkpoint paths in manifests describe the source runs and are
not needed for table generation. The overhead calculation uses recorded
timings; it does not run a new hardware benchmark.

To calculate an individual table from another results root with the same layout:

```bash
uv run --locked python -m paper_data.main \
    --label tab:alerts --results-root results
```

The additional trace/gradient complementarity analysis uses the bundled rerun
records by default:

```bash
uv run --locked python -m paper_data.tables.complementarity
```

## Run experiments

Recomputing training trajectories is separate from rebuilding tables. It
requires dataset downloads, checkpoint storage, and substantially more compute.
Run a short training/comparison smoke test before launching a full sweep:

```bash
make diagnostic
```

Train the experiment matrix, then extend the clean calibration ensembles:

```bash
make exp_b GPUS=0,1 WORKERS=1
make phase1_extension GPUS=0,1 WORKERS=1
```

- `exp_b` trains the original 240-run matrix: three architectures, two
  datasets, four noise rates, and seeds 0 through 9, for 30 epochs.
- `phase1_extension` adds 60 clean runs with seeds 10 through 19, giving
  20 clean seeds per architecture/dataset pair and 300 runs in total.
- `full` runs the original matrix plus post-hoc estimator comparisons.
- `exp_a` compares estimators using existing checkpoints; the bundled
  records alone are not sufficient for this stage.

`GPUS` selects devices, `WORKERS` sets concurrent workers per GPU, and
`SEEDS="0 1 2"` restricts the original sweep's seed axis. The extension uses
`EXT_SEEDS` instead, which must not overlap the original seeds.

A single training cell and its post-hoc comparison can also be run directly:

```bash
uv run --locked python -m main train \
    --arch resnet18 --dataset cifar10 --noise 0.40 --seed 0
uv run --locked python -m main compare \
    --arch resnet18 --dataset cifar10 --noise 0.40 --seed 0
```

Training writes `final.json`, `steps.jsonl`, and snapshots under
`results/exp_b/runs/<cell>/`. Working outputs are separate from the immutable
paper inputs in `paper_data/artifacts/`.

Use `make help`, `bash run_all.sh --help`, or the CLI's `--help` options to
inspect available commands.

## Estimators and reproducibility notes

All trace estimators implement `hessian.trace.TraceEstimator`.

| Estimator                    | Interpretation                                                                                                                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Hutchinson                   | Mean Rademacher quadratic forms; one global HVP per probe, with layer-wise slices.                                                                                    |
| Hutch++ (Meyer et al., 2021) | Low-rank plus projected residual trace estimation. The one-third budget split uses 13 HVPs for `n_total=10`.                                                          |
| Becker-LeCun diagonal        | The sum of the stochastic diagonal estimate is algebraically Hutchinson at the same probe count.                                                                      |
| K-FAC through ASDL           | Factorized Fisher/Gauss-Newton curvature, not the full Hessian. The Hessian-Fisher gap is not captured; unsupported layers such as BatchNorm have no factor estimate. |

Training enables deterministic algorithms and fixed cuBLAS workspace settings.
Probe and Hessian-batch generators are separate from the training sampler.
Reproducibility still depends on the software and hardware environment:
retraining is not a promise of cross-environment bitwise identity. Use the
bundled records to reproduce the reported table values.

## Repository structure

| Path                | Contents                                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `hessian/`          | Trace estimators, exact/stochastic Hessian blocks, and Gauss-Newton decomposition. |
| `stochastic_check/` | Models, data, training, detection, and experiment runners.                         |
| `lab_orchestrator/` | Single-node GPU scheduling and worker processes.                                   |
| `paper_data/`       | Record loading, table calculations, and bundled experimental inputs.               |
| `tests/`            | Refactor-preservation, CLI, data-integrity, and paper-table regression tests.      |
| `main.py`           | Training, comparison, sweep, and aggregation CLI.                                  |
| `run_all.sh`        | Experiment presets used by the Makefile.                                           |

## Development

```bash
make test       # CPU tests, including numerical and paper-table regressions
make lint       # Ruff lint, docstrings, formatting, and shell syntax
make typecheck  # mypy
make check      # All of the above plus bundled-data checksums
make format     # Safe Ruff fixes and formatting
```

`make clean` removes generated tables and project caches. It preserves the
bundled data, virtual environment, manuscript, and checkpoints.

## Citation

If you use this code or the experimental records, please cite the paper.
BibTeX for the arXiv preprint:

```bibtex
@misc{bolshim2026stochastic,
  title         = {Stochastic Estimation of the Layer-wise Hessian Trace
                   for Monitoring Neural-network Training},
  author        = {Bolshim, Maxim and Kugaevskikh, Alexander},
  year          = {2026},
  eprint        = {2605.25674},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi           = {10.48550/arXiv.2605.25674},
  url           = {https://arxiv.org/abs/2605.25674}
}
```

## License

The source code and bundled experimental records are licensed under the
[MIT License](LICENSE). If you use them in research, please cite the paper;
scientific citation is requested, not an additional license condition.

The datasets and dependencies are not
covered by this license and remain subject to their respective terms.
