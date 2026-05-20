# stochastic-check

Experiments on **stochastic layer-wise Hessian trace estimation under
label noise** for the Нейроинформатика-2026 paper.

The codebase is organised by clear separation of concerns:

| package / dir          | responsibility                                                 |
|------------------------|----------------------------------------------------------------|
| `hessian/`             | numerical kernel (estimator algorithms)                        |
| `stochastic_check/`    | training loop, data, models, runners (experiment A and B)      |
| `lab_orchestrator/`    | multi-GPU scheduler (verbatim copy)                            |
| `paper_data/`          | LaTeX-table generators (one module per table)                  |
| `main.py`              | command-line entry point (train / compare / sweep / aggregate) |
| `run_all.sh`           | top-level driver: `diagnostic_run` and `full_run` modes        |

## Estimators

All estimators implement `hessian.trace.TraceEstimator` and operate
**layer-wise** on the parameter-space block-diagonal trace
`T_ell = tr(H_{ell,ell})` via a single-pass HVP trick:

  - `HutchinsonTraceEstimator`  -- single-pass Rademacher,
    `T_ell ~= (1/K) sum_k z_{k,ell}^T (H z_k)_ell`.
  - `HutchPlusPlusTraceEstimator` (Meyer 2021) -- low-rank +
    residual Hutchinson; `O(1/n^2)` variance on rank-deficient H.
  - `BeckerLeCunDiagonalEstimator` -- stochastic diagonal; algebraically
    identical to Hutchinson at the same K, exposed separately to mirror
    the cited literature.
  - `KFACTraceEstimator` -- `tr(F_ell) ~= tr(A_ell) * tr(G_ell)`
    via ASDL (`asdfghjkl`). **Caveat:** this approximates the
    Fisher / Gauss-Newton matrix, not the full Hessian; the gap
    `H - F` is not captured.

## Quickstart

```bash
# 1. Install
uv sync

# 2. Smoke test (single GPU, ~5 min)
bash run_all.sh diagnostic_run

# 3. Full sweep (240 cells across 8 GPUs)
bash run_all.sh full_run

# 4. Rebuild LaTeX tables
bash run_all.sh tables
```

Useful overrides:

```bash
GPUS="0,1,2,3" WORKERS=2 bash run_all.sh full_run
```

## Single-cell train + compare

```bash
python -m main train  --arch resnet18 --dataset cifar10 --noise 0.40 --seed 0 --epochs 30
python -m main compare --arch resnet18 --dataset cifar10 --noise 0.40 --seed 0
```

## Output layout

```
results/
  exp_b/
    runs/arch=resnet18_dataset=cifar10_noise=0.40_seed=0/
      final.json          # config + train summary + checkpoint meta
      steps.jsonl         # per-step trace trajectory (streaming)
      snapshots/{init,mid,final}.pt
    aggregate.json        # index over all runs
  exp_a/
    runs/<slug>/comparison.json
    aggregate.json

paper_data/tables_out/
  tab_comparison.tex
  tab_cohens_d.tex
```

## Reproducibility

Every run sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`torch.use_deterministic_algorithms(True)`, and cudnn deterministic
mode; the same seed gives bit-identical results on the same hardware.

## Testing

```bash
make test    # runs the synthetic-H estimator unit tests
make check   # lint + typecheck + tests
```
# stochastic_check
