#!/usr/bin/env bash
# Top-level runner for the stochastic-check experiments.
#
# Modes
# -----
#   diagnostic_run  --  small smoke test, single GPU, 1 cell, 2 epochs.
#                       Used to validate the full pipeline before
#                       launching a multi-day sweep.
#   full_run        --  the publication sweep:
#                         archs    = {resnet18, vgg11, resnet34}
#                         datasets = {cifar10, cifar100}
#                         noises   = {0.0, 0.25, 0.40, 0.60}
#                         seeds    = {0, ..., 9}
#                       240 cells; dispatched across all visible GPUs
#                       via lab_orchestrator.
#
# Auxiliary modes
# ---------------
#   exp_a   --  rerun only the post-hoc estimator-comparison stage on
#               an already-trained set of cells.
#   exp_b   --  alias for full_run (training-only).
#   tables  --  rebuild every paper table from results/.
#
# Environment overrides
# ---------------------
#   GPUS="0,1,2,3"   restrict to a subset of GPUs.
#   WORKERS=2        workers per GPU (default: 1).

set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-diagnostic_run}"

# ---- GPU selection ---------------------------------------------------------

if [[ -n "${GPUS:-}" ]]; then
    GPU_ARGS=(--gpu-ids ${GPUS//,/ })
else
    GPU_ARGS=()
fi
WORKERS="${WORKERS:-1}"

# ---- Mode dispatch ---------------------------------------------------------

run_diagnostic() {
    echo "[diagnostic] single-cell smoke test: ResNet-18 / CIFAR-10 / eta=0.40 / seed=0"
    python -m main train \
        --arch resnet18 \
        --dataset cifar10 \
        --noise 0.40 \
        --seed 0 \
        --epochs 2 \
        --metric-interval 50 \
        --output-dir results/diagnostic/runs
    python -m main compare \
        --arch resnet18 \
        --dataset cifar10 \
        --noise 0.40 \
        --seed 0 \
        --runs-root results/diagnostic/runs \
        --output-dir results/diagnostic/comparison \
        --hessian-batch 64 \
        --reference-probes 30
    python -m main aggregate \
        --kind exp_b \
        --root results/diagnostic/runs \
        --out  results/diagnostic/aggregate.json
    echo "[diagnostic] OK"
}

run_full_training() {
    echo "[full_run] dispatching full sweep (archs x datasets x noises x seeds)"
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 0.25 0.40 0.60 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --epochs 30 \
        --workers-per-gpu "$WORKERS" \
        --output-root results/exp_b \
        --also-compare \
        "${GPU_ARGS[@]}"
    python -m main aggregate --kind exp_b \
        --root results/exp_b/runs --out results/exp_b/aggregate.json
    python -m main aggregate --kind exp_a \
        --root results/exp_a/runs --out results/exp_a/aggregate.json
    echo "[full_run] OK"
}

run_exp_a_only() {
    echo "[exp_a] re-running estimator-comparison stage only"
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 0.25 0.40 0.60 \
        --seeds 0 1 2 3 4 5 6 7 8 9 \
        --epochs 30 \
        --workers-per-gpu "$WORKERS" \
        --output-root results/exp_b \
        --also-compare \
        --dry-run \
        "${GPU_ARGS[@]}" \
        | grep '^.* compare:' \
        | awk '{$1=""; $2=""; print}' \
        | xargs -I {} -P 1 bash -c '{}'
    python -m main aggregate --kind exp_a \
        --root results/exp_a/runs --out results/exp_a/aggregate.json
    echo "[exp_a] OK"
}

run_tables() {
    echo "[tables] regenerating paper artefacts"
    python -m paper_data.main --all
    echo "[tables] OK"
}

case "$MODE" in
    diagnostic_run) run_diagnostic ;;
    full_run)       run_full_training ;;
    exp_a)          run_exp_a_only ;;
    exp_b)          run_full_training ;;
    tables)         run_tables ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Usage: $0 {diagnostic_run|full_run|exp_a|exp_b|tables}" >&2
        exit 2
        ;;
esac
