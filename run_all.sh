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
#   exp_b   --  training stage only (no estimator comparison); useful
#               when you want to split the budget over multiple sessions
#               and run the comparison separately via ``exp_a``.
#   tables  --  rebuild every paper table from results/.
#
# Environment overrides
# ---------------------
#   GPUS="0,1,2,3"   restrict to a subset of GPUs.
#   WORKERS=2        workers per GPU (default: 1).
#   SEEDS="0 1 2"    restrict the seed axis of the sweep (default: 0..9).
#                    Lets you partition the 240-cell matrix across
#                    sessions / machines without editing the script.
set -euo pipefail

cd "$(dirname "$0")"

usage() {
    cat <<'EOF'
Usage: run_all.sh [MODE]

Modes:
  diagnostic_run  Single-cell smoke test (ResNet-18 / CIFAR-10 / eta=0.40,
                  2 epochs). Default when no MODE is given.
  full_run        Publication sweep (240 cells: 3 archs x 2 datasets x
                  4 noises x 10 seeds, 30 epochs). Also runs the
                  estimator-comparison stage via --also-compare.
  exp_a           Re-run only the estimator-comparison stage on a
                  matrix that has already been trained.
  exp_b           Training stage only (no estimator comparison); pair
                  with a later ``exp_a`` to split the budget over
                  multiple sessions.
  tables          Regenerate every paper artefact from results/.

Environment overrides:
  GPUS="0,1,2,3"  Restrict to a subset of GPU ids.
  WORKERS=2       Concurrent tasks per GPU (default: 1).
  SEEDS="0 1 2"   Restrict the seed axis of the sweep (default: 0..9).
                  Use to partition the 240-cell matrix across sessions
                  or machines without editing the script.

Examples:
  ./run_all.sh diagnostic_run
  GPUS=0,1 WORKERS=2 ./run_all.sh full_run
  SEEDS="0 1 2" ./run_all.sh exp_b      # train first third of seeds
  SEEDS="0 1 2" ./run_all.sh exp_a      # compare just those seeds
  ./run_all.sh tables
EOF
}

case "${1:-}" in
    -h|--help|help)
        usage
        exit 0
        ;;
esac

MODE="${1:-diagnostic_run}"

# ---- GPU selection ---------------------------------------------------------

if [[ -n "${GPUS:-}" ]]; then
    GPU_ARGS=(--gpu-ids ${GPUS//,/ })
else
    GPU_ARGS=()
fi
WORKERS="${WORKERS:-1}"

# ---- Seed selection --------------------------------------------------------
#
# ``SEEDS`` lets the operator partition the seed axis of the sweep across
# sessions / machines without editing the script.  Accepts whitespace- or
# comma-separated integers; the default reproduces the paper's 0..9 grid.
SEEDS_RAW="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
# Split on whitespace OR comma; ``read -ra`` honours IFS.
IFS=$' \t,' read -ra SEED_LIST <<< "$SEEDS_RAW"
if [[ ${#SEED_LIST[@]} -eq 0 ]]; then
    echo "SEEDS is empty after parsing: '$SEEDS_RAW'" >&2
    exit 2
fi

# ---- Mode dispatch ---------------------------------------------------------

run_diagnostic() {
    echo "[diagnostic] single-cell smoke test: ResNet-18 / CIFAR-10 / eta=0.40 / seed=0"
    python -m main train \
        --arch resnet18 \
        --dataset cifar10 \
        --noise 0.40 \
        --seed 0 \
        --epochs 2 \
        --checkpoint-epochs 0 1 2 \
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
        --reference-probes 100
    python -m main aggregate \
        --kind exp_b \
        --root results/diagnostic/runs \
        --out  results/diagnostic/aggregate.json
    echo "[diagnostic] OK"
}

run_full_training() {
    echo "[full_run] dispatching full sweep (archs x datasets x noises x seeds=${SEED_LIST[*]})"
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 0.25 0.40 0.60 \
        --seeds "${SEED_LIST[@]}" \
        --epochs 30 \
        --checkpoint-epochs 0 10 30 \
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

run_exp_b_only() {
    echo "[exp_b] training stage only (seeds=${SEED_LIST[*]})"
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 0.25 0.40 0.60 \
        --seeds "${SEED_LIST[@]}" \
        --epochs 30 \
        --checkpoint-epochs 0 10 30 \
        --only-stage train \
        --workers-per-gpu "$WORKERS" \
        --output-root results/exp_b \
        "${GPU_ARGS[@]}"
    python -m main aggregate --kind exp_b \
        --root results/exp_b/runs --out results/exp_b/aggregate.json
    echo "[exp_b] OK"
}

run_exp_a_only() {
    echo "[exp_a] re-running estimator-comparison stage only (seeds=${SEED_LIST[*]})"
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 0.25 0.40 0.60 \
        --seeds "${SEED_LIST[@]}" \
        --epochs 30 \
        --checkpoint-epochs 0 10 30 \
        --only-stage compare \
        --workers-per-gpu "$WORKERS" \
        --output-root results/exp_b \
        "${GPU_ARGS[@]}"
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
    exp_b)          run_exp_b_only ;;
    tables)         run_tables ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac
