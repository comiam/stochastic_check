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
#   phase1_extension
#           --  extend the Phase-I clean-label ensemble from S=10 to
#               S=20 by training 60 additional (arch x dataset x seed)
#               cells at eta=0.  Lowers CV(sigma_0) of the CUSUM
#               calibration from 1/sqrt(2*9) ~ 0.24 to 1/sqrt(2*19)
#               ~ 0.16.  Seeds default to 10..19 and can be overridden
#               via ``EXT_SEEDS``.
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
  phase1_extension
                  Train 60 extra clean-label (eta=0) cells -- 3 archs x
                  2 datasets x 10 fresh seeds -- to grow the CUSUM
                  Phase-I baseline ensemble from S=10 to S=20.  Only
                  the training stage is dispatched (no Exp-A); the
                  default ``EXT_SEEDS`` is ``10 11 ... 19`` and avoids
                  collisions with the seeds 0..9 from ``full_run``.
  tables          Regenerate every paper artefact from results/.

Environment overrides:
  GPUS="0,1,2,3"  Restrict to a subset of GPU ids.
  WORKERS=2       Concurrent tasks per GPU (default: 1).
  SEEDS="0 1 2"   Restrict the seed axis of the sweep (default: 0..9).
                  Use to partition the 240-cell matrix across sessions
                  or machines without editing the script.
  EXT_SEEDS="10 ... 19"
                  Seed axis for ``phase1_extension`` (default: 10..19).
                  Must be disjoint from the seeds used by ``full_run``
                  to avoid overwriting existing run directories.

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

# Phase-I extension seeds: disjoint from SEEDS by default so reruns of
# ``phase1_extension`` cannot clobber the publication sweep.
EXT_SEEDS_RAW="${EXT_SEEDS:-10 11 12 13 14 15 16 17 18 19}"
IFS=$' \t,' read -ra EXT_SEED_LIST <<< "$EXT_SEEDS_RAW"
if [[ ${#EXT_SEED_LIST[@]} -eq 0 ]]; then
    echo "EXT_SEEDS is empty after parsing: '$EXT_SEEDS_RAW'" >&2
    exit 2
fi

# ---- Dataset pre-download --------------------------------------------------
#
# Torchvision races on its marker file when ``download=True`` is called
# concurrently from multiple workers; pre-fetching once in this process
# avoids losing the occasional cell to a half-written archive.
ensure_datasets() {
    python -c \
"from stochastic_check.data import ensure_datasets_downloaded; ensure_datasets_downloaded()"
}

# ---- Mode dispatch ---------------------------------------------------------

run_diagnostic() {
    echo "[diagnostic] single-cell smoke test: ResNet-18 / CIFAR-10 / eta=0.40 / seed=0"
    ensure_datasets
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
    ensure_datasets
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
    ensure_datasets
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
    ensure_datasets
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

run_phase1_extension() {
    # Extend the CUSUM Phase-I clean ensemble: 3 archs x 2 datasets x
    # |EXT_SEED_LIST| extra eta=0 cells.  Only the training stage is
    # scheduled -- the post-hoc Exp-A estimator comparison does not
    # touch clean runs that are added solely for calibration.
    echo "[phase1_extension] training extra clean-label cells (seeds=${EXT_SEED_LIST[*]})"
    # Sanity check: refuse to overlap with the publication seed grid.
    for s in "${EXT_SEED_LIST[@]}"; do
        for s0 in "${SEED_LIST[@]}"; do
            if [[ "$s" == "$s0" ]]; then
                echo "[phase1_extension] EXT_SEEDS overlap SEEDS at seed=$s; aborting to avoid overwriting existing runs." >&2
                exit 3
            fi
        done
    done
    ensure_datasets
    python -m main sweep \
        --archs resnet18 vgg11 resnet34 \
        --datasets cifar10 cifar100 \
        --noises 0.0 \
        --seeds "${EXT_SEED_LIST[@]}" \
        --epochs 30 \
        --checkpoint-epochs 0 10 30 \
        --only-stage train \
        --workers-per-gpu "$WORKERS" \
        --output-root results/exp_b \
        "${GPU_ARGS[@]}"
    python -m main aggregate --kind exp_b \
        --root results/exp_b/runs --out results/exp_b/aggregate.json
    echo "[phase1_extension] OK"
}

run_tables() {
    echo "[tables] regenerating paper artefacts"
    python -m paper_data.main --all
    echo "[tables] OK"
}

case "$MODE" in
    diagnostic_run)    run_diagnostic ;;
    full_run)          run_full_training ;;
    exp_a)             run_exp_a_only ;;
    exp_b)             run_exp_b_only ;;
    phase1_extension)  run_phase1_extension ;;
    tables)            run_tables ;;
    *)
        echo "Unknown mode: $MODE" >&2
        usage >&2
        exit 2
        ;;
esac
