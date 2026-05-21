.PHONY: help install lint format typecheck test check clean \
        diagnostic full exp_a exp_b tables

GPU ?=

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Dev
# ---------------------------------------------------------------------------

install: ## Install project + dev deps via uv
	uv sync

lint: ## Run ruff + flake8
	uv run ruff check .
	uv run flake8 . --exclude=.venv

format: ## Format with black + ruff --fix
	uv run black .
	uv run ruff check --fix .

typecheck: ## Run mypy
	uv run mypy . --exclude '.venv'

test: ## Run pytest
	uv run pytest hessian/tests/ stochastic_check/tests/ -v

check: lint typecheck test ## Run lint + typecheck + test

# ---------------------------------------------------------------------------
# Experiments
#
# Customise via environment variables (forwarded to run_all.sh):
#   GPUS="0,1,2,3"   subset of GPU ids (default: all visible)
#   WORKERS=2        concurrent tasks per GPU (default: 1)
#   SEEDS="0 1 2"    subset of the seed axis (default: 0..9)
#
# Examples:
#   make full   GPUS=0,1,2,3 WORKERS=2
#   make exp_b  GPUS=2,3     WORKERS=2
#   make exp_a  GPUS=0,1,2,3 WORKERS=2 SEEDS="0 1 2"
# ---------------------------------------------------------------------------

RUN := GPUS="$(GPUS)" WORKERS="$(WORKERS)" SEEDS="$(SEEDS)" bash run_all.sh

diagnostic: ## Smoke run: 1 seed, 1 arch, 1 noise level, short training
	$(RUN) diagnostic_run

full: ## Full Springer sweep (3 archs x 2 datasets x 4 noise x 10 seeds)
	$(RUN) full_run

exp_a: ## Only estimator comparison (post-hoc on existing checkpoints)
	$(RUN) exp_a

exp_b: ## Only training sweep (no estimator comparison)
	$(RUN) exp_b

tables: ## Generate LaTeX tables from results/
	uv run python -m paper_data.main

# ---------------------------------------------------------------------------

clean: ## Remove caches
	find . -type d -name __pycache__   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache   -exec rm -rf {} + 2>/dev/null || true
