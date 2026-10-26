.PHONY: help install lint format style typecheck test check clean \
        diagnostic full exp_a exp_b phase1_extension tables data-check paper-check

.DEFAULT_GOAL := help
UV ?= uv
PYTHON = $(UV) run --locked python
SOURCES = main.py hessian stochastic_check lab_orchestrator paper_data scripts_adhoc tests
RUN = GPUS="$(GPUS)" WORKERS="$(WORKERS)" SEEDS="$(SEEDS)" bash run_all.sh

help: ## Show available targets
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  %-18s %s\n", $$1, $$2}'

install: ## Install locked dependencies with Python 3.12
	$(UV) sync --locked --python 3.12

lint: ## Check Python lint, documentation, formatting, and shell syntax
	$(UV) run --locked ruff check $(SOURCES)
	$(UV) run --locked ruff format --check $(SOURCES)
	bash -n run_all.sh

format: ## Apply safe Ruff fixes and format the maintained Python files
	$(UV) run --locked ruff check --fix $(SOURCES)
	$(UV) run --locked ruff format $(SOURCES)

style: lint ## Check the source style

typecheck: ## Check types without changing numerical expressions
	$(PYTHON) -m mypy $(SOURCES)

test: ## Run the CPU tests, including paper regressions
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PYTHON) -m pytest -q

data-check: ## Verify all bundled result checksums
	cd paper_data/artifacts && sha256sum --quiet -c SHA256SUMS

check: lint typecheck test data-check ## Run all development and reproduction checks

tables: ## Rebuild the three article tables without checkpoints or a GPU
	$(PYTHON) -m paper_data.main --label tab:alerts --label tab:baselines --label tab:overhead

diagnostic: ## Run the short training and comparison smoke test
	$(RUN) diagnostic_run

full: ## Run the original 240-cell sweep
	$(RUN) full_run

exp_a: ## Compare estimators on existing checkpoints
	$(RUN) exp_a

exp_b: ## Run only the training sweep
	$(RUN) exp_b

phase1_extension: ## Add the 60 clean runs needed for the paper's full ensemble
	$(RUN) phase1_extension

clean: ## Remove generated tables and caches, never source data or checkpoints
	rm -rf build/paper_tables .pytest_cache .ruff_cache .mypy_cache
	find hessian stochastic_check lab_orchestrator paper_data scripts_adhoc tests \
		-type d -name __pycache__ -prune -exec rm -rf {} +
