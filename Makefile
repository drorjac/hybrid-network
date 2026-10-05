# hybrid-network: development tasks.
#
# PYTHON is the interpreter of the environment the package is installed in
# (`pip install -e '.[dev]'`). Override it if needed:
#     make test PYTHON=/path/to/python
PYTHON ?= python
export OMP_NUM_THREADS ?= 1
export VECLIB_MAXIMUM_THREADS ?= 1
export PYTHONPATH := src

.PHONY: help test lint format typecheck build run quick doc diagrams

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-10s %s\n", $$1, $$2}'

test:  ## unit tests (dataset tests skip when the data are not built)
	$(PYTHON) -m pytest -p no:faulthandler -q

lint:  ## ruff check and format check, mypy
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m mypy src tests

format:  ## apply ruff fixes and formatting
	$(PYTHON) -m ruff check --fix src tests
	$(PYTHON) -m ruff format src tests

build:  ## reduce the archives under CML_DATA_ROOT to the common format
	$(PYTHON) -m hybrid_network build

run:  ## the full study, then docs/RESULTS.md (hours)
	$(PYTHON) -m hybrid_network run

quick:  ## short runs, one seed, into results/cml_quick/
	$(PYTHON) -m hybrid_network run --quick

doc:  ## docs/RESULTS.md and figures/cml/ from results/cml/
	$(PYTHON) -m hybrid_network doc

diagrams:  ## block diagrams in figures/diagrams/
	$(PYTHON) -m hybrid_network diagrams
