
SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
NAME := dacboenv
PACKAGE_NAME := dacboenv
VERSION := 0.0.1
DIST := dist
UV ?= uv
SMACBRANCH ?= development 
CARPSBRANCH ?= plot_update

env:
	$(PIP) install uv
	$(PYTHON) -m $(UV) venv --python=3.12 .env --clear
	. .env/bin/activate && $(PYTHON) -m ensurepip --upgrade && $(PYTHON) -m $(PIP) install uv --upgrade && $(UV) $(PIP) install setuptools wheel
	# Manually activate env. Does not work with make somehow

install:
	$(UV) $(PIP) install setuptools wheel swig
	$(UV) $(PIP) install -e ".[dev]"
	pre-commit install
	$(MAKE) carps
	$(MAKE) smac
	$(MAKE) optbench

carps:
	git clone --branch $(CARPSBRANCH) git@github.com:automl/CARP-S.git lib/CARP-S
	cd lib/CARP-S && $(UV) pip install -e '.[dev]' && pre-commit install
	export PIP="uv pip" && $(PYTHON) -m carps.build.make benchmark_bbob #benchmark_yahpo benchmark_mfpbench optimizer_optuna optimizer_ax
	$(PYTHON) -m carps.utils.index_configs

smac:
	git clone --branch $(SMACBRANCH) git@github.com:automl/SMAC3.git lib/SMAC3
	$(UV) pip install swig
	cd lib/SMAC3 && $(UV) pip install -e '.[dev]' && pre-commit install

check:
	pre-commit run --all-files

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

clean-build:
	rm -rf ${DIST}

# Build a distribution in ./dist
build:
	$(PYTHON) -m $(PIP) install build
	$(PYTHON) -m build --sdist

collect_incumbents:
	$(PYTHON) -m dacboenv.experiment.collect_incumbents runs

optbench:
	git clone git@github.com:automl/OptBench.git lib/OptBench
	cd lib/OptBench && uv pip install -e .
	python -m carps.utils.index_configs '--extra_task_paths=["lib/OptBench/optbench/configs/task"]'

metabo:
	conda env create -f lib/MetaBO/environment.yml
	conda activate metabo

# tensorboard regex: (?s:.*?)finished(?s:.*?)SAWEI(?s:.*?)fid8(?s:.*?)log_2
