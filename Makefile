
SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
NAME := dacboenv
PACKAGE_NAME := dacboenv
VERSION := 0.0.1
DIST := dist

env:
	pip install uv
	uv venv --python=3.12 venvdacboenv

welcome:
	. venvdacboenv/bin/activate

install:
	uv pip install setuptools wheel swig
	uv pip install -e ".[dev]"
	pre-commit install

test:
	$(PYTHON) -m pytest tests/test_configs.py tests/test_optimizers.py tests/test_tasks.py -n 8

docs:
	$(PYTHON) -m webbrowser -t "http://127.0.0.1:8000/"
	$(PYTHON) -m mkdocs serve --clean

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