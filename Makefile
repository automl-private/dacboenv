
SHELL := /bin/bash
PYTHON ?= python
PIP ?= pip
NAME := dacboenv
PACKAGE_NAME := dacboenv
VERSION := 0.0.1
DIST := dist
UV ?= uv
SMACBRANCH ?= development 
CARPSBRANCH ?= development

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
	$(MAKE) adaptaf

carps:
	git clone --branch $(CARPSBRANCH) git@github.com:automl/CARP-S.git lib/CARP-S
	cd lib/CARP-S && $(UV) pip install -e '.[dev]' && pre-commit install
	export PIP="uv pip" && $(PYTHON) -m carps.build.make benchmark_bbob #benchmark_yahpo benchmark_mfpbench optimizer_optuna optimizer_ax
	$(PYTHON) -m carps.utils.index_configs

smac:
	git clone --branch $(SMACBRANCH) git@github.com:automl/SMAC3.git lib/SMAC3
	$(UV) pip install swig
	cd lib/SMAC3 && $(UV) pip install -e '.[dev]' && pre-commit install


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

collect_incumbents:
	$(PYTHON) -m dacboenv.experiment.collect_incumbents runs

adaptaf:
	git clone git@github.com:automl-private/adaptaf.git lib/adaptaf
	cd lib/adaptaf && uv pip install -e .

optbench:
	git clone git@github.com:automl/OptBench.git lib/OptBench
	cd lib/OptBench && uv pip install -e .
	python -m carps.utils.index_configs '--extra_task_paths=["lib/OptBench/optbench/configs/task"]'

gather-data:
	python -m carps.analysis.gather_data '--rundir=["runs_eval","/scratch/hpc-prf-intexml/tklenke/experiment_runs/dacboenv_ppo_semi"]' --outdir=results

gather-data-small:
	python -m carps.analysis.gather_data \
	'--rundir=["runs_eval/PPO-AlphaNet*","runs_eval/NoOpPolicy","runs_eval/SAWEI","runs_eval/SMAC-AC--dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled*","runs_eval/SMAC-AC--dacbo_Csymlogregret_AWEI-cont_Ssawei_Rsymlogregret*"]' \
	--outdir=results_alphanet2
# 	--n_processes=1

gather-data-sawei:
	python -m carps.analysis.gather_data '--rundir=runs_eval/SAWEI-P' --outdir=resultssawei --n_processes=1

testppoalpha:
	python -m dacboenv.experiment.ppo_norm_alphanet \
		+opt=ppo_alphanet2 \
		experiment.n_workers=4 \
		experiment.n_episodes=50 \
		dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.logging_level=9999 \
		+instances=bbob2d_3seeds \
		+task=dacboenv_sawei \
		seed=3 \
		baserundir=tmprun \
		+env/instance_selector=random \
		task.optimization_resources.n_trials=10

testppo:
	python -m dacboenv.experiment.ppo_norm \
		+opt=ppo \
		experiment.n_workers=4 \
		experiment.n_episodes=50 \
		dacboenv.optimizer_cfg.smac_cfg.smac_kwargs.logging_level=9999 \
		+instances=bbob2d_3seeds \
		+task=dacboenv_epdonescaledpluslogregret \
		seed=3 \
		baserundir=tmprun \
		+env/instance_selector=random \
		task.optimization_resources.n_trials=10

tmpeval:
	python -m carps.run hydra.searchpath=[pkg://dacboenv/configs] \
		+eval=base \
		+env=base \
		+env/obs=sawei \
		+env/reward=ep_done_scaled \
		+env/opt=base \
		seed=2 \
		+task/BBOB=cfg_8_5_0 \
		+env/action=wei_alpha_continuous \
		+policy/optimized/PPO-AlphaNet3/dacbo_Cepisode_length_scaled_plus_logregret_AWEI-cont_Ssawei_Repisode_finished_scaled_Ibbob2d_3seeds=seed1 \
		dacboenv.terminate_after_reference_performance_reached=False


# TODO METABO
# Fix 'scikit-learn=0.21.3' in environment.yml
# For testing metabo, run with gpu.
#  cd lib/MetaBO; python evaluate_metabo_gprice.py
# Interactive job for GPU testing: salloc -t 02:00:00 --qos=devel --partition=dgx --gres=gpu:a100:1
metabo:
	git clone https://github.com/LUH-AI/MetaBO.git lib/MetaBO
	conda env create -f lib/MetaBO/environment.yml
	conda activate metabo

# tensorboard regex: (?s:.*?)finished(?s:.*?)SAWEI(?s:.*?)fid8(?s:.*?)log_2

runsaweitest:
	python -m carps.run hydra.searchpath=[pkg://dacboenv/configs,pkg://adaptaf/configs] +task/BNNBO=TensionCompressionString +method=sawei_20p seed=3 baserundir=runstmp

runsaweiptest:
	python -m carps.run hydra.searchpath=[pkg://dacboenv/configs] \
		+eval=base +env=base +env/obs=sawei +env/opt=base \
		+env/action=wei_alpha_continuous +env/reward=ep_done_scaled +env/refperf=saweip \
		+policy=sawei \
		+task/BNNBO=TensionCompressionString \
		seed=3 \
		baserundir=runstmp \
		dacboenv.evaluation_mode=false

ppotest:
	python -m dacboenv.experiment.ppo_norm_alphanet \
		+task=dacboenv_sawei_symlog +instances=bbob2d_3seeds +opt=ppo_alphanet +env/refperf=saweip \
		experiment.n_workers=1 \
		experiment.n_episodes=2 \
		seed=1 \
		+env/instance_selector=roundrobin \
		baserundir=tmp_runs_opt