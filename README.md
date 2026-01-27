### DACBO Env

## Installation
Create a `uv` environment.
```bash
make env
```
Activate that environment before proceeding:
```bash
source .env/bin/activate
```
Install all requirements:
```bash
make install
```
In order to install `MetaBO`, deactivate your environment and run:
```bash
make metabo
```
MetaBO's requirements are very old and incompatible.

## Experiment Workflow
Warning: This creates a many many compute jobs. Check the scripts for syntax of running a single run.

1. Train PPO with `bash scripts/launch_ppo.sh`.
1. Prepare evaluation configs with `python -m dacboenv.experiment.collect_ppo runsicml2`.
    Now in `dacboenv/configs/policy/optimized` configurations for the single policies should appear.
1. Evaluate baselines and PPO with `bash scripts/eval_policies.sh`.
1. Gather data with `sbatch scripts/gather_data.sh`.
1. Plot using `notebooks/plot_results_icml.ipynb`.