# Copyright (c) 2019 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# ******************************************************************
# evaluate_metabo_gprice.py
# Reproduce results from MetaBO paper on GPrice-Function
# For convenience, we provide the pretrained weights resulting from the experiments described in the paper.
# These weights can be reproduced using train_metabo_gprice.py
# ******************************************************************

import argparse

import os
from datetime import datetime

from gym.envs.registration import register, registry
from metabo.eval.evaluate import eval_experiment
from metabo.eval.plot_results import plot_results
from metabo.policies.taf.generate_taf_data_gprice import generate_taf_data_gprice  # TODO change
from metabo.ppo.util import get_best_iter_idx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--fideval", type=int, default=-1)
parser.add_argument("--d", type=int, default=0)
parser.add_argument("--fidtrain", type=int, default=-1)
args = parser.parse_args()
fideval = args.fideval
fidtrain = args.fidtrain
d = args.d

if fidtrain == -1:
    fidtrain = fideval

# set evaluation parameters
afs_to_evaluate = ["MetaBO"]#, "TAF-ME", "TAF-RANKING", "EI", "Random"]
# afs_to_evaluate = ["MetaBO", "EPS-GREEDY", "GMM-UCB", "EI"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bbob")
logpath = os.path.join(rootdir, "log", f"MetaBO-BBOB-{fidtrain}-{d}D-v0")
savepath = os.path.join(logpath, "eval", f"MetaBO-BBOB-{fideval}-{d}D-v0")
n_workers = 1  # TODO change back to 10
n_episodes = 10

# evaluate all afs
for af in afs_to_evaluate:
    # set af-specific parameters
    if af == "MetaBO":
        features = ["posterior_mean", "posterior_std", "timestep", "budget", "x"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = get_best_iter_idx(logpath)  # best ppo iteration during training, determined via metabo/ppo/util/get_best_iter_idx
        deterministic = True
        policy_specs = {}  # will be loaded from the logfiles
    elif af == "TAF-ME" or af == "TAF-RANKING":
        generate_taf_data_gprice(M=50, N=100)
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
        pass_X_to_pi = True
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"TAF_datafile": os.path.join(rootdir, "policies", "taf", "taf_gprice_M_50_N_100.pkl")}
    elif af == "EPS-GREEDY":
        # we use the same data as we used for TAF/METABO-M
        generate_taf_data_gprice(M=50, N=100)
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "budget", "x"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"datafile": os.path.join(rootdir, "policies", "taf", "taf_gprice_M_50_N_100.pkl"),
                        "eps": 0.55}
    elif af == "GMM-UCB":
        # we use the same data as we used for TAF/METABO-M
        generate_taf_data_gprice(M=50, N=100)
        features = ["posterior_mean", "posterior_std", "x", "timestep", "budget"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"datafile": os.path.join(rootdir, "policies", "taf", "taf_gprice_M_50_N_100.pkl"),
                        "ucb_kappa": 2.0, "w": 0.22, "n_components": 1}
    else:
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep"]
        pass_X_to_pi = False
        n_init_samples = 1
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        if af == "EI":
            policy_specs = {}
        elif af == "Random":
            policy_specs = {}
        else:
            raise ValueError("Unknown AF!")

    # define environment
    env_spec = {
        "env_id": f"MetaBO-BBOB-{fideval}-{d}D-v0",
        "D": d,
        "f_type": f"bbob-var-fid{fideval}",
        "f_opts": {"bound_translation": 0.1,
                   "bound_scaling": 0.1,
                           "fid": fideval,
                    "log_path": savepath},
        "features": features,
        "T": int(np.ceil(20 + 40 * np.sqrt(d))),
        "n_init_samples": n_init_samples,
        "pass_X_to_pi": pass_X_to_pi,
        # parameters were determined offline via type-2-ML on a GP with 100 datapoints
        "kernel_lengthscale": [0.2] * d,
        "kernel_variance": 0.616,
        "noise_variance": 1e-6,
        "use_prior_mean_function": False,
        "local_af_opt": True,
        "N_MS": 1000,
        "N_LS": 1000,
        "k": 5,
        "reward_transformation": "none",
    }

    # register gym environment
    if env_spec["env_id"] in registry.env_specs:
        del registry.env_specs[env_spec["env_id"]]
    register(
        id=env_spec["env_id"],
        entry_point="metabo.environment.metabo_gym:MetaBO",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # define evaluation run
    eval_spec = {
        "env_id": env_spec["env_id"],
        "env_seed_offset": 100,
        "policy": af,
        "logpath": logpath,
        "load_iter": load_iter,
        "deterministic": deterministic,
        "policy_specs": policy_specs,
        "savepath": savepath,
        "n_workers": n_workers,
        "n_episodes": n_episodes,
        "T": env_spec["T"],
    }

    # perform evaluation
    print("Evaluating {} on {}...".format(af, env_spec["env_id"]))
    eval_experiment(eval_spec)
    print("Done! Saved result in {}".format(savepath))
    print("**********************\n\n")

# plot (plot is saved to savepath)
print("Plotting...")
plot_results(path=savepath, logplot=True)
print("Done! Saved plot in {}".format(savepath))
