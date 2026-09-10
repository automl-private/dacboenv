"""Bounded real CARP-S/SMAC checks for initial fitted-data verification."""

from __future__ import annotations

import numpy as np
from dacboenv.env.observations.gp_hyperparameters import probe_synchronized_gp, verify_synchronized_gp
from dacboenv.experiment.collect_snapshots import completed_evaluations
from dacboenv.experiment.real_env import real_structured_bbob_smoke_env
from dacboenv.selective_wei.initial_context import finite_probe_ubr, initial_output_scale


def test_real_bbob_initial_gp_verification_is_observer_neutral() -> None:
    """Repeated verification preserves the next f5 proposal/cost sequence."""
    observed = real_structured_bbob_smoke_env("bbob/2/3/0", 197, interaction_frequency=5)
    control = real_structured_bbob_smoke_env("bbob/2/3/0", 197, interaction_frequency=5)
    try:
        observation, _ = observed.reset()
        control.reset()
        assert observed.observation_space.contains(observation)
        first = verify_synchronized_gp(observed._smac_instance)
        assert first == verify_synchronized_gp(observed._smac_instance)
        assert first["completed_observations"] == 2
        panel = probe_synchronized_gp(observed._smac_instance, probe_count=8, seed=17)
        repeated = probe_synchronized_gp(observed._smac_instance, probe_count=8, seed=17)
        assert panel["probe_digest"] == repeated["probe_digest"]
        assert np.array_equal(panel["mean"], repeated["mean"])
        scale, _ = initial_output_scale(panel["initial_costs"])
        diagnostic = finite_probe_ubr(
            panel["mean"], panel["variance"], initial_indices=panel["initial_indices"], scale=scale
        )
        assert np.isfinite(diagnostic["raw_proxy"])
        before = completed_evaluations(observed)
        assert before == completed_evaluations(control)
        next_observation, reward, _, _, _ = observed.step(2)
        control.step(2)
        assert np.isfinite(reward)
        assert observed.observation_space.contains(next_observation)
        assert completed_evaluations(observed) == completed_evaluations(control)
        assert verify_synchronized_gp(observed._smac_instance)["model_digest"] != first["model_digest"]
    finally:
        observed.close()
        control.close()
