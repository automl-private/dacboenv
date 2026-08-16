"""Contracts for fixed-shape GP hyperparameter observations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ConfigSpace import ConfigurationSpace, Float
from dacboenv.env.observation import ObservationSpace
from dacboenv.env.observations.gp_hyperparameters import (
    GP_HP_CHANGE_INDEX,
    GP_HP_ROLE_INDEX,
    GP_HP_SUMMARY_INDEX,
    GPHyperparameterFeatureProvider,
    GPHyperparameterSettings,
    classify_kernel_parameter_role,
    extract_free_kernel_parameters,
)
from hydra import compose, initialize_config_module
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from smac import BlackBoxFacade, Scenario
from smac.runhistory.dataclasses import TrialValue


class GaussianProcessProbe:
    """Small fitted-state stand-in exposing the SMAC GP compatibility API."""

    def __init__(self, kernel: object, *, fitted: bool = True) -> None:
        self._gp = SimpleNamespace(kernel_=kernel)
        self._kernel = kernel
        self._is_trained = fitted


def fake_smbo(model: object, finished: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        intensifier=SimpleNamespace(config_selector=SimpleNamespace(_model=model)),
        runhistory=SimpleNamespace(finished=finished),
    )


def test_fitted_isotropic_gp_summary_and_roles() -> None:
    kernel = ConstantKernel(2.0, (1e-3, 1e3)) * RBF(1.5, (1e-3, 1e3)) + WhiteKernel(0.1, (1e-5, 1e1))
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    bundle = provider.features(fake_smbo(GaussianProcessProbe(kernel)))

    summary = bundle.summary
    assert summary[GP_HP_SUMMARY_INDEX["available"]] == 1
    assert summary[GP_HP_SUMMARY_INDEX["is_gp"]] == 1
    assert summary[GP_HP_SUMMARY_INDEX["uses_ard"]] == 0
    assert summary[GP_HP_SUMMARY_INDEX["lengthscale_available"]] == 1
    assert summary[GP_HP_SUMMARY_INDEX["signal_available"]] == 1
    assert summary[GP_HP_SUMMARY_INDEX["noise_available"]] == 1
    assert bundle.raw_mask.sum() == 3
    assert bundle.raw_roles[:3].sum() == 3
    assert bundle.raw_roles[0, GP_HP_ROLE_INDEX["signal"]] == 1
    assert bundle.raw_roles[1, GP_HP_ROLE_INDEX["lengthscale"]] == 1
    assert bundle.raw_roles[2, GP_HP_ROLE_INDEX["noise"]] == 1
    for key, space in provider.observation_spaces().items():
        assert space.contains(provider.value(bundle, key))


def test_fitted_ard_gp_statistics_effective_dimension_and_order() -> None:
    kernel = RBF([0.1, 1.0, 10.0], (1e-3, 1e3))
    parameters = extract_free_kernel_parameters(kernel)
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    bundle = provider.features(fake_smbo(GaussianProcessProbe(kernel)))

    assert [(parameter.name, parameter.element_index) for parameter in parameters] == [
        ("length_scale", 0),
        ("length_scale", 1),
        ("length_scale", 2),
    ]
    assert bundle.summary[GP_HP_SUMMARY_INDEX["uses_ard"]] == 1
    assert bundle.summary[GP_HP_SUMMARY_INDEX["lengthscale_std"]] > 0
    assert bundle.summary[GP_HP_SUMMARY_INDEX["lengthscale_anisotropy"]] > 0
    effective = bundle.summary[GP_HP_SUMMARY_INDEX["lengthscale_effective_dimension_fraction"]]
    assert 0 < effective < 1
    np.testing.assert_allclose(bundle.raw[:3], [parameter.normalized for parameter in parameters])


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("k1__length_scale", "lengthscale"),
        ("lengthscale", "lengthscale"),
        ("constant_value", "signal"),
        ("outputscale", "signal"),
        ("white__noise_level", "noise"),
        ("mystery_parameter", "other"),
    ],
)
def test_role_classifier_is_conservative(name: str, expected: str) -> None:
    assert classify_kernel_parameter_role(name) == expected


def test_non_gp_and_unfitted_gp_use_zero_conventions() -> None:
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    non_gp = provider.features(fake_smbo(SimpleNamespace(_is_trained=True)))
    assert non_gp.summary[GP_HP_SUMMARY_INDEX["available"]] == 0
    assert non_gp.summary[GP_HP_SUMMARY_INDEX["is_gp"]] == 0
    assert non_gp.raw_mask.sum() == 0

    kernel = RBF(1.0)
    unfitted = provider.features(fake_smbo(GaussianProcessProbe(kernel, fitted=False)))
    assert unfitted.summary[GP_HP_SUMMARY_INDEX["available"]] == 0
    assert unfitted.summary[GP_HP_SUMMARY_INDEX["is_gp"]] == 1
    assert unfitted.raw_mask.sum() == 0


def test_repeated_state_is_stable_update_has_delta_and_reset_clears_history() -> None:
    kernel = RBF([1.0, 2.0], (1e-3, 1e3))
    smbo = fake_smbo(GaussianProcessProbe(kernel))
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    first = provider.features(smbo)
    repeated = provider.features(smbo)
    np.testing.assert_array_equal(first.change, repeated.change)
    assert repeated.change[GP_HP_CHANGE_INDEX["previous_available"]] == 0

    kernel.theta = kernel.theta + 0.25
    smbo.runhistory.finished += 1
    updated = provider.features(smbo)
    assert updated.change[GP_HP_CHANGE_INDEX["previous_available"]] == 1
    assert updated.change[GP_HP_CHANGE_INDEX["theta_delta_l2"]] > 0
    provider.reset()
    after_reset = provider.features(smbo)
    assert after_reset.change[GP_HP_CHANGE_INDEX["previous_available"]] == 0


def test_padding_overflow_and_explicit_truncation() -> None:
    kernel = RBF(np.ones(65), (1e-3, 1e3))
    smbo = fake_smbo(GaussianProcessProbe(kernel))
    with pytest.raises(OverflowError, match="exceeding configured maximum"):
        GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True, max_raw_parameters=64)).features(smbo)

    provider = GPHyperparameterFeatureProvider(
        GPHyperparameterSettings(enabled=True, max_raw_parameters=64, overflow_policy="truncate")
    )
    with pytest.warns(RuntimeWarning, match="Truncating"):
        bundle = provider.features(smbo)
    assert bundle.raw.shape == (64,)
    assert bundle.raw_mask.sum() == 64
    assert provider.diagnostics.truncation_count == 1


def test_invalid_bound_fallback_is_finite_and_diagnostic() -> None:
    hyperparameter = SimpleNamespace(name="length_scale", fixed=False, n_elements=1)
    kernel = SimpleNamespace(
        theta=np.asarray([8.0]),
        bounds=np.asarray([[np.nan, np.inf]]),
        hyperparameters=[hyperparameter],
    )
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    bundle = provider.features(fake_smbo(GaussianProcessProbe(kernel)))
    assert np.isfinite(bundle.raw).all()
    assert bundle.parameters[0].bound_available is False
    assert provider.diagnostics.fallback_bound_count == 1


def test_extraction_is_side_effect_free() -> None:
    kernel = RBF([1.0, 2.0])
    smbo = fake_smbo(GaussianProcessProbe(kernel), finished=9)
    theta_before = kernel.theta.copy()
    finished_before = smbo.runhistory.finished
    provider = GPHyperparameterFeatureProvider(GPHyperparameterSettings(enabled=True))
    provider.features(smbo)
    np.testing.assert_array_equal(kernel.theta, theta_before)
    assert smbo.runhistory.finished == finished_before


def _real_probe_smbo(output: Path):
    configspace = ConfigurationSpace(seed=71, space={"x": Float("x", (-2.0, 2.0))})
    scenario = Scenario(configspace, deterministic=True, n_trials=8, seed=71, output_directory=output)
    facade = BlackBoxFacade(
        scenario,
        lambda config, seed=0: float(config["x"] ** 2),  # noqa: ARG005
        initial_design=BlackBoxFacade.get_initial_design(scenario, n_configs=2),
        overwrite=True,
        logging_level=False,
    )
    smbo = facade.optimizer
    for _ in range(2):
        trial = smbo.ask()
        smbo.tell(trial, TrialValue(cost=float(trial.config["x"] ** 2), time=0.0))
    return smbo


def test_real_observation_is_stable_and_does_not_change_next_candidate(tmp_path: Path) -> None:
    observed = _real_probe_smbo(tmp_path / "observed")
    control = _real_probe_smbo(tmp_path / "control")
    observation_space = ObservationSpace(
        observed,
        keys=["global_state", "action_features", "gp_hp_summary", "gp_hp_change"],
        gp_hyperparameters={"enabled": True},
    )
    first = observation_space.get_initial_observation()
    repeated = observation_space.get_observation()
    assert observation_space.space.contains(first)
    for key in first:
        np.testing.assert_array_equal(first[key], repeated[key])
    assert observation_space.get_gp_hyperparameter_diagnostics()["gp_hp/extraction_calls"] == 1
    observed_candidate = observed.ask().config.get_array()
    control_space = ObservationSpace(control, keys=["global_state", "action_features"])
    control_space.get_initial_observation()
    control_candidate = control.ask().config.get_array()
    np.testing.assert_array_equal(observed_candidate, control_candidate)


def test_fixed_shapes_are_kernel_dimension_and_non_gp_independent() -> None:
    """Vector workers get identical GP-key spaces across heterogeneous tasks."""
    settings = GPHyperparameterSettings(enabled=True, max_raw_parameters=64)
    shapes = []
    for model in (
        GaussianProcessProbe(RBF(np.ones(2))),
        GaussianProcessProbe(RBF(np.ones(8))),
        SimpleNamespace(_is_trained=True),
    ):
        provider = GPHyperparameterFeatureProvider(settings)
        bundle = provider.features(fake_smbo(model))
        spaces = provider.observation_spaces()
        shapes.append({key: space.shape for key, space in spaces.items()})
        assert all(space.contains(provider.value(bundle, key)) for key, space in spaces.items())
    assert shapes[0] == shapes[1] == shapes[2]


@pytest.mark.parametrize(
    ("config_name", "expected_keys"),
    [
        ("structured", ["global_state", "action_features"]),
        ("structured_gp_summary", ["global_state", "action_features", "gp_hp_summary"]),
        (
            "structured_gp_summary_change",
            ["global_state", "action_features", "gp_hp_summary", "gp_hp_change"],
        ),
        (
            "structured_gp_raw",
            ["global_state", "action_features", "gp_hp_raw", "gp_hp_raw_mask", "gp_hp_raw_roles"],
        ),
    ],
)
def test_observation_hydra_groups_are_explicit_opt_ins(config_name: str, expected_keys: list[str]) -> None:
    """New Hydra groups never change the existing structured default."""
    overrides = ["+training=bbob_double_dqn_smoke"]
    if config_name != "structured":
        overrides.append(f"+env/obs={config_name}")
    with initialize_config_module(version_base=None, config_module="dacboenv.configs"):
        cfg = compose(config_name=None, overrides=overrides)
    assert list(cfg.dacboenv.observation_keys) == expected_keys
    assert bool(cfg.dacboenv.gp_hyperparameters.enabled) is (config_name != "structured")
