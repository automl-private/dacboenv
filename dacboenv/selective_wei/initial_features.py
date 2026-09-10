"""Shared initial-design feature extraction for collection and deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np
from ConfigSpace import CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.hyperparameters import FloatHyperparameter, IntegerHyperparameter

from dacboenv.env.observation import _synchronize_model
from dacboenv.env.observations.gp_hyperparameters import (
    GP_HP_SUMMARY_NAMES,
    GPHyperparameterFeatureProvider,
    GPHyperparameterSettings,
    probe_synchronized_gp,
    verify_synchronized_gp,
)
from dacboenv.selective_wei.context import YAHPO_SCENARIOS, canonical_hash, context_from_task_id
from dacboenv.selective_wei.initial_context import InitialContext, finite_probe_ubr, initial_output_scale


@dataclass(frozen=True, slots=True)
class InitialFeatureSettings:
    """Scientific recipe; seeds and probe counts are never inferred from paths."""

    probe_count: int = 128
    probe_seed: int = 1729
    confidence_multiplier: float = 2.0
    scale_floor: float = 1e-12
    scenario_features: bool = True
    require_gp: bool = True
    require_ubr: bool = True
    version: str = "initial-mdgu-v2"

    def __post_init__(self) -> None:
        """Reject invalid diagnostic recipes before touching the optimizer."""
        if self.version != "initial-mdgu-v2":
            raise ValueError("Initial-feature protocol changed; legacy pilot features require explicit reconstruction.")
        if self.probe_count < 1 or self.probe_seed < 0:
            raise ValueError("Initial probes require a positive count and nonnegative seed.")
        if not np.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("Initial output scale floor must be positive and finite.")
        if not np.isfinite(self.confidence_multiplier) or self.confidence_multiplier < 0:
            raise ValueError("Posterior confidence multiplier must be finite and nonnegative.")
        if self.require_ubr and not self.require_gp:
            raise ValueError("Required UBR requires verified GP fitting.")


def extract_initial_context(env: Any, settings: InitialFeatureSettings) -> tuple[InitialContext, dict[str, Any]]:  # noqa: C901, PLR0912, PLR0915
    """Extract from all completed D0, without objective calls or live ask.

    The environment enforces that this is its original reset boundary. Input
    coverage is mean shared-active coordinate distance: normalized numeric
    distances and categorical/ordinal mismatch, never category-code Euclidean
    distance. Pairs without a shared active coordinate are masked out.
    """
    extraction_started = perf_counter()
    smbo = env._smac_instance
    records = [(smbo.runhistory.get_config(key.config_id), smbo.runhistory[key]) for key in smbo.runhistory]
    configurations = [item[0] for item in records]
    costs = np.asarray([item[1].cost for item in records], dtype=np.float64).reshape(-1)
    if not configurations or len(costs) != len(configurations) or not np.isfinite(costs).all():
        raise ValueError("Initial features require complete finite single-objective D0.")
    n0 = len(costs)
    space = configurations[0].config_space
    parameters = list(space.values())
    dimension = len(parameters)
    vectors = np.asarray([configuration.get_array() for configuration in configurations])
    active = np.isfinite(vectors)
    context = context_from_task_id(env.current_task_id)
    scale, scale_reason = initial_output_scale(costs, floor=settings.scale_floor)
    names: list[str] = []
    values: list[float] = []
    masks: list[bool] = []
    groups: list[str] = []

    def add(group: str, name: str, value: float, available: bool = True) -> None:  # noqa: FBT001, FBT002
        names.append(name)
        values.append(float(value) if available else 0.0)
        masks.append(available)
        groups.append(group)

    for domain in ("bbob", "yahpo", "optbench"):
        add("M", f"domain_{domain}", context.domain == domain)
    if settings.scenario_features:
        for scenario in sorted(YAHPO_SCENARIOS):
            add("M", f"scenario_{scenario}", context.scenario == scenario)
    categorical = np.asarray([isinstance(hp, (CategoricalHyperparameter, OrdinalHyperparameter)) for hp in parameters])
    integer = np.asarray([isinstance(hp, IntegerHyperparameter) for hp in parameters])
    continuous = np.asarray([isinstance(hp, FloatHyperparameter) for hp in parameters])
    total = int(env._n_trials)
    metadata = {
        "nominal_dimension": dimension,
        "mean_active_parameter_count": active.sum(axis=1).mean(),
        "continuous_fraction": continuous.mean(),
        "integer_fraction": integer.mean(),
        "categorical_fraction": categorical.mean(),
        "conditional_fraction": len(space.conditional_hyperparameters) / dimension,
        "total_budget": total,
        "actual_n0": n0,
        "remaining_budget": total - n0,
        "n0_per_d_plus_one": n0 / (dimension + 1),
        "remaining_per_d_plus_one": (total - n0) / (dimension + 1),
    }
    for name, value in metadata.items():
        add("M", name, value)
    median = float(np.median(costs))
    for name, value in {
        "output_scale": scale,
        "output_iqr": np.quantile(costs, 0.75) - np.quantile(costs, 0.25),
        "incumbent_minus_median_scaled": (costs.min() - median) / scale,
        "q10_minus_median_scaled": (np.quantile(costs, 0.1) - median) / scale,
        "q90_minus_median_scaled": (np.quantile(costs, 0.9) - median) / scale,
    }.items():
        add("D", name, value)
    designs = [dict(configuration) for configuration in configurations]
    add("D", "duplicate_fraction", 1 - len({canonical_hash(row) for row in designs}) / n0)
    distances = []
    for left in range(n0):
        for right in range(left):
            shared = active[left] & active[right]
            if shared.any():
                delta = np.where(categorical, vectors[left] != vectors[right], np.abs(vectors[left] - vectors[right]))
                distances.append(float(delta[shared].mean()))
    for name, quantile in (
        ("coverage_distance_q10", 0.1),
        ("coverage_distance_median", 0.5),
        ("coverage_distance_q90", 0.9),
    ):
        add("D", name, float(np.quantile(distances, quantile)) if distances else 0.0, bool(distances))
    synchronization_started = perf_counter()
    _synchronize_model(smbo)
    synchronization_seconds = perf_counter() - synchronization_started
    identity: dict[str, Any]
    try:
        identity = verify_synchronized_gp(smbo)
        provider = GPHyperparameterFeatureProvider(
            GPHyperparameterSettings(enabled=True, strict_kernel_validation=True)
        )
        summary = provider.features(smbo).summary
    except RuntimeError as error:
        if settings.require_gp:
            raise
        identity = {"model_digest": "unavailable", "training_data_digest": "unavailable", "failure": str(error)}
        summary = np.zeros(len(GP_HP_SUMMARY_NAMES))
    for name, value in zip(GP_HP_SUMMARY_NAMES, summary, strict=True):
        add("G", f"gp_{name}", value, bool(summary[0]))
    diagnostic: dict[str, Any] = {}
    try:
        panel = probe_synchronized_gp(smbo, probe_count=settings.probe_count, seed=settings.probe_seed)
        diagnostic = finite_probe_ubr(
            panel["mean"],
            panel["variance"],
            initial_indices=panel["initial_indices"],
            scale=scale,
            confidence_multiplier=settings.confidence_multiplier,
        )
        diagnostic["probe_digest"] = panel["probe_digest"]
    except RuntimeError as error:
        if settings.require_ubr:
            raise
        diagnostic["failure"] = str(error)
    for name in ("raw_proxy", "scaled_proxy"):
        add("U", f"ubr_{name}", diagnostic.get(name, 0.0), "valid" in diagnostic)
    for index, name in enumerate(("posterior_std_q10", "posterior_std_median", "posterior_std_q90")):
        add("U", name, diagnostic.get("posterior_std_quantiles", [0.0] * 3)[index], "valid" in diagnostic)
    anchor = InitialContext(
        episode_id=canonical_hash({"task": env.current_task_id, "seed": env.current_seed, "total": total}),
        names=tuple(names),
        values=tuple(values),
        available=tuple(masks),
        groups=tuple(groups),
        n_initial=n0,
        total_budget=total,
        ordered_design_digest=canonical_hash(designs),
        ordered_cost_digest=canonical_hash(costs.tolist()),
        model_digest=identity["model_digest"],
        training_data_digest=identity["training_data_digest"],
        diagnostic_protocol_digest=canonical_hash(asdict(settings)),
        diagnostic_seed=settings.probe_seed,
    )
    return anchor, {
        "settings": asdict(settings),
        "timing": {
            "initial_synchronization_seconds": synchronization_seconds,
            "initial_diagnostic_seconds": perf_counter() - extraction_started - synchronization_seconds,
        },
        "identity": identity,
        "diagnostic": diagnostic,
        "scale_reason": scale_reason,
        "initial_configurations": designs,
        "initial_costs": costs.tolist(),
        "input_distance": "shared-active-normalized-numeric-and-categorical-mismatch-v1",
    }
