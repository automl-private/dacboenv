"""Fail-closed production CLI for manifest-paired DACBO evaluation.

The CLI executes structured DACBO methods, native SAWEI through a dedicated
paired environment factory, and native default SMAC through a native-to-tidy
adapter. It rejects every requested method before the first episode if any
adapter or artifact is unavailable. Dynamic-oracle execution remains in the
saved-snapshot analyzer, where its state-conditional semantics are defined.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import OmegaConf
from stable_baselines3 import PPO

from dacboenv.experiment.collect_snapshots import configured_structured_action_space
from dacboenv.experiment.evaluation_runner import (
    make_dacbo_method_runner,
    make_default_smac_method_runner,
)
from dacboenv.experiment.paired_evaluator import (
    BEST_VALIDATION_STATIC,
    DEFAULT_SMAC,
    DYNAMIC_ORACLE,
    LEARNED_FINAL,
    LEARNED_VALIDATION_SELECTED,
    MARGINAL_RANDOM_CONTROL,
    MODAL_STATIC_CLONE,
    SAWEI,
    STATIC_ACTION_PREFIX,
    UNIFORM_RANDOM,
    EvaluationContext,
    MethodRegistry,
    authorize_manifest_execution,
    available_method_cells,
    derive_validation_controls,
    evaluate_registered_methods,
    hierarchical_paired_bootstrap,
    paired_method_comparison,
    select_validation_static_action,
    validate_contexts_against_manifest,
    write_evaluation_records_csv,
)
from dacboenv.experiment.protocol import load_manifest
from dacboenv.policy.random import MarginalRandomPolicy, RandomPolicy
from dacboenv.policy.sawei import SAWEIPolicy
from dacboenv.policy.sb3_model import ModelPolicy
from dacboenv.policy.static import StaticParameterPolicy
from dacboenv.reference import ManifestReferenceProvider, ObjectiveReference

_UNAVAILABLE_METHOD_REASONS = {
    DYNAMIC_ORACLE: "dynamic oracle values require saved snapshot branches and are analysis-only",
}
_MISSING_SAWEI_FACTORY_REASON = "native SAWEI requires its method-specific UBR/WEI-term environment factory"
_SHA256_HEX_LENGTH = 64
_DEFAULT_ENV_FACTORY_BY_DOMAIN = {
    "bbob": "dacboenv.experiment.real_env:real_structured_bbob_env",
    "yahpo": "dacboenv.experiment.real_env:real_structured_yahpo_env",
    "mixed": "dacboenv.experiment.real_env:real_structured_mixed_env",
}


class ProductionEvaluationUnavailableError(RuntimeError):
    """Raised before execution when one requested production adapter is absent."""

    def __init__(self, readiness: Mapping[str, Any]) -> None:
        self.readiness = dict(readiness)
        reasons = "; ".join(f"{method}: {reason}" for method, reason in self.readiness.get("unavailable", {}).items())
        super().__init__(f"Unified evaluation is fail-closed: {reasons}")


@dataclass(frozen=True)
class StageARunArtifacts:
    """Validated Stage-A checkpoints and checkpoint-specific normalization."""

    run_root: Path
    outer_seed: int
    best_model: Path
    best_normalization: Path | None
    final_model: Path
    final_normalization: Path | None
    vecnormalize: bool
    action_family: str
    interaction_frequency: int


def _existing_first(paths: Sequence[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def inspect_stage_a_run(run_root: Path) -> StageARunArtifacts:
    """Validate both learned checkpoints before any evaluation episode starts."""
    if not run_root.is_dir():
        raise FileNotFoundError(f"Stage-A run root does not exist: {run_root}")
    config_path = run_root / ".hydra" / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Stage-A Hydra config is missing: {config_path}")
    cfg = OmegaConf.load(config_path)
    outer_seed = int(cfg.seed)
    vecnormalize = bool(cfg.experiment.get("vecnormalize", False))
    action_family = configured_structured_action_space(cfg)
    interaction_frequency = int(OmegaConf.select(cfg, "dacboenv.interaction_frequency"))
    if interaction_frequency not in {1, 5, 10}:
        raise ValueError(f"Unsupported Stage-A interaction frequency {interaction_frequency!r}.")

    best_model = _existing_first(
        (run_root / "validation" / "best_balanced_model.zip", run_root / "validation" / "best_model.zip")
    )
    final_model = _existing_first((run_root / "model.zip",))
    missing: list[str] = []
    if best_model is None:
        missing.append("validation/best_balanced_model.zip (or validation/best_model.zip)")
    if final_model is None:
        missing.append("model.zip")

    best_normalization = _existing_first(
        (
            run_root / "validation" / "best_balanced_vecnormalize.pkl",
            run_root / "validation" / "best_vecnormalize.pkl",
        )
    )
    final_normalization = _existing_first((run_root / "vecnormalize.pkl",))
    if vecnormalize and best_normalization is None:
        missing.append("validation/best_balanced_vecnormalize.pkl")
    if vecnormalize and final_normalization is None:
        missing.append("vecnormalize.pkl")
    if missing:
        raise FileNotFoundError(f"Incomplete Stage-A run {run_root}: missing {', '.join(missing)}")
    assert best_model is not None
    assert final_model is not None
    return StageARunArtifacts(
        run_root=run_root,
        outer_seed=outer_seed,
        best_model=best_model,
        best_normalization=best_normalization,
        final_model=final_model,
        final_normalization=final_normalization,
        vecnormalize=vecnormalize,
        action_family=action_family,
        interaction_frequency=interaction_frequency,
    )


def load_evaluation_contexts(path: Path) -> list[EvaluationContext]:
    """Load explicit reference/budget conventions for every manifest context."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("contexts") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("Context JSON must be a non-empty list or {'contexts': [...]} mapping.")
    return [EvaluationContext(**row) for row in rows]


def _load_environment_factory(
    specification: str,
    action_family: str,
    context_split: str,
    *,
    reference_table: Path | None = None,
    allow_sealed_test: bool = False,
) -> Callable[..., Any]:
    try:
        module_name, attribute = specification.split(":", maxsplit=1)
    except ValueError as error:
        raise ValueError("Environment factory must use module:callable syntax.") from error
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"Environment factory {specification!r} is not callable.")

    def environment_factory(task_id: str, inner_seed: int, *, interaction_frequency: int = 1) -> Any:
        kwargs: dict[str, Any] = {"context_split": context_split}
        if task_id.lower().startswith("yahpo/"):
            if reference_table is None:
                raise ValueError("YAHPO evaluation requires an explicit provenance-complete reference table.")
            kwargs["reference_table"] = reference_table
            kwargs["allow_sealed_test"] = allow_sealed_test
        kwargs["interaction_frequency"] = interaction_frequency
        return factory(task_id, inner_seed, action_family, **kwargs)

    return environment_factory


def _preflight_yahpo_references(
    reference_table: Path | None,
    contexts: Sequence[EvaluationContext],
) -> tuple[Path | None, ManifestReferenceProvider | None]:
    """Validate every requested YAHPO reference before the first episode."""
    yahpo_contexts = [context for context in contexts if context.domain == "yahpo"]
    if not yahpo_contexts:
        return None, None
    if reference_table is None:
        raise ValueError("YAHPO or mixed evaluation requires --reference-table.")
    provider = ManifestReferenceProvider(
        reference_table,
        expected_runtime_objective_transform="negative_accuracy",
        expected_reporting_objective_transform="one_minus_accuracy",
        expected_fidelity="fixed_maximum",
    )
    for context in yahpo_contexts:
        try:
            reference = provider.references[context.task_id]
        except KeyError as error:
            raise ValueError(f"Reference table has no row for evaluation task {context.task_id!r}.") from error
        if reference.kind != context.reference_kind:
            raise ValueError(f"Reference kind mismatch for {context.task_id!r}.")
        if reference.runtime_objective_transform != context.objective_transform:
            raise ValueError(f"Runtime objective-transform mismatch for {context.task_id!r}.")
        tolerance = max(float(reference.tolerance), 1e-12)
        if not np.isclose(reference.value, context.reference_value, rtol=0.0, atol=tolerance):
            raise ValueError(f"Reference value mismatch for {context.task_id!r}.")
    return reference_table, provider


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _load_control_provenance(  # noqa: C901
    path: Path,
    n_actions: int,
    action_family: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "validation":
        raise ValueError("Modal/marginal provenance must be derived from validation data only.")
    if payload.get("source_method") not in {LEARNED_VALIDATION_SELECTED, LEARNED_FINAL}:
        raise ValueError("Control provenance must identify its learned source method.")
    if payload.get("source_action_family") != action_family:
        raise ValueError("Control provenance action family differs from the requested evaluator family.")
    if payload.get("source_checkpoint") not in {"best", "final"}:
        raise ValueError("Control provenance must identify a best or final source checkpoint.")
    outer_seed = payload.get("source_outer_ppo_seed")
    if isinstance(outer_seed, bool) or not isinstance(outer_seed, int) or outer_seed < 0:
        raise ValueError("Control provenance must contain a non-negative outer PPO seed.")
    if not _valid_sha256(payload.get("source_validation_manifest_hash")):
        raise ValueError("Control provenance has no valid source validation manifest hash.")
    if not isinstance(payload.get("source_code_commit"), str) or not payload["source_code_commit"]:
        raise ValueError("Control provenance has no source-code revision.")
    frequencies = payload.get("source_action_frequencies")
    if not isinstance(frequencies, list) or len(frequencies) != n_actions:
        raise ValueError(f"Control provenance must contain {n_actions} action frequencies.")
    frequency_array = np.asarray(frequencies, dtype=float)
    if (
        not np.isfinite(frequency_array).all()
        or np.any(frequency_array < 0.0)
        or not np.isclose(np.sum(frequency_array), 1.0)
    ):
        raise ValueError("Control action frequencies must be finite, non-negative, and sum to one.")
    counts = payload.get("source_action_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != n_actions
        or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in counts)
    ):
        raise ValueError(f"Control provenance must contain {n_actions} non-negative integer action counts.")
    total = sum(counts)
    if total <= 0 or not np.allclose(frequency_array, np.asarray(counts, dtype=float) / total):
        raise ValueError("Control action frequencies do not match their source counts.")
    modal_action = payload.get("modal_action")
    if isinstance(modal_action, bool) or not isinstance(modal_action, int) or not 0 <= modal_action < n_actions:
        raise ValueError("Control provenance contains an invalid modal_action.")
    return payload


def _load_static_selection_provenance(path: Path, n_actions: int, action_family: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("source_split") != "validation":
        raise ValueError("Best-static provenance must be derived from validation data only.")
    if payload.get("source_action_family") != action_family:
        raise ValueError("Best-static provenance action family differs from the requested evaluator family.")
    selected_action = payload.get("selected_action")
    if (
        isinstance(selected_action, bool)
        or not isinstance(selected_action, int)
        or not 0 <= selected_action < n_actions
    ):
        raise ValueError("Best-static provenance contains an invalid selected_action.")
    if not _valid_sha256(payload.get("source_validation_manifest_hash")):
        raise ValueError("Best-static provenance has no valid source validation manifest hash.")
    if not isinstance(payload.get("source_code_commit"), str) or not payload["source_code_commit"]:
        raise ValueError("Best-static provenance has no source-code revision.")
    if payload.get("method") != BEST_VALIDATION_STATIC or not isinstance(payload.get("source_metric"), str):
        raise ValueError("Best-static provenance has incomplete method/metric metadata.")
    raw_scores = payload.get("source_static_scores")
    if not isinstance(raw_scores, dict) or set(raw_scores) != {str(action) for action in range(n_actions)}:
        raise ValueError("Best-static provenance must contain one score for every action.")
    scores = {int(action): float(score) for action, score in raw_scores.items()}
    if not np.isfinite(list(scores.values())).all():
        raise ValueError("Best-static provenance contains a non-finite validation score.")
    expected_action = min(scores, key=lambda action: (-scores[action], action))
    if selected_action != expected_action:
        raise ValueError("Best-static selected_action is inconsistent with the persisted validation scores.")
    return payload


def production_readiness(  # noqa: C901, PLR0912
    method_names: Sequence[str],
    *,
    run_root: Path | None,
    control_provenance_path: Path | None,
    static_selection_provenance_path: Path | None,
    action_family: str,
    interaction_frequency: int,
    n_actions: int,
    sawei_env_factory_available: bool = False,
) -> tuple[dict[str, Any], StageARunArtifacts | None, dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve every required artifact without creating an environment."""
    unavailable = {
        method: _UNAVAILABLE_METHOD_REASONS[method] for method in method_names if method in _UNAVAILABLE_METHOD_REASONS
    }
    if SAWEI in method_names and not sawei_env_factory_available:
        unavailable[SAWEI] = _MISSING_SAWEI_FACTORY_REASON
    learned_requested = any(method in {LEARNED_VALIDATION_SELECTED, LEARNED_FINAL} for method in method_names)
    controls_requested = any(method in {MODAL_STATIC_CLONE, MARGINAL_RANDOM_CONTROL} for method in method_names)
    best_static_requested = BEST_VALIDATION_STATIC in method_names
    artifacts = None
    controls = None
    static_selection = None
    if learned_requested:
        if run_root is None:
            unavailable["stage_a_run_root"] = "learned methods require an explicit --run-root"
        else:
            try:
                artifacts = inspect_stage_a_run(run_root)
                if artifacts.action_family != action_family:
                    raise ValueError(
                        "Stage-A checkpoint action family differs from the evaluator: "
                        f"{artifacts.action_family!r} != {action_family!r}."
                    )
                if artifacts.interaction_frequency != interaction_frequency:
                    raise ValueError(
                        "Stage-A checkpoint interaction frequency differs from the evaluator: "
                        f"{artifacts.interaction_frequency} != {interaction_frequency}."
                    )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                unavailable["stage_a_run_root"] = str(error)
    if controls_requested:
        if control_provenance_path is None:
            unavailable["control_provenance"] = "modal/marginal methods require validation-derived --control-provenance"
        else:
            try:
                controls = _load_control_provenance(control_provenance_path, n_actions, action_family)
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                unavailable["control_provenance"] = str(error)
    if best_static_requested:
        if static_selection_provenance_path is None:
            unavailable["static_selection_provenance"] = (
                "best_validation_static requires validation-derived --static-selection-provenance"
            )
        else:
            try:
                static_selection = _load_static_selection_provenance(
                    static_selection_provenance_path,
                    n_actions,
                    action_family,
                )
            except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
                unavailable["static_selection_provenance"] = str(error)
    if controls is not None and learned_requested:
        learned_methods = [method for method in method_names if method in {LEARNED_VALIDATION_SELECTED, LEARNED_FINAL}]
        if len(learned_methods) != 1:
            unavailable["control_provenance"] = (
                "One modal/marginal provenance file can accompany exactly one learned checkpoint per invocation."
            )
        elif controls["source_method"] != learned_methods[0]:
            unavailable["control_provenance"] = "Control provenance source method differs from the learned method."
        elif controls["source_checkpoint"] != (
            "best" if learned_methods[0] == LEARNED_VALIDATION_SELECTED else "final"
        ):
            unavailable["control_provenance"] = "Control provenance checkpoint label differs from the learned method."
        elif artifacts is not None and int(controls["source_outer_ppo_seed"]) != artifacts.outer_seed:
            unavailable["control_provenance"] = "Control provenance outer seed differs from the Stage-A run."
    readiness = {
        "ready": not unavailable,
        "requested_methods": list(method_names),
        "unavailable": unavailable,
        "explicitly_unimplemented": {
            **_UNAVAILABLE_METHOD_REASONS,
            **({SAWEI: _MISSING_SAWEI_FACTORY_REASON} if not sawei_env_factory_available else {}),
        },
    }
    return readiness, artifacts, controls, static_selection


def build_production_registry(  # noqa: C901, PLR0913
    method_names: Sequence[str],
    *,
    env_factory: Callable[[str, int], Any],
    action_family: str,
    trace_directory: Path,
    run_root: Path | None = None,
    control_provenance_path: Path | None = None,
    static_selection_provenance_path: Path | None = None,
    sawei_env_factory: Callable[[str, int], Any] | None = None,
    n_actions: int = 5,
    policy_seed: int = 0,
    interaction_frequency: int = 1,
    default_smac_references: Mapping[str, ObjectiveReference] | None = None,
    context_split: str = "validation",
) -> MethodRegistry:
    """Attach all runnable production methods, or fail before any callback."""
    registry = MethodRegistry(n_static_actions=n_actions)
    unknown = sorted(set(method_names) - set(registry.method_names))
    if unknown:
        raise KeyError(f"Unknown unified evaluator methods: {unknown}")
    readiness, artifacts, controls, static_selection = production_readiness(
        method_names,
        run_root=run_root,
        control_provenance_path=control_provenance_path,
        static_selection_provenance_path=static_selection_provenance_path,
        action_family=action_family,
        interaction_frequency=interaction_frequency,
        n_actions=n_actions,
        sawei_env_factory_available=sawei_env_factory is not None,
    )
    if not readiness["ready"]:
        raise ProductionEvaluationUnavailableError(readiness)

    def register(
        method: str,
        policy_factory: Callable[[Any, EvaluationContext, Any], Any],
        *,
        checkpoint_type: str = "none",
        outer_seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        method_env_factory: Callable[[str, int], Any] | None = None,
        method_action_family: str | None = None,
    ) -> None:
        registry.register_runner(
            method,
            make_dacbo_method_runner(
                env_factory=method_env_factory or env_factory,
                policy_factory=policy_factory,
                action_family=method_action_family or action_family,
                checkpoint_type=checkpoint_type,
                outer_ppo_seed=outer_seed,
                trace_directory=trace_directory,
                policy_seed=policy_seed,
                policy_metadata=metadata,
            ),
        )

    for method in method_names:
        if method.startswith(STATIC_ACTION_PREFIX):
            action = int(method.removeprefix(STATIC_ACTION_PREFIX))
            register(method, lambda env, _context, _method, action=action: StaticParameterPolicy(env, action))
        elif method == UNIFORM_RANDOM:
            register(method, lambda env, _context, _method: RandomPolicy(env))
        elif method in {LEARNED_VALIDATION_SELECTED, LEARNED_FINAL}:
            assert artifacts is not None
            is_best = method == LEARNED_VALIDATION_SELECTED
            model_path = artifacts.best_model if is_best else artifacts.final_model
            normalization = artifacts.best_normalization if is_best else artifacts.final_normalization
            checkpoint = "best" if is_best else "final"
            register(
                method,
                lambda env, _context, _method, model_path=model_path, normalization=normalization: ModelPolicy(
                    env,
                    model=str(model_path),
                    model_class=PPO,
                    normalization_wrapper=None if normalization is None else str(normalization),
                ),
                checkpoint_type=checkpoint,
                outer_seed=artifacts.outer_seed,
                metadata={"run_root": str(artifacts.run_root), "checkpoint": checkpoint},
            )
        elif method in {MODAL_STATIC_CLONE, MARGINAL_RANDOM_CONTROL}:
            assert controls is not None
            if method == MODAL_STATIC_CLONE:
                action = int(controls["modal_action"])
                policy_factory = lambda env, _context, _method, action=action: StaticParameterPolicy(env, action)
            else:
                frequencies = list(controls["source_action_frequencies"])
                policy_factory = lambda env, _context, _method, frequencies=frequencies: MarginalRandomPolicy(
                    env, frequencies
                )
            register(
                method,
                policy_factory,
                checkpoint_type=str(controls["source_checkpoint"]),
                outer_seed=int(controls["source_outer_ppo_seed"]),
                metadata={"control_provenance": str(control_provenance_path)},
            )
        elif method == BEST_VALIDATION_STATIC:
            assert static_selection is not None
            action = int(static_selection["selected_action"])
            register(
                method,
                lambda env, _context, _method, action=action: StaticParameterPolicy(env, action),
                metadata={"static_selection_provenance": str(static_selection_provenance_path)},
            )
        elif method == SAWEI:
            assert sawei_env_factory is not None
            register(
                method,
                lambda env, _context, _method: SAWEIPolicy(env),
                metadata={
                    "alpha": 0.5,
                    "delta": 0.1,
                    "window_size": 7,
                    "atol_rel": 0.1,
                    "track_attitude": "last",
                    "auto_alpha": False,
                },
                method_env_factory=sawei_env_factory,
                method_action_family="wei_continuous",
            )
        elif method == DEFAULT_SMAC:
            registry.register_runner(
                method,
                make_default_smac_method_runner(
                    output_directory=trace_directory.parent / "native_default_smac",
                    trace_directory=trace_directory,
                    objective_references=default_smac_references,
                    context_split=context_split,
                ),
            )
    return registry


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary(records: Sequence[Any], baseline_method: str | None, bootstrap_resamples: int) -> dict[str, Any]:
    cells = available_method_cells(records)
    payload: dict[str, Any] = {
        "record_count": len(records),
        "method_cells": [asdict(cell) for cell in cells],
        "paired_comparisons": [],
        "hierarchical_bootstrap": [],
    }
    if baseline_method is None:
        return payload
    baseline_cells = [cell for cell in cells if cell.method == baseline_method]
    if len(baseline_cells) != 1:
        raise ValueError(f"Expected one baseline cell named {baseline_method!r}, found {baseline_cells}.")
    baseline = baseline_cells[0]
    for cell in cells:
        if cell == baseline:
            continue
        comparison = paired_method_comparison(records, cell, baseline)
        payload["paired_comparisons"].append(_jsonable(asdict(comparison)))
        if bootstrap_resamples > 0:
            bootstrap = hierarchical_paired_bootstrap(
                records,
                cell,
                baseline,
                n_resamples=bootstrap_resamples,
                seed=0,
            )
            payload["hierarchical_bootstrap"].append(_jsonable(asdict(bootstrap)))
    return payload


def _persist_validation_derived_controls(
    records: Sequence[Any],
    manifest: Mapping[str, Any],
    output_directory: Path,
    *,
    n_actions: int,
) -> dict[str, Any]:
    """Persist deployable controls using validation records only."""
    if manifest["split"] != "validation":
        return {}
    artifacts: dict[str, Any] = {}
    expected_static = {f"{STATIC_ACTION_PREFIX}{action}" for action in range(n_actions)}
    represented_methods = {record.method for record in records}
    if expected_static.issubset(represented_methods):
        static_records = [record for record in records if record.method in expected_static]
        path = output_directory / "best_static_selection.json"
        provenance = select_validation_static_action(
            static_records,
            manifest,
            n_actions=n_actions,
            output_path=path,
        )
        artifacts[BEST_VALIDATION_STATIC] = {"path": str(path), "provenance": provenance.to_dict()}

    for learned_method in (LEARNED_VALIDATION_SELECTED, LEARNED_FINAL):
        learned_records = [record for record in records if record.method == learned_method]
        if not learned_records:
            continue
        path = output_directory / f"controls_{learned_method}.json"
        provenance = derive_validation_controls(learned_records, manifest, output_path=path)
        artifacts[learned_method] = {"path": str(path), "provenance": provenance.to_dict()}
    return artifacts


def main() -> None:
    """Execute one explicit manifest/run-root evaluation cell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--control-provenance", type=Path)
    parser.add_argument("--static-selection-provenance", type=Path)
    parser.add_argument("--action-family", default="wei")
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--env-factory", help="Optional module:callable override; defaults by manifest domain.")
    parser.add_argument(
        "--reference-table",
        type=Path,
        help="Required provenance-complete best-known table for YAHPO or mixed contexts.",
    )
    parser.add_argument("--baseline-method")
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--allow-sealed-test", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    readiness_path = args.output_dir / "production_readiness.json"
    manifest = load_manifest(args.manifest)
    contexts = load_evaluation_contexts(args.contexts)
    try:
        authorize_manifest_execution(manifest, allow_sealed_test=args.allow_sealed_test)
        validate_contexts_against_manifest(contexts, manifest)
        interaction_frequencies = {context.interaction_frequency for context in contexts}
        if len(interaction_frequencies) != 1:
            raise ValueError(
                "One unified-evaluator invocation must use one interaction frequency; "
                f"got {sorted(interaction_frequencies)}."
            )
        interaction_frequency = next(iter(interaction_frequencies))
        reference_table, yahpo_reference_provider = _preflight_yahpo_references(args.reference_table, contexts)
        factory_specification = args.env_factory or _DEFAULT_ENV_FACTORY_BY_DOMAIN[str(manifest["domain"])]
        environment_factory = _load_environment_factory(
            factory_specification,
            args.action_family,
            str(manifest["split"]),
            reference_table=reference_table,
            allow_sealed_test=args.allow_sealed_test,
        )
        sawei_environment_factory = _load_environment_factory(
            "dacboenv.experiment.real_env:real_sawei_env",
            "wei",
            str(manifest["split"]),
            reference_table=reference_table,
            allow_sealed_test=args.allow_sealed_test,
        )
        registry = build_production_registry(
            args.methods,
            env_factory=environment_factory,
            action_family=args.action_family,
            trace_directory=args.output_dir / "traces",
            run_root=args.run_root,
            control_provenance_path=args.control_provenance,
            static_selection_provenance_path=args.static_selection_provenance,
            sawei_env_factory=sawei_environment_factory,
            n_actions=args.n_actions,
            policy_seed=args.policy_seed,
            interaction_frequency=interaction_frequency,
            default_smac_references=(None if yahpo_reference_provider is None else yahpo_reference_provider.references),
            context_split=str(manifest["split"]),
        )
    except ProductionEvaluationUnavailableError as error:
        readiness_path.write_text(
            json.dumps(error.readiness, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(str(error)) from error
    except (FileNotFoundError, KeyError, PermissionError, TypeError, ValueError) as error:
        readiness = {
            "ready": False,
            "requested_methods": args.methods,
            "unavailable": {"preflight": str(error)},
        }
        readiness_path.write_text(json.dumps(readiness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise SystemExit(f"Unified evaluation preflight failed: {error}") from error

    readiness_path.write_text(
        json.dumps(
            {
                "ready": True,
                "requested_methods": args.methods,
                "explicitly_unimplemented": _UNAVAILABLE_METHOD_REASONS,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    records = evaluate_registered_methods(
        manifest,
        contexts,
        args.methods,
        registry,
        allow_sealed_test=args.allow_sealed_test,
    )
    write_evaluation_records_csv(records, args.output_dir / "evaluation_records.csv")
    summary = _summary(records, args.baseline_method, args.bootstrap_resamples)
    summary["derived_validation_artifacts"] = _persist_validation_derived_controls(
        records,
        manifest,
        args.output_dir,
        n_actions=args.n_actions,
    )
    (args.output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
