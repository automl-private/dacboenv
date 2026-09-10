"""Model-agnostic delayed fixed-action value predictors."""

from __future__ import annotations

import hashlib
import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

from dacboenv.offline.models.shared_dueling_q import OfflineQModelConfig, OfflineQNetwork, build_offline_q_model
from dacboenv.offline.normalization import ObservationNormalizer
from dacboenv.selective_wei.context import canonical_hash
from dacboenv.selective_wei.schemas import ACTION_COUNT, ValuePrediction

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class DelayedValuePredictor(ABC):
    """Predict fixed-action H=5/H=10 values from deployable observations."""

    @abstractmethod
    def predict(self, observation: dict[str, NDArray[np.floating]]) -> ValuePrediction:
        """Return one value per discrete WEI action."""

    @abstractmethod
    def artifact_metadata(self) -> dict[str, Any]:
        """Return immutable model identity and feature provenance."""


class SharedBranchQPredictor(DelayedValuePredictor):
    """Adapter around the existing permutation-equivariant branch-Q model."""

    def __init__(
        self,
        model: OfflineQNetwork,
        *,
        model_id: str,
        model_hash: str,
        normalizer: ObservationNormalizer | None,
        normalizer_hash: str,
        feature_schema_hash: str,
        has_q10: bool = True,
    ) -> None:
        self.model = model.eval()
        self.model_id = model_id
        self.model_hash = model_hash
        self.normalizer = normalizer
        self.normalizer_hash = normalizer_hash
        self.feature_schema_hash = feature_schema_hash
        self.has_q10 = has_q10

    def _arrays(self, observation: dict[str, NDArray[np.floating]]) -> tuple[torch.Tensor, torch.Tensor]:
        state = np.asarray(observation["global_state"], dtype=np.float32)
        actions = np.asarray(observation["action_features"], dtype=np.float32)
        if state.shape != (13,) or actions.shape != (ACTION_COUNT, 4):
            raise ValueError("Selective predictor expects global_state (13,) and action_features (5,4).")
        if self.normalizer is not None:
            state = self.normalizer.global_state.transform(state)
            actions = self.normalizer.action_features.transform(actions)
        return torch.from_numpy(state).unsqueeze(0), torch.from_numpy(actions).unsqueeze(0)

    def predict(self, observation: dict[str, NDArray[np.floating]]) -> ValuePrediction:
        """Run deterministic shared-head inference without environment side effects."""
        state, actions = self._arrays(observation)
        with torch.no_grad():
            q5 = self.model(state, actions, head="branch_q5").squeeze(0).double().numpy()
            q10 = self.model(state, actions, head="branch_q10").squeeze(0).double().numpy() if self.has_q10 else None
        return ValuePrediction(
            q5_mean=q5,
            q10_mean=q10,
            model_ids=(self.model_id,),
            model_hashes=(self.model_hash,),
            normalizer_hash=self.normalizer_hash,
            feature_schema_hash=self.feature_schema_hash,
        )

    def artifact_metadata(self) -> dict[str, Any]:
        """Return neural model metadata."""
        return {
            "predictor_type": "shared_branch_q",
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "model_config": asdict(self.model.config),
            "normalizer_hash": self.normalizer_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "has_q10": self.has_q10,
            "parameter_count": self.model.parameter_count,
        }


class EnsembleDelayedValuePredictor(DelayedValuePredictor):
    """Aggregate independent-seed or task-bootstrap predictor members."""

    def __init__(self, members: list[DelayedValuePredictor], *, ensemble_kind: str) -> None:
        if not members:
            raise ValueError("A delayed-value ensemble requires at least one member.")
        self.members = members
        self.ensemble_kind = ensemble_kind

    def predict(self, observation: dict[str, NDArray[np.floating]]) -> ValuePrediction:
        """Preserve member values for paired residual uncertainty."""
        predictions = [member.predict(observation) for member in self.members]
        q5 = np.stack([value.q5_mean for value in predictions])
        q10_values = [value.q10_mean for value in predictions]
        q10 = (
            None
            if any(value is None for value in q10_values)
            else np.stack(cast("list[NDArray[np.float64]]", q10_values))
        )
        return ValuePrediction(
            q5_mean=q5.mean(axis=0),
            q10_mean=None if q10 is None else q10.mean(axis=0),
            q5_member_values=q5,
            q10_member_values=q10,
            q5_std=q5.std(axis=0, ddof=1) if len(q5) > 1 else np.zeros(ACTION_COUNT),
            q10_std=(None if q10 is None else q10.std(axis=0, ddof=1) if len(q10) > 1 else np.zeros(ACTION_COUNT)),
            model_ids=tuple(item.model_ids[0] for item in predictions),
            model_hashes=tuple(item.model_hashes[0] for item in predictions),
            normalizer_hash=predictions[0].normalizer_hash,
            feature_schema_hash=predictions[0].feature_schema_hash,
        )

    def artifact_metadata(self) -> dict[str, Any]:
        """Return ordered member identities."""
        return {
            "predictor_type": "ensemble",
            "ensemble_kind": self.ensemble_kind,
            "members": [member.artifact_metadata() for member in self.members],
        }


def _long_form(state: NDArray[np.floating], actions: NDArray[np.floating]) -> NDArray[np.float64]:
    states = np.asarray(state, dtype=np.float64)
    features = np.asarray(actions, dtype=np.float64)
    if states.ndim == 1:
        states = states[None, :]
        features = features[None, :, :]
    repeated = np.repeat(states[:, None, :], ACTION_COUNT, axis=1)
    return np.concatenate((repeated, features), axis=2).reshape(-1, states.shape[1] + features.shape[2])


class SklearnDelayedValuePredictor(DelayedValuePredictor):
    """Long-form Extra Trees or histogram-gradient diagnostic predictor."""

    def __init__(
        self,
        q5_model: Any,
        q10_model: Any | None,
        *,
        predictor_type: str,
        model_id: str,
        model_hash: str,
        feature_schema_hash: str,
    ) -> None:
        self.q5_model = q5_model
        self.q10_model = q10_model
        self.predictor_type = predictor_type
        self.model_id = model_id
        self.model_hash = model_hash
        self.feature_schema_hash = feature_schema_hash

    def predict(self, observation: dict[str, NDArray[np.floating]]) -> ValuePrediction:
        """Score the five state/action rows in stable alpha order."""
        rows = _long_form(observation["global_state"], observation["action_features"])
        q5 = np.asarray(self.q5_model.predict(rows), dtype=np.float64)
        q10 = None if self.q10_model is None else np.asarray(self.q10_model.predict(rows), dtype=np.float64)
        return ValuePrediction(
            q5_mean=q5,
            q10_mean=q10,
            model_ids=(self.model_id,),
            model_hashes=(self.model_hash,),
            feature_schema_hash=self.feature_schema_hash,
        )

    def artifact_metadata(self) -> dict[str, Any]:
        """Return diagnostic model identity."""
        return {
            "predictor_type": self.predictor_type,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "feature_schema_hash": self.feature_schema_hash,
            "has_q10": self.q10_model is not None,
        }


def fit_sklearn_predictor(
    state: NDArray[np.floating],
    action_features: NDArray[np.floating],
    q5: NDArray[np.floating],
    q10: NDArray[np.floating] | None,
    *,
    kind: Literal["extra_trees", "hist_gradient_boosting"],
    seed: int,
    model_id: str,
    sample_weight: NDArray[np.floating] | None = None,
) -> SklearnDelayedValuePredictor:
    """Fit a prespecified long-form tree diagnostic on training states only."""
    rows = _long_form(state, action_features)
    weights: NDArray[np.float64] | None = (
        None if sample_weight is None else np.repeat(np.asarray(sample_weight, dtype=np.float64), ACTION_COUNT)
    )

    def fit(target: NDArray[np.floating]) -> Any:
        if kind == "extra_trees":
            model: Any = ExtraTreesRegressor(
                n_estimators=256,
                min_samples_leaf=2,
                max_features=0.75,
                n_jobs=1,
                random_state=seed,
            )
        else:
            model = HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=200,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                random_state=seed,
            )
        model.fit(rows, np.asarray(target, dtype=np.float64).reshape(-1), sample_weight=weights)
        return model

    q5_model = fit(q5)
    q10_model = None if q10 is None else fit(q10)
    model_hash = canonical_hash(
        {"kind": kind, "id": model_id, "seed": seed, "n": len(state), "q5_mean": np.asarray(q5).mean().item()}
    )
    return SklearnDelayedValuePredictor(
        q5_model,
        q10_model,
        predictor_type=kind,
        model_id=model_id,
        model_hash=model_hash,
        feature_schema_hash=canonical_hash({"global_state": [13], "action_features": [5, 4]}),
    )


def fit_neural_branch_predictor(
    state: NDArray[np.floating],
    action_features: NDArray[np.floating],
    q5: NDArray[np.floating],
    q10: NDArray[np.floating] | None,
    *,
    seed: int,
    updates: int,
    learning_rate: float,
    model_id: str,
    normalizer: ObservationNormalizer | None = None,
) -> SharedBranchQPredictor:
    """Fit only branch heads; this is not offline Bellman/FQI training."""
    torch.manual_seed(seed)
    model = OfflineQNetwork(OfflineQModelConfig())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = np.random.default_rng(seed)
    normalized_state = np.asarray(state, dtype=np.float32)
    normalized_actions = np.asarray(action_features, dtype=np.float32)
    if normalizer is not None:
        normalized_state = normalizer.global_state.transform(normalized_state)
        normalized_actions = normalizer.action_features.transform(normalized_actions)
    for _update in range(updates):
        indices = generator.integers(len(state), size=min(256, len(state)))
        state_tensor = torch.as_tensor(normalized_state[indices])
        action_tensor = torch.as_tensor(normalized_actions[indices])
        loss = torch.nn.functional.huber_loss(
            model(state_tensor, action_tensor, head="branch_q5"),
            torch.as_tensor(np.asarray(q5)[indices], dtype=torch.float32),
        )
        if q10 is not None:
            loss = loss + torch.nn.functional.huber_loss(
                model(state_tensor, action_tensor, head="branch_q10"),
                torch.as_tensor(np.asarray(q10)[indices], dtype=torch.float32),
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
    buffer = pickle.dumps({key: value.detach().cpu().numpy() for key, value in model.state_dict().items()})
    model_hash = hashlib.sha256(buffer).hexdigest()
    return SharedBranchQPredictor(
        model,
        model_id=model_id,
        model_hash=model_hash,
        normalizer=normalizer,
        normalizer_hash="" if normalizer is None else canonical_hash(normalizer.to_dict()),
        feature_schema_hash=canonical_hash({"global_state": [13], "action_features": [5, 4]}),
        has_q10=q10 is not None,
    )


def save_predictor(predictor: DelayedValuePredictor, root: Path) -> dict[str, Any]:
    """Persist a predictor with explicit type metadata, never filename inference."""
    root.mkdir(parents=True, exist_ok=True)
    metadata = predictor.artifact_metadata()
    if isinstance(predictor, EnsembleDelayedValuePredictor):
        members = [
            save_predictor(member, root / f"member_{index:02d}") for index, member in enumerate(predictor.members)
        ]
        metadata["member_artifacts"] = [
            {**member, "manifest_path": f"member_{index:02d}/predictor.json"} for index, member in enumerate(members)
        ]
    elif isinstance(predictor, SharedBranchQPredictor):
        path = root / "model.pt"
        torch.save({"model_config": asdict(predictor.model.config), "model_state": predictor.model.state_dict()}, path)
        metadata["model_path"] = path.name
        metadata["artifact_sha256"] = _sha256(path)
        if predictor.normalizer is not None:
            normalizer = root / "normalizer.json"
            normalizer.write_text(json.dumps(predictor.normalizer.to_dict(), indent=2, sort_keys=True) + "\n")
            metadata["normalizer_path"] = normalizer.name
            metadata["normalizer_artifact_sha256"] = _sha256(normalizer)
    elif isinstance(predictor, SklearnDelayedValuePredictor):
        path = root / "model.pkl"
        path.write_bytes(pickle.dumps({"q5": predictor.q5_model, "q10": predictor.q10_model}))
        metadata["model_path"] = path.name
        metadata["artifact_sha256"] = _sha256(path)
    else:
        raise TypeError(f"Unsupported predictor type {type(predictor).__name__}.")
    manifest = root / "predictor.json"
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    metadata["manifest_path"] = str(manifest)
    metadata["manifest_sha256"] = _sha256(manifest)
    return metadata


def load_predictor(path: Path) -> DelayedValuePredictor:
    """Load a predictor from explicit algorithm metadata and verified hashes."""
    metadata = json.loads(path.read_text(encoding="utf-8"))
    kind = metadata["predictor_type"]
    if kind == "ensemble":
        for item in metadata["member_artifacts"]:
            if _sha256(path.parent / item["manifest_path"]) != item["manifest_sha256"]:
                raise ValueError("Ensemble member manifest hash mismatch.")
        members = [load_predictor(path.parent / item["manifest_path"]) for item in metadata["member_artifacts"]]
        return EnsembleDelayedValuePredictor(members, ensemble_kind=str(metadata["ensemble_kind"]))
    model_path = path.parent / metadata["model_path"]
    if _sha256(model_path) != metadata["artifact_sha256"]:
        raise ValueError("Predictor model hash mismatch.")
    if kind == "shared_branch_q":
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
        model = build_offline_q_model(payload["model_config"])
        model.load_state_dict(payload["model_state"])
        normalizer = None
        normalizer_path = metadata.get("normalizer_path")
        if normalizer_path:
            normalizer_file = path.parent / normalizer_path
            if _sha256(normalizer_file) != metadata["normalizer_artifact_sha256"]:
                raise ValueError("Predictor normalizer hash mismatch.")
            normalizer = ObservationNormalizer.from_dict(json.loads(normalizer_file.read_text(encoding="utf-8")))
        return SharedBranchQPredictor(
            model,
            model_id=str(metadata["model_id"]),
            model_hash=str(metadata["model_hash"]),
            normalizer=normalizer,
            normalizer_hash=str(metadata["normalizer_hash"]),
            feature_schema_hash=str(metadata["feature_schema_hash"]),
            has_q10=bool(metadata["has_q10"]),
        )
    payload = pickle.loads(model_path.read_bytes())  # noqa: S301 - hash-verified local scientific artifact
    return SklearnDelayedValuePredictor(
        payload["q5"],
        payload["q10"],
        predictor_type=str(kind),
        model_id=str(metadata["model_id"]),
        model_hash=str(metadata["model_hash"]),
        feature_schema_hash=str(metadata["feature_schema_hash"]),
    )
