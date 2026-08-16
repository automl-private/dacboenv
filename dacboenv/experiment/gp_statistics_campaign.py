"""CARP-S-native GP-hyperparameter trajectory campaign utilities.

The optimization jobs themselves are deliberately executed through
``python -m carps.run``.  This module supplies the side-effect-free SMAC
callback, deterministic packaged-task inventory, status validation, and
post-run plotting/consolidation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any

import carps
import pandas as pd
from omegaconf import OmegaConf
from smac.callback.callback import Callback

from dacboenv.env.observations.gp_hyperparameters import (
    GP_HP_CHANGE_NAMES,
    GP_HP_ROLE_NAMES,
    GP_HP_SUMMARY_NAMES,
    GPHyperparameterFeatureProvider,
    GPHyperparameterSettings,
)
from dacboenv.experiment.evaluation_determinism import canonical_sha256, file_sha256

if TYPE_CHECKING:
    from smac.main.smbo import SMBO
    from smac.runhistory.dataclasses import TrialInfo

CAMPAIGN_VERSION = "carps-gp-statistics-v1"
EXPECTED_BBOB_CONFIGS = 24
EXPECTED_YAHPO_CONFIGS = 20
CARPS_SEEDS = (0, 1)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _configuration_payload(info: TrialInfo) -> dict[str, Any]:
    config = info.config
    values = config.get_dictionary() if hasattr(config, "get_dictionary") else dict(config)
    return {str(name): values[name] for name in sorted(values)}


class GPHyperparameterTrajectoryCallback(Callback):
    """Record one immutable GP observation bundle after every SMAC ``ask``."""

    def __init__(
        self,
        output_path: str,
        task_id: str,
        seed: int,
        *,
        max_raw_parameters: int = 64,
        overflow_policy: str = "error",
        near_bound_fraction: float = 0.05,
        strict_kernel_validation: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.task_id = str(task_id)
        self.seed = int(seed)
        self.ask_index = 0
        self._previous_state_key: tuple[int, str] | None = None
        self._provider = GPHyperparameterFeatureProvider(
            GPHyperparameterSettings(
                enabled=True,
                max_raw_parameters=int(max_raw_parameters),
                overflow_policy=overflow_policy,  # type: ignore[arg-type]
                near_bound_fraction=float(near_bound_fraction),
                unsupported_model_policy="zeros",
                strict_kernel_validation=bool(strict_kernel_validation),
            )
        )

    def on_ask_end(self, smbo: SMBO, info: TrialInfo) -> None:
        """Read, but never fit or modify, the model used for this proposal."""
        bundle = self._provider.features(smbo)
        summary = {name: float(bundle.summary[index]) for index, name in enumerate(GP_HP_SUMMARY_NAMES)}
        change = {name: float(bundle.change[index]) for index, name in enumerate(GP_HP_CHANGE_NAMES)}
        parameters = [asdict(parameter) for parameter in bundle.parameters]
        state_key = None if bundle.state_key is None else [bundle.state_key[0], bundle.state_key[1]]
        record = {
            "schema_version": CAMPAIGN_VERSION,
            "timestamp": _utc_now(),
            "task_id": self.task_id,
            "seed": self.seed,
            "ask_index": self.ask_index,
            "runhistory_finished": int(smbo.runhistory.finished),
            "state_key": state_key,
            "model_revision_changed": bundle.state_key is not None and bundle.state_key != self._previous_state_key,
            "candidate": _configuration_payload(info),
            "summary": summary,
            "change": change,
            "raw": bundle.raw.tolist(),
            "raw_mask": bundle.raw_mask.tolist(),
            "raw_roles": bundle.raw_roles.tolist(),
            "parameters": parameters,
            "diagnostics": asdict(self._provider.diagnostics),
        }
        _append_jsonl(self.output_path, record)
        self._previous_state_key = bundle.state_key
        self.ask_index += 1

    def on_end(self, smbo: SMBO) -> None:
        """Write a small atomic callback-completion marker."""
        _atomic_json(
            self.output_path.with_name("gp_statistics_callback_status.json"),
            {
                "schema_version": CAMPAIGN_VERSION,
                "status": "success",
                "task_id": self.task_id,
                "seed": self.seed,
                "ask_count": self.ask_index,
                "runhistory_finished": int(smbo.runhistory.finished),
                "diagnostics": asdict(self._provider.diagnostics),
                "completed_at": _utc_now(),
            },
        )


def _carps_task_root() -> Path:
    return Path(carps.__file__).resolve().parent / "configs" / "task"


def _bbob_sort_key(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"cfg_2_(\d+)_(\d+)\.yaml", path.name)
    if match is None:
        raise ValueError(f"Unexpected BBOB-2D task filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def _read_task(path: Path, task_group: str, config_name: str) -> dict[str, Any]:
    cfg = OmegaConf.load(path)
    n_trials = int(cfg.task.optimization_resources.n_trials)
    task_id = str(cfg.task.name)
    if n_trials <= 0:
        raise ValueError(f"Packaged task has a nonpositive n_trials: {path}")
    return {
        "task_group": task_group,
        "config_name": config_name,
        "task_config_path": str(path),
        "task_config_sha256": file_sha256(path),
        "task_id": task_id,
        "benchmark_id": str(cfg.benchmark_id),
        "default_n_trials": n_trials,
    }


def discover_packaged_tasks() -> list[dict[str, Any]]:
    """Enumerate every packaged BBOB-2D and YAHPO/SO configuration."""
    root = _carps_task_root()
    bbob_paths = sorted((root / "BBOB").glob("cfg_2_*_0.yaml"), key=_bbob_sort_key)
    yahpo_paths = sorted((root / "YAHPO" / "SO").glob("cfg_*.yaml"), key=lambda path: path.name)
    if len(bbob_paths) != EXPECTED_BBOB_CONFIGS:
        raise RuntimeError(f"Expected {EXPECTED_BBOB_CONFIGS} packaged BBOB-2D configs, found {len(bbob_paths)}.")
    if len(yahpo_paths) != EXPECTED_YAHPO_CONFIGS:
        raise RuntimeError(f"Expected {EXPECTED_YAHPO_CONFIGS} packaged YAHPO/SO configs, found {len(yahpo_paths)}.")
    tasks = [_read_task(path, "BBOB", path.stem) for path in bbob_paths]
    tasks.extend(_read_task(path, "YAHPO/SO", path.stem) for path in yahpo_paths)
    if len({task["task_id"] for task in tasks}) != len(tasks):
        raise RuntimeError("Packaged GP-statistics task IDs are not unique.")
    return tasks


def build_inventory(output_root: Path) -> dict[str, Any]:
    """Freeze the 44-task by two-seed CARP-S job inventory."""
    output_root = output_root.resolve()
    tasks = discover_packaged_tasks()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        domain = "bbob" if task["task_group"] == "BBOB" else "yahpo"
        for seed in CARPS_SEEDS:
            output_directory = output_root / "runs" / domain / task["config_name"] / f"seed_{seed}"
            rows.append(
                {
                    "job_index": len(rows) + 1,
                    **task,
                    "domain": domain,
                    "seed": seed,
                    "output_directory": str(output_directory),
                }
            )
    scientific_rows = [{name: value for name, value in row.items() if name != "output_directory"} for row in rows]
    manifest_hash = canonical_sha256(
        {
            "campaign_version": CAMPAIGN_VERSION,
            "carps_version": version("carps"),
            "optimizer": "SMAC3 BlackBoxFacade",
            "seeds": list(CARPS_SEEDS),
            "rows": scientific_rows,
        }
    )
    payload = {
        "campaign_version": CAMPAIGN_VERSION,
        "created_at": _utc_now(),
        "manifest_hash": manifest_hash,
        "carps_version": version("carps"),
        "smac_version": version("smac"),
        "optimizer_config": "optimizer/smac20=blackbox",
        "optimizer_facade": "smac.facade.blackbox_facade.BlackBoxFacade",
        "task_config_root": str(_carps_task_root()),
        "seeds": list(CARPS_SEEDS),
        "bbob_task_count": EXPECTED_BBOB_CONFIGS,
        "yahpo_task_count": EXPECTED_YAHPO_CONFIGS,
        "unique_task_count": len(tasks),
        "job_count": len(rows),
        "native_task_budgets_preserved": True,
        "jobs": rows,
    }
    destination = output_root / "inventory.json"
    if destination.is_file():
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing.get("manifest_hash") != manifest_hash or existing.get("jobs") != rows:
            raise RuntimeError(f"Refusing to replace a different campaign inventory at {destination}.")
        return existing
    _atomic_json(destination, payload)
    return payload


def load_inventory(path: Path, expected_hash: str | None = None) -> dict[str, Any]:
    """Load and validate one frozen campaign inventory."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("campaign_version") != CAMPAIGN_VERSION:
        raise ValueError(f"Unsupported campaign inventory: {payload.get('campaign_version')!r}.")
    if expected_hash is not None and payload.get("manifest_hash") != expected_hash:
        raise ValueError("GP-statistics inventory hash mismatch.")
    if len(payload.get("jobs", [])) != int(payload.get("job_count", -1)):
        raise ValueError("GP-statistics inventory job count is inconsistent.")
    return payload


def inventory_row(inventory: dict[str, Any], index: int) -> dict[str, Any]:
    """Return one one-based job row after checking frozen ordering."""
    if not 1 <= index <= len(inventory["jobs"]):
        raise IndexError(f"Job index {index} is outside 1..{len(inventory['jobs'])}.")
    row = inventory["jobs"][index - 1]
    if int(row["job_index"]) != index:
        raise ValueError("GP-statistics inventory ordering is corrupt.")
    return row


def _status_path(row: dict[str, Any]) -> Path:
    return Path(row["output_directory"]) / "status.json"


def job_decision(row: dict[str, Any], manifest_hash: str) -> str:
    """Return run, skip, or corrupt for one expected output cell."""
    status_path = _status_path(row)
    if not status_path.is_file():
        return "run"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    identity_matches = all(
        status.get(name) == value
        for name, value in {
            "manifest_hash": manifest_hash,
            "job_index": row["job_index"],
            "task_id": row["task_id"],
            "seed": row["seed"],
            "default_n_trials": row["default_n_trials"],
        }.items()
    )
    if status.get("status") == "success" and identity_matches:
        stats_path = Path(row["output_directory"]) / "gp_statistics.jsonl"
        return "skip" if stats_path.is_file() and stats_path.stat().st_size > 0 else "corrupt"
    if status.get("status") == "success" and not identity_matches:
        return "corrupt"
    return "run"


def write_job_status(
    row: dict[str, Any],
    manifest_hash: str,
    state: str,
    *,
    exit_code: int | None = None,
    message: str | None = None,
) -> None:
    """Atomically persist a terminal or in-progress job status."""
    if state not in {"running", "success", "failed"}:
        raise ValueError(f"Unknown job state {state!r}.")
    payload = {
        "campaign_version": CAMPAIGN_VERSION,
        "manifest_hash": manifest_hash,
        "status": state,
        "job_index": row["job_index"],
        "task_group": row["task_group"],
        "config_name": row["config_name"],
        "task_id": row["task_id"],
        "seed": row["seed"],
        "default_n_trials": row["default_n_trials"],
        "task_config_sha256": row["task_config_sha256"],
        "output_directory": row["output_directory"],
        "timestamp": _utc_now(),
        "exit_code": exit_code,
        "message": message,
    }
    _atomic_json(_status_path(row), payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON in {path}:{line_number}.") from error
    return records


def _safe_task_name(task_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", task_id).strip("_")


def _plot_task(task_frame: pd.DataFrame, output_path: Path) -> None:
    """Plot one task with feature columns arranged by seaborn FacetGrid."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import seaborn as sns  # noqa: PLC0415
    from matplotlib import pyplot as plt  # noqa: PLC0415
    from matplotlib.backends.backend_pdf import PdfPages  # noqa: PLC0415

    feature_names = [*GP_HP_SUMMARY_NAMES, *GP_HP_CHANGE_NAMES]
    missing = set(feature_names) - set(task_frame)
    if missing:
        raise ValueError(f"Plotted GP feature columns are missing: {sorted(missing)}")

    with PdfPages(output_path) as pdf:
        for group_name, names in (("GP summary", GP_HP_SUMMARY_NAMES), ("GP change", GP_HP_CHANGE_NAMES)):
            long_frame = task_frame.melt(
                id_vars=("runhistory_finished", "seed"),
                value_vars=names,
                var_name="feature",
                value_name="value",
            )
            long_frame["CARP-S seed"] = long_frame["seed"].map(lambda seed: str(int(seed)))
            grid = sns.FacetGrid(
                long_frame,
                col="feature",
                hue="CARP-S seed",
                col_wrap=4,
                col_order=list(names),
                sharex=True,
                sharey=False,
                height=2.8,
                aspect=1.25,
            )
            grid.map_dataframe(
                sns.lineplot,
                x="runhistory_finished",
                y="value",
                estimator=None,
                errorbar=None,
                linewidth=1.2,
            )
            grid.add_legend(title="CARP-S seed")
            grid.set_axis_labels("completed objective evaluations", "value")
            grid.set_titles("{col_name}")
            for axis in grid.axes.flat:
                axis.grid(alpha=0.25)
            grid.figure.suptitle(f"{task_frame['task_id'].iloc[0]} — {group_name}")
            grid.figure.tight_layout(rect=(0, 0, 1, 0.98))
            pdf.savefig(grid.figure)
            plt.close(grid.figure)


def consolidate(inventory_path: Path, output_root: Path) -> dict[str, Any]:
    """Validate all jobs, create tidy tables, and plot every task."""
    inventory = load_inventory(inventory_path)
    audit_rows: list[dict[str, Any]] = []
    wide_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for row in inventory["jobs"]:
        decision = job_decision(row, inventory["manifest_hash"])
        status_path = _status_path(row)
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else {}
        stats_path = Path(row["output_directory"]) / "gp_statistics.jsonl"
        records = _read_jsonl(stats_path) if stats_path.is_file() else []
        audit_rows.append(
            {
                "job_index": row["job_index"],
                "task_id": row["task_id"],
                "seed": row["seed"],
                "status": status.get("status", "missing"),
                "decision": decision,
                "record_count": len(records),
                "expected_n_trials": row["default_n_trials"],
            }
        )
        if decision != "skip":
            continue
        if len(records) != int(row["default_n_trials"]):
            audit_rows[-1]["decision"] = "incomplete-record-count"
            continue
        for record in records:
            wide = {
                "manifest_hash": inventory["manifest_hash"],
                "domain": row["domain"],
                "task_group": row["task_group"],
                "config_name": row["config_name"],
                "task_id": row["task_id"],
                "seed": row["seed"],
                "ask_index": int(record["ask_index"]),
                "runhistory_finished": int(record["runhistory_finished"]),
                "state_key": json.dumps(record["state_key"], separators=(",", ":")),
                "model_revision_changed": bool(record["model_revision_changed"]),
                **{name: float(record["summary"][name]) for name in GP_HP_SUMMARY_NAMES},
                **{name: float(record["change"][name]) for name in GP_HP_CHANGE_NAMES},
            }
            wide_rows.append(wide)
            for parameter in record["parameters"]:
                raw_rows.append(
                    {
                        **{
                            key: wide[key]
                            for key in (
                                "manifest_hash",
                                "domain",
                                "task_id",
                                "seed",
                                "ask_index",
                                "runhistory_finished",
                            )
                        },
                        **parameter,
                    }
                )

    audit = pd.DataFrame(audit_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_root / "completion_audit.csv", index=False)
    incomplete = audit[audit["decision"] != "skip"]
    if not incomplete.empty:
        _atomic_json(
            output_root / "consolidation_status.json",
            {
                "status": "incomplete",
                "manifest_hash": inventory["manifest_hash"],
                "expected_jobs": len(inventory["jobs"]),
                "complete_jobs": int((audit["decision"] == "skip").sum()),
                "incomplete_jobs": incomplete.to_dict(orient="records"),
            },
        )
        raise RuntimeError(f"GP-statistics campaign has {len(incomplete)} incomplete/corrupt jobs.")

    wide_frame = pd.DataFrame(wide_rows).sort_values(["domain", "task_id", "seed", "ask_index"])
    raw_frame = pd.DataFrame(raw_rows)
    wide_frame.to_parquet(output_root / "gp_statistics.parquet", index=False)
    wide_frame.to_csv(output_root / "gp_statistics.csv", index=False)
    raw_frame.to_parquet(output_root / "gp_raw_parameters.parquet", index=False)
    plot_index = []
    for (domain, task_id), task_frame in wide_frame.groupby(["domain", "task_id"], sort=True):
        plot_path = output_root / "plots" / domain / f"{_safe_task_name(task_id)}.pdf"
        _plot_task(task_frame, plot_path)
        plot_index.append(
            {
                "domain": domain,
                "task_id": task_id,
                "plot_path": str(plot_path),
                "plot_type": "seaborn.FacetGrid",
                "facet_variable": "feature",
                "hue_variable": "seed",
                "records": len(task_frame),
                "seeds": sorted(int(seed) for seed in task_frame["seed"].unique()),
                "available_fraction": float(task_frame["available"].mean()),
            }
        )
    result = {
        "status": "success",
        "campaign_version": CAMPAIGN_VERSION,
        "manifest_hash": inventory["manifest_hash"],
        "job_count": len(inventory["jobs"]),
        "task_count": len(plot_index),
        "observation_count": len(wide_frame),
        "raw_parameter_row_count": len(raw_frame),
        "summary_names": list(GP_HP_SUMMARY_NAMES),
        "change_names": list(GP_HP_CHANGE_NAMES),
        "role_names": list(GP_HP_ROLE_NAMES),
        "plot_type": "seaborn.FacetGrid",
        "plots": plot_index,
        "gp_available_fraction": float(wide_frame.iloc[:, wide_frame.columns.get_loc("available")].mean()),
        "completed_at": _utc_now(),
    }
    _atomic_json(output_root / "plot_index.json", result)
    _atomic_json(output_root / "consolidation_status.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output-root", type=Path, required=True)
    fields = subparsers.add_parser("job-fields")
    fields.add_argument("--inventory", type=Path, required=True)
    fields.add_argument("--manifest-hash", required=True)
    fields.add_argument("--index", type=int, required=True)
    decision = subparsers.add_parser("job-decision")
    decision.add_argument("--inventory", type=Path, required=True)
    decision.add_argument("--manifest-hash", required=True)
    decision.add_argument("--index", type=int, required=True)
    status = subparsers.add_parser("write-status")
    status.add_argument("--inventory", type=Path, required=True)
    status.add_argument("--manifest-hash", required=True)
    status.add_argument("--index", type=int, required=True)
    status.add_argument("--state", choices=("running", "success", "failed"), required=True)
    status.add_argument("--exit-code", type=int)
    status.add_argument("--message")
    merge = subparsers.add_parser("consolidate")
    merge.add_argument("--inventory", type=Path, required=True)
    merge.add_argument("--output-root", type=Path, required=True)
    submission = subparsers.add_parser("record-submission")
    submission.add_argument("--output", type=Path, required=True)
    submission.add_argument("--inventory", type=Path, required=True)
    submission.add_argument("--manifest-hash", required=True)
    submission.add_argument("--job-id", required=True)
    submission.add_argument("--group", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute inventory, worker-helper, or consolidation commands."""
    args = _parser().parse_args(argv)
    if args.command == "build":
        payload = build_inventory(args.output_root)
        print(json.dumps({key: payload[key] for key in ("manifest_hash", "job_count", "unique_task_count")}))
        return
    if args.command == "consolidate":
        print(json.dumps(consolidate(args.inventory, args.output_root), indent=2))
        return
    if args.command == "record-submission":
        inventory = load_inventory(args.inventory, args.manifest_hash)
        if args.output.exists():
            raise FileExistsError(f"Refusing duplicate submission record: {args.output}")
        _atomic_json(
            args.output,
            {
                "campaign_version": CAMPAIGN_VERSION,
                "manifest_hash": inventory["manifest_hash"],
                "job_count": inventory["job_count"],
                "job_id": args.job_id,
                "group": args.group,
                "submitted_at": _utc_now(),
            },
        )
        return
    inventory = load_inventory(args.inventory, args.manifest_hash)
    row = inventory_row(inventory, args.index)
    if args.command == "job-fields":
        for value in (
            row["task_group"],
            row["config_name"],
            row["task_id"],
            row["seed"],
            row["default_n_trials"],
            row["output_directory"],
        ):
            print(value)
        return
    if args.command == "job-decision":
        print(job_decision(row, inventory["manifest_hash"]))
        return
    if args.command == "write-status":
        write_job_status(
            row,
            inventory["manifest_hash"],
            args.state,
            exit_code=args.exit_code,
            message=args.message,
        )
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
