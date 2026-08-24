#!/usr/bin/env bash
# Prepare and submit optional offline-initialized DDQN-f5 fine-tuning.
set -euo pipefail
if (( $# != 4 )); then
  echo "Usage: $0 OFFLINE_RUN_ROOT FINAL_DATASET_ROOT OUTPUT_ROOT yahpo|mixed" >&2
  exit 2
fi
run_root="$(realpath "$1")" final_root="$(realpath "$2")"
mkdir -p "$3"; output_root="$(realpath "$3")"; domain="$4"
[[ "${domain}" == "yahpo" || "${domain}" == "mixed" ]] || { echo "Domain must be yahpo or mixed." >&2; exit 2; }
repository="$(git rev-parse --show-toplevel)"; cd "${repository}"
python_bin="${repository}/.venv/bin/python"
[[ -x "${python_bin}" ]] || { echo "Otus .venv is missing." >&2; exit 1; }
manifest="${output_root}/offline_finetune_manifest.tsv"
[[ ! -e "${manifest}" ]] || { echo "Manifest exists; refusing duplicate submission." >&2; exit 3; }
printf 'run_id\tsource_policy_id\tcheckpoint\tcheckpoint_hash\tnormalizer\tnormalizer_hash\tseed\n' > "${manifest}"
"${python_bin}" - "${run_root}" "${domain}" >> "${manifest}" <<'PY'
import hashlib,json,pathlib,sys,torch
from dacboenv.offline.deployment import deployment_head_for_mode
from dacboenv.offline.identity import offline_policy_id
root=pathlib.Path(sys.argv[1]); domain=sys.argv[2]
for complete in sorted(root.rglob("training_complete.json")):
    summary=json.load(open(complete)); run=complete.parent
    if summary["domain"] != domain: continue
    checkpoint=run/"best_branch_dev.pt"
    payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
    resolved=payload["resolved_config"]
    normalizer=pathlib.Path(resolved["offline_dataset"]["root"])/"normalization_schema.json"
    digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    mode=str(resolved["offline_training"].get("algorithm_mode",resolved["offline_algorithm"]["mode"]))
    selection=payload.get("deployment_selection",{})
    if selection.get("deployment_head") != deployment_head_for_mode(mode):
        raise RuntimeError(f"Deployment/checkpoint head mismatch in {checkpoint}")
    if selection.get("checkpoint_selection_metric") != "dev/deployment_selected_value" or not selection.get("deployment_selection_eligible"):
        raise RuntimeError(f"Checkpoint is not an eligible deployment-selected artifact: {checkpoint}")
    if int(selection.get("selected_update",-1)) != int(payload["update"]):
        raise RuntimeError(f"Best checkpoint update disagrees with selection metadata: {checkpoint}")
    coefficient=float(resolved["offline_training"].get("cql_coefficient",resolved["offline_algorithm"]["cql_coefficient"]))
    source_id=offline_policy_id(experiment_id=str(resolved["offline_training"]["experiment_id"]),algorithm_mode=mode,cql_coefficient=coefficient,training_seed=int(resolved["seed"]),selected_update=int(payload["update"]),checkpoint_mode="best_branch_dev",model_sha256=digest(checkpoint))
    run_id=f"finetune-{source_id}"
    print("\t".join(map(str,[run_id,source_id,checkpoint.resolve(),digest(checkpoint),normalizer.resolve(),digest(normalizer),resolved["seed"]])))
PY
count="$(( $(wc -l < "${manifest}") - 1 ))"
(( count > 0 )) || { echo "No completed ${domain} offline runs found." >&2; exit 1; }
"${python_bin}" -c 'import csv,sys; p=sys.argv[1]; rows=list(csv.DictReader(open(p),delimiter="\t")); identifiers=[r["run_id"] for r in rows]; sources=[r["source_policy_id"] for r in rows]; assert len(identifiers)==len(set(identifiers)),"duplicate fine-tuning run IDs"; assert len(sources)==len(set(sources)),"duplicate source policy IDs"; print("manifest={}".format(p)); print("rows={}".format(len(rows))); print("source_policy_ids={}".format(sorted(sources))); print("output_root={}".format(sys.argv[2]))' "${manifest}" "${output_root}/runs"
mkdir -p slurmlogs/offline-finetune
# No %N suffix: Otus schedules all eligible fine-tuning cells.
sbatch --array="0-$((count - 1))" scripts/otus/opt_offline_finetune.sh \
  "${repository}" "${manifest}" "${final_root}" "${output_root}/runs" "${domain}"
