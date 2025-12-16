BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK1="+task/BBOB=cfg_2_1_0"
TASK20="+task/BBOB=cfg_2_20_0"
TASK8="+task/BBOB=cfg_2_8_0"
TASKALL="+task/BBOB=glob(cfg_2_*_0)"

if command -v sinfo >/dev/null 2>&1 || [ -f /etc/slurm/slurm.conf ]; then
    CLUSTER="+cluster=cpu_noctua"
else
    CLUSTER=""
fi

TASK="$1"
if [ -z "$TASK" ]; then
    echo "Error: TASK is empty. Please provide a task as the first argument (+task...)." >&2
    exit 1
fi
POLICYPATH="$2"
if [ -z "$POLICYPATH" ]; then
    echo "Error: POLICYPATH is empty. Please provide a task as the second argument (e.g., dacboenv/configs/policy/optimized/SMAC-AC-CostInc/dacbo_default_on_2dbbob)." >&2
    exit 1
fi
policy_opt_folder=$POLICYPATH
policy_override_base="${policy_opt_folder/dacboenv\/configs\//+}"

file_list=$(find "$policy_opt_folder" -maxdepth 1 -type f -printf "%f\n" \
  | sed 's/\.[^.]*$//' \
  | paste -sd, -)

policy_override="$policy_override_base=$file_list"
echo "Found configs of policies: $file_list"
echo "Policy override: $policy_override"

python -m $BASE $TASK $SEED +eval=base +eval/opt=base $policy_override $CLUSTER --multirun