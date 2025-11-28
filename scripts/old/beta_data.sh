#!/bin/bash

BASE="carps.run hydra.searchpath=[pkg://dacboenv/configs]"
SEED="seed=range(1,11)"
TASK="+task/BBOB=glob(cfg_2_*_0)"
CLUSTER="+cluster=cpu_noctua"

MAX_JOBS=5000
JOBS=0

modes="htl lth"
jumps="025 05 075"
static="low mid high"

# Jumps

for mode in $modes; do
    for jump in $jumps; do
        python -m $BASE +base=dacboenv_beta_jump_$mode\_$jump $TASK $CLUSTER $SEED --multirun &
        ((JOBS+=240))
        if [[ $JOBS -ge $MAX_JOBS ]]; then
            wait
            JOBS=0
        fi
    done
done

# Linear

for mode in $modes; do
    python -m $BASE +base=dacboenv_beta_linear_$mode $TASK $CLUSTER $SEED --multirun &
    ((JOBS+=240))
    if [[ $JOBS -ge $MAX_JOBS ]]; then
        wait
        JOBS=0
    fi
done

# Static

for stat in $static; do
    python -m $BASE +base=dacboenv_beta_static_$stat $TASK $CLUSTER $SEED --multirun &
    ((JOBS+=240))
    if [[ $JOBS -ge $MAX_JOBS ]]; then
        wait
        JOBS=0
    fi
done

# Random

python -m $BASE +base=dacboenv_beta_random $TASK $CLUSTER $SEED --multirun