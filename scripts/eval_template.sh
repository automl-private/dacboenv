export HYDRA_FULL_ERROR=1

seeds="seed=range(1,21)"
methods=(
    # "+optimizer/smac20=blackbox"
    "+method/schedule=cslog_rf"

)
tasks=(
    "+task/BBOB=glob(cfg_8_*_0)"
    "+task/YAHPO/SO=glob(*)"
    "+task/BNNBO=glob(*)  hydra.launcher.mem_per_cpu=16G"
)

for method in "${methods[@]}"; do    
    for task in "${tasks[@]}"; do    
        python -m carps.run "hydra.searchpath=['pkg://dacboenv/configs']" +cluster=cpu_noctua \
            $seeds \
            $task \
            $method \
            -m &
    done
done