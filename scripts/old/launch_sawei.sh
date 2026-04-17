export BASEENV="+env=base +env/opt=base +env/action=wei_alpha_continuous +env/obs=sawei +env/reward=ep_done_scaled +env/refperf=saweip dacboenv.evaluation_mode=true"

python -m carps.run "hydra.searchpath=[pkg://dacboenv/configs]" \
    '+task/BBOB=glob(cfg_16_*_0)' \
    +eval=base \
    baserundir=runs_static \
    $BASEENV \
    +policy=sawei \
    +cluster=cpu_noctua 'seed=range(1,11)' \
    -m &
