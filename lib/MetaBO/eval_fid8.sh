for fid in {1..24}; do
    for d in 2 8; do
        sbatch eval_metabo_bbob.sh --fideval $fid --d $d --fidtrain 8
    done
done