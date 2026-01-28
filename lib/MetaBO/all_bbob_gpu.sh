for fid in {1..24}; do
    for d in 2 8; do
        sbatch train_metabo_bbob_gpu.sh --fid $fid --d $d
    done
done