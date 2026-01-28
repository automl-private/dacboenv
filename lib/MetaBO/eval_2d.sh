for fid in {1..24}; do
    sbatch eval_metabo_bbob2d.sh --fideval $fid --d 2
done