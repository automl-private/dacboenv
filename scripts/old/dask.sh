sbatch dask_cluster.sh seed=1 experiment.n_episodes=1000
sbatch dask_cluster.sh seed=2 experiment.n_episodes=1000
sbatch dask_cluster.sh seed=3 experiment.n_episodes=1000
sbatch dask_cluster.sh seed=4 experiment.n_episodes=1000
sbatch dask_cluster.sh seed=5 experiment.n_episodes=1000

sbatch dask_cluster.sh seed=6 experiment.instances=[\"8_2\",\"11_2\",\"19_2\"] experiment.experiment_id=8_2-11_2-19_2
sbatch dask_cluster.sh seed=7 experiment.instances=[\"8_2\",\"11_2\",\"19_2\"] experiment.experiment_id=8_2-11_2-19_2
sbatch dask_cluster.sh seed=8 experiment.instances=[\"8_2\",\"11_2\",\"19_2\"] experiment.experiment_id=8_2-11_2-19_2
sbatch dask_cluster.sh seed=9 experiment.instances=[\"8_2\",\"11_2\",\"19_2\"] experiment.experiment_id=8_2-11_2-19_2
sbatch dask_cluster.sh seed=10 experiment.instances=[\"8_2\",\"11_2\",\"19_2\"] experiment.experiment_id=8_2-11_2-19_2