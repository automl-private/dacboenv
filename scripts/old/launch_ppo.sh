for instance in {1..24}; do
    sbatch opt_ppo.sh +instances=bbob2d_${instance}_3seeds +task=dacboenv_epdonescaledpluslogregret
    sbatch opt_ppo.sh +instances=bbob2d_${instance}_3seeds +task=dacboenv_epdonescaledpluslogregret_wei

    sbatch opt_ppo_norm.sh +instances=bbob2d_${instance}_3seeds +task=dacboenv_epdonescaledpluslogregret optimizer_id=PPO-norm-Perceptron
    sbatch opt_ppo_norm.sh +instances=bbob2d_${instance}_3seeds +task=dacboenv_epdonescaledpluslogregret_wei optimizer_id=PPO-norm-Perceptron
done

sbatch opt_ppo.sh +instances=bbob2d_3seeds +task=dacboenv_epdonescaledpluslogregret
sbatch opt_ppo.sh +instances=bbob2d_3seeds +task=dacboenv_epdonescaledpluslogregret_wei

sbatch opt_ppo_norm.sh +instances=bbob2d_3seeds +task=dacboenv_epdonescaledpluslogregret optimizer_id=PPO-norm-Perceptron
sbatch opt_ppo_norm.sh +instances=bbob2d_3seeds +task=dacboenv_epdonescaledpluslogregret_wei optimizer_id=PPO-norm-Perceptron