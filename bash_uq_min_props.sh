#!/bin/bash
#SBATCH --job-name=wandbtest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:10:00
#SBATCH --mem=32Gb
#SBATCH --array=0-50
source $HOME/python_envs/alsgs/bin/activate
PROPS=(1e-3 5e-3 7e-3 1e-2 3e-2 0.1 0.2 0.3 0.4 0.5)
N_PROPS=${#PROPS[@]}
N_SEEDS=5
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))
PROP_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_PROPS ))
python uq_tests.py --seed $SEED --minority_prop ${PROPS[$PROP_IDX]}  
