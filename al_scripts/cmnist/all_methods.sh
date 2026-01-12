#!/bin/bash
#SBATCH --job-name=cmnistmethods
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-139%30
source $SCRATCH/python_envs/al_wg/bin/activate
METHODS=(entropy_per_source_n_largest entropy_per_source_top_k entropy_per_source_soft_max entropy_per_source_soft_rank entropy_per_source entropy_per_source_power n_largest_top_k n_largest_soft_max n_largest_soft_rank n_largest_power random random_gdro uniform_sources entropy) #14

N_SEEDS=10
N_METHODS=${#METHODS[@]}
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))
METHOD_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_METHODS ))
python al_main.py --mode cmnist  --project_name cmnist_dino_jan --acquisition ${METHODS[$METHOD_IDX]} --seed $SEED 

