#!/bin/bash
#SBATCH --job-name=wb_seeds_methods
#SBATCH --ntasks=1
#SBATCH --gres=gpu:48gb
#SBATCH --time=12:10:00
#SBATCH --mem=32Gb
#SBATCH --array=0-139%30
data_dir=/network/scratch/m/mizu.nishikawa-toomey/
module load python/3.9
module load cuda/11.8
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
source $HOME/python_envs/alsgs/bin/activate

METHODS=(entropy_per_source_n_largest entropy_per_source_top_k entropy_per_source_soft_max entropy_per_source_soft_rank entropy_per_source entropy_per_source_power n_largest_top_k n_largest_soft_max n_largest_soft_rank n_largest_power random random_gdro uniform_sources entropy)

N_METHODS=${#METHODS[@]}
N_SEEDS=10
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))
METHOD_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_METHODS ))

python al_main.py --mode wb --acquisition ${METHODS[$METHOD_IDX]} --seed $SEED --project_name wb_results_jan


