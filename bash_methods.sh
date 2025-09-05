#!/bin/bash
#SBATCH --job-name=wandbtest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:10:00
#SBATCH --mem=10Gb
#SBATCH --array=0-20
data_dir=/network/scratch/m/mizu.nishikawa-toomey/
module load python/3.9
module load cuda/11.8
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
source $HOME/python_envs/alsgs/bin/activate
METHODS=(entropy entropy_per_group accuracy uniform_groups)
N_METHODS=${#METHODS[@]}
N_SEEDS=5
SEED=$(( SLURM_ARRAY_TASK_ID % N_SEEDS))
METHOD_IDX=$(( (SLURM_ARRAY_TASK_ID / N_SEEDS) % N_METHODS ))
python cmnist_al.py --project_name ten_groups_mult_int --seed $SEED --data_mode ten_groups_multiple_int --al_size 30 --acquisition ${METHODS[$METHOD_IDX]}  --start_acquisition uniform_groups --causal_noise 0.0 --spurious_noise 0.0 
