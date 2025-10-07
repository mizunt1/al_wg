#!/bin/bash
#SBATCH --job-name=wandbtest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:10:00
#SBATCH --mem=32Gb
#SBATCH --array=0-24
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
source $HOME/python_envs/alsgs/bin/activate
LR=(1e-2 1e-3 1e-4 1e-5 1e-6)
N_LR=${#LR[@]}
BATCH=(8 16 32 64 128)
N_BATCH=${#BATCH[@]}
BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_BATCH))
LR_IDX=$(( (SLURM_ARRAY_TASK_ID / N_BATCH) % N_LR ))
python uq_tests.py --minority_prop 0.01 --data_mode celeba --lr ${LR[$LR_IDX]}  --batch_size ${BATCH[$BATCH_IDX]} --num_epochs 40 --save_dir results_hps_celeba --project_name uq_test_celeba_hps2
