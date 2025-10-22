#!/bin/bash
#SBATCH --job-name=wandbtest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:80gb
#SBATCH --time=10:10:00
#SBATCH --mem=48Gb
#SBATCH --array=0-45
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
source $HOME/python_envs/alsgs/bin/activate

LR=(1e-3 1e-4 1e-5 1e-6 1e-7)
DROP_OUT=(0.5 0.7 0.9)
BATCH=(2 4 6)
N_LR=${#LR[@]}
N_BATCH=${#BATCH[@]}
N_DROP=${#DROP_OUT[@]}

BATCH_IDX=$(( SLURM_ARRAY_TASK_ID % N_BATCH))
LR_IDX=$(( (SLURM_ARRAY_TASK_ID / N_BATCH) % N_LR ))
DROP_IDX=$(( (SLURM_ARRAY_TASK_ID / (N_BATCH * N_LR)) % N_DROP ))

python uq_tests.py --minority_prop 0.1 --data_mode wb --lr ${LR[$LR_IDX]}  --batch_size ${BATCH[$BATCH_IDX]} --num_epochs 100 --save_dir results --project_name dino_hps_wb --model_name BayesianNetDino --mc_drop_p ${DROP_OUT[$DROP_IDX]} --frozen_weights
