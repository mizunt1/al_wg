#!/bin/bash
#SBATCH --job-name=resnet
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:10:00
#SBATCH --mem=32Gb
source $HOME/python_envs/alsgs/bin/activate
python waterbirds4g_al.py --model_name BayesianNetRes50ULarger --lr 1e-4 --batch_size 30 --acquisition random --num_minority_points 100

