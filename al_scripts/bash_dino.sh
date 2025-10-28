#!/bin/bash
#SBATCH --job-name=dino_ent
#SBATCH --ntasks=1
#SBATCH --gres=gpu:48gb
#SBATCH --time=10:10:00
#SBATCH --mem=32Gb
source $HOME/python_envs/alsgs/bin/activate
python waterbirds4g_al.py --model_name BayesianNetDino --lr 1e-6 --batch_size 2 --acquisition mi --num_minority_points 100

