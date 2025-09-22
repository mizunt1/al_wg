#!/bin/bash
#SBATCH --job-name=wandbtest
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:10:00
#SBATCH --mem=32Gb
#SBATCH --array=0-11
data_dir=/network/scratch/m/mizu.nishikawa-toomey/
module load python/3.9
module load cuda/11.8
WANDB_API_KEY=$17a113b4804951bde9c66b2002fe378c0209fb64
source $HOME/python_envs/alsgs/bin/activate
wandb agent mizunt/al_wg/ffrd81eg
