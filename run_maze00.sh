#!/bin/bash
#SBATCH --job-name=38
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=2G

module load PyTorch

source /data/$USER/.envs/maze/bin/activate


echo Starting Python program
python3 /home/p285087/Environments/Maze/main_maze2.py

deactivate