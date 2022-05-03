#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Ensures all allocated cores are on the same node
#SBATCH -N 1

# Request 4 CPU core(s)
#SBATCH -n 4
# Memory is evenly distributed amongst number of cores

#SBATCH --mem=70G
#SBATCH -t 24:00:00

#SBATCH -o transformer_output.out
#SBATCH -e transformer_err.out

## Provide a job name
#SBATCH -J csci1430_cchen207

module load python/3.8.12_gcc8.3

source ../../cs1430_env/bin/activate 

python3 model.py --evaluate True --load-checkpoint checkpoints/model-016-0.991479-0.971264.h5
