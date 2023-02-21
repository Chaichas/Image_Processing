#!/bin/bash
#SBATCH --job-name=image_processing
#SBATCH --output=%x.o%j
#SBATCH --error=%x.e%j
#SBATCH --time=01:00:00 

# Compute nodes
#SBATCH -N 1

# Compute cores
#SBATCH -c 1

# Compute the number of GPU 
#SBATCH --gres=gpu:1
#SBATCH --partition=instant

# Echo of commands
set -x

# Execution
./modif_img.exe
