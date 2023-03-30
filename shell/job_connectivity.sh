#!/bin/bash
#SBATCH -J CM2
#SBATCH -p short
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH --time=2:00
#SBATCH --array=1-21:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python3 /homedtic/rzucca/epigame/code/connectivity_matrices.py /homedtic/rzucca/epigame/data/${SLURM_ARRAY_TASK_ID}-preseizure5.prep
