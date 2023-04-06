#!/bin/bash
#SBATCH -J SVMp
#SBATCH -p short
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH --array=1-21:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python /homedtic/rzucca/epigame/code/kfold_pairs.py ${SLURM_ARRAY_TASK_ID} "preseizure1"