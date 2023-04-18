#!/bin/bash
#SBATCH -J EPIG
#SBATCH -p middle
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH --array=1-21:1
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python /homedtic/rzucca/epigame/code/matches.py ${SLURM_ARRAY_TASK_ID} "preseizure1"