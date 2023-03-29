#!/bin/bash
#SBATCH -J CM2
#SBATCH -p short
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH --time=2:00
#SBATCH --array=1-21:1
#SBATCH --mem=4G
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

INPUT_SEIZURE = $(ls /homedtic/rzucca/data/seizure/${SLURM_ARRAY_TASK_ID}-seizure.EDF)
INPUT_BASELINE = $(ls /homedtic/rzucca/data/baseline/${SLURM_ARRAY_TASK_ID}-baseline.EDF)

python3 /homedtic/rzucca/epigame/code/connectivity_matrices_2.py $INPUT_SEIZURE $INPUT_BASELINE 2
