#!/bin/bash
#SBATCH -J rdg
#SBATCH -p medium
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python /homedtic/rzucca/epigame/code/rd_groups_match.py "seizure1" 2