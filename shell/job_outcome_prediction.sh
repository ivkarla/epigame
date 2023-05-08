#!/bin/bash
#SBATCH -J SOP
#SBATCH -p high
#SBATCH -N 4
#SBATCH -n 16
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python /homedtic/rzucca/epigame/code/outcome_prediction.py "preseizure1"