#!/bin/bash
#SBATCH -J GDT
#SBATCH -p short
#SBATCH -N 2
#SBATCH -n 8
#SBATCH --chdir=/homedtic/rzucca/epigame
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python /homedtic/rzucca/epigame/code/make_game_data_table.py ${SLURM_ARRAY_TASK_ID} "preseizure1"