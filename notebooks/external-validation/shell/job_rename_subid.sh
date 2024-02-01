#!/bin/bash
#SBATCH -J FNAME
#SBATCH -p short
#SBATCH -N 1
#SBATCH --chdir=/homes/users/kivankovic/ext-val
#SBATCH --time=2:00
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2
#SBATCH -o %N.%J.out #STDOUT
#SBATCH -e %N.%J.err #STDERR

python3 /homes/users/kivankovic/ext-val/scripts/rename_sub_id.py /homes/users/kivankovic/ext-val/
