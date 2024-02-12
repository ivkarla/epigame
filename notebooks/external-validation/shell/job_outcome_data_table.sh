#!/bin/bash
#SBATCH -J s2
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=16GB
#SBATCH -t 0-2:00

#SBATCH -o %J_%A_%a.out #STDOUT
#SBATCH -e %J_%A_%a.err #STDERR

python3 /homes/users/kivankovic/ext-val/scripts/outcome_data_table_ext.py ${SLURM_ARRAY_TASK_ID} 2
