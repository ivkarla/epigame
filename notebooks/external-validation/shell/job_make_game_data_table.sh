#!/bin/bash
#SBATCH -J tab
#SBATCH -p normal
#SBATCH -N 2
#SBATCH -n 4
#SBATCH --mem=16GB
#SBATCH -t 0-2:00

#SBATCH -o %J_%A_%a.out #STDOUT
#SBATCH -e %J_%A_%a.err #STDERR

# Load the Singularity module
module load Singularity/3.7.1-foss-2016b

echo "Virtual Environment Path: /opt/venv"
# Activate the virtual env
source /opt/venv/bin/activate

echo "Python Version: $(python --version)"

# Run the Python script within the virtual environment
singularity exec epigame.sif /opt/venv/bin/python /homes/users/kivankovic/ext-val/scripts/make_game_data_table_ext.py