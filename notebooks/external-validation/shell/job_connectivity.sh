#!/bin/bash
#SBATCH -J CM14
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
echo "NumPy Version: $(python -c 'import numpy; print(numpy.__version__)')"

# Run the Python script within the virtual environment
singularity exec /homes/users/kivankovic/ext-val/epigame.sif /opt/venv/bin/python3 /homes/users/kivankovic/ext-val/scripts/connectivity_matrices_ext.py /homes/users/kivankovic/ext-val/preprocessed/${SLURM_ARRAY_TASK_ID}.prep "(1,4)"

