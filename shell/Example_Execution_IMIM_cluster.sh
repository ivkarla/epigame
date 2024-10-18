#!/bin/bash

#SBATCH --job-name=MACS2_peak_calling 
#SBATCH --partition=long
#SBATCH --cpus-per-task=2 
#SBATCH --mem=12G
#SBATCH --nodes=1  
#SBATCH --output=logs/MACS2.out
#SBATCH --error=logs/MACS2.err
#SBATCH --array=1-2%2

# REMARK!!!! Adapt the number of array tasks to the number of samples i.e. if you have
# 12 samples to analyze you need to specify 12 as indicated. %X means refers to the number 
# of tasks will be sent to cluster execution simultaneously. 
# Each task meaning one sample to be analyzed

#=========================
# User defined parameters: relevant paths and analysis type
#=========================

# SPECIFY Root directory in the cluster (usually /projects/cancer)
ROOTDIR="/projects/cancer"

# SPECIFY your project working directory. 
WKD=$ROOTDIR'/IkBa_mESCs_ABigas/ChIPseq_IkBa'

# SPECIFY the file name where the sample;input is included.
SAMPLESHEET=$WKD"/MACS2_peak_calling/Samples_Input.txt"

#=========================
# General configuration
#=========================
START=$(date +%s)
# Enable Singularity image to look into the general path (equivalent to -B)
export SINGULARITY_BIND=$ROOTDIR 
# Path to images folder in cluster
IMAGES_PATH=$ROOTDIR"/images"
# Path to databases folder in cluster
DB_PATH=$ROOTDIR"/db_files"

################################################################################
##       MACS2 peak calling
################################################################################

# MACS2 peak caller requires aligned reads in BAM format (typically from Bowtie2)
# For ChIPseq: BAM should only include unique mappers and duplicates should be marked

# Link to MACS2 peak caller manual
# https://pypi.org/project/MACS2/

###########################
## 1. Other relevant paths
###########################

# Folder where input BAM files are available
DATA=$WKD'/Bowtie_align/BAM'

# Folder where MACS2 output results will be stored: 
OUTPEAKS=$WKD'/MACS2_peak_calling/Other_results'

#################################################
## 2. Singularity image and Tool Parametrization
#################################################

# Specify image/s name to be used (tool-related)
MACS2='macs2_v2.2.7.1.sif '  #This image inludes MACS2 2.2.7.1

# Specify any particular tool parameters

# Effective genome size
GSIZE=2308125349 # mm10 50bp read length 

# Adj p-val (q-val) to be used as threshold criteria: 5% by default
FDR=0.05

# Criteria (cut-off) for broad regions, if considered: 10% by default
BROAD_CUTOFF=0.1

# NOTE: MACS2 does not consider duplicates for peak calling
KEEPDUP="" # "--keep-dup all" if you wanna change this behaviour

################################################################################
## 3. Command file preparation: to execute batch array
################################################################################

while IFS=";" read -r sample input; 
do
  # Sample name
  NAME=${sample%_trimmed.sorted.unique.bam}
  
  # Peak calling with MACS2

  echo "singularity exec $IMAGES_PATH/$MACS2 macs2 callpeak -B $KEEPDUP --nomodel -g $GSIZE -q $FDR -t $DATA/$sample -c $DATA/$input --outdir $OUTPEAKS -n $NAME"
  
done < $SAMPLESHEET > $WKD'/scripts/cmds/'$RUNNAME'_MACS2_peak_calling_samples.cmd'


################################################################################
## 4. MACS2 Peak calling
################################################################################

echo "-----------------------------------------"
echo "Starting MACS2 Peak Calling"
echo "-----------------------------------------"

START=$(date +%s)
echo "  Samples peak calling in array mode: $DATE"
echo " "

SEEDFILE=$WKD'/scripts/cmds/'$RUNNAME'_MACS2_peak_calling_samples.cmd'
SEED=$(sed -n ${SLURM_ARRAY_TASK_ID}p $SEEDFILE)
eval $SEED

END=$(date +%s)
DIFF=$(( $END - $START ))
echo 'MACS2 peak calling completed' 
echo "Processing Time: $DIFF seconds"


