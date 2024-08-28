#!/bin/bash
#SBATCH --job-name=surrogates_parallel         # Job name
#SBATCH --account= add this             # Project account
#SBATCH --partition=small                     # Partition (queue)
#SBATCH --cpus-per-task=10                    # Number of CPU cores per task
#SBATCH --mem=16G                             # Memory per node
#SBATCH --time=02:00:00                       # Set wall-clock time to 2 hours (adjust as needed)
#SBATCH --output=surrogates_%A_%a.out         # Standard output file for all tasks, %a for array ID
#SBATCH --error=surrogates_%A_%a.err          # Standard error file for all tasks, %a for array ID
#SBATCH --array=1-3                          # Array indices for all tasks (now 1 to 3)

# Load necessary modules
module load python-data

# Define a consolidated log file
LOG_FILE="surrogates_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output and errors to a single log file
exec > >(tee -a $LOG_FILE) 2>&1

# Array tasks
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # Set data directory and script path for DTLZ
    DATA_FOLDER=/scratch/project_2010752/dtlz_data/dtlz_data
    SCRIPT_PATH=/projappl/project_2010752/dtlz_surrogates.py
    
    # Run the DTLZ script
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    # Set data directory and script path for WFG
    DATA_FOLDER=/scratch/project_2010752/wfg_data
    SCRIPT_PATH=/projappl/project_2010752/WFG_surrogates.py
    
    # Run the WFG script
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    # Define data directories and script path for Engineering
    DATA_FOLDERS=(
        "/scratch/project_2010752/re21/re21"
        "/scratch/project_2010752/re22/re22"
        "/scratch/project_2010752/re23/re23"
        "/scratch/project_2010752/re24/re24"
        "/scratch/project_2010752/re25/re25"
        "/scratch/project_2010752/re31/re31"
        "/scratch/project_2010752/re32/re32"
        "/scratch/project_2010752/re33/re33"
    )
    SCRIPT_PATH=/projappl/project_2010752/engineering_surrogates.py
    OUTPUT_DIR=/scratch/project_2010752/modelling_results/surrogate_optimization_results

    # Loop over each data directory
    for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
        echo "Processing directory: $DATA_FOLDER"
        srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER --output_dir=$OUTPUT_DIR
    done

else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID. Exiting."
    exit 1
fi
