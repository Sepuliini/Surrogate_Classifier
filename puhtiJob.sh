#!/bin/bash
#SBATCH --job-name=surrogates_parallel         # Job name
#SBATCH --account=project_2011092              # Project account
#SBATCH --partition=longrun                      # Partition (queue)
#SBATCH --cpus-per-task=10                     # Number of CPU cores per task
#SBATCH --mem=16G                              # Memory per node
#SBATCH --time=time=14-00:00:00                 # Set wall-clock time to 2 hours (adjust as needed)
#SBATCH --output=/projappl/project_2011092/err_logs/surrogates_%A_%a.out  # Standard output file for all tasks
#SBATCH --error=/projappl/project_2011092/err_logs/surrogates_%A_%a.err   # Standard error file for all tasks
#SBATCH --array=1-7                            # Array indices for all tasks (now 1 to 7)

# Load necessary modules
module load python-data
module load r-env

# Define a consolidated log file
LOG_FILE="/projappl/project_2011092/err_logs/surrogates_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log"

# Redirect all output and errors to a single log file
exec > >(tee -a $LOG_FILE) 2>&1

# Array tasks
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    # Set data directory and script path for DTLZ
    DATA_FOLDER=/scratch/project_2011092/Data/dtlz_data
    SCRIPT_PATH=/projappl/project_2011092/dtlz_surrogates.py
    
    # Run the DTLZ script
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER

elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    # Set data directory and script path for WFG
    DATA_FOLDER=/scratch/project_2011092/Data/wfg_data
    SCRIPT_PATH=/projappl/project_2011092/wfg_surrogates.py
    
    # Run the WFG script
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER

elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    # Define data directories and script path for Engineering
    DATA_FOLDERS=(
        "/scratch/project_2011092/Data/re21"
        "/scratch/project_2011092/Data/re22"
        "/scratch/project_2011092/Data/re23"
        "/scratch/project_2011092/Data/re24"
        "/scratch/project_2011092/Data/re25"
        "/scratch/project_2011092/Data/re31"
        "/scratch/project_2011092/Data/re32"
        "/scratch/project_2011092/Data/re33"
    )
    SCRIPT_PATH=/projappl/project_2011092/engineering_surrogates.py
    OUTPUT_DIR=/scratch/project_2011092/modelling_results/surrogate_optimization_results

    # Loop over each data directory
    for DATA_FOLDER in "${DATA_FOLDERS[@]}"; do
        echo "Processing directory: $DATA_FOLDER"
        srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER --output_dir=$OUTPUT_DIR
    done
    
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    # Set data directory and script path for Multiple Clutch Brakes
    DATA_FOLDER=/scratch/project_2011092/Data/multiple_clutch_brakes
    SCRIPT_PATH=/projappl/project_2011092/multiple_clutch_brakes_surrogates.py
    OUTPUT_DIR=/scratch/project_2011092/modelling_results
    
    # Run the Multiple Clutch Brakes script with arguments
    echo "Processing directory: $DATA_FOLDER"
    srun python3 $SCRIPT_PATH --data-dir=$DATA_FOLDER --output-dir=$OUTPUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    # Set data directory and script path for Vehicle Crashworthiness
    DATA_FOLDER=/scratch/project_2011092/Data/vehicle_crashworthiness
    SCRIPT_PATH=/projappl/project_2011092/vehicle_crashworthiness_surrogates.py
    OUTPUT_DIR=/scratch/project_2011092/modelling_results/surrogate_optimization_results

    # Run the Vehicle Crashworthiness script
    echo "Processing directory: $DATA_FOLDER"
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER --output_dir=$OUTPUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 6 ]; then
    # Set data directory and script path for River Pollution
    DATA_FOLDER=/scratch/project_2011092/Data/river_pollution_problem
    SCRIPT_PATH=/projappl/project_2011092/river_pollution_surrogates.py
    OUTPUT_DIR=/scratch/project_2011092/modelling_results/surrogate_optimization_results

    # Run the River Pollution script
    echo "Processing directory: $DATA_FOLDER"
    srun python3 $SCRIPT_PATH --data_dir=$DATA_FOLDER --output_dir=$OUTPUT_DIR

elif [ $SLURM_ARRAY_TASK_ID -eq 7 ]; then
    # R script processing for all datasets
    R_SCRIPT_PATH=/projappl/project_2011092/features.R 

    # Run the R script
    echo "Running R script"
    srun Rscript $R_SCRIPT_PATH

else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID. Exiting."
    exit 1
fi

# Log the completion of the job
echo "Job completed at $(date)"

# Show the efficiency report of the job
seff $SLURM_JOBID
