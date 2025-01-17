#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import path, makedirs, listdir
from datetime import datetime
import logging
import random
import argparse
import warnings
import re
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_problem.testproblems.CarSideImpact import car_side_impact

from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.RVEA import RVEA

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# Setup logging
base_folder = '/scratch/project_2012636'
log_dir = path.join(base_folder, "logs")
if not path.exists(log_dir):
    makedirs(log_dir)

log_file = datetime.now().strftime("car_side_impact_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(
    filename=path.join(log_dir, log_file),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging.captureWarnings(True)

# Parse command-line arguments for data and output directories
parser = argparse.ArgumentParser(description='Run car side impact surrogates')
parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

# Start timing the script
script_start = datetime.now()

# Define surrogate modeling techniques
algorithms = {
    "SVM": svm.SVR,
    "NN": lambda: MLPRegressor(max_iter=1000, tol=1e-4),  # Increased iterations and relaxed tolerance
    "Ada": ensemble.AdaBoostRegressor,
    "GPR": GaussianProcessRegressor,
    "SGD": SGD,
    "KNR": KNR,
    "DTR": DTR,
    "RFR": ensemble.RandomForestRegressor,
    "ExTR": ensemble.ExtraTreesRegressor,
    "GBR": ensemble.GradientBoostingRegressor,
    "XGB": XGBRegressor
}

# Initialize an empty DataFrame to store R² and MSE results
R2results = pd.DataFrame(columns=["File", "Model", "Objective", "R2_Score", "MSE"])

# List all files in the data directory (only CSVs)
files = [f for f in listdir(data_dir) if f.endswith('.csv')]

# Set this to False if you want to select a subset of files
use_full_dataset = True

# Select exactly one file if not using the full dataset
selected_files = files if use_full_dataset else random.sample(files, 1) if files else []

print(f"Script started at {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Found {len(files)} files in the dataset folder: {'Full dataset' if use_full_dataset else 'Selected subset'}")

# Regex for extracting sample size from filename
def extract_sample_size_from_filename(filename):
    pattern = re.compile(r"car_side_impact_(\d+)_samples\.csv")
    match = pattern.match(filename)
    if match:
        sample_size = match.group(1)
        return sample_size
    else:
        logging.error(f"Filename {filename} does not match the expected pattern.")
        return None

def train_models_for_file(file, algo):
    print(f"Processing file: {file}")
    try:
        # Load dataset
        data = pd.read_csv(path.join(data_dir, file))

        # Extract sample size from the filename
        sample_size = extract_sample_size_from_filename(file)
        if not sample_size:
            return None, None, None, None

        # Identify objective columns (those starting with 'f_')
        objective_columns = [col for col in data.columns if col.startswith('f_')]
        input_columns = [col for col in data.columns if not col.startswith('f_')]

        # Extract inputs (x_1, x_2, etc.) and number of variables (inputs)
        inputs = data[input_columns]
        num_vars = inputs.shape[1]

        # Extract the number of objectives (those starting with 'f_')
        num_obj = len(objective_columns)

        trained_models = []

        # Train models for each objective
        for objective_index, obj_col in enumerate(objective_columns):
            target = data[obj_col]  # e.g. data['f_1']
            X_train, X_test, y_train, y_test = tts(inputs, target, test_size=0.3, random_state=42)

            # Train the model using the selected algorithm
            clf = algo()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Calculate R² (score) and MSE
            r2_score_value = clf.score(X_test, y_test)
            mse = mean_squared_error(y_test, y_pred)

            # Log and store results
            R2results.loc[len(R2results)] = [file, algo.__name__, obj_col, r2_score_value, mse]

            trained_models.append(clf)

        logging.info(f"{file} [success] - models trained successfully.")
        return trained_models, num_vars, num_obj, sample_size

    except Exception as e:
        logging.error(f"{file} [failed] - error during model training: {str(e)}")
        return None, None, None, None

def optimization_part(models, problem_name, algorithm_name, num_vars, num_obj, output_folder, sample_size):
    print(f"Starting optimization with {algorithm_name} for problem {problem_name}.")
    logging.info(f"Starting optimization for {algorithm_name} on {problem_name}")

    try:
        # Define variables with the correct ranges for car_side_impact
        variable_ranges = [
            (0.5, 1.5),    # x_1
            (0.45, 1.35),  # x_2
            (0.5, 1.5),    # x_3
            (0.5, 1.5),    # x_4
            (0.875, 2.625),# x_5
            (0.4, 1.2),    # x_6
            (0.4, 1.2),    # x_7
        ]

        # Create the Variable objects
        variables = [
            Variable(
                f"x_{i+1}",
                lower_bound=variable_ranges[i][0],
                upper_bound=variable_ranges[i][1],
                initial_value=(variable_ranges[i][0] + variable_ranges[i][1]) / 2
            ) 
            for i in range(num_vars)
        ]

        # Create surrogate objectives for each model
        surrogate_objectives = [
            ScalarObjective(
                name=f'f{i+1}',
                evaluator=lambda x, m=models[i]: m.predict(np.atleast_2d(x)).flatten()
            )
            for i in range(num_obj)
        ]

        # Use constraints from the CarSideImpact problem
        problem = MOProblem(
            variables=variables,
            objectives=surrogate_objectives,
            constraints=car_side_impact().constraints
        )

        # Common parameters for all EAs
        common_params = {'n_iterations': 10, 'n_gen_per_iter': 50}

        # Algorithms to run
        eas = {
            "NSGAIII": (NSGAIII, {}),
            "IBEA": (IBEA, {'population_size': 100}),
            "RVEA": (RVEA, {'population_size': 100})
        }

        for ea_name, (ea, ea_specific_params) in eas.items():
            print(f"\nRunning {ea_name} for {algorithm_name} on problem {problem_name}.")
            logging.info(f"\nStarting {ea_name} optimization for {algorithm_name} on {problem_name}")

            evolver = ea(problem, **{**common_params, **ea_specific_params})

            while evolver.continue_evolution():
                evolver.iterate()

            solutions = evolver.end()[1]
            print(f"Optimization complete for {ea_name} on {problem_name}. Saving results...")
            save_optimization_results(
                solutions,
                problem_name,
                algorithm_name,
                ea_name,
                num_obj,
                output_folder,
                sample_size
            )
            logging.info(f"Optimization for {ea_name} on {problem_name} completed successfully.")

    except Exception as e:
        logging.error(f"Optimization for {algorithm_name} on {problem_name} [failed] - error: {str(e)}")

def save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_folder, sample_size):
    # Ensure there's a subdirectory for this problem in the output folder
    optimization_results_dir = path.join(output_folder, problem_name)
    if not path.exists(optimization_results_dir):
        makedirs(optimization_results_dir)

    # Build the results filename
    results_filename = path.join(
        optimization_results_dir,
        f"{algorithm_name}_{ea_name}_{sample_size}samples_optimization_results.csv"
    )

    # Convert solutions to a DataFrame and save
    solutions_df = pd.DataFrame(solutions, columns=[f"f{i+1}" for i in range(num_obj)])
    solutions_df.to_csv(results_filename, index=False)

    print(f"Optimization results for {ea_name} saved to {results_filename}")
    logging.info(f"Optimization results for {ea_name} saved to {results_filename}")

# -------------------------------------------------
# Fixed section: handle the processed_files.csv
# -------------------------------------------------
processed_files_log = "/scratch/project_2012636/modelling_results/processed_files.csv"

# Create the file if it doesn't exist
if not path.exists(processed_files_log):
    with open(processed_files_log, "w") as log_file:
        log_file.write("")

# Check if the file has data before reading
if path.exists(processed_files_log) and path.getsize(processed_files_log) > 0:
    processed_files_df = pd.read_csv(processed_files_log, header=None)
    processed_files = set(processed_files_df[0].tolist())
else:
    processed_files = set()

def log_processed_file(file):
    with open(processed_files_log, "a") as log_file:
        log_file.write(f"{file}\n")
# -------------------------------------------------

# Iterate over selected files
for file in selected_files:
    if file in processed_files:
        print(f"Skipping file {file}, already processed.")
        continue

    print(f"\nProcessing file: {file}")
    for algo_name, algo in algorithms.items():
        models, num_vars, num_obj, sample_size = train_models_for_file(file, algo)
        if models and num_vars is not None and num_obj is not None:
            optimization_part(models, "car_side_impact", algo_name, num_vars, num_obj, output_dir, sample_size)

    # Log the processed file
    log_processed_file(file)

# Save R² and MSE results
R2results_filename = path.join(output_dir, "car_side_impact_R2_MSE_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("R² and MSE results saved successfully.")
print(f"R² and MSE results saved to {R2results_filename}")

# End timing the script
script_end = datetime.now()
print(f"Script ended at {script_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {script_end - script_start}")
