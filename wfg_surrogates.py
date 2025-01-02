#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import path, makedirs, listdir
from datetime import datetime
import logging
import re
import random
import argparse

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as tts
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.RVEA import RVEA

import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# Setup logging
base_folder = '/scratch/project_2012636'
log_dir = path.join(base_folder, "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)

# Parse command-line arguments for data directory
parser = argparse.ArgumentParser(description='Run WFG surrogates')
parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
args = parser.parse_args()

data_dir = args.data_dir

# Start timing the script
script_start = datetime.now()

# Define surrogate modeling techniques
algorithms = {
    "SVM": svm.SVR,
    "NN": MLPRegressor,
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

# Folder paths
inner_data_folder = data_dir  # Data directory from command-line argument

output_folder = path.join(base_folder, "modelling_results")
if not path.exists(output_folder):
    makedirs(output_folder)

# Initialize an empty DataFrame to store R² and MSE results
R2results = pd.DataFrame(columns=["File", "Model", "Objective", "R2_Score", "MSE"])

# List all files in the data directory
files = listdir(inner_data_folder)

# Set this to False if you want to select a subset of files
use_full_dataset = True

# Select exactly one file if not using the full dataset
if not use_full_dataset:
    # Ensure there is at least one file
    if files:
        selected_files = [random.choice(files)]  # Select exactly one file
    else:
        selected_files = []
else:
    selected_files = files

print(f"Script started at {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Found {len(files)} files in the dataset folder: {'Full dataset' if use_full_dataset else 'Selected subset'}")

def extract_details_from_filename(filename):
    pattern = re.compile(r"(WFG\d+)_(\d+)var_(\d+)obj_(\d+)samples_.*")
    match = pattern.match(filename)
    if match:
        problem_name, num_vars, num_obj, num_samples = match.groups()
        return problem_name, int(num_vars), int(num_obj), int(num_samples)
    else:
        logging.error(f"Filename {filename} does not match the expected pattern.")
        return None, None, None, None


def train_models_for_file(file, algo):
    print(f"Processing file: {file}")
    problem_name, num_vars, num_obj, num_samples = extract_details_from_filename(file)
    if not problem_name:
        print(f"Skipping file {file}, unable to extract details.")
        logging.info(f"{file} [failed] - unable to extract details.")
        return None, None, None, None  # Skipping this file

    try:
        data = pd.read_csv(path.join(inner_data_folder, file))  # Load dataset from inner data directory
        inputs = data.iloc[:, :-num_obj]
        trained_models = []

        for objective_index in range(num_obj):
            target = data.iloc[:, -(objective_index + 1)]
            X_train, X_test, y_train, y_test = tts(inputs, target, test_size=0.3, random_state=42)

            clf = algo()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            # Calculate R² score
            r2_score = clf.score(X_test, y_test)

            # Calculate MSE
            mse = mean_squared_error(y_test, y_pred)

            # Save R² and MSE results
            R2results.loc[len(R2results)] = [file, algo.__name__, f"f{objective_index + 1}", r2_score, mse]

            trained_models.append(clf)

        logging.info(f"{file} [success] - models trained successfully.")
        return trained_models, problem_name, num_vars, num_obj, num_samples

    except Exception as e:
        logging.error(f"{file} [failed] - error during model training: {str(e)}")
        return None, None, None, None  # Skipping this file


# Function for optimization with EAs
def optimization_part(models, problem_name, algorithm_name, num_vars, num_obj, num_samples, output_folder):
    print(f"Starting optimization with {algorithm_name} for problem {problem_name}.")
    logging.info(f"Starting optimization for {algorithm_name} on {problem_name}")

    try:
        # Define variables
        initial_value = 0.5  # Assuming a generic placeholder
        variables = [Variable(f"x_{i + 1}", lower_bound=0.1, upper_bound=1.0, initial_value=initial_value) for i in range(num_vars)]

        # Use surrogate models as objective functions
        surrogate_objectives = [
            ScalarObjective(name=f'f{i + 1}', evaluator=(lambda x, m=models[i]: m.predict(np.atleast_2d(x)).flatten()))
            for i in range(num_obj)
        ]

        # Setup the optimization problem with surrogate objectives
        problem = MOProblem(variables=variables, objectives=surrogate_objectives)

        # Define the common parameters for all EAs
        common_params = {'n_iterations': 10, 'n_gen_per_iter': 50}

        # Optimization algorithms to use with their specific additional parameters
        eas = {
            "NSGAIII": (NSGAIII, {}),
            "IBEA": (IBEA, {'population_size': 100}),
            "RVEA": (RVEA, {'population_size': 100})  # Assuming RVEA also requires population_size
        }

        for ea_name, (ea, ea_specific_params) in eas.items():
            print(f"\nRunning {ea_name} for {algorithm_name} on problem {problem_name}.")
            logging.info(f"\nStarting {ea_name} optimization for {algorithm_name} on {problem_name}")

            # Initialize the evolutionary algorithm with both common and specific parameters
            evolver = ea(problem, **{**common_params, **ea_specific_params})

            iteration_count = 0
            while evolver.continue_evolution():
                evolver.iterate()
                iteration_count += 1

            solutions = evolver.end()[1]
            print(f"Optimization complete for {ea_name} on {problem_name}. Saving results...")
            save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_vars, num_obj, num_samples, output_folder)
            logging.info(f"Optimization for {ea_name} on {problem_name} completed successfully.")

    except Exception as e:
        logging.error(f"Optimization for {algorithm_name} on {problem_name} [failed] - error: {str(e)}")

def save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_vars, num_obj, num_samples, output_folder):
    optimization_results_dir = path.join(output_folder, "surrogate_optimization_results", problem_name)
    if not path.exists(optimization_results_dir):
        makedirs(optimization_results_dir)

    # Include additional details in the filename
    results_filename = path.join(optimization_results_dir, f"{algorithm_name}_{ea_name}_{num_obj}obj_{num_vars}vars_{num_samples}samples_optimization_results.csv")

    solutions_df = pd.DataFrame(solutions, columns=[f"f{i + 1}" for i in range(num_obj)])
    solutions_df.to_csv(results_filename, index=False)
    print(f"Optimization results for {ea_name} saved to {results_filename}")
    logging.info(f"Optimization results for {ea_name} saved to {results_filename}")


# Load list of already processed files
processed_files_log = path.join(output_folder, "processed_files.csv")
if path.exists(processed_files_log):
    processed_files_df = pd.read_csv(processed_files_log)
    processed_files = set(processed_files_df["File"].tolist())
else:
    processed_files = set()

# Function to append processed file to the log
def log_processed_file(file):
    with open(processed_files_log, "a") as log_file:
        log_file.write(f"{file}\n")

# Iterate over selected files
for file in selected_files:
    if file in processed_files:
        print(f"Skipping file {file}, already processed.")
        continue  # Skip already processed files

    print(f"\nProcessing file: {file}")
    problem_name, num_vars, num_obj, num_samples = extract_details_from_filename(file)

    if not problem_name:
        continue  # Skip this file if details extraction failed

    # For each file, iterate through each algorithm
    for algo_name, algo in algorithms.items():
        print(f"\n--- Training models using {algo_name} for {file} ---")
        trained_models, problem_name, num_vars, num_obj, num_samples = train_models_for_file(file, algo)

        if trained_models:
            optimization_part(trained_models, problem_name, algo_name, num_vars, num_obj, num_samples, output_folder)
        else:
            logging.info(f"{file} [failed] - training failed, skipping optimization.")
    
    # Log the processed file
    log_processed_file(file)

# Save R² and MSE results
R2results_filename = path.join(output_folder, "WFG_R2_MSE_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("R² and MSE results saved successfully.")
print(f"R² and MSE results saved to {R2results_filename}")


script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
print(f"Script finished in {elapsed_time:.2f} seconds.")
