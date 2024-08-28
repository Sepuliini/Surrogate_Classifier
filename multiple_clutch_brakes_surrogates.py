#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import path, makedirs, listdir
from datetime import datetime
import logging
import random
import argparse
import warnings

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
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes

from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.RVEA import RVEA

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# Setup logging
base_folder = '/scratch/project_2010752'
log_dir = path.join(base_folder, "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("multiple_clutch_brakes_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)

# Parse command-line arguments for data and output directories
parser = argparse.ArgumentParser(description='Run multiple clutch brakes surrogates')
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

# Initialize an empty DataFrame to store R² and MSE results
R2results = pd.DataFrame(columns=["File", "Model", "Objective", "R2_Score", "MSE"])

# List all files in the data directory
files = [f for f in listdir(data_dir) if f.endswith('.csv')]

# Set this to False if you want to select a subset of files
use_full_dataset = False

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

# Function to train models for each file
def train_models_for_file(file, algo):
    print(f"Processing file: {file}")

    try:
        data = pd.read_csv(path.join(data_dir, file))  # Load dataset from data directory
        num_obj = len(data.columns) - data.shape[1]  # Assuming the number of objectives is the difference
        inputs = data.iloc[:, :-num_obj]
        trained_models = []

        for objective_index in range(num_obj):
            target = data.iloc[:, -(objective_index+1)]
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
        return trained_models, num_obj  # Only return the number of objectives

    except Exception as e:
        logging.error(f"{file} [failed] - error during model training: {str(e)}")
        return None, None  # Skipping this file

# Function for optimization with EAs
def optimization_part(models, problem_name, algorithm_name, num_vars, num_obj, output_folder):
    print(f"Starting optimization with {algorithm_name} for problem {problem_name}.")
    logging.info(f"Starting optimization for {algorithm_name} on {problem_name}")

    try:
        # Define variables
        initial_value = 0.5  # Assuming a generic placeholder
        variables = [Variable(f"x_{i+1}", lower_bound=0.1, upper_bound=1.0, initial_value=initial_value) for i in range(num_vars)]

        # Use surrogate models as objective functions
        surrogate_objectives = [
            ScalarObjective(name=f'f{i+1}', evaluator=(lambda x, m=models[i]: m.predict(np.atleast_2d(x)).flatten()))
            for i in range(num_obj)
        ]

        # Setup the optimization problem with surrogate objectives and multiple clutch brakes constraints
        problem = MOProblem(variables=variables, objectives=surrogate_objectives, constraints=multiple_clutch_brakes().constraints)

        # Define the common parameters for all EAs
        common_params = {'n_iterations': 10, 'n_gen_per_iter': 50}

        # Optimization algorithms to use with their specific additional parameters
        eas = {
            "NSGAIII": (NSGAIII, {}),
            "IBEA": (IBEA, {'population_size': 100}),
            "RVEA": (RVEA, {'population_size': 100})
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
            save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_folder)
            logging.info(f"Optimization for {ea_name} on {problem_name} completed successfully.")

    except Exception as e:
        logging.error(f"Optimization for {algorithm_name} on {problem_name} [failed] - error: {str(e)}")

def save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_folder):
    optimization_results_dir = path.join(output_folder, "surrogate_optimization_results", problem_name)
    if not path.exists(optimization_results_dir):
        makedirs(optimization_results_dir)
    results_filename = path.join(optimization_results_dir, f"{algorithm_name}_{ea_name}_optimization_results.csv")
    solutions_df = pd.DataFrame(solutions, columns=[f"f{i+1}" for i in range(num_obj)])
    solutions_df.to_csv(results_filename, index=False)
    print(f"Optimization results for {ea_name} saved to {results_filename}")
    logging.info(f"Optimization results for {ea_name} saved to {results_filename}")

# Iterate over selected files
for file in selected_files:
    print(f"\nProcessing file: {file}")

    for algo_name, algo in algorithms.items():
        models, num_obj = train_models_for_file(file, algo)
        if models:
            optimization_part(models, "multiple_clutch_brakes", algo_name, 0, num_obj, output_dir)

# Save R² and MSE results
R2results_filename = path.join(output_dir, "multiple_clutch_brakes_R2_MSE_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("R² and MSE results saved successfully.")
print(f"R² and MSE results saved to {R2results_filename}")

# End timing the script
script_end = datetime.now()
print(f"Script ended at {script_end.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total runtime: {script_end - script_start}")
