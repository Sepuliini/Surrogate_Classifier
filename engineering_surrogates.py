#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import path, makedirs, listdir
from datetime import datetime
import logging
import random
import argparse
import re 
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_emo.EAs import NSGAIII, IBEA, RVEA
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# Setup logging
base_folder = '/scratch/project_2010752'
log_dir = path.join(base_folder, "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("engineering_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run Engineering surrogates')
parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

# Define surrogate modeling techniques
model = {
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

# Engineering problems mapping
problem_mapping = {
    "re21": re21,
    "re22": re22,
    "re23": re23,
    "re24": re24,
    "re25": re25,
    "re31": re31,
    "re32": re32,
    "re33": re33
}

# Initialize an empty DataFrame to store R² and MSE results
R2results = pd.DataFrame(columns=["Problem", "Model", "Objective", "R2_Score", "MSE"])

# List all files in the data directory
files = listdir(data_dir)

# Option to use the full dataset or a subset
use_full_dataset = False  # Set to True to use the full dataset
selected_files = files if use_full_dataset else random.sample(files, min(len(files), 1)) if files else []

print(f"Script started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Found {len(files)} files in the dataset folder: {'Full dataset' if use_full_dataset else 'Selected subset'}")

def extract_details_from_filename(filename):
    pattern = re.compile(r"(re\d+)_\d+_samples\.csv")
    match = pattern.match(filename)
    if match:
        problem_name = match.group(1)
        return problem_name
    else:
        logging.error(f"Filename {filename} does not match the expected pattern.")
        return None

def train_models_for_file(file, algo):
    print(f"Processing file: {file}")
    problem_name = extract_details_from_filename(file)
    if not problem_name:
        print(f"Skipping file {file}, unable to extract details.")
        logging.info(f"{file} [failed] - unable to extract details.")
        return None, None, None, None  # Skipping this file

    try:
        data = pd.read_csv(path.join(data_dir, file))  # Load dataset
        #Change this!!
        inputs = data.iloc[:, :-2]  # Assumes last two columns are objectives
        num_vars = inputs.shape[1]
        num_obj = len(data.columns) - num_vars
        
        trained_models = []
        for objective_index in range(num_obj):
            target = data.iloc[:, -(objective_index+1)]
            X_train, X_test, y_train, y_test = tts(inputs, target, test_size=0.3, random_state=42)

            clf = algo()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            # Save R² and MSE results
            R2results.loc[len(R2results)] = [problem_name, algo.__name__, f"f{objective_index + 1}", r2, mse]

            trained_models.append(clf)

        logging.info(f"{file} [success] - models trained successfully.")
        return trained_models, problem_name, num_vars, num_obj

    except Exception as e:
        logging.error(f"{file} [failed] - error during model training: {str(e)}")
        return None, None, None, None  # Skipping this file

def optimization_part(models, problem_name, algorithm_name, num_vars, num_obj, output_dir):
    print(f"Starting optimization with {algorithm_name} for problem {problem_name}.")
    logging.info(f"Starting optimization for {algorithm_name} on {problem_name}")
    try:
        # Define variables with an initial value
        initial_value = 0.5
        variables = [Variable(f"x_{i+1}", lower_bound=0.1, upper_bound=1.0, initial_value=initial_value) for i in range(num_vars)]

        # Use surrogate models as objective functions
        surrogate_objectives = [
            ScalarObjective(name=f'f{i+1}', evaluator=lambda x, m=models[i]: m.predict(np.atleast_2d(x)).flatten())
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
            save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_dir)
            logging.info(f"Optimization for {ea_name} on {problem_name} completed successfully.")

    except Exception as e:
        logging.error(f"Optimization for {algorithm_name} on {problem_name} [failed] - error: {str(e)}")


def save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_dir):
    # Directly use the provided output_dir for saving results, ensuring it does not repeat folder names
    # Add a subfolder for the problem name to organize results by problem
    optimization_results_dir = path.join(output_dir, problem_name)
    
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
    problem_name = extract_details_from_filename(file)
    if problem_name is None:
        continue  # Skip the file if details cannot be extracted

    for algo_name, algo in model.items():
        models, problem_name, num_vars, num_obj = train_models_for_file(file, algo)
        if models and problem_name and num_vars is not None and num_obj is not None:
            optimization_part(models, problem_name, algo_name, num_vars, num_obj, output_dir)

# Save the R² and MSE results to a CSV file
r2_results_filename = path.join(output_dir, "R2_MSE_results.csv")
R2results.to_csv(r2_results_filename, index=False)
print(f"R² and MSE results saved to {r2_results_filename}")
logging.info(f"R² and MSE results saved to {r2_results_filename}")

# End timing the script
script_end = datetime.now()
print(f"Script ended at {script_end.strftime('%Y-%m-%d %H:%M:%S')}")
logging.info(f"Script ended at {script_end.strftime('%Y-%m-%d %H:%M:%S')}")
