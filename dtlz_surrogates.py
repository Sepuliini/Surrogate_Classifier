import numpy as np
import pandas as pd
from os import getcwd, listdir, path, makedirs
from datetime import datetime
import logging
import warnings
import re
import random

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

from desdeo_problem import Variable, ScalarObjective, MOProblem, variable_builder
from desdeo_problem.testproblems.TestProblems import test_problem_builder

from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.RVEA import RVEA

# Setup logging
log_dir = path.join(getcwd(), "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)

# Start timing the script
script_start = datetime.now()

problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

# Folder paths
base_folder = getcwd()
data_folder = path.join(base_folder, "data", "dtlz_data")
output_folder = path.join(base_folder, "modelling_results")
if not path.exists(output_folder):
    makedirs(output_folder)

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

files = listdir(data_folder)
use_full_dataset = False
selected_files = files if use_full_dataset else random.sample(files, max(int(len(files) * 0.1), 1))

# Initialize DataFrame to store R2 results
R2results = pd.DataFrame(columns=["File", "Model", "Objective", "R2_Score"])

print(f"Script started at {script_start.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Found {len(files)} files in the dataset folder: {'Full dataset' if use_full_dataset else 'Selected subset'}")

def optimization_part(models, problem_name, algorithm_name, num_vars, num_obj, output_folder):
    print(f"Starting optimization with {algorithm_name} for problem {problem_name}.")
    logging.info(f"Starting optimization for {algorithm_name} on {problem_name}")
    
    print("problem name: ", problem_name)
    # Create DTLZ problem instance to extract constraints
    dtlz_problem = test_problem_builder(problem_name, n_of_objectives=num_obj, n_of_variables=num_vars)
    
    # Define variables
    initial_value = 0.5  # Assuming a generic placeholder
    variables = [Variable(f"x_{i+1}", lower_bound=0.1, upper_bound=1.0, initial_value=initial_value) for i in range(num_vars)]
    
    # Use surrogate models as objective functions
    surrogate_objectives = [
        ScalarObjective(name=f'f{i+1}', evaluator=(lambda x, m=models[i]: m.predict(np.atleast_2d(x)).flatten()))
        for i in range(num_obj)
    ]
    
    # Extract constraints from the DTLZ problem
    constraints = dtlz_problem.constraints if dtlz_problem.constraints else []
    print("Const: ", constraints)
    
    # Setup the optimization problem with surrogate objectives and DTLZ constraints
    problem = MOProblem(variables=variables, objectives=surrogate_objectives, constraints=constraints)

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

        # Setup the optimization problem with surrogate objectives and DTLZ constraints
        problem = MOProblem(variables=variables, objectives=surrogate_objectives, constraints=constraints)
        
        # Merge common parameters with EA-specific parameters
        ea_params = {**common_params, **ea_specific_params}

        # Initialize the evolutionary algorithm with both common and specific parameters
        evolver = ea(problem, **ea_params)

        iteration_count = 0
        while evolver.continue_evolution():
            evolver.iterate()
            iteration_count += 1

        solutions = evolver.end()[1]
        print(f"Optimization complete for {ea_name} on {problem_name}. Saving results...")
        save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_folder)
        
def save_optimization_results(solutions, problem_name, algorithm_name, ea_name, num_obj, output_folder):
    optimization_results_dir = path.join(output_folder, "surrogate_optimization_results", problem_name)
    if not path.exists(optimization_results_dir):
        makedirs(optimization_results_dir)
    # Include ea_name in the filename to differentiate results
    results_filename = path.join(optimization_results_dir, f"{algorithm_name}_{ea_name}_optimization_results.csv")
    solutions_df = pd.DataFrame(solutions, columns=[f"f{i+1}" for i in range(num_obj)])
    solutions_df.to_csv(results_filename, index=False)
    print(f"Optimization results for {ea_name} saved to {results_filename}")


def extract_details_from_filename(filename):
    pattern = re.compile(r"(DTLZ\d+)_(\d+)var_(\d+)obj_\d+samples_.*")
    match = pattern.match(filename)
    if match:
        problem_name, num_vars, num_obj = match.groups()
        return problem_name, int(num_vars), int(num_obj)
    else:
        logging.error(f"Filename {filename} does not match the expected pattern.")
        return None, None, None
    
files = listdir(data_folder)
use_full_dataset = False
selected_files = files if use_full_dataset else random.sample(files, max(int(len(files) * 0.1), 1))

def train_models_for_file(file, algo):
    print(f"Processing file: {file}")
    problem_name, num_vars, num_obj = extract_details_from_filename(file)
    if not problem_name:
        print(f"Skipping file {file}, unable to extract details.")
        return None, None, None  # Skipping this file

    data = pd.read_csv(path.join(data_folder, file))
    inputs = data.iloc[:, :-num_obj]
    trained_models = []

    for objective_index in range(num_obj):
        target = data.iloc[:, -(objective_index+1)]
        X_train, X_test, y_train, y_test = tts(inputs, target, test_size=0.3, random_state=42)

        clf = algo()
        clf.fit(X_train, y_train)
        trained_models.append(clf)

    return trained_models, problem_name, num_vars, num_obj

# Iterate over selected files
for file in selected_files:
    print(f"\nProcessing file: {file}")
    problem_name, num_vars, num_obj = extract_details_from_filename(file)

    if not problem_name:
        continue  # Skip this file if details extraction failed
    
    # For each file, iterate through each algorithm
    for algo_name, algo in algorithms.items():
        print(f"\n--- Training models using {algo_name} for {file} ---")
        trained_models, problem_name, num_vars, num_obj = train_models_for_file(file, algo)
        
        if trained_models:
            optimization_part(trained_models, problem_name, algo_name, num_vars, num_obj, output_folder)

# Save R^2 results
R2results_filename = path.join(output_folder, "DTLZ_R2_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("Surrogate modeling for DTLZ problem completed.")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
print(f"Script finished in {elapsed_time:.2f} seconds.")
