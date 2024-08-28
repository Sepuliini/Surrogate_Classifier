#!/usr/bin/env python3

import numpy as np
import pandas as pd
from os import getcwd, listdir, path, makedirs
from datetime import datetime
import logging
import argparse
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
from desdeo_emo.EAs.NSGAIII import NSGAIII

# Suppress specific warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Setup logging
base_folder = getcwd()
log_dir = path.join(base_folder, "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("river_pollution_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run river pollution model training and optimization')
parser.add_argument('--data-dir', type=str, required=True, help='Path to the data directory')
parser.add_argument('--output-dir', type=str, required=True, help='Path to the output directory')
args = parser.parse_args()

data_dir = args.data_dir
output_dir = args.output_dir

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
columns = ['File', 'Model', 'Objective', 'R2_Score', 'MSE']
R2results = pd.DataFrame(columns=columns)

# List all files in the data directory
files = [f for f in listdir(data_dir) if f.endswith('.csv')]

# Process each file
for file in files:
    data = pd.read_csv(path.join(data_dir, file))
    feature_columns = [col for col in data.columns if col.startswith('x_')]
    objective_columns = [col for col in data.columns if col.startswith('f_')]
    num_vars = len(feature_columns)
    num_obj = len(objective_columns)

    logging.info(f"Processing file {file} with {num_vars} features and {num_obj} objectives.")

    models = []
    for objective_index, obj_col in enumerate(objective_columns):
        target = data[obj_col]
        X_train, X_test, y_train, y_test = tts(inputs, target, test_size=0.3, random_state=42)
        for algo_name, algo in algorithms.items():
            clf = algo()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            R2results.loc[len(R2results)] = [file, algo_name, obj_col, r2, mse]
            models.append(clf)
            logging.info(f"Trained {algo_name} for objective {obj_col} with R2: {r2}, MSE: {mse}")

    # Log number of trained models for objectives
    logging.info(f"Trained models for {num_obj} objectives from file {file}.")

    # Optimization for each model
    for model, obj_col in zip(models, objective_columns):
        variables = [Variable(f"x{i+1}", lower_bound=0.1, upper_bound=1.0) for i in range(num_vars)]
        objectives = [ScalarObjective(name=obj_col, evaluator=(lambda x, m=model: m.predict(np.atleast_2d(x)).flatten()))]
        problem = MOProblem(variables=variables, objectives=objectives, constraints=[])

        ea = NSGAIII(problem, n_iterations=10, n_gen_per_iter=50)
        ea.solve()
        solutions = ea.end()[1]
        results_df = pd.DataFrame(solutions, columns=[obj_col])
        
        results_filename = path.join(output_dir, f"{file[:-4]}_{obj_col}_optimization_results.csv")
        results_df.to_csv(results_filename, index=False)
        logging.info(f"Optimization results for objective {obj_col} saved to {results_filename}")

# Save and log R² and MSE results
R2results_filename = path.join(output_dir, "R2_MSE_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("All R² and MSE results saved and script completed successfully.")
