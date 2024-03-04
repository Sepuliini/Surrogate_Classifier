import numpy as np
import pandas as pd
from os import getcwd, listdir, path, makedirs
import pickle
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import random
from datetime import datetime
import logging
import warnings

# Setup logging
log_dir = path.join(getcwd(), "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("wfg_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.captureWarnings(False)

# Start timing the script
script_start = datetime.now()

# Folder paths
base_folder = getcwd()
wfg_base_folder = path.join(base_folder, "data", "wfg_data")
output_folder = path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not path.exists(output_folder):
    makedirs(output_folder)

# Option to use the full dataset or a subset for testing
use_full_dataset = False  # Change to True to run on the entire dataset

# Define surrogate modeling techniques
algorithms = ["SVM", "NN", "Ada", "GPR", "SGD", "KNR", "DTR", "RFR", "ExTR", "GBR", "XGB"]
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

# Initialize R2results DataFrame outside the main loop
columns = ['File', 'Model', 'Objective', 'R2_Score']
R2results = pd.DataFrame(columns=columns)

# Collecting files from WFG subdirectories, skipping non-directory items such as .DS_Store
selected_files = []  # This will be a list of tuples (problem_folder, file)
problem_folders = [f for f in listdir(wfg_base_folder) if path.isdir(path.join(wfg_base_folder, f))]
for problem_folder in problem_folders:
    problem_path = path.join(wfg_base_folder, problem_folder)
    files = [f for f in listdir(problem_path) if path.isfile(path.join(problem_path, f))]
    if not use_full_dataset:
        num_files_to_select = max(int(len(files) * 0.1), 1)
        files = random.sample(files, num_files_to_select)
    for file in files:
        selected_files.append((problem_folder, file))


# Iterate over selected files and train each algorithm
for i, (problem_folder, file) in enumerate(selected_files):
    problem_path = path.join(wfg_base_folder, problem_folder)
    fullfilename = path.join(problem_path, file)
    logging.info(f"Processing file {i + 1} of {len(selected_files)}: {fullfilename}")

    data = pd.read_csv(fullfilename)
    inputs = data.iloc[:, :-2]  
    f1 = data.iloc[:, -2]       # Second-to-last column as the first target
    f2 = data.iloc[:, -1]       # Last column as the second target

    inputs_train_f1, inputs_test_f1, f1_train, f1_test = tts(inputs, f1, test_size=0.3, random_state=42)
    inputs_train_f2, inputs_test_f2, f2_train, f2_test = tts(inputs, f2, test_size=0.3, random_state=42)

    for algo in algorithms:
        model_start = datetime.now()
        clf_f1 = model[algo]()
        clf_f1.fit(inputs_train_f1, f1_train)
        pred_f1 = clf_f1.predict(inputs_test_f1)
        score_f1 = r2_score(f1_test, pred_f1)
        
        clf_f2 = model[algo]()
        clf_f2.fit(inputs_train_f2, f2_train)
        pred_f2 = clf_f2.predict(inputs_test_f2)
        score_f2 = r2_score(f2_test, pred_f2)

        model_end = datetime.now()
        training_duration = (model_end - model_start).total_seconds()
        logging.info(f"Trained {algo} on {file} for f1 & f2 in {training_duration:.2f} seconds")

        R2results = R2results.append({'File': file, 'Model': algo, 'Objective': 'f1', 'R2_Score': score_f1}, ignore_index=True)
        R2results = R2results.append({'File': file, 'Model': algo, 'Objective': 'f2', 'R2_Score': score_f2}, ignore_index=True)

# Save R^2 results to CSV specifically named for WFG results
R2results_filename = path.join(output_folder, "WFG_R2_results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info(f"WFG R^2 results saved to {R2results_filename}")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
