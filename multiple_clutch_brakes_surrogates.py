import numpy as np
import pandas as pd
from os import getcwd, listdir, path, makedirs
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import random

# Setup logging
log_dir = path.join(getcwd(), "logs")
if not path.exists(log_dir):
    makedirs(log_dir)
log_file = datetime.now().strftime("multiple_clutch_brakes_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.captureWarnings(False)

# Start timing the script
script_start = datetime.now()

# Folder paths
base_folder = getcwd()
problem_base_folder = path.join(base_folder, "data", "multiple_clutch_brakes")
output_folder = path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not path.exists(output_folder):
    makedirs(output_folder)

# Option to use the full dataset or a subset for testing
use_full_dataset = False  # Change to True to run on the entire dataset

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

# Initialize R2results DataFrame outside the main loop
columns = ['Dataset', 'Model', 'Objective', 'R2_Score']
R2results = pd.DataFrame(columns=columns)

# Collecting datasets
files = [f for f in listdir(problem_base_folder) if path.isfile(path.join(problem_base_folder, f))]
if not use_full_dataset:
    num_files_to_select = max(int(len(files) * 0.1), 1)
    files = random.sample(files, num_files_to_select)

# Process each dataset
for file in files:
    fullfilename = path.join(problem_base_folder, file)
    logging.info(f"Processing file: {fullfilename}")

    data = pd.read_csv(fullfilename)
    inputs = data.iloc[:, :-5] 
    objectives = data.columns[-5:]  # Capture the 5 objective column names

    for objective in objectives:
        obj_data = data[objective]
        inputs_train, inputs_test, obj_train, obj_test = tts(inputs, obj_data, test_size=0.3, random_state=42)

        for algo_name, algo in algorithms.items():
            model_start = datetime.now()
            clf = algo()
            clf.fit(inputs_train, obj_train)
            pred = clf.predict(inputs_test)
            score = r2_score(obj_test, pred)

            model_end = datetime.now()
            training_duration = (model_end - model_start).total_seconds()
            logging.info(f"Trained {algo_name} on {objective} in {training_duration:.2f} seconds with R^2 score: {score}")

            R2results = R2results.append({
                'Dataset': file, 
                'Model': algo_name, 
                'Objective': objective, 
                'R2_Score': score
            }, ignore_index=True)

# Save R^2 results to CSV 
R2results_filename = path.join(output_folder, "Multiple_Clutch_Brakes_R2_results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info(f"Multiple Clutch Brakes R2 results saved to {R2results_filename}")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
