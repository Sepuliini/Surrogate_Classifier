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
log_file = datetime.now().strftime("vehicle_crashworthiness_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)

# Start timing the script
script_start = datetime.now()

# Folder paths
base_folder = getcwd()
data_folder = path.join(base_folder, "data", "vehicle_crashworthiness")
output_folder = path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not path.exists(output_folder):
    makedirs(output_folder)

# Define surrogate modeling techniques
models = {
    "SVM": svm.SVR(),
    "NN": MLPRegressor(),
    "Ada": ensemble.AdaBoostRegressor(),
    "GPR": GaussianProcessRegressor(),
    "SGD": SGD(),
    "KNR": KNR(),
    "DTR": DTR(),
    "RFR": ensemble.RandomForestRegressor(),
    "ExTR": ensemble.ExtraTreesRegressor(),
    "GBR": ensemble.GradientBoostingRegressor(),
    "XGB": XGBRegressor()
}

# Option to use the full dataset or a subset for testing
use_full_dataset = False  # Change to True to run on the entire dataset

# Initialize R2results DataFrame outside the main loop
columns = ['File', 'Model', 'Objective', 'R2_Score']
R2results = pd.DataFrame(columns=columns)

# Collecting files and optionally selecting a subset
files = [f for f in listdir(data_folder) if f.endswith('.csv')]
selected_files = random.sample(files, int(len(files) * 0.1)) if not use_full_dataset else files

# Process each dataset
for file in selected_files:
    fullfilename = path.join(data_folder, file)
    logging.info(f"Processing file: {fullfilename}")

    data = pd.read_csv(fullfilename)
    num_objectives = 3 
    objectives_columns = data.columns[-num_objectives:]
    inputs = data.iloc[:, :-num_objectives]  # Exclude objective columns from inputs

    for objective in objectives_columns:
        y = data[objective]
        inputs_train, inputs_test, y_train, y_test = tts(inputs, y, test_size=0.3, random_state=42)

        for algo_name, algo in models.items():
            model_start = datetime.now()
            algo.fit(inputs_train, y_train)
            y_pred = algo.predict(inputs_test)
            score = r2_score(y_test, y_pred)

            model_end = datetime.now()
            training_duration = (model_end - model_start).total_seconds()
            logging.info(f"Trained {algo_name} on {objective} in {training_duration:.2f} seconds with R^2 score: {score}")

            R2results = R2results.append({'File': file, 'Model': algo_name, 'Objective': objective, 'R2_Score': score}, ignore_index=True)

# Save R^2 results to CSV
R2results_filename = path.join(output_folder, "VehicleCrashworthiness_R2_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("Surrogate modeling for Vehicle Crashworthiness problem completed.")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
