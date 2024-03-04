
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
log_file = datetime.now().strftime("river_pollution_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=path.join(log_dir, log_file), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

logging.captureWarnings(True)

# Start timing the script
script_start = datetime.now()

# Option to use the full dataset or a subset for testing
use_full_dataset = False  # Change to True to run on the entire dataset

# Folder paths
base_folder = getcwd()
data_folder = path.join(base_folder, "data", "river_pollution_problem")
output_folder = path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not path.exists(output_folder):
    makedirs(output_folder)

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

# Collecting files and optionally selecting a subset
files = [f for f in listdir(data_folder) if f.endswith('.csv')]
selected_files = random.sample(files, int(len(files) * 0.1)) if not use_full_dataset else files

# Process each dataset
for file in selected_files:
    fullfilename = path.join(data_folder, file)
    logging.info(f"Processing file: {fullfilename}")

    data = pd.read_csv(fullfilename)
    inputs = data.iloc[:, :-5] 
    for i in range(1, 6):
        objective = f'f_{i}'
        y = data[objective]
        inputs_train, inputs_test, y_train, y_test = tts(inputs, y, test_size=0.3, random_state=42)

        for algo_name, algo_class in model.items():  # Corrected line
            model_start = datetime.now()
            clf = algo_class()  # Instantiate the model
            clf.fit(inputs_train, y_train)
            y_pred = clf.predict(inputs_test)
            score = r2_score(y_test, y_pred)

            model_end = datetime.now()
            training_duration = (model_end - model_start).total_seconds()
            logging.info(f"Trained {algo_name} on {objective} in {training_duration:.2f} seconds with R^2 score: {score}")

            R2results = R2results.append({
                'File': file, 
                'Model': algo_name, 
                'Objective': objective, 
                'R2_Score': score
            }, ignore_index=True)

# Save R^2 results to CSV specifically named for river pollution results
R2results_filename = path.join(output_folder, "RiverPollution_R2_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("Surrogate modeling for River Pollution problem completed.")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
