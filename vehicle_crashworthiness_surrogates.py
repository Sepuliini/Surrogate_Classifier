import numpy as np
import pandas as pd
import os
import random
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import r2_score
from sklearn import ensemble, svm, linear_model, gaussian_process, neighbors, tree, neural_network
from xgboost import XGBRegressor

# Setup logging
log_dir = os.path.join(os.getcwd(), "logs")
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = datetime.now().strftime("vehicle_crashworthiness_model_training_%Y-%m-%d_%H-%M-%S.log")
logging.basicConfig(filename=os.path.join(log_dir, log_file), level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logging.captureWarnings(True)

# Start timing the script
script_start = datetime.now()

# Folder paths
base_folder = os.getcwd()
data_folder = os.path.join(base_folder, "data", "vehicle_crashworthiness")
output_folder = os.path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define surrogate modeling techniques
algorithms = {
    "SVM": svm.SVR(),
    "NN": neural_network.MLPRegressor(),
    "Ada": ensemble.AdaBoostRegressor(),
    "GPR": gaussian_process.GaussianProcessRegressor(),
    "SGD": linear_model.SGDRegressor(),
    "KNR": neighbors.KNeighborsRegressor(),
    "DTR": tree.DecisionTreeRegressor(),
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
files = [f for f in os.listdir(data_folder) if f.endswith('.csv')]
selected_files = random.sample(files, int(len(files) * 0.1)) if not use_full_dataset else files

# Process each dataset
for file in selected_files:
    fullfilename = os.path.join(data_folder, file)
    logging.info(f"Processing file: {fullfilename}")

    data = pd.read_csv(fullfilename)
    # Assuming the last 3 columns are objectives
    num_objectives = 3
    objectives_columns = data.columns[-num_objectives:]
    inputs = data.iloc[:, :-num_objectives]  # Exclude objective columns from inputs

    for objective in objectives_columns:
        y = data[objective]
        inputs_train, inputs_test, y_train, y_test = tts(inputs, y, test_size=0.3, random_state=42)

        for algo_name, algo in algorithms.items():
            model_start = datetime.now()
            algo.fit(inputs_train, y_train)
            y_pred = algo.predict(inputs_test)
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


# Save R^2 results to CSV specifically named for vehicle crashworthiness results
R2results_filename = os.path.join(output_folder, "VehicleCrashworthiness_R2_Results.csv")
R2results.to_csv(R2results_filename, index=False)
logging.info("Surrogate modeling for Vehicle Crashworthiness problem completed.")

script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
logging.info(f"Script finished in {elapsed_time:.2f} seconds.")
