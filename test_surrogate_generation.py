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

import warnings
warnings.filterwarnings("ignore")

# Start timing the script
script_start = datetime.now()

# Folder paths
base_folder = getcwd()
data_folder = path.join(base_folder, "data", "dtlz_data")
output_folder = path.join(base_folder, "modelling_results")

# Ensure the output folder exists
if not path.exists(output_folder):
    makedirs(output_folder)

# Control whether to use the full dataset or a subset
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

files = listdir(data_folder)

if use_full_dataset:
    selected_files = files
else:
    num_files_to_select = max(int(len(files) * 0.1), 1)
    selected_files = random.sample(files, num_files_to_select)

# Initialize DataFrame to store results
R2results = pd.DataFrame(np.zeros((len(selected_files), 2*len(algorithms) + 1)), columns=["file"] + [f"{algo}_{fn}" for algo in algorithms for fn in ['f1', 'f2']])

# Iterate over selected files and train each algorithm
for i, file in enumerate(selected_files):
    print(f"Processing file {i + 1} of {len(selected_files)}: {file}")
    fullfilename = path.join(data_folder, file)
    data = pd.read_csv(fullfilename)
    inputs = data.iloc[:, :-2]  # Assuming the last two columns are targets
    f1 = data.iloc[:, -2]       # Second-to-last column as the first target
    f2 = data.iloc[:, -1]       # Last column as the second target
    
    inputs_train_f1, inputs_test_f1, f1_train, f1_test = tts(inputs, f1, test_size=0.3, random_state=42)
    inputs_train_f2, inputs_test_f2, f2_train, f2_test = tts(inputs, f2, test_size=0.3, random_state=42)
    
    R2results.loc[i, "file"] = file
    
    for algo in algorithms:
        model_start = datetime.now()
        clf_f1 = model[algo]()
        clf_f1.fit(inputs_train_f1, f1_train)
        pred_f1 = clf_f1.predict(inputs_test_f1)
        score_f1 = r2_score(f1_test, pred_f1)
        R2results.loc[i, f"{algo}_f1"] = score_f1

        clf_f2 = model[algo]()
        clf_f2.fit(inputs_train_f2, f2_train)
        pred_f2 = clf_f2.predict(inputs_test_f2)
        score_f2 = r2_score(f2_test, pred_f2)
        R2results.loc[i, f"{algo}_f2"] = score_f2

        model_end = datetime.now()
        training_duration = (model_end - model_start).total_seconds()
        print(f"Trained {algo} on {file} for f1 & f2 in {training_duration:.2f} seconds")

# Save R^2 results to CSV in the output folder
R2results_filename = path.join(output_folder, "R2results.csv")
R2results.to_csv(R2results_filename, index=False)
print(f"R^2 results saved to {R2results_filename}")

# End timing the script and print elapsed time
script_end = datetime.now()
elapsed_time = (script_end - script_start).total_seconds()
print(f"Script finished in {elapsed_time:.2f} seconds.")
