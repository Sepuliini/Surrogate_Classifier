import numpy as np
import pandas as pd
import time
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


import warnings
warnings.filterwarnings("ignore")

# Start timing the script
script_start = time.time()

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

R2results = pd.DataFrame(np.zeros((num_files_to_select, len(algorithms) + 1)), columns=["file"] + algorithms)

# Iterate over selected files and train each algorithm
for i, file in enumerate(selected_files):
    print(f"Processing file {i + 1} of {num_files_to_select}: {file}")
    fullfilename = path.join(data_folder, file)
    data = pd.read_csv(fullfilename)
    inputs = data.iloc[:, :-2]  # Assuming the last two columns are targets
    f2 = data.iloc[:, -1]       # Using the last column as the target
    
    inputs_train, inputs_test, f2_train, f2_test = tts(inputs, f2, test_size=0.3, random_state=42)
    
    R2results.loc[i, "file"] = file
    
    for algo in algorithms:
        clf = model[algo]()
        clf.fit(inputs_train, f2_train)
        pred = clf.predict(inputs_test)
        score = r2_score(f2_test, pred)
        R2results.loc[i, algo] = score

        # Save models
        # model_filename = path.join(output_folder, f"{file}_{algo}_model.pkl")
        # with open(model_filename, 'wb') as f:
        #     pickle.dump(clf, f)

# Save R^2 results to CSV in the output folder
R2results_filename = path.join(output_folder, "R2results.csv")
R2results.to_csv(R2results_filename, index=False)
print(f"R^2 results saved to {R2results_filename}")

# End timing the script and print elapsed time
script_end = time.time()
elapsed_time = script_end - script_start
print(f"Script finished in {elapsed_time:.2f} seconds.")