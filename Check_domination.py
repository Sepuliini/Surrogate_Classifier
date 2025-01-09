#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import numba

# Define the path to the surrogate optimization results directory
results_dir = "/scratch/project_2012636/modelling_results/surrogate_optimization_results"

# Numba optimized function to check dominance
@numba.njit()
def dominates(x, y):
    dom = False
    for i in range(len(x)):
        if x[i] > y[i]:
            return False
        elif x[i] < y[i]:
            dom = True
    return dom

@numba.njit()
def non_dominated(data):
    num_solutions = len(data)
    index = np.zeros(num_solutions, dtype=np.bool_)
    index[0] = True
    for i in range(1, num_solutions):
        index[i] = True
        for j in range(i):
            if not index[j]:
                continue
            if dominates(data[i], data[j]):
                index[j] = False
            elif dominates(data[j], data[i]):
                index[i] = False
                break
    return index

# Function to remove dominated solutions and save the cleaned Pareto front
def clean_pareto_front(file_path):
    # Check if file has already been processed
    if "_cleaned.csv" in file_path:
        print(f"Skipping already processed file: {file_path}")
        return

    print(f"Processing file: {file_path}")
    df = pd.read_csv(file_path)

    # Convert data to numerical numpy array, forcing float conversion
    try:
        data = df.to_numpy(dtype=np.float64)  # Force numerical conversion
    except ValueError:
        print(f"Error: Non-numeric data found in {file_path}. Skipping.")
        return

    # Identify non-dominated solutions
    non_dominated_indices = non_dominated(data)
    non_dominated_solutions = data[non_dominated_indices]

    # Convert back to DataFrame and save with a modified filename
    cleaned_df = pd.DataFrame(non_dominated_solutions, columns=df.columns)
    cleaned_file_path = file_path.replace(".csv", "_cleaned.csv")
    cleaned_df.to_csv(cleaned_file_path, index=False)
    print(f"Cleaned and saved: {cleaned_file_path}")

# Traverse all folders and CSV files in the results directory
for root, dirs, files in os.walk(results_dir):
    for file in files:
        if file.endswith(".csv") and not file.endswith("_cleaned.csv"):
            file_path = os.path.join(root, file)
            clean_pareto_front(file_path)

print("All Pareto fronts have been processed and cleaned.")

