#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem import ScalarConstraint
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem

# Define the base data directory for Puhti
base_data_dir = "/scratch/project_2012636/Data"

# Ensure the base data directory exists
os.makedirs(base_data_dir, exist_ok=True)

# Variable ranges for sampling
variable_ranges = {
    "multiple_clutch_brakes": [(55, 80), (75, 110), (1.5, 3), (300, 1000), (2, 10)],
    "car_side_impact": [(0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5), (0.875, 2.625), (0.4, 1.2), (0.4, 1.2)],
    "river_pollution_problem": [(0.3, 1.0), (0.3, 1.0)],
    "vehicle_crashworthiness": [(1.0, 3.0)] * 5,
    "re21": [(0.1, 1.0)] * 4,
    "re22": [(0.2, 15), (0, 20), (0, 40)],
    "re23": [(1, 100), (1, 100), (10, 200), (10, 240)],
    "re24": [(0.5, 4), (4, 50)],
    "re25": [(1, 70), (0.6, 30), (0.009, 0.5)],
    "re31": [(0.00001, 100), (0.00001, 100), (1.0, 3.0)],
    "re32": [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)],
    "re33": [(55, 80), (75, 110), (1000, 3000), (11, 20)]
}

# Function to create a directory for each problem
def create_directory_for_problem(problem_name):
    problem_dir = os.path.join(base_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)
    return problem_dir

# Generate and save datasets for each engineering problem
def create_dataset(problem_function, num_samples):
    problem = problem_function()
    num_variables = len(problem.variables)
    num_objectives = len(problem.objectives)
    problem_name = problem_function.__name__

    # Create a folder for the problem
    problem_dir = create_directory_for_problem(problem_name)

    # Generate random samples within the variable ranges
    samples = np.empty((num_samples, num_variables))
    for i, var in enumerate(problem.variables):
        lower_bound, upper_bound = var.get_bounds()
        samples[:, i] = np.random.uniform(lower_bound, upper_bound, num_samples)

    # Evaluate objective values for the samples
    objectives = np.empty((num_samples, num_objectives))
    for i, sample in enumerate(samples):
        evaluation_result = problem.evaluate(sample)
        objectives[i, :] = evaluation_result.objectives

    # Combine inputs and outputs into a single dataframe
    data = np.hstack((samples, objectives))
    variable_column_names = [f"x_{i+1}" for i in range(num_variables)]
    objective_column_names = [f"f_{i+1}" for i in range(num_objectives)]
    column_names = variable_column_names + objective_column_names

    # Save the dataset
    df = pd.DataFrame(data, columns=column_names)
    csv_filename = os.path.join(problem_dir, f"{problem_name}_{num_samples}_samples.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved dataset to {csv_filename}")

# Number of samples for each problem
num_samples_options = [100, 500, 1000, 2000]

# List of engineering problem functions
problem_functions = [re21, re22, re23, re24, re25, re31, re32, re33, multiple_clutch_brakes, car_side_impact, vehicle_crashworthiness, river_pollution_problem]

# Generate datasets for all problems and sample sizes
for num_samples in num_samples_options:
    for problem_function in problem_functions:
        create_dataset(problem_function, num_samples)
