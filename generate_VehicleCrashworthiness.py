import numpy as np
import os
import pandas as pd
from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem

# Import your problem function here
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness

# Define the base data directory
base_data_dir = os.path.join(os.getcwd(), "data")

# Ensure the base data directory exists
os.makedirs(base_data_dir, exist_ok=True)

# Function to create a directory for a specific problem inside the base data directory
def create_directory_for_problem(problem_name):
    problem_dir = os.path.join(base_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)
    return problem_dir

# Function to create datasets
def create_dataset(problem_function, num_samples):
    problem = problem_function()
    num_variables = len(problem.variables)
    problem_name = problem_function.__name__
    problem_dir = create_directory_for_problem(problem_name)
    samples = np.empty((num_samples, num_variables))

    # Generate random samples within the variable bounds
    for i, var in enumerate(problem.variables):
        lower_bound, upper_bound = var.get_bounds()
        samples[:, i] = np.random.uniform(lower_bound, upper_bound, num_samples)

    # Evaluate objectives for each sample
    objectives = np.empty((num_samples, len(problem.objectives)))
    for i, sample in enumerate(samples):
        evaluation_result = problem.evaluate(sample)
        objectives[i, :] = evaluation_result.objectives

    # Combine the variables and objectives into one dataset
    data = np.hstack((samples, objectives))

    # Adjusted to use a generic naming scheme
    var_names = [f'x{i+1}' for i in range(num_variables)]
    obj_names = [f'f{j+1}' for j in range(len(problem.objectives))]
    column_names = var_names + obj_names
    df = pd.DataFrame(data, columns=column_names)

    # Save as CSV in the problem's directory
    csv_filename = os.path.join(problem_dir, f"{problem_name}_{num_samples}_samples.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved dataset to {csv_filename}")

# Define the number of samples for each problem
num_samples_options = [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]

# List of your problem functions
problem_functions = [vehicle_crashworthiness]

# Generate and save the dataset for each sample size
for num_samples in num_samples_options:
    for problem_function in problem_functions:
        create_dataset(problem_function, num_samples)
