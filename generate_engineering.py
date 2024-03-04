import numpy as np
import os
import pandas as pd
from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem import ScalarConstraint

# Import the engineering problem functions here
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33

# Define the base data directory
base_data_dir = os.path.join(os.getcwd(), "data")

# Make sure the base data directory exists
os.makedirs(base_data_dir, exist_ok=True)

# Function to create a directory for a specific problem inside the base data directory
def create_directory_for_problem(problem_name):
    # Create a directory for the problem
    problem_dir = os.path.join(base_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)
    return problem_dir

def create_dataset(problem_function, num_samples):
    problem = problem_function()
    num_variables = len(problem.variables)
    num_objectives = len(problem.objectives)
    problem_name = problem_function.__name__

    problem_dir = create_directory_for_problem(problem_name)

    samples = np.empty((num_samples, num_variables))
    for i, var in enumerate(problem.variables):
        lower_bound, upper_bound = var.get_bounds()
        samples[:, i] = np.random.uniform(lower_bound, upper_bound, num_samples)

    objectives = np.empty((num_samples, num_objectives))
    for i, sample in enumerate(samples):
        evaluation_result = problem.evaluate(sample)
        objectives[i, :] = evaluation_result.objectives

    data = np.hstack((samples, objectives))

    # Generate standardized column names
    variable_column_names = [f"x_{i+1}" for i in range(num_variables)]
    objective_column_names = [f"f_{i+1}" for i in range(num_objectives)]
    column_names = variable_column_names + objective_column_names

    df = pd.DataFrame(data, columns=column_names)

    csv_filename = os.path.join(problem_dir, f"{problem_name}_{num_samples}_samples.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved dataset to {csv_filename}")


# Define the number of samples for each problem
num_samples_options = [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]

# List of your problem functions
problem_functions = [re21, re22, re23, re24, re25, re31, re32, re33]

# Loop through all problem functions and sample sizes
for num_samples in num_samples_options:
    for problem_function in problem_functions:
        # Generate and save the dataset for the current problem and sample size
        create_dataset(problem_function, num_samples)