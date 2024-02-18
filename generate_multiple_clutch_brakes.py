import numpy as np
import os
import pandas as pd
from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem import ScalarConstraint

# Import your problem function here
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes

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

# Function to create datasets
def create_dataset(problem_function, num_samples):
    # Generate the problem instance
    problem = problem_function()
    num_variables = len(problem.variables)
    problem_name = problem_function.__name__

    # Create directory for the current problem
    problem_dir = create_directory_for_problem(problem_name)

    # Generate random samples within the variable bounds
    samples = np.empty((num_samples, num_variables))
    for i, var in enumerate(problem.variables):
        lower_bound, upper_bound = var.get_bounds()
        samples[:, i] = np.random.uniform(lower_bound, upper_bound, num_samples)

    # Initialize an array for objectives
    objectives = np.empty((num_samples, len(problem.objectives)))

    # Evaluate objectives for each sample
    for i, sample in enumerate(samples):
        evaluation_result = problem.evaluate(sample)
        objectives[i, :] = evaluation_result.objectives  # Directly access the 'objectives' attribute

    # Combine the variables and objectives into one dataset
    data = np.hstack((samples, objectives))

    # Convert to DataFrame
    column_names = [var.name for var in problem.variables] + [obj.name for obj in problem.objectives]
    df = pd.DataFrame(data, columns=column_names)

    # Save as CSV in the problem's directory
    csv_filename = os.path.join(problem_dir, f"{problem_name}_{num_samples}_samples.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved dataset to {csv_filename}")

# Define the number of samples for each problem
num_samples_options = [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]

# Use your problem function here
problem_function = multiple_clutch_brakes

# Loop through all problem functions and sample sizes
for num_samples in num_samples_options:
    # Generate and save the dataset for the current problem and sample size
    create_dataset(problem_function, num_samples)
