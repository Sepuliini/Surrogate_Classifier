import numpy as np
import pandas as pd
import os
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
from scipy.stats.distributions import norm

# Parameters
problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
num_vars_list = [10, 20, 30]
num_obj = 2
num_samples_list = [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
distribution_types = ["uniform", "normal"]
noise_mean = 0
noise_std = 0.1

# Define the path for the 'data' and 'dtlz_data' folders
data_dir = os.path.join(os.getcwd(), "data")
dtlz_data_dir = os.path.join(data_dir, "dtlz_data")

# Create/Clear the 'dtlz_data' directory
if os.path.exists(dtlz_data_dir):
    # Clear existing files in the directory
    for file in os.listdir(dtlz_data_dir):
        os.remove(os.path.join(dtlz_data_dir, file))
else:
    # Create the directory if it does not exist
    os.makedirs(dtlz_data_dir)

# Function to generate datasets
def generate_datasets(problem, num_vars, num_obj, num_samples, distribution):
    dtlz_problem = test_problem_builder(problem, n_of_objectives=num_obj, n_of_variables=num_vars)
    
    # Generate sample data
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
        np.clip(sample_data, 0, 1, out=sample_data)

    objective_values_list = []
    for sample in sample_data:
        evaluated = dtlz_problem.evaluate(sample)
        # Extract objective values from the EvaluationResults object
        if hasattr(evaluated, 'objectives'):
            objective_values = evaluated.objectives
            if objective_values is not None and len(objective_values.shape) == 2 and objective_values.shape[1] == num_obj:
                objective_values_list.append(objective_values[0])  # Assuming the first index holds the objective values
            else:
                raise ValueError("Invalid shape or content in objective values.")
        else:
            raise TypeError("EvaluationResults object does not contain objective values.")

    objective_values = np.array(objective_values_list)

    # Process each element in objective_values_list
    for i, value in enumerate(objective_values_list):
        if isinstance(value, list) or isinstance(value, np.ndarray):
            if len(value) != num_obj:
                raise ValueError(f"Inconsistent objective value size at index {i}")
        else:
            raise TypeError(f"Invalid type for objective values at index {i}")

    objective_values = np.array(objective_values_list)

    # Check shape consistency
    if objective_values.shape[0] != num_samples or objective_values.shape[1] != num_obj:
        raise ValueError("Inconsistent shape for objective values.")

    # Concatenate sample data with objective values
    data = np.hstack((sample_data, objective_values))

    # Add noise if needed
    if data is not None:
        try:
            data += np.random.normal(noise_mean, noise_std, data.shape)
        except Exception as e:
            raise Exception("Error adding noise to data: " + str(e))
    else:
        raise Exception("Data is None. Failed to concatenate sample data with objective values.")

    # Save data to CSV
    columns = [f"x{i}" for i in range(1, num_vars + 1)] + [f"f{j}" for j in range(1, num_obj + 1)]
    df = pd.DataFrame(data, columns=columns)
    filename = os.path.join(dtlz_data_dir, f"{problem}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}.csv")
    df.to_csv(filename, index=False)
    print(f"Saved dataset to {filename}")

# Generating datasets
for problem in problems:
    for num_vars in num_vars_list:
        for num_samples in num_samples_list:
            for distribution in distribution_types:
                generate_datasets(problem, num_vars, num_obj, num_samples, distribution)
