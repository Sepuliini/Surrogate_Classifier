import os
import numpy as np
import pandas as pd
from optproblems import wfg
from pyDOE import lhs
from scipy.stats.distributions import norm

# Parameters
problems = {
    "WFG1": wfg.WFG1,
    "WFG2": wfg.WFG2,
    "WFG3": wfg.WFG3,
    "WFG4": wfg.WFG4,
    "WFG5": wfg.WFG5,
    "WFG6": wfg.WFG6,
    "WFG7": wfg.WFG7,
    "WFG8": wfg.WFG8,
    "WFG9": wfg.WFG9,
}
num_vars_list = [10, 20, 30]
num_obj = 2
num_samples_list = [100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000]
distribution_types = ["uniform", "normal"]
noise_mean = 0
noise_std = 0.1
base_data_dir = os.path.join(os.getcwd(), "data")
wfg_data_dir = os.path.join(base_data_dir, "wfg_data")

# Create base data directories if they don't exist
os.makedirs(wfg_data_dir, exist_ok=True)

# Function to generate and save datasets
def generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution):
    problem_class = problems[problem_name]
    # Base calculation of k
    k = num_obj - 1

    # Ensure (num_vars - k) is even for all WFG problems
    if (num_vars - k) % 2 != 0:
        k += 1

    # Ensure k is within the bounds
    if k >= num_vars:
        k = num_vars - 2 if num_vars > 1 else 1
    
    objective = problem_class(num_objectives=num_obj, num_variables=num_vars, k=k)

    # Generate sample data
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
        np.clip(sample_data, 0, 1, out=sample_data)

    objective_values = np.array([objective(sample) for sample in sample_data])

    # Save data in problem-specific subfolder
    problem_data_dir = os.path.join(wfg_data_dir, problem_name.lower())
    os.makedirs(problem_data_dir, exist_ok=True)

    save_data(problem_name, problem_data_dir, num_vars, num_obj, num_samples, distribution, sample_data, objective_values, False)
    noisy_objective_values = objective_values + np.random.normal(noise_mean, noise_std, objective_values.shape)
    save_data(problem_name, problem_data_dir, num_vars, num_obj, num_samples, distribution, sample_data, noisy_objective_values, True)

def save_data(problem_name, problem_data_dir, num_vars, num_obj, num_samples, distribution, variables, objectives, noisy):
    var_names = [f"x{i+1}" for i in range(num_vars)]
    obj_names = [f"f{j+1}" for j in range(num_obj)]
    columns = var_names + obj_names
    data = np.hstack((variables, objectives))
    df = pd.DataFrame(data, columns=columns)
    noise_suffix = "_noisy" if noisy else ""
    filename = f"{problem_name}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}{noise_suffix}.csv"
    df.to_csv(os.path.join(problem_data_dir, filename), index=False)
    print(f"Saved dataset to {os.path.join(problem_data_dir, filename)}")

# Generate datasets
for problem_name in problems:
    for num_vars in num_vars_list:
        for num_samples in num_samples_list:
            for distribution in distribution_types:
                generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution)
