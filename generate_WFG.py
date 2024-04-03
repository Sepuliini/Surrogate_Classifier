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
num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5]
num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]
noise_mean = 0
noise_std = 0.1

base_data_dir = os.path.join(os.getcwd(), "data")
wfg_data_dir = os.path.join(base_data_dir, "wfg_data")
os.makedirs(wfg_data_dir, exist_ok=True)

def generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution):
    problem_class = problems[problem_name]
    k_base = num_obj - 1

    # Initial 'k' calculation ensures it's at least as large as 'k_base' and accounts for problem requirements
    k = max(k_base, num_vars - num_obj)

    # Ensure 'k' is a multiple of (num_objectives - 1) and adjust to be even if necessary
    if k % k_base != 0:
        k = ((k // k_base) + 1) * k_base
    if k % 2 != 0:
        k += k_base

    # Adjust 'num_vars' to ensure 'k < num_vars' and '(num_vars - k) % 2 == 0'
    if not (k < num_vars and (num_vars - k) % 2 == 0):
        # Increment 'num_vars' to satisfy both conditions
        while not (k < num_vars and (num_vars - k) % 2 == 0):
            num_vars += 1
        print(f"Adjusted 'num_vars' to {num_vars} to accommodate 'k' = {k} and maintain (num_vars - k) % 2 == 0.")

    objective = problem_class(num_objectives=num_obj, num_variables=num_vars, k=k)

    # Generate sample data
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
    np.clip(sample_data, 0, 1, out=sample_data)

    objective_values = np.array([objective(sample) for sample in sample_data])

    for noisy in [False, True]:
        objectives_noisy = objective_values + np.random.normal(noise_mean, noise_std, objective_values.shape) if noisy else objective_values
        file_suffix = "_noisy" if noisy else ""
        save_data(problem_name, wfg_data_dir, num_vars, num_obj, num_samples, distribution, sample_data, objectives_noisy, file_suffix)

def save_data(problem_name, wfg_data_dir, num_vars, num_obj, num_samples, distribution, variables, objectives, file_suffix):
    var_names = [f"x{i+1}" for i in range(num_vars)]
    obj_names = [f"f{j+1}" for j in range(num_obj)]
    columns = var_names + obj_names
    data = np.hstack((variables, objectives))
    df = pd.DataFrame(data, columns=columns)
    filename = f"{problem_name}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}{file_suffix}.csv"
    df.to_csv(os.path.join(wfg_data_dir, filename), index=False)
    print(f"Saved dataset to {filename}")

# Generate datasets
for problem_name in problems:
    for num_vars in num_vars_list:
        for num_obj in num_obj_list:
            for num_samples in num_samples_list:
                for distribution in distribution_types:
                    generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution)
