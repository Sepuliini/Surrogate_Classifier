# --DTLZ--
import numpy as np
import pandas as pd
import os
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from pyDOE import lhs
from scipy.stats.distributions import norm

# Problem configurations
problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5]
num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]

# Noise parameters
noise_mean = 0
noise_std = 0.1

# Base output directory
base_dir = "/scratch/project_2012636/Data"
dtlz_root_dir = os.path.join(base_dir, "DTLZ")
os.makedirs(dtlz_root_dir, exist_ok=True)

def generate_datasets(problem, num_vars, num_obj, num_samples, distribution):
    # Build the DTLZ problem instance
    dtlz_problem = test_problem_builder(problem, n_of_objectives=num_obj, n_of_variables=num_vars)

    # Create problem-specific directory
    problem_dir = os.path.join(dtlz_root_dir, problem)
    os.makedirs(problem_dir, exist_ok=True)

    # Generate input samples
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
        np.clip(sample_data, 0, 1, out=sample_data)

    # Evaluate objectives
    objective_values_list = []
    for sample in sample_data:
        evaluated = dtlz_problem.evaluate(sample)
        if hasattr(evaluated, 'objectives'):
            objective_values = evaluated.objectives
            if objective_values is not None:
                objective_values_list.append(objective_values[0])
        else:
            raise TypeError("EvaluationResults object does not contain objective values.")

    objectives_array = np.array(objective_values_list)
    columns = [f"x{i}" for i in range(1, num_vars + 1)] + [f"f{j}" for j in range(1, num_obj + 1)]

    # Save clean dataset
    clean_data = np.hstack((sample_data, objectives_array))
    clean_df = pd.DataFrame(clean_data, columns=columns)
    clean_filename = f"{problem}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}.csv"
    clean_df.to_csv(os.path.join(problem_dir, clean_filename), index=False)
    print(f"Saved clean dataset to {os.path.join(problem_dir, clean_filename)}")

    # Save noisy dataset
    noisy_objectives = objectives_array + np.random.normal(noise_mean, noise_std, objectives_array.shape)
    noisy_data = np.hstack((sample_data, noisy_objectives))
    noisy_df = pd.DataFrame(noisy_data, columns=columns)
    noisy_filename = f"{problem}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}_noise.csv"
    noisy_df.to_csv(os.path.join(problem_dir, noisy_filename), index=False)
    print(f"Saved noisy dataset to {os.path.join(problem_dir, noisy_filename)}")

# Loop over all configurations
for problem in problems:
    for num_vars in num_vars_list:
        for num_obj in num_obj_list:
            for num_samples in num_samples_list:
                for distribution in distribution_types:
                    generate_datasets(problem, num_vars, num_obj, num_samples, distribution)
# --WFG--
import os
import numpy as np
import pandas as pd
from optproblems import wfg
from pyDOE import lhs
from scipy.stats.distributions import norm

# Problem definitions
problems = {
    "WFG1": wfg.WFG1, "WFG2": wfg.WFG2, "WFG3": wfg.WFG3,
    "WFG4": wfg.WFG4, "WFG5": wfg.WFG5, "WFG6": wfg.WFG6,
    "WFG7": wfg.WFG7, "WFG8": wfg.WFG8, "WFG9": wfg.WFG9,
}

num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5]
num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]
noise_mean = 0
noise_std = 0.1

# Base directory
base_data_dir = "/scratch/project_2012636/Data"
wfg_data_dir = os.path.join(base_data_dir, "WFG")
os.makedirs(wfg_data_dir, exist_ok=True)

def generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution):
    problem_class = problems[problem_name]
    k_base = num_obj - 1
    k = max(k_base, num_vars - num_obj)

    if k % k_base != 0:
        k = ((k // k_base) + 1) * k_base
    if k % 2 != 0:
        k += k_base

    while not (k < num_vars and (num_vars - k) % 2 == 0):
        num_vars += 1
        print(f"Adjusted 'num_vars' to {num_vars} to accommodate 'k' = {k} and keep (num_vars - k) % 2 == 0.")

    # Create problem instance
    objective = problem_class(num_objectives=num_obj, num_variables=num_vars, k=k)

    # Generate input samples
    if distribution == "uniform":
        sample_data = lhs(num_vars, samples=num_samples)
    elif distribution == "normal":
        sample_data = norm(loc=0.5, scale=0.15).ppf(lhs(num_vars, samples=num_samples))
    np.clip(sample_data, 0, 1, out=sample_data)

    # Evaluate true objective values
    objective_values = np.array([objective(sample) for sample in sample_data])

    # Save clean version
    save_data(
        problem_name, num_vars, num_obj, num_samples,
        distribution, sample_data, objective_values, is_noisy=False
    )

    # Save noisy version (add noise to objectives only)
    noisy_objectives = objective_values + np.random.normal(noise_mean, noise_std, objective_values.shape)
    save_data(
        problem_name, num_vars, num_obj, num_samples,
        distribution, sample_data, noisy_objectives, is_noisy=True
    )

def save_data(problem_name, num_vars, num_obj, num_samples, distribution, variables, objectives, is_noisy=False):
    # Make per-problem directory
    problem_dir = os.path.join(wfg_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    var_names = [f"x{i+1}" for i in range(variables.shape[1])]
    obj_names = [f"f{j+1}" for j in range(objectives.shape[1])]
    columns = var_names + obj_names
    data = np.hstack((variables, objectives))
    df = pd.DataFrame(data, columns=columns)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{num_vars}var_{num_obj}obj_{num_samples}samples_{distribution}{suffix}.csv"
    full_path = os.path.join(problem_dir, filename)
    df.to_csv(full_path, index=False)
    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset to {full_path}")

# Generate all combinations
for problem_name in problems:
    for num_vars in num_vars_list:
        for num_obj in num_obj_list:
            for num_samples in num_samples_list:
                for distribution in distribution_types:
                    generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution)

# --Engineering--
import numpy as np
import os
import pandas as pd
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact

# Base data directory
base_data_dir = "/scratch/project_2012636/Data"

# Noise parameters (for objective values only)
noise_mean = 0
noise_std = 0.1

# Sample sizes to generate
num_samples_options = [100, 500, 1000, 2000]

# Engineering problem functions
problem_functions = [
    re21, re22, re23, re24, re25,
    re31, re32, re33,
    multiple_clutch_brakes,
    river_pollution_problem,
    vehicle_crashworthiness,
    car_side_impact,
]

def generate_datasets(problem_function, num_samples):
    problem = problem_function()
    problem_name = problem_function.__name__
    num_vars = len(problem.variables)
    num_objs = len(problem.objectives)

    # Create output folder: /scratch/project_2012636/Data/problem_name/
    problem_dir = os.path.join(base_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    # Generate uniform samples
    X = np.empty((num_samples, num_vars))
    for i, var in enumerate(problem.variables):
        lb, ub = var.get_bounds()
        X[:, i] = np.random.uniform(lb, ub, num_samples)

    # Evaluate objective values
    Y = np.empty((num_samples, num_objs))
    for i in range(num_samples):
        res = problem.evaluate(X[i])
        Y[i, :] = res.objectives

    # Save clean version
    save_dataset(X, Y, problem_dir, problem_name, num_samples, is_noisy=False)

    # Save noisy version
    Y_noisy = Y + np.random.normal(noise_mean, noise_std, Y.shape)
    save_dataset(X, Y_noisy, problem_dir, problem_name, num_samples, is_noisy=True)

def save_dataset(X, Y, out_dir, problem_name, num_samples, is_noisy=False):
    var_names = [f"x{i+1}" for i in range(X.shape[1])]
    obj_names = [f"f{i+1}" for i in range(Y.shape[1])]
    columns = var_names + obj_names

    data = np.hstack((X, Y))
    df = pd.DataFrame(data, columns=columns)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{num_samples}_samples{suffix}.csv"
    df.to_csv(os.path.join(out_dir, filename), index=False)

    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset to {os.path.join(out_dir, filename)}")

# Main loop to generate all datasets
for num_samples in num_samples_options:
    for problem_function in problem_functions:
        generate_datasets(problem_function, num_samples)
