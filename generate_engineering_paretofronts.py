#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
from desdeo_problem.problem.Variable import Variable
from desdeo_problem.problem.Objective import ScalarObjective
from desdeo_problem.problem.Problem import MOProblem
from desdeo_problem import ScalarConstraint
from desdeo_emo.EAs import NSGAIII, RVEA
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem


# Engineering problems mapping
problem_mapping = {
    "multiple_clutch_brakes": multiple_clutch_brakes,
    "car_side_impact": car_side_impact,
    "vehicle_crashworthniess": vehicle_crashworthiness,
    "river_pollution_problem": river_pollution_problem,
    "re21": re21,
    "re22": re22,
    "re23": re23,
    "re24": re24,
    "re25": re25,
    "re31": re31,
    "re32": re32,
    "re33": re33
}

# Define necessary input variables
num_vars = 5
output_dir = "/scratch/project_2012636/modelling_results/real_paretofronts"
sample_size = 100
algorithm_name = "desdeo_ea"

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define the common parameters for all EAs (Increased for better Pareto front generation)
common_params = {'n_iterations': 100, 'n_gen_per_iter': 500}

# Optimization algorithms to use with their specific additional parameters (Increased population size)
eas = {
    "NSGAIII": (NSGAIII, {'population_size': 500}),
    "RVEA": (RVEA, {'population_size': 500})
}

# Function to check if a solution dominates another
import numba
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

# Function to remove dominated solutions and save results
def save_optimization_results(solutions, problem_name, algorithm_name, output_dir):
    num_obj = solutions.shape[1]
    # Remove dominated solutions
    non_dominated_indices = non_dominated(solutions)
    non_dominated_solutions = solutions[non_dominated_indices]

    filename = f"{problem_name}_combined_real_paretofront.csv"
    file_path = os.path.join(output_dir, filename)
    
    # Convert solutions to a DataFrame for easier saving
    df = pd.DataFrame(non_dominated_solutions, columns=[f"obj_{i+1}" for i in range(num_obj)])
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

# Loop through each problem and run the optimization loop
for problem_name, problem_function in problem_mapping.items():
    print(f"Processing problem: {problem_name}")

    # Define variables with an initial value
    initial_value = 0.5
    variables = [Variable(f"x_{i+1}", lower_bound=0.1, upper_bound=1.0, initial_value=initial_value) for i in range(num_vars)]

    # Instantiate the engineering problem
    problem = problem_function()

    combined_solutions = []

    for ea_name, (ea, ea_specific_params) in eas.items():
        print(f"Running {ea_name} on {problem_name}")
        evolver = ea(problem, **{**common_params, **ea_specific_params})

        all_solutions = []

        while evolver.continue_evolution():
            evolver.iterate()
            # Collect solutions at each generation
            all_solutions.append(evolver.population.objectives)

        # Flatten all collected solutions across generations
        all_solutions = np.vstack(all_solutions)
        combined_solutions.append(all_solutions)

    # Combine solutions from both algorithms and filter non-dominated solutions
    combined_solutions = np.vstack(combined_solutions)
    save_optimization_results(combined_solutions, problem_name, algorithm_name, output_dir)
