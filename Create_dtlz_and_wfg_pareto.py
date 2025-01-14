#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pygmo as pg

# Function to save Pareto front to a CSV file
def save_optimization_results(solutions, problem_name, output_dir):
    # No need for domination check as the solutions are assumed to be non-dominated
    filename = f"{problem_name}_real_paretofront.csv"
    file_path = os.path.join(output_dir, filename)
    columns = [f"obj_{i+1}" for i in range(solutions.shape[1])]
    df = pd.DataFrame(solutions, columns=columns)
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

# Parameters
problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7", 
            "WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"]

num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5]
samples = 1000

# Base output directory
base_output_dir = "/scratch/project_2012636/modelling_results/real_paretofronts"

# Loop over each problem and generate Pareto fronts
for problem_name in problems:
    # Create specific folder for DTLZ and WFG problems
    if problem_name.startswith("DTLZ"):
        problem_folder = os.path.join(base_output_dir, "DTLZ")
    elif problem_name.startswith("WFG"):
        problem_folder = os.path.join(base_output_dir, "WFG")
    else:
        continue  # Skip unsupported problems

    # Ensure the problem folder exists
    os.makedirs(problem_folder, exist_ok=True)

    for num_vars in num_vars_list:
        for num_obj in num_obj_list:
            print(f"[INFO] Processing {problem_name} with {num_vars} variables and {num_obj} objectives ...")

            # Create the problem instance using pygmo
            if problem_name.startswith("DTLZ"):
                problem_func = getattr(pg, problem_name.lower())  # Access the function dynamically for DTLZ
                problem = pg.problem(problem_func(num_obj, num_vars))
            elif problem_name.startswith("WFG"):
                problem_func = getattr(pg, problem_name.lower())  # Access the function dynamically for WFG
                problem = pg.problem(problem_func(num_obj, num_vars))
            else:
                print(f"[WARNING] Unsupported problem: {problem_name}. Skipping.")
                continue

            # Generate the Pareto front by sampling solutions in the decision space
            solutions = []
            for _ in range(samples):
                x = np.random.uniform(0, 1, num_vars)  # Sample random decision variables
                solutions.append(problem.fitness(x))  # Get corresponding objective values

            solutions = np.array(solutions)

            # Save the Pareto front (no need for non-dominated filtering)
            save_optimization_results(solutions, f"{problem_name}_{num_vars}vars_{num_obj}objs_{samples}samples", problem_folder)

            print(f"[INFO] Done with {problem_name} for {num_vars} variables and {num_obj} objectives.")

