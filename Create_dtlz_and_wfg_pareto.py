import pygmo as pg
import os
import pandas as pd

# List of problems to generate pareto fronts for
problems = ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7", 
            "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]

# Loop over each problem and generate pareto front
for problem_name in problems:
    print(f"Generating Pareto front for {problem_name}")

    # Dynamically access the problem using pygmo
    if problem_name.startswith("dtlz"):
        # Access DTLZ problems dynamically
        problem_func = getattr(pg, "dtlz")  # Access DTLZ problems
        problem_instance = problem_func(int(problem_name[-1]), 10, 3)  # (problem_number, num_vars, num_objs)
    elif problem_name.startswith("wfg"):
        # Access WFG problems dynamically
        problem_func = getattr(pg, "wfg")  # Access WFG problems
        problem_instance = problem_func(int(problem_name[-1]), 10, 3)  # (problem_number, num_vars, num_objs)
    else:
        print(f"Problem {problem_name} is not supported.")
        continue

    # Setup the problem
    print(f"[INFO] Solving {problem_name} with 10 variables and 3 objectives.")

    # Create an algorithm for solving the problem (NSGA-II with 100 generations)
    algo = pg.algorithm(pg.nsga2(gen=100))  # Use NSGA-II with 100 generations

    # Create a population for the algorithm
    pop = pg.population(problem_instance, 100)  # Population size of 100

    # Evolve the population using the algorithm
    pop = algo.evolve(pop)

    # Extract and save the pareto front (non-dominated solutions)
    pareto_front = pop.get_f()

    # Create output directory for each problem
    output_dir = f"/projappl/project_2012636/paretofronts/{problem_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Save the Pareto front to a CSV file
    file_path = os.path.join(output_dir, f"{problem_name}_pareto_front.csv")
    pd.DataFrame(pareto_front).to_csv(file_path, index=False)
    print(f"Pareto front for {problem_name} saved to {file_path}")
