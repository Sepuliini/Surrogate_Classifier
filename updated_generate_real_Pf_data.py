#--DTLZ and WFG--
import numpy as np
import pandas as pd
import os
import logging
from os import path
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.termination import get_termination
from pymoo.optimize import minimize

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ---------- Utility ----------

def save_pareto_front(F, folder, filename):
    os.makedirs(folder, exist_ok=True)
    df = pd.DataFrame(F)
    df.to_csv(path.join(folder, filename), index=False, header=False)

def comb(n, k):
    from math import comb as math_comb
    return math_comb(n, k)

# ---------- DTLZ Front Generators ----------

def generate_dtlz1_pareto_front(n_points, n_obj):
    def uniform_points_on_simplex(n, m):
        H1 = 1
        while comb(H1 + m - 1, m - 1) <= n:
            H1 += 1
        from itertools import combinations
        pts = []
        for c in combinations(range(H1 + m - 1), m - 1):
            pt = np.zeros(m)
            pt[0] = c[0]
            for i in range(1, m - 1):
                pt[i] = c[i] - c[i - 1] - 1
            pt[-1] = H1 + m - 2 - c[-1]
            pts.append(pt / H1)
        return np.array(pts)
    points = uniform_points_on_simplex(n_points, n_obj)
    return 0.5 * points

def generate_dtlz23_pareto_front(n_points, n_obj):
    vec = np.random.normal(0, 1, (n_points, n_obj))
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec

def generate_dtlz7_pareto_front(n_points, n_obj):
    f = np.random.rand(n_points, n_obj - 1)
    g = 1 + 9 * np.mean(f[:, 1:], axis=1)
    h = n_obj - np.sum((f[:, :-1] / (1 + g[:, None])) * (1 + np.sin(3 * np.pi * f[:, :-1])), axis=1)
    f_last = (1 + g) * h
    return np.hstack((f, f_last[:, None]))

def generate_dtlz_fronts(base_folder, n_points=1000):
    logging.info("Generate DTLZ fronts")
    dtlz_problems = [f"DTLZ{i}" for i in range(1, 8)]
    obj_list = [3, 4, 5]

    for problem_name in dtlz_problems:
        for n_obj in obj_list:
            try:
                logging.info(f"Generating real PF for {problem_name} with {n_obj} objs")

                if problem_name == "DTLZ1":
                    F = generate_dtlz1_pareto_front(n_points, n_obj)
                elif problem_name in ["DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6"]:
                    F = generate_dtlz23_pareto_front(n_points, n_obj)
                elif problem_name == "DTLZ7":
                    F = generate_dtlz7_pareto_front(n_points, n_obj)
                else:
                    raise ValueError(f"Unknown DTLZ problem: {problem_name}")

                # Filter out dominated points
                nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
                F_nd = F[nds]

                folder = path.join(base_folder, problem_name)
                filename = f"real_pareto_front_{n_obj}obj.csv"
                save_pareto_front(F_nd, folder, filename)

            except Exception as e:
                logging.error(f"Failed for {problem_name} with {n_obj} objs: {e}")


# ---------- WFG Approximation ----------

def run_nsgaiii_on_wfg(problem_name, n_var, n_obj, n_points=1000, seed=1):
    logging.info(f"Runnning optimization (seed={seed})")
    name = problem_name.lower()
    problem = get_problem(name, n_var, n_obj)
    n_partitions = {
        3: 12,
        4: 10,
        5: 8,
    }.get(n_obj, 6)

    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    algorithm = NSGA3(pop_size=len(ref_dirs), ref_dirs=ref_dirs)
    termination = get_termination("n_gen", 500)

    res = minimize(problem,
                   algorithm,
                   termination,
                   seed=seed,
                   save_history=False,
                   verbose=False)

    logging.info("Optimization DONE")
    return res.F


def is_valid_wfg_combination(n_var, n_obj):
    k = 2 * (n_obj - 1)
    l = n_var - k
    return n_var > k and l % 2 == 0

def generate_wfg_fronts(base_folder, n_points=1000):
    logging.info("Generate WFG")
    wfg_problems = [f"WFG{i}" for i in range(1, 10)]
    obj_list = [3, 4, 5]
    var_list = [6, 10, 20, 30]

    for problem_name in wfg_problems:
        for n_obj in obj_list:
            for n_var in var_list:
                if not is_valid_wfg_combination(n_var, n_obj):
                    logging.info(f"Skipping invalid WFG config: {problem_name} with {n_var} vars and {n_obj} objs")
                    continue
                try:
                    logging.info(f"Running 3x optimization for {problem_name} with {n_var} vars and {n_obj} objs")

                    all_F = []
                    for seed in [1, 2, 3]:
                        F = run_nsgaiii_on_wfg(problem_name, n_var, n_obj, n_points=n_points)
                        all_F.append(F)

                    combined_F = np.vstack(all_F)
                    nds = NonDominatedSorting().do(combined_F, only_non_dominated_front=True)
                    F_nd = combined_F[nds]

                    folder = os.path.join(base_folder, problem_name)
                    filename = f"pareto_front_{n_obj}obj_{n_var}vars.csv"
                    save_pareto_front(F_nd, folder, filename)

                except Exception as e:
                    logging.error(f"Failed for {problem_name} with {n_obj} objs and {n_var} vars: {e}")

# ---------- Main ----------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Script started...")

    base_folder = "/scratch/project_2012636/modelling_results/real_paretofronts"
    logging.info("Starting DTLZ PF generation...")
    generate_dtlz_fronts(base_folder)

    logging.info("Starting WFG PF approximation...")
    generate_wfg_fronts(base_folder)

#--Engineering--
#!/usr/bin/env python3

from desdeo_problem.testproblems.EngineeringRealWorld import (
    re21, re22, re23, re24, re25, re31, re32, re33
)
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact

from desdeo_emo.EAs import NSGAIII
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting



import numpy as np
import os
import pandas as pd

# List of problem instances and names
problems = [
    ("re21", re21), ("re22", re22), ("re23", re23), ("re24", re24),
    ("re25", re25), ("re31", re31), ("re32", re32), ("re33", re33),
    ("mcb", multiple_clutch_brakes), ("river", river_pollution_problem),
    ("vcrash", vehicle_crashworthiness), ("car", car_side_impact)
]

# Output directory
output_base = "/scratch/project_2012636/modelling_results/real_paretofronts/engineering"
os.makedirs(output_base, exist_ok=True)

# Optimization parameters
n_iterations = 100
n_gen_per_iter = 10
population_size = 100
n_runs = 3

for name, problem_fn in problems:
    print(f"\nOptimizing {name} with {n_runs} independent runs...")

    # Initialize problem
    problem = problem_fn()
    all_objectives = []

    for run in range(n_runs):
        print(f"  Run {run + 1}/{n_runs}")
        algo = NSGAIII(problem, n_iterations=n_iterations, n_gen_per_iter=n_gen_per_iter, population_size=population_size)

        while algo.continue_iteration():
            algo.iterate()

        objectives = algo.end()[0]

        all_objectives.append(objectives)

    # Combine all solutions
    combined = np.vstack(all_objectives)

    # Filter non-dominated solutions
    front = NonDominatedSorting().do(combined, only_non_dominated_front=True)
    non_dominated = combined[front]

    # Save results
    save_path = os.path.join(output_base, f"{name}_NSGAIII_combinedSeeds.csv")
    pd.DataFrame(non_dominated).to_csv(save_path, index=False, header=False)

    print(f"Saved {name} Pareto front with shape: {non_dominated.shape} to {save_path}")
