#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd

from desdeo_problem.testproblems.EngineeringRealWorld import (
    re21, re22, re23, re24, re25, re31, re32, re33
)
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact

# =========================================================
# CONFIG
# =========================================================
SEED = 42
np.random.seed(SEED)

base_data_dir = "/scratch/project_2017216/Data/Engineering/"

noise_mean = 0.0
noise_std = 0.1
SAVE_NOISY_DATA = True
CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE = False

num_samples_options = [100, 500, 1000, 2000]

problem_functions = [
    re21, re22, re23, re24, re25,
    re31, re32, re33,
    multiple_clutch_brakes,
    river_pollution_problem,
    vehicle_crashworthiness,
    car_side_impact,
]

# =========================================================
# HELPERS
# =========================================================
def get_problem_name(problem_function):
    return problem_function.__name__


def sample_inputs(problem, num_samples):
    num_vars = len(problem.variables)
    X = np.empty((num_samples, num_vars), dtype=float)

    for i, var in enumerate(problem.variables):
        lb, ub = var.get_bounds()
        X[:, i] = np.random.uniform(lb, ub, num_samples)

    return X


def evaluate_problem(problem, X):
    num_samples = X.shape[0]
    num_objs = len(problem.objectives)
    Y = np.empty((num_samples, num_objs), dtype=float)

    for i in range(num_samples):
        res = problem.evaluate(X[i])
        Y[i, :] = np.asarray(res.objectives).reshape(-1)

    return Y


def check_generated_data(problem_name, X, Y, Y_noisy=None):
    print(f"\n=== SANITY CHECK: {problem_name} ===")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    print(f"Decision variable min: {np.min(X):.6f}")
    print(f"Decision variable max: {np.max(X):.6f}")

    print(f"Objective min: {np.min(Y):.6f}")
    print(f"Objective max: {np.max(Y):.6f}")
    print(f"Objective mean: {np.mean(Y):.6f}")

    if not np.all(np.isfinite(X)):
        print("[WARN] Non-finite values found in X.")
    if not np.all(np.isfinite(Y)):
        print("[WARN] Non-finite values found in clean objectives.")

    if Y_noisy is not None:
        print(f"Noisy objective min: {np.min(Y_noisy):.6f}")
        print(f"Noisy objective max: {np.max(Y_noisy):.6f}")
        print(f"Noisy objective mean: {np.mean(Y_noisy):.6f}")

        if not np.all(np.isfinite(Y_noisy)):
            print("[WARN] Non-finite values found in noisy objectives.")

        neg_count = int(np.sum(Y_noisy < 0))
        print(f"Noisy objective negative values: {neg_count}")


def save_dataset(X, Y, out_dir, problem_name, num_samples, is_noisy=False):
    var_names = [f"x{i+1}" for i in range(X.shape[1])]
    obj_names = [f"f{i+1}" for i in range(Y.shape[1])]
    columns = var_names + obj_names

    data = np.hstack((X, Y))
    df = pd.DataFrame(data, columns=columns)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{X.shape[1]}var_{Y.shape[1]}obj_{num_samples}samples{suffix}.csv"
    full_path = os.path.join(out_dir, filename)

    df.to_csv(full_path, index=False)
    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset to {full_path}")


def generate_datasets(problem_function, num_samples):
    problem = problem_function()
    problem_name = get_problem_name(problem_function)

    num_vars = len(problem.variables)
    num_objs = len(problem.objectives)

    problem_dir = os.path.join(base_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    X = sample_inputs(problem, num_samples)
    Y = evaluate_problem(problem, X)

    save_dataset(X, Y, problem_dir, problem_name, num_samples, is_noisy=False)

    Y_noisy = None
    if SAVE_NOISY_DATA:
        Y_noisy = Y + np.random.normal(noise_mean, noise_std, Y.shape)

        if CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE:
            Y_noisy = np.clip(Y_noisy, 0.0, None)

        save_dataset(X, Y_noisy, problem_dir, problem_name, num_samples, is_noisy=True)

    check_generated_data(problem_name, X, Y, Y_noisy=Y_noisy)


# =========================================================
# MAIN
# =========================================================
def main():
    print(f"Saving engineering datasets under: {base_data_dir}")
    print(f"Random seed: {SEED}")
    print(f"Noise mean: {noise_mean}")
    print(f"Noise std: {noise_std}")
    print(f"Save noisy data: {SAVE_NOISY_DATA}")
    print(f"Clip noisy objectives to nonnegative: {CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE}")

    for num_samples in num_samples_options:
        for problem_function in problem_functions:
            try:
                generate_datasets(problem_function, num_samples)
            except Exception as e:
                print(
                    f"[ERROR] Failed on {problem_function.__name__} "
                    f"with {num_samples} samples: {e}"
                )


if __name__ == "__main__":
    main()
