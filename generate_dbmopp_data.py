#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator

# =========================================================
# CONFIG
# =========================================================
SEED = 42
np.random.seed(SEED)

base_data_dir = "/scratch/project_2017216/Data"
dbmopp_data_dir = os.path.join(base_data_dir, "DBMOPP")
os.makedirs(dbmopp_data_dir, exist_ok=True)

noise_mean = 0.0
noise_std = 0.1

SAVE_NOISY_DATA = True
CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE = False

num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]
num_vars_list = [6, 10, 20, 30]
num_objectives_list = [3, 4, 5, 6, 7]

# =========================================================
# DBMOPP PROBLEM DEFINITIONS
# =========================================================
dbmopp_problems = {
    "DBMOPP1": dict(
        n_local_pareto_regions=2,
        n_dominance_res_regions=0,
        n_global_pareto_regions=3,
        pareto_set_type=0,
        constraint_type=1,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.0,
    ),
    "DBMOPP2": dict(
        n_local_pareto_regions=2,
        n_dominance_res_regions=1,
        n_global_pareto_regions=3,
        pareto_set_type=1,
        constraint_type=3,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.0,
    ),
    "DBMOPP3": dict(
        n_local_pareto_regions=3,
        n_dominance_res_regions=2,
        n_global_pareto_regions=4,
        pareto_set_type=2,
        constraint_type=5,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.0,
    ),
    "DBMOPP4": dict(
        n_local_pareto_regions=3,
        n_dominance_res_regions=4,
        n_global_pareto_regions=5,
        pareto_set_type=2,
        constraint_type=8,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.1,
    ),
    "DBMOPP5": dict(
        n_local_pareto_regions=1,
        n_dominance_res_regions=0,
        n_global_pareto_regions=3,
        pareto_set_type=0,
        constraint_type=1,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.0,
    ),
    "DBMOPP6": dict(
        n_local_pareto_regions=2,
        n_dominance_res_regions=1,
        n_global_pareto_regions=4,
        pareto_set_type=1,
        constraint_type=4,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.2,
    ),
    "DBMOPP7": dict(
        n_local_pareto_regions=2,
        n_dominance_res_regions=2,
        n_global_pareto_regions=5,
        pareto_set_type=2,
        constraint_type=7,
        ndo=0,
        vary_sol_density=False,
        vary_objective_scales=False,
        prop_neutral=0.0,
    ),
}

# =========================================================
# HELPERS
# =========================================================
def lhs_sample(n_samples, n_vars, seed=None):
    """
    Simple Latin hypercube sampling in [0,1]^n_vars
    without scipy.stats.qmc dependency.
    """
    rng = np.random.RandomState(seed)
    result = np.empty((n_samples, n_vars), dtype=float)

    for j in range(n_vars):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.rand(n_samples)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points)
        result[:, j] = points

    return result


def sample_uniform_lhs(n_samples, n_vars):
    return lhs_sample(n_samples, n_vars, seed=SEED)


def sample_truncated_normal(n_samples, n_vars, mean=0.5, std=0.15):
    """
    Independent truncated normal samples in [0,1]^n_vars.
    Note: this is not LHS.
    """
    a, b = (0.0 - mean) / std, (1.0 - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=(n_samples, n_vars))


def build_dbmopp_problem(n_var, n_obj, params):
    """
    Build DBMOPP problem with a robust fallback for version differences.
    """
    try:
        return DBMOPP_generator(
            k=n_obj,
            n=n_var,
            nlp=params["n_local_pareto_regions"],
            ndr=params["n_dominance_res_regions"],
            ngp=params["n_global_pareto_regions"],
            prop_constraint_checker=params.get("prop_neutral", 0.0),
            pareto_set_type=params["pareto_set_type"],
            constraint_type=params["constraint_type"],
            ndo=params["ndo"],
            vary_sol_density=params["vary_sol_density"],
            vary_objective_scales=params["vary_objective_scales"],
            prop_neutral=params["prop_neutral"],
            nm=10000,
        )
    except TypeError:
        # fallback for slightly different installed signatures
        return DBMOPP_generator(
            k=n_obj,
            n=n_var,
            nlp=params["n_local_pareto_regions"],
            ndr=params["n_dominance_res_regions"],
            ngp=params["n_global_pareto_regions"],
            prop_constraint_checker=params.get("prop_neutral", 0.0),
            pareto_set_type=params["pareto_set_type"],
            constraint_type=params["constraint_type"],
        )


def extract_objectives(eval_result, n_obj):
    """
    Robustly extract objective array from DBMOPP evaluation.
    """
    # Case 1: already ndarray-like
    if isinstance(eval_result, np.ndarray):
        arr = np.asarray(eval_result, dtype=float)

    # Case 2: tuple/list return, first item might be objectives
    elif isinstance(eval_result, (tuple, list)):
        if len(eval_result) == 0:
            raise ValueError("Evaluation returned empty tuple/list.")
        arr = np.asarray(eval_result[0], dtype=float)

    # Case 3: object with 'objectives'
    elif hasattr(eval_result, "objectives"):
        arr = np.asarray(eval_result.objectives, dtype=float)

    else:
        raise TypeError(f"Unsupported evaluation return type: {type(eval_result)}")

    if arr.ndim == 1:
        if arr.size == n_obj:
            arr = arr.reshape(1, n_obj)
        else:
            raise ValueError(f"1D objective output has size {arr.size}, expected {n_obj}.")

    if arr.ndim != 2:
        raise ValueError(f"Objective output must be 2D, got shape {arr.shape}.")

    if arr.shape[1] != n_obj:
        raise ValueError(f"Objective output has {arr.shape[1]} objectives, expected {n_obj}.")

    return arr


def evaluate_dbmopp(problem, sample_data, n_obj):
    """
    Evaluate all sample points robustly.
    Tries vectorized evaluation first, falls back to row-by-row.
    """
    # Try vectorized
    try:
        result = None
        if hasattr(problem, "evaluate_objectives"):
            result = problem.evaluate_objectives(sample_data)
        elif hasattr(problem, "evaluate"):
            result = problem.evaluate(sample_data)
        else:
            raise AttributeError("Problem has neither evaluate_objectives nor evaluate.")

        arr = extract_objectives(result, n_obj)

        # If vectorized worked correctly, row count should match
        if arr.shape[0] == sample_data.shape[0]:
            return arr

    except Exception:
        pass

    # Fallback: evaluate one row at a time
    rows = []
    for sample in sample_data:
        if hasattr(problem, "evaluate_objectives"):
            result = problem.evaluate_objectives(sample.reshape(1, -1))
        elif hasattr(problem, "evaluate"):
            result = problem.evaluate(sample.reshape(1, -1))
        else:
            raise AttributeError("Problem has neither evaluate_objectives nor evaluate.")

        arr = extract_objectives(result, n_obj)
        if arr.shape[0] != 1:
            raise ValueError(f"Single-sample evaluation returned shape {arr.shape}, expected (1, {n_obj}).")
        rows.append(arr[0])

    return np.asarray(rows, dtype=float)


def check_generated_data(problem_name, distribution_tag, sample_data, objective_values, noisy_objectives=None):
    print(f"\n=== SANITY CHECK: {problem_name} / {distribution_tag} ===")
    print(f"X shape: {sample_data.shape}")
    print(f"Y shape: {objective_values.shape}")

    print(f"Decision variable global min: {sample_data.min():.6f}")
    print(f"Decision variable global max: {sample_data.max():.6f}")

    if np.any(sample_data < 0.0):
        print("[WARN] Some decision variables are below 0.")
    if np.any(sample_data > 1.0):
        print("[WARN] Some decision variables exceed 1.")

    print(f"Objective min: {np.min(objective_values):.6f}")
    print(f"Objective max: {np.max(objective_values):.6f}")
    print(f"Objective mean: {np.mean(objective_values):.6f}")

    if not np.all(np.isfinite(objective_values)):
        print("[WARN] Non-finite values found in clean objectives.")

    if noisy_objectives is not None:
        print(f"Noisy objective min: {np.min(noisy_objectives):.6f}")
        print(f"Noisy objective max: {np.max(noisy_objectives):.6f}")
        print(f"Noisy objective mean: {np.mean(noisy_objectives):.6f}")

        if not np.all(np.isfinite(noisy_objectives)):
            print("[WARN] Non-finite values found in noisy objectives.")

        neg_count = int(np.sum(noisy_objectives < 0))
        print(f"Noisy objective negative values: {neg_count}")


def save_data(problem_name, n_var, n_obj, num_samples, dist_tag, variables, objectives, is_noisy=False):
    problem_dir = os.path.join(dbmopp_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    var_names = [f"x{i+1}" for i in range(n_var)]
    obj_names = [f"f{j+1}" for j in range(n_obj)]

    df = pd.DataFrame(np.hstack((variables, objectives)), columns=var_names + obj_names)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{n_var}var_{n_obj}obj_{num_samples}samples_{dist_tag}{suffix}.csv"
    full_path = os.path.join(problem_dir, filename)

    df.to_csv(full_path, index=False)
    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset -> {full_path}")


# =========================================================
# MAIN DATASET GENERATION
# =========================================================
def generate_datasets():
    print(f"Saving DBMOPP datasets under: {dbmopp_data_dir}")
    print(f"Random seed: {SEED}")
    print(f"Noise mean: {noise_mean}")
    print(f"Noise std: {noise_std}")
    print(f"Save noisy data: {SAVE_NOISY_DATA}")
    print(f"Clip noisy objectives to nonnegative: {CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE}")

    for problem_name, params in dbmopp_problems.items():
        for n_var in num_vars_list:
            for n_obj in num_objectives_list:
                try:
                    problem = build_dbmopp_problem(n_var, n_obj, params)
                    print(f"\n[{problem_name}] Initialized ({n_var} vars, {n_obj} objs).")

                except Exception as e:
                    print(f"[ERROR] Failed to initialize {problem_name} ({n_var} vars, {n_obj} objs): {e}")
                    continue

                for num_samples in num_samples_list:
                    for distribution in distribution_types:
                        try:
                            # Sampling
                            if distribution == "uniform":
                                sample_data = sample_uniform_lhs(num_samples, n_var)
                                dist_tag = "uniform"
                            elif distribution == "normal":
                                sample_data = sample_truncated_normal(num_samples, n_var, mean=0.5, std=0.15)
                                dist_tag = "truncnorm"
                            else:
                                raise ValueError(f"Unsupported distribution type: {distribution}")

                            np.clip(sample_data, 0.0, 1.0, out=sample_data)

                            # Evaluation
                            f_values = evaluate_dbmopp(problem, sample_data, n_obj)

                            if f_values.shape[0] != num_samples:
                                raise ValueError(
                                    f"Evaluated row count {f_values.shape[0]} does not match sample count {num_samples}."
                                )

                            # Save clean
                            save_data(
                                problem_name=problem_name,
                                n_var=n_var,
                                n_obj=n_obj,
                                num_samples=num_samples,
                                dist_tag=dist_tag,
                                variables=sample_data,
                                objectives=f_values,
                                is_noisy=False
                            )

                            noisy_f = None

                            # Save noisy
                            if SAVE_NOISY_DATA:
                                noisy_f = f_values + np.random.normal(
                                    loc=noise_mean,
                                    scale=noise_std,
                                    size=f_values.shape
                                )

                                if CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE:
                                    noisy_f = np.clip(noisy_f, 0.0, None)

                                save_data(
                                    problem_name=problem_name,
                                    n_var=n_var,
                                    n_obj=n_obj,
                                    num_samples=num_samples,
                                    dist_tag=dist_tag,
                                    variables=sample_data,
                                    objectives=noisy_f,
                                    is_noisy=True
                                )

                            # Sanity check
                            check_generated_data(
                                problem_name=problem_name,
                                distribution_tag=dist_tag,
                                sample_data=sample_data,
                                objective_values=f_values,
                                noisy_objectives=noisy_f
                            )

                        except Exception as e:
                            print(
                                f"[ERROR] Failed on {problem_name}, "
                                f"n_var={n_var}, n_obj={n_obj}, "
                                f"num_samples={num_samples}, distribution={distribution}: {e}"
                            )
                            continue


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    generate_datasets()
    print("\nAll DBMOPP datasets attempted.")
