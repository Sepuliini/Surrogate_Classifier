#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from desdeo_problem.testproblems.TestProblems import test_problem_builder

# =========================================================
# CONFIG
# =========================================================
SEED = 42
np.random.seed(SEED)

problems = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]
num_vars_list = [6, 10, 20, 30]
num_obj_list = [3, 4, 5]
num_samples_list = [100, 500, 1000, 2000]
distribution_types = ["uniform", "normal"]  # "normal" = truncated normal in [0,1]

noise_mean = 0.0
noise_std = 0.1

SAVE_NOISY_DATA = True
CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE = False

base_dir = "/scratch/project_2017216/Data"
dtlz_root_dir = os.path.join(base_dir, "DTLZ")
os.makedirs(dtlz_root_dir, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def lhs_sample(n_samples, n_vars, seed=None):
    """
    Simple Latin hypercube sampling in [0,1]^n_vars.
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


def evaluate_dtlz_problem(problem, sample_data):
    """
    Evaluate all samples for a DESDEO DTLZ problem.
    """
    objective_values_list = []

    for sample in sample_data:
        evaluated = problem.evaluate(sample)

        if not hasattr(evaluated, "objectives"):
            raise TypeError("Evaluation result does not contain 'objectives'.")

        obj = np.asarray(evaluated.objectives)

        # Common case: shape (1, n_obj)
        if obj.ndim == 2 and obj.shape[0] == 1:
            objective_values_list.append(obj[0])

        # Fallback: already 1D
        elif obj.ndim == 1:
            objective_values_list.append(obj)

        else:
            raise ValueError(f"Unexpected objective shape: {obj.shape}")

    return np.asarray(objective_values_list, dtype=float)


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


def save_data(problem_name, num_vars, num_obj, num_samples, dist_tag, variables, objectives, is_noisy=False):
    problem_dir = os.path.join(dtlz_root_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    var_names = [f"x{i}" for i in range(1, variables.shape[1] + 1)]
    obj_names = [f"f{j}" for j in range(1, objectives.shape[1] + 1)]
    columns = var_names + obj_names

    df = pd.DataFrame(np.hstack((variables, objectives)), columns=columns)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{num_vars}var_{num_obj}obj_{num_samples}samples_{dist_tag}{suffix}.csv"
    full_path = os.path.join(problem_dir, filename)

    df.to_csv(full_path, index=False)
    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset to {full_path}")


def generate_datasets(problem_name, num_vars, num_obj, num_samples, distribution):
    dtlz_problem = test_problem_builder(
        problem_name,
        n_of_objectives=num_obj,
        n_of_variables=num_vars
    )

    if distribution == "uniform":
        sample_data = sample_uniform_lhs(num_samples, num_vars)
        dist_tag = "uniform"

    elif distribution == "normal":
        sample_data = sample_truncated_normal(num_samples, num_vars, mean=0.5, std=0.15)
        dist_tag = "truncnorm"

    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    np.clip(sample_data, 0.0, 1.0, out=sample_data)

    objective_values = evaluate_dtlz_problem(dtlz_problem, sample_data)

    if objective_values.ndim != 2:
        raise ValueError(
            f"Objective evaluation returned shape {objective_values.shape}, "
            f"expected (n_samples, num_obj)"
        )

    if objective_values.shape[1] != num_obj:
        raise ValueError(
            f"Objective evaluation returned {objective_values.shape[1]} objectives, "
            f"expected {num_obj}"
        )

    save_data(
        problem_name=problem_name,
        num_vars=num_vars,
        num_obj=num_obj,
        num_samples=num_samples,
        dist_tag=dist_tag,
        variables=sample_data,
        objectives=objective_values,
        is_noisy=False
    )

    noisy_objectives = None

    if SAVE_NOISY_DATA:
        noisy_objectives = objective_values + np.random.normal(
            loc=noise_mean,
            scale=noise_std,
            size=objective_values.shape
        )

        if CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE:
            noisy_objectives = np.clip(noisy_objectives, 0.0, None)

        save_data(
            problem_name=problem_name,
            num_vars=num_vars,
            num_obj=num_obj,
            num_samples=num_samples,
            dist_tag=dist_tag,
            variables=sample_data,
            objectives=noisy_objectives,
            is_noisy=True
        )

    check_generated_data(
        problem_name=problem_name,
        distribution_tag=dist_tag,
        sample_data=sample_data,
        objective_values=objective_values,
        noisy_objectives=noisy_objectives
    )


# =========================================================
# MAIN LOOP
# =========================================================
def main():
    print(f"Saving DTLZ datasets under: {dtlz_root_dir}")
    print(f"Random seed: {SEED}")
    print(f"Noise mean: {noise_mean}")
    print(f"Noise std: {noise_std}")
    print(f"Save noisy data: {SAVE_NOISY_DATA}")
    print(f"Clip noisy objectives to nonnegative: {CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE}")

    for problem_name in problems:
        for num_vars in num_vars_list:
            for num_obj in num_obj_list:
                for num_samples in num_samples_list:
                    for distribution in distribution_types:
                        try:
                            generate_datasets(
                                problem_name=problem_name,
                                num_vars=num_vars,
                                num_obj=num_obj,
                                num_samples=num_samples,
                                distribution=distribution
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] Failed on "
                                f"{problem_name}, num_vars={num_vars}, "
                                f"num_obj={num_obj}, num_samples={num_samples}, "
                                f"distribution={distribution}: {e}"
                            )


if __name__ == "__main__":
    main()
