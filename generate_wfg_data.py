#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from optproblems import wfg
from scipy.stats import truncnorm

# =========================================================
# CONFIG
# =========================================================
SEED = 42
np.random.seed(SEED)

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

noise_mean = 0.0
noise_std = 0.1

SAVE_NOISY_DATA = True
CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE = False

base_data_dir = "/scratch/project_2017216/Data"
wfg_data_dir = os.path.join(base_data_dir, "WFG")
os.makedirs(wfg_data_dir, exist_ok=True)

# =========================================================
# HELPERS
# =========================================================
def lhs_sample(n_samples, n_vars, seed=None):
    """
    Simple Latin hypercube sampling in [0,1]^n_vars
    without pyDOE or scipy.stats.qmc.
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


def scale_to_wfg_domain(samples_unit):
    """
    Scale samples from [0,1] to standard WFG domain:
    x_i in [0, 2*(i+1)] for i=0,...,n_var-1
    """
    n_var = samples_unit.shape[1]
    upper_bounds = np.array([2.0 * (i + 1) for i in range(n_var)], dtype=float)
    return samples_unit * upper_bounds


def get_valid_wfg_nvar(original_nvars, num_obj):
    """
    For WFG, k = 2*(M-1), and n_var must satisfy:
      n_var > k
      (n_var - k) % 2 == 0
    """
    k = 2 * (num_obj - 1)
    n_var = original_nvars

    while not (n_var > k and (n_var - k) % 2 == 0):
        n_var += 1

    l = n_var - k
    return n_var, k, l


def check_generated_data(problem_name, distribution_tag, sample_data, objective_values, noisy_objectives=None):
    print(f"\n=== SANITY CHECK: {problem_name} / {distribution_tag} ===")
    print(f"X shape: {sample_data.shape}")
    print(f"Y shape: {objective_values.shape}")

    upper_bounds = np.array([2.0 * (i + 1) for i in range(sample_data.shape[1])], dtype=float)
    x_min = sample_data.min(axis=0)
    x_max = sample_data.max(axis=0)

    print(f"Decision variable global min: {sample_data.min():.6f}")
    print(f"Decision variable global max: {sample_data.max():.6f}")

    if np.any(x_min < 0):
        print("[WARN] Some decision variables are below 0.")
    if np.any(x_max > upper_bounds + 1e-12):
        print("[WARN] Some decision variables exceed WFG upper bounds.")

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


def save_data(problem_name, n_var, num_obj, num_samples, dist_tag, variables, objectives, is_noisy=False):
    problem_dir = os.path.join(wfg_data_dir, problem_name)
    os.makedirs(problem_dir, exist_ok=True)

    var_names = [f"x{i+1}" for i in range(variables.shape[1])]
    obj_names = [f"f{j+1}" for j in range(objectives.shape[1])]
    columns = var_names + obj_names

    df = pd.DataFrame(np.hstack((variables, objectives)), columns=columns)

    suffix = "_noise" if is_noisy else ""
    filename = f"{problem_name}_{n_var}var_{num_obj}obj_{num_samples}samples_{dist_tag}{suffix}.csv"
    full_path = os.path.join(problem_dir, filename)

    df.to_csv(full_path, index=False)
    print(f"Saved {'noisy' if is_noisy else 'clean'} dataset to {full_path}")


def generate_datasets(problem_name, original_nvars, num_obj, num_samples, distribution):
    problem_class = problems[problem_name]

    n_var, k, l = get_valid_wfg_nvar(original_nvars, num_obj)

    print(
        f"[{problem_name}] requested_n_var={original_nvars}, "
        f"adjusted_n_var={n_var}, k={k}, l={l}, num_obj={num_obj}, "
        f"samples={num_samples}, distribution={distribution}"
    )

    # Create WFG problem instance
    objective = problem_class(num_objectives=num_obj, num_variables=n_var, k=k)

    # Sample in [0,1]^n_var first
    if distribution == "uniform":
        samples_unit = sample_uniform_lhs(num_samples, n_var)
        dist_tag = "uniform"

    elif distribution == "normal":
        samples_unit = sample_truncated_normal(num_samples, n_var, mean=0.5, std=0.15)
        dist_tag = "truncnorm"

    else:
        raise ValueError(f"Unsupported distribution type: {distribution}")

    # safety clip to [0,1]
    np.clip(samples_unit, 0.0, 1.0, out=samples_unit)

    # scale to WFG domain
    sample_data = scale_to_wfg_domain(samples_unit)

    # Evaluate objectives
    objective_values = np.array([objective(sample) for sample in sample_data], dtype=float)

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

    # Save clean data
    save_data(
        problem_name=problem_name,
        n_var=n_var,
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
            n_var=n_var,
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
    print(f"Saving WFG datasets under: {wfg_data_dir}")
    print(f"Random seed: {SEED}")
    print(f"Noise mean: {noise_mean}")
    print(f"Noise std: {noise_std}")
    print(f"Save noisy data: {SAVE_NOISY_DATA}")
    print(f"Clip noisy objectives to nonnegative: {CLIP_NOISY_OBJECTIVES_TO_NONNEGATIVE}")

    for problem_name in problems:
        for original_nvars in num_vars_list:
            for num_obj in num_obj_list:
                for num_samples in num_samples_list:
                    for distribution in distribution_types:
                        try:
                            generate_datasets(
                                problem_name=problem_name,
                                original_nvars=original_nvars,
                                num_obj=num_obj,
                                num_samples=num_samples,
                                distribution=distribution
                            )
                        except Exception as e:
                            print(
                                f"[ERROR] Failed on "
                                f"{problem_name}, requested_n_var={original_nvars}, "
                                f"num_obj={num_obj}, num_samples={num_samples}, "
                                f"distribution={distribution}: {e}"
                            )


if __name__ == "__main__":
    main()
