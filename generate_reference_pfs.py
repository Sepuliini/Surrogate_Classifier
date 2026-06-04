#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd

from desdeo_problem.testproblems.TestProblems import test_problem_builder
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
from desdeo_problem.testproblems.EngineeringRealWorld import (
    re21, re22, re23, re24, re25, re31, re32, re33,
)
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from optproblems import wfg


# =========================================================
# PROBLEM REGISTRIES
# =========================================================

DTLZ_PROBLEMS = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

WFG_PROBLEMS = {
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

ENGINEERING_PROBLEMS = {
    "re21": re21,
    "re22": re22,
    "re23": re23,
    "re24": re24,
    "re25": re25,
    "re31": re31,
    "re32": re32,
    "re33": re33,
    "multiple_clutch_brakes": multiple_clutch_brakes,
    "river_pollution_problem": river_pollution_problem,
    "vehicle_crashworthiness": vehicle_crashworthiness,
    "car_side_impact": car_side_impact,
}

DBMOPP_PROBLEMS = {
    "DBMOPP1": dict(
        n_local_pareto_regions=2, n_dominance_res_regions=0, n_global_pareto_regions=3,
        pareto_set_type=0, constraint_type=1, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0
    ),
    "DBMOPP2": dict(
        n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=3,
        pareto_set_type=1, constraint_type=3, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0
    ),
    "DBMOPP3": dict(
        n_local_pareto_regions=3, n_dominance_res_regions=2, n_global_pareto_regions=4,
        pareto_set_type=2, constraint_type=5, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0
    ),
    "DBMOPP4": dict(
        n_local_pareto_regions=3, n_dominance_res_regions=4, n_global_pareto_regions=5,
        pareto_set_type=2, constraint_type=8, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.1
    ),
    "DBMOPP5": dict(
        n_local_pareto_regions=1, n_dominance_res_regions=0, n_global_pareto_regions=3,
        pareto_set_type=0, constraint_type=1, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0
    ),
    "DBMOPP6": dict(
        n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=4,
        pareto_set_type=1, constraint_type=4, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.2
    ),
    "DBMOPP7": dict(
        n_local_pareto_regions=2, n_dominance_res_regions=2, n_global_pareto_regions=5,
        pareto_set_type=2, constraint_type=7, ndo=0,
        vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0
    ),
}

DTLZ_NVARS = {3: 12, 4: 14, 5: 16}
WFG_NVARS = {3: 10, 4: 12, 5: 14}
DBMOPP_NVARS = {3: 10, 4: 10, 5: 10, 6: 10, 7: 10}


# =========================================================
# HELPERS
# =========================================================

def lhs_sample(n_samples, n_vars, seed=None):
    rng = np.random.RandomState(seed)
    result = np.empty((n_samples, n_vars), dtype=float)
    for j in range(n_vars):
        cut = np.linspace(0.0, 1.0, n_samples + 1)
        u = rng.rand(n_samples)
        points = cut[:-1] + u * (cut[1:] - cut[:-1])
        rng.shuffle(points)
        result[:, j] = points
    return result


def nondominated_filter(F):
    """
    Simple O(N^2) nondominated filter.
    Fine for moderate sizes; for very large runs you may later want pymoo/pygmo.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2:
        raise ValueError(f"Expected 2D objective array, got shape {F.shape}")

    n = F.shape[0]
    keep = np.ones(n, dtype=bool)

    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                keep[i] = False
                break

    return F[keep]


def save_pf(points, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, filename)
    df = pd.DataFrame(points, columns=[f"f{i+1}" for i in range(points.shape[1])])
    df.to_csv(full_path, index=False)
    print(f"Saved PF -> {full_path} | shape={points.shape}")


def sanity_report(name, F):
    print(f"\n=== {name} ===")
    print(f"PF shape: {F.shape}")
    print(f"Min: {np.min(F):.6f}")
    print(f"Max: {np.max(F):.6f}")
    print(f"Mean: {np.mean(F):.6f}")
    if not np.all(np.isfinite(F)):
        print("[WARN] Non-finite values found.")


def read_task_from_file(task_file, task_id):
    """
    task_id is assumed 1-based, matching SLURM_ARRAY_TASK_ID if array starts at 1.
    File format:
      FAMILY PROBLEM NOBJ
    Example:
      DBMOPP DBMOPP1 3
    """
    with open(task_file, "r", encoding="utf-8") as f:
        lines = [
            line.strip() for line in f
            if line.strip() and not line.strip().startswith("#")
        ]

    if task_id < 1 or task_id > len(lines):
        raise IndexError(f"task_id={task_id} out of range 1..{len(lines)}")

    parts = lines[task_id - 1].split()
    if len(parts) < 2:
        raise ValueError(f"Invalid task line: {lines[task_id - 1]!r}")

    family = parts[0]
    problem = parts[1]
    n_obj = int(parts[2]) if len(parts) >= 3 else None

    return family, problem, n_obj


# =========================================================
# DTLZ
# =========================================================

def generate_dtlz_reference(problem_name, n_obj, n_samples, seed, base_out_dir):
    if problem_name not in DTLZ_PROBLEMS:
        raise ValueError(f"Unknown DTLZ problem: {problem_name}")
    if n_obj not in DTLZ_NVARS:
        raise ValueError(f"Unsupported DTLZ objective count: {n_obj}")

    n_var = DTLZ_NVARS[n_obj]
    problem = test_problem_builder(
        problem_name,
        n_of_objectives=n_obj,
        n_of_variables=n_var,
    )

    X = lhs_sample(n_samples, n_var, seed=seed)
    rows = []
    for x in X:
        res = problem.evaluate(x)
        obj = np.asarray(res.objectives).reshape(-1)
        rows.append(obj)

    F = np.asarray(rows, dtype=float)
    F_nd = nondominated_filter(F)

    out_dir = os.path.join(base_out_dir, "DTLZ", problem_name)
    filename = f"{problem_name}_{n_obj}obj_reference_pf.csv"
    save_pf(F_nd, out_dir, filename)
    sanity_report(f"{problem_name} {n_obj}obj", F_nd)


# =========================================================
# WFG
# =========================================================

def get_valid_wfg_nvar(requested_nvar, n_obj):
    k = 2 * (n_obj - 1)
    n_var = requested_nvar
    while not (n_var > k and (n_var - k) % 2 == 0):
        n_var += 1
    return n_var, k


def scale_to_wfg_domain(X_unit):
    n_var = X_unit.shape[1]
    upper_bounds = np.array([2.0 * (i + 1) for i in range(n_var)], dtype=float)
    return X_unit * upper_bounds


def generate_wfg_reference(problem_name, n_obj, n_samples, seed, base_out_dir):
    if problem_name not in WFG_PROBLEMS:
        raise ValueError(f"Unknown WFG problem: {problem_name}")
    if n_obj not in WFG_NVARS:
        raise ValueError(f"Unsupported WFG objective count: {n_obj}")

    problem_class = WFG_PROBLEMS[problem_name]
    requested_nvar = WFG_NVARS[n_obj]
    n_var, k = get_valid_wfg_nvar(requested_nvar, n_obj)

    problem = problem_class(num_objectives=n_obj, num_variables=n_var, k=k)

    X_unit = lhs_sample(n_samples, n_var, seed=seed)
    X = scale_to_wfg_domain(X_unit)

    F = np.asarray([problem(x) for x in X], dtype=float)
    F_nd = nondominated_filter(F)

    out_dir = os.path.join(base_out_dir, "WFG", problem_name)
    filename = f"{problem_name}_{n_obj}obj_reference_pf.csv"
    save_pf(F_nd, out_dir, filename)
    sanity_report(f"{problem_name} {n_obj}obj", F_nd)


# =========================================================
# ENGINEERING
# =========================================================

def sample_engineering_inputs(problem, n_samples, seed):
    rng = np.random.RandomState(seed)
    n_var = len(problem.variables)
    X = np.empty((n_samples, n_var), dtype=float)

    for i, var in enumerate(problem.variables):
        lb, ub = var.get_bounds()
        X[:, i] = rng.uniform(lb, ub, n_samples)

    return X


def generate_engineering_reference(problem_name, n_samples, seed, base_out_dir):
    if problem_name not in ENGINEERING_PROBLEMS:
        raise ValueError(f"Unknown engineering problem: {problem_name}")

    problem = ENGINEERING_PROBLEMS[problem_name]()
    n_obj = len(problem.objectives)

    X = sample_engineering_inputs(problem, n_samples, seed)

    rows = []
    for x in X:
        res = problem.evaluate(x)
        obj = np.asarray(res.objectives).reshape(-1)
        rows.append(obj)

    F = np.asarray(rows, dtype=float)
    F_nd = nondominated_filter(F)

    out_dir = os.path.join(base_out_dir, "engineering", problem_name)
    filename = f"{problem_name}_{n_obj}obj_reference_pf.csv"
    save_pf(F_nd, out_dir, filename)
    sanity_report(problem_name, F_nd)


# =========================================================
# DBMOPP
# =========================================================

def build_dbmopp_problem(n_var, n_obj, params):
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


def extract_dbmopp_objectives(eval_result, n_obj):
    if isinstance(eval_result, np.ndarray):
        arr = np.asarray(eval_result, dtype=float)
    elif isinstance(eval_result, (tuple, list)):
        arr = np.asarray(eval_result[0], dtype=float)
    elif hasattr(eval_result, "objectives"):
        arr = np.asarray(eval_result.objectives, dtype=float)
    else:
        raise TypeError(f"Unsupported DBMOPP return type: {type(eval_result)}")

    if arr.ndim == 1:
        if arr.size == n_obj:
            arr = arr.reshape(1, n_obj)
        else:
            raise ValueError(f"Unexpected DBMOPP output shape: {arr.shape}")

    return arr


def evaluate_dbmopp(problem, X, n_obj):
    try:
        if hasattr(problem, "evaluate_objectives"):
            result = problem.evaluate_objectives(X)
        elif hasattr(problem, "evaluate"):
            result = problem.evaluate(X)
        else:
            raise AttributeError("No evaluation method found.")

        arr = extract_dbmopp_objectives(result, n_obj)
        if arr.shape[0] == X.shape[0]:
            return arr
    except Exception:
        pass

    rows = []
    for x in X:
        if hasattr(problem, "evaluate_objectives"):
            result = problem.evaluate_objectives(x.reshape(1, -1))
        elif hasattr(problem, "evaluate"):
            result = problem.evaluate(x.reshape(1, -1))
        else:
            raise AttributeError("No evaluation method found.")

        arr = extract_dbmopp_objectives(result, n_obj)
        rows.append(arr.reshape(-1))

    return np.asarray(rows, dtype=float)


def generate_dbmopp_reference(problem_name, n_obj, n_samples, seed, base_out_dir):
    if problem_name not in DBMOPP_PROBLEMS:
        raise ValueError(f"Unknown DBMOPP problem: {problem_name}")
    if n_obj not in DBMOPP_NVARS:
        raise ValueError(f"Unsupported DBMOPP objective count: {n_obj}")

    params = DBMOPP_PROBLEMS[problem_name]
    n_var = DBMOPP_NVARS[n_obj]
    problem = build_dbmopp_problem(n_var, n_obj, params)

    X = lhs_sample(n_samples, n_var, seed=seed)
    F = evaluate_dbmopp(problem, X, n_obj)
    F_nd = nondominated_filter(F)

    out_dir = os.path.join(base_out_dir, "DBMOPP", problem_name)
    filename = f"{problem_name}_{n_obj}obj_reference_pf.csv"
    save_pf(F_nd, out_dir, filename)
    sanity_report(f"{problem_name} {n_obj}obj", F_nd)


# =========================================================
# DISPATCH
# =========================================================

def run_one_task(family, problem_name, n_obj, n_samples, seed, base_out_dir):
    family_upper = family.upper()

    print("--------------------------------------------------")
    print(f"Family     : {family_upper}")
    print(f"Problem    : {problem_name}")
    print(f"Objectives : {n_obj}")
    print(f"Samples    : {n_samples}")
    print(f"Seed       : {seed}")
    print(f"Output dir : {base_out_dir}")
    print("--------------------------------------------------")

    if family_upper == "DTLZ":
        if n_obj is None:
            raise ValueError("DTLZ task requires n_obj")
        generate_dtlz_reference(problem_name, n_obj, n_samples, seed, base_out_dir)

    elif family_upper == "WFG":
        if n_obj is None:
            raise ValueError("WFG task requires n_obj")
        generate_wfg_reference(problem_name, n_obj, n_samples, seed, base_out_dir)

    elif family_upper == "ENGINEERING":
        generate_engineering_reference(problem_name, n_samples, seed, base_out_dir)

    elif family_upper == "DBMOPP":
        if n_obj is None:
            raise ValueError("DBMOPP task requires n_obj")
        generate_dbmopp_reference(problem_name, n_obj, n_samples, seed, base_out_dir)

    else:
        raise ValueError(f"Unknown family: {family}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate reference Pareto front for one task."
    )

    parser.add_argument("--family", type=str, help="DTLZ / WFG / ENGINEERING / DBMOPP")
    parser.add_argument("--problem", type=str, help="Problem name, e.g. DTLZ1, WFG4, DBMOPP2")
    parser.add_argument("--n-obj", type=int, default=None, help="Number of objectives")

    parser.add_argument("--task-file", type=str, default=None,
                        help="Path to task map file")
    parser.add_argument("--task-id", type=int, default=None,
                        help="1-based task index in task-file")

    parser.add_argument("--n-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-out-dir", type=str,
                        default="/scratch/project_2017216/modelling_results/real_paretofronts")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task_file is not None:
        if args.task_id is None:
            raise ValueError("--task-id is required when --task-file is used")
        family, problem_name, n_obj = read_task_from_file(args.task_file, args.task_id)
    else:
        if args.family is None or args.problem is None:
            raise ValueError("Either use --task-file/--task-id or provide --family and --problem")
        family = args.family
        problem_name = args.problem
        n_obj = args.n_obj

    run_one_task(
        family=family,
        problem_name=problem_name,
        n_obj=n_obj,
        n_samples=args.n_samples,
        seed=args.seed,
        base_out_dir=args.base_out_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
