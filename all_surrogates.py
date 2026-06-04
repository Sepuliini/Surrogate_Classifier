#!/usr/bin/env python3
"""
NSGA-III / RVEA / IBEA surrogate evaluator with stabilized metrics and RunTag versioning.
Supports DTLZ, WFG, Engineering, and DBMOPP problems.

"""
import logging

import os, re, argparse, logging, warnings, traceback
from os import path, makedirs, listdir
from datetime import datetime
import numpy as np

if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "bool"):
    np.bool = bool
    
import pandas as pd
import matplotlib.pyplot as plt

from pymoo.indicators.hv import HV

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from pymoo.indicators.hv import HV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import qmc

from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_emo.EAs import NSGAIII, RVEA, IBEA
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_emo.population import Population
from optproblems import wfg

warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# ---------------------------
# Argument parsing
# ---------------------------

parser = argparse.ArgumentParser(description="Run surrogate evaluation for selected problem suite.")
parser.add_argument(
    "--suite",
    type=str,
    default="ALL",
    choices=["ALL", "DTLZ", "WFG", "Engineering", "DBMOPP"],
    help="Select which suite to process (default: ALL)."
)
parser.add_argument(
    "--problem",
    type=str,
    default=None,
    help="Optional: run a specific problem (default: all problems in the suite)."
)
parser.add_argument(
    "--n_obj",
    type=int,
    default=None,
    help="Optional: process only datasets with this number of objectives."
)

args = parser.parse_args()

# ================== EXPERIMENT KNOBS ==================
BASE_SEED   = 2025
POP_SIZE    = 150
N_GEN       = 50
EPS_CLEAN       = 1e-3
PF_RANGE_FLOOR  = 1e-9
RUN_TAG = f"EA_pop{POP_SIZE}_gen{N_GEN}_seed{BASE_SEED}_eps{EPS_CLEAN}"
# ======================================================

# ----- Paths & Logging -----
base_folder = '/scratch/project_2017216'
out_dir = path.join(base_folder, 'modelling_results'); makedirs(out_dir, exist_ok=True)
log_dir = path.join(base_folder, "logs"); makedirs(log_dir, exist_ok=True)
task_id = os.getenv("SLURM_ARRAY_TASK_ID", str(os.getpid()))
log_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_task{task_id}.log"
logging.basicConfig(
    filename=path.join(log_dir, log_file),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler(); console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logging.getLogger().addHandler(console)
logging.captureWarnings(True)
logging.info("========== SCRIPT START ==========")
logging.info(f"RUN_TAG={RUN_TAG}  SELECTED_SUITE={args.suite}")

# ----- Output files -----
pf_fp = path.join(out_dir, 'surrogate_metrics.csv')
perf_fp = path.join(out_dir, 'surrogate_perf.csv')

pf_cols = [
    "problem", "n_var", "n_obj", "n_samples", "dataset_file",
    "ea", "algo", "noise", "iteration",
    "IGD", "HV", "EpsAdd", "EpsMulti", "RUN_TAG"
]

perf_cols = [
    "problem", "n_var", "n_obj", "dataset_file", "algo", "noise",
    "objective", "mean_r2", "std_r2", "mean_mse", "n_samples", "n_splits"
]

def load_existing(fp, cols):
    if path.exists(fp):
        df = pd.read_csv(fp)
        for c in cols:
            if c not in df.columns: df[c] = np.nan
        return df[cols].drop_duplicates()
    return pd.DataFrame(columns=cols)

def plot_and_save_pf(
    approx_pf,
    ref_pf,
    problem_name,
    ea_name,
    surrogate,
    n_vars,
    n_objs,
    output_dir="/scratch/project_2017216/modelling_results/approx_pfs_images"
):
    """Unified and self-contained PF plotting function with corrected normalization + auto-hide reference PF."""
    
    # -----------------------------
    # Validate PF input
    # -----------------------------
    if approx_pf is None or approx_pf.size == 0:
        print(f"[WARN] Empty PF for {problem_name}, {ea_name}, skipping plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # -----------------------------
    # Safe filename cleaner
    # -----------------------------
    def clean(s):
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

    # -----------------------------
    # Build unique filename parts
    # -----------------------------
    parts = [
        clean(problem_name),
        clean(surrogate),
        clean(ea_name),
        f"{n_vars}vars",
        f"{n_objs}objs"
    ]
    base_name = "_".join(parts)

    # -----------------------------
    # Union-based normalization
    # -----------------------------
    if ref_pf is not None and ref_pf.size > 0:
        combined = np.vstack([approx_pf, ref_pf])
    else:
        combined = approx_pf

    min_vals = np.min(combined, axis=0)
    max_vals = np.max(combined, axis=0)
    eps = 1e-12

    norm_pf = (approx_pf - min_vals) / (max_vals - min_vals + eps)
    
    norm_ref = None
    if ref_pf is not None and ref_pf.size > 0:
        norm_ref = (ref_pf - min_vals) / (max_vals - min_vals + eps)

    # -----------------------------
    # Auto-hide reference PF if collapsed
    # -----------------------------
    hide_ref = False
    if norm_ref is not None:
        ref_range = norm_ref.max(axis=0) - norm_ref.min(axis=0)
        max_range = ref_range.max()

        if max_range < 0.03:
            hide_ref = True
            logging.info(
                f"[PLOT] Reference PF collapsed after normalization (max range={max_range:.5f}). "
                f"Hiding reference PF."
            )

    # -----------------------------
    # Debug logging
    # -----------------------------
    logging.info(f"[DEBUG-PLOT] min vals: {min_vals}")
    logging.info(f"[DEBUG-PLOT] max vals: {max_vals}")
    logging.info(f"[DEBUG-PLOT] ranges: {max_vals - min_vals}")
    logging.info(f"[DEBUG-PLOT] approx_pf unique rows: {np.unique(approx_pf.round(6), axis=0).shape}")

    if norm_ref is not None and not hide_ref:
        logging.info(f"[DEBUG-PLOT] ref_pf unique rows: {np.unique(ref_pf.round(6), axis=0).shape}")

    logging.info(f"[DEBUG-PLOT] norm_pf unique rows: {np.unique(norm_pf.round(6), axis=0).shape}")

    num_obj = norm_pf.shape[1]

    # -----------------------------
    # Title
    # -----------------------------
    title_main = f"{problem_name} – {surrogate} – {ea_name}"
    full_title = f"{title_main} (Approx vs Reference)"

    # -----------------------------
    # 2D plot
    # -----------------------------
    if num_obj == 2:
        plt.figure(figsize=(8,6))
        plt.scatter(norm_pf[:,0], norm_pf[:,1], c='deepskyblue', s=18, label='Approx PF')

        if norm_ref is not None and not hide_ref:
            plt.scatter(norm_ref[:,0], norm_ref[:,1], c='orange', s=20, marker='x', label='Reference PF')

        all_x = np.concatenate([norm_pf[:,0], norm_ref[:,0]]) if norm_ref is not None and not hide_ref else norm_pf[:,0]
        all_y = np.concatenate([norm_pf[:,1], norm_ref[:,1]]) if norm_ref is not None and not hide_ref else norm_pf[:,1]

        plt.xlim(all_x.min(), all_x.max())
        plt.ylim(all_y.min(), all_y.max())

        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title(full_title)
        plt.legend()
        fname = os.path.join(output_dir, f"{base_name}_2D.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        return

    # -----------------------------
    # 3D plot
    # -----------------------------
    if num_obj == 3:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(norm_pf[:,0], norm_pf[:,1], norm_pf[:,2],
                   c='deepskyblue', s=16, label='Approx PF')

        if norm_ref is not None and not hide_ref:
            ax.scatter(norm_ref[:,0], norm_ref[:,1], norm_ref[:,2],
                       c='orange', s=24, marker='x', label='Reference PF')

        all_x = np.concatenate([norm_pf[:,0], norm_ref[:,0]]) if norm_ref is not None and not hide_ref else norm_pf[:,0]
        all_y = np.concatenate([norm_pf[:,1], norm_ref[:,1]]) if norm_ref is not None and not hide_ref else norm_pf[:,1]
        all_z = np.concatenate([norm_pf[:,2], norm_ref[:,2]]) if norm_ref is not None and not hide_ref else norm_pf[:,2]

        ax.set_xlim(all_x.min(), all_x.max())
        ax.set_ylim(all_y.min(), all_y.max())
        ax.set_zlim(all_z.min(), all_z.max())

        ax.set_xlabel('f1')
        ax.set_ylabel('f2')
        ax.set_zlabel('f3')
        ax.set_title(full_title)
        ax.legend()

        fname = os.path.join(output_dir, f"{base_name}_3D.png")
        plt.savefig(fname, dpi=300)
        plt.close()
        return

    # -----------------------------
    # Parallel coordinates (n_obj > 3)
    # -----------------------------
    df_pf = pd.DataFrame(norm_pf, columns=[f'f{i+1}' for i in range(num_obj)])
    df_pf['type'] = 'Approx PF'

    if norm_ref is not None and not hide_ref:
        df_ref = pd.DataFrame(norm_ref, columns=[f'f{i+1}' for i in range(num_obj)])
        df_ref['type'] = 'Reference PF'
        df_combined = pd.concat([df_pf, df_ref], ignore_index=True)
    else:
        df_combined = df_pf

    plt.figure(figsize=(10,6))
    pd.plotting.parallel_coordinates(
        df_combined, 'type',
        color=['deepskyblue', 'orange'] if not hide_ref else ['deepskyblue'],
        alpha=0.6
    )
    plt.title(full_title)
    plt.ylabel("Normalized objective value")
    plt.xlabel("Objectives")
    fname = os.path.join(output_dir, f"{base_name}_parallel.png")
    plt.savefig(fname, dpi=300)
    plt.close()

def get_surrogate_algorithm_name(ml_list):
    """Extract the surrogate algorithm name from ml_list."""
    if ml_list is None:
        return None
    
    for model in ml_list:
        if model is None:
            continue
        try:
            # inside TransformedTargetRegressor → Pipeline → ('regressor', algo_instance)
            algo = model.regressor.named_steps['regressor']
            return algo.__class__.__name__
        except:
            continue
    return None

# --- Distance ---
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# --- Hypervolume ---
def hypervolume(A, ref_point=None):
    """
    Compute hypervolume (minimization assumed).
    A: ndarray (n_points, n_objectives)
    ref_point: ndarray of same length as objectives. If None, set to max(A, axis=0) * 1.01
    Returns float or np.nan on empty input.
    """
    if A is None or A.size == 0:
        return np.nan

    A = np.atleast_2d(A)
    A = A[np.isfinite(A).all(axis=1)]
    if A.size == 0:
        return np.nan

    if ref_point is None:
        ref_point = np.max(A, axis=0) * 1.01
    else:
        ref_point = np.asarray(ref_point, dtype=float)

    try:
        hv_calc = HV(ref_point=ref_point)
        hv = hv_calc.do(A)
        return float(hv) if np.isfinite(hv) else np.nan
    except Exception as e:
        logging.warning(f"HV computation failed: {e}")
        return np.nan

# --- Inverted Generational Distance (IGD) ---
def igd(A, R):
    """
    Average distance from reference front R to closest point in approximation front A.
    """
    if A.size == 0 or R.size == 0:
        return np.nan

    A = np.atleast_2d(A)
    R = np.atleast_2d(R)
    # Compute all pairwise distances (vectorized)
    diffs = R[:, None, :] - A[None, :, :]      # shape (|R|, |A|, n_obj)
    dists = np.linalg.norm(diffs, axis=2)      # shape (|R|, |A|)
    min_dists = np.min(dists, axis=1)          # closest distance for each reference point
    return float(np.mean(min_dists))

# --- Epsilon cleaning ---
def epsilon_cleanup(nd: np.ndarray, eps: float) -> np.ndarray:
    if not np.isfinite(eps) or eps <= 0 or nd.size == 0:
        return nd
    grid = np.floor(nd / eps + 0.5).astype(np.int64)
    _, uniq_idx = np.unique(grid, axis=0, return_index=True)
    return nd[np.sort(uniq_idx)]

# --- Additive epsilon ---
def eps_additive(A: np.ndarray, R: np.ndarray) -> float:
    """
    Additive epsilon indicator: minimum value to add to A to weakly dominate R
    """
    if A.size == 0 or R.size == 0:
        return np.nan
    A = np.atleast_2d(A)
    R = np.atleast_2d(R)
    # Vectorized: for each r in R, compute max(A - r, axis=1) -> min over A
    eps_r = np.min(np.max(A[None, :, :] - R[:, None, :], axis=2), axis=1)
    return float(np.max(eps_r))

# --- Multiplicative epsilon ---
def eps_multiplicative(A: np.ndarray, R: np.ndarray, tiny=1e-8) -> float:
    """
    Multiplicative epsilon indicator: min factor to multiply A to weakly dominate R.
    Assumes A, R >= 0
    """
    if A.size == 0 or R.size == 0:
        return np.nan

    A = np.atleast_2d(A)
    R = np.atleast_2d(R)
    # Avoid division by zero
    R_safe = np.clip(R, tiny, None)
    eps_r = np.min(np.max(A[None, :, :] / R_safe[:, None, :], axis=2), axis=1)
    return float(np.max(eps_r))


# ----- Engineering problem registry -----
ENGINEERING_PROBLEMS = {
    "re21": re21,
    "re22": re22,
    "re23": re23,
    "re24": re24,
    "re25": re25,
    "re31": re31,
    "re32": re32,
    "re33": re33,
    "car_side_impact": car_side_impact,
    "river_pollution_problem": river_pollution_problem,
    "vehicle_crashworthiness": vehicle_crashworthiness,
    "multiple_clutch_brakes": multiple_clutch_brakes,
}

# ----- Full engineering instance map for maximize flags / constraints -----
ED_MAP = {
    "re21": re21,
    "re22": re22,
    "re23": re23,
    "re24": re24,
    "re25": re25,
    "re31": re31,
    "re32": re32,
    "re33": re33,
    "car_side_impact": car_side_impact,
    "river_pollution_problem": river_pollution_problem,
    "vehicle_crashworthiness": vehicle_crashworthiness,
    "multiple_clutch_brakes": multiple_clutch_brakes,
}



variable_ranges = {
    "multiple_clutch_brakes": [(55, 80), (75, 110), (1.5, 3), (300, 1000), (2, 10)],
    "car_side_impact": [(0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5), (0.875, 2.625), (0.4, 1.2), (0.4, 1.2)],
    "river_pollution_problem": [(0.3, 1.0), (0.3, 1.0)],
    "vehicle_crashworthiness": [(1.0, 3.0)] * 5,
    "re21": [(0.1, 1.0)] * 4,
    "re22": [(0.2, 15), (0, 20), (0, 40)],
    "re23": [(1, 100), (1, 100), (10, 200), (10, 240)],
    "re24": [(0.5, 4), (4, 50)],
    "re25": [(1, 70), (0.6, 30), (0.009, 0.5)],
    "re31": [(0.00001, 100), (0.00001, 100), (1.0, 3.0)],
    "re32": [(0.125, 5), (0.1, 10), (0.1, 10), (0.125, 5)],
    "re33": [(55, 80), (75, 110), (1000, 3000), (11, 20)]
}

# ----- Surrogate builders -----
def build_algorithms(nv):
    rs = BASE_SEED
    return {
        "SVM": svm.SVR,
        "NN": lambda: MLPRegressor(max_iter=2000, tol=1e-4, random_state=rs),
        "Ada": lambda: ensemble.AdaBoostRegressor(random_state=rs),
        "GPR": GaussianProcessRegressor,
        "SGD": lambda: SGD(random_state=rs),
        "KNR": KNR,
        "DTR": lambda: DTR(random_state=rs),
        "RFR": lambda: ensemble.RandomForestRegressor(random_state=rs),
        "ExTR": lambda: ensemble.ExtraTreesRegressor(random_state=rs),
        "GBR": lambda: ensemble.GradientBoostingRegressor(random_state=rs),
        "XGB": lambda: XGBRegressor(random_state=rs, n_estimators=200, verbosity=0)
    }
    
def generate_initial_population(pop_size, nv, lb, ub, seed=None):
    """Generate a well-spread initial population (numpy array)"""
    sampler = qmc.LatinHypercube(d=nv, seed=seed)
    sample = sampler.random(pop_size)  # [0,1]
    lb = np.array(lb)
    ub = np.array(ub)
    return qmc.scale(sample, lb, ub)  # scale to bounds

def build_desdeo_population(problem, pop_size, lb, ub, seed=None):
    """Return a desdeo Population object initialized with LHS points"""
    init_array = generate_initial_population(pop_size, len(problem.variables), lb, ub, seed)
    pop = Population(problem)
    pop.add_individuals(init_array)
    return pop
    
def normalize(A, ref):
    """Min-max normalize A using ref PF bounds (safe for NaNs and zero span)."""
    if A is None or ref is None or A.size == 0 or ref.size == 0:
        return np.empty_like(A)
    A = np.atleast_2d(A).astype(float)
    ref = np.atleast_2d(ref).astype(float)

    # Remove NaN/inf rows from ref to avoid invalid min/max
    ref = ref[np.isfinite(ref).all(axis=1)]
    if ref.size == 0:
        return np.empty_like(A)

    mins = np.nanmin(ref, axis=0)
    maxs = np.nanmax(ref, axis=0)
    span = np.where((maxs - mins) < 1e-12, 1.0, maxs - mins)

    normed = (A - mins) / span
    return np.clip(normed, 0.0, 1.0)

# ---------- Helper functions (single definitions) ----------
def nondominated_filter(arr):
    """Return the non-dominated subset of arr (rows = points, cols = objectives)."""
    if arr.size == 0:
        return np.empty((0, arr.shape[1] if arr.ndim == 2 else 0))
    arr = np.asarray(arr, dtype=float)
    npts = arr.shape[0]
    is_nd = np.ones(npts, dtype=bool)
    for i in range(npts):
        if not is_nd[i]:
            continue
        for j in range(npts):
            if i == j or not is_nd[j]:
                continue
            try:
                if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                    is_nd[i] = False
                    break
            except Exception:
                continue
    return arr[is_nd]

def sanitize_pf(pf, no, floor=1e-8, problem_name=None):
    """Ensure pf is finite, 2D, and apply a tiny floor to avoid zero issues."""
    if pf is None or pf.size == 0:
        return np.empty((0, no))
    pf = np.atleast_2d(np.asarray(pf, dtype=float))
    mask = np.isfinite(pf).all(axis=1)
    if not mask.all():
        logging.warning(
            f"Dropping {np.sum(~mask)} non-finite rows from PF"
            + (f" for {problem_name}" if problem_name is not None else "")
        )
        pf = pf[mask]
    pf = np.where(np.abs(pf) < floor, np.sign(pf) * floor, pf)
    return pf

def apply_maximize_inversion(arr, maximize_flags):
    """If maximize_flags[i] is True, negate that objective column."""
    if arr is None or arr.size == 0:
        return arr
    arr = np.asarray(arr, dtype=float).copy()
    for i, mf in enumerate(maximize_flags):
        if mf:
            arr[:, i] = -arr[:, i]
    return arr

def compute_metrics(nd_arr, pf_arr, problem_name=None):
    """
    Compute IGD, HV, additive epsilon, and multiplicative epsilon.
    Inputs must already be in the SAME objective convention (all minimized).

    Normalization is based ONLY on the reference PF, so metrics remain comparable
    across algorithms/runs for the same problem.
    """
    if nd_arr is None or pf_arr is None or nd_arr.size == 0 or pf_arr.size == 0:
        return dict(IGD=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan)

    nd = np.atleast_2d(np.asarray(nd_arr, dtype=float))
    pf = np.atleast_2d(np.asarray(pf_arr, dtype=float))

    nd = nd[np.isfinite(nd).all(axis=1)]
    pf = pf[np.isfinite(pf).all(axis=1)]

    if nd.size == 0 or pf.size == 0:
        return dict(IGD=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan)

    nd = nondominated_filter(nd)
    pf = nondominated_filter(pf)

    # normalize using REFERENCE PF only
    mins = np.min(pf, axis=0)
    maxs = np.max(pf, axis=0)
    spans = maxs - mins
    spans[spans < 1e-12] = 1.0

    nd_norm = (nd - mins) / spans
    pf_norm = (pf - mins) / spans

    # do not clip for IGD / epsilon; let bad solutions stay outside [0,1]
    # but HV needs a valid reference point beyond the worst accepted region
    hv_ref_point = np.ones(pf.shape[1]) * 1.1

    hv_val = np.nan
    igd_val = np.nan
    eps_a = np.nan
    eps_m = np.nan

    try:
        # for HV, keep only points not worse than ref point in every objective
        nd_hv = nd_norm[np.all(nd_norm <= hv_ref_point, axis=1)]
        nd_hv = nd_hv[np.all(np.isfinite(nd_hv), axis=1)]
        if nd_hv.size > 0:
            hv_val = hypervolume(nd_hv, ref_point=hv_ref_point)
    except Exception as e:
        logging.warning(f"Hypervolume computation failed for {problem_name}: {e}")

    try:
        igd_val = igd(nd_norm, pf_norm)
    except Exception as e:
        logging.warning(f"IGD computation failed for {problem_name}: {e}")

    try:
        eps_a = eps_additive(nd_norm, pf_norm)
    except Exception as e:
        logging.warning(f"Additive epsilon computation failed for {problem_name}: {e}")

    try:
        # multiplicative epsilon assumes nonnegative values
        nd_mult = np.clip(nd_norm, 1e-12, None)
        pf_mult = np.clip(pf_norm, 1e-12, None)
        eps_m = eps_multiplicative(nd_mult, pf_mult)
    except Exception as e:
        logging.warning(f"Multiplicative epsilon computation failed for {problem_name}: {e}")

    return dict(IGD=igd_val, HV=hv_val, EpsAdd=eps_a, EpsMulti=eps_m)
    
def make_obj_from_model(m):
    def surrogate(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        y = np.asarray(m.predict(x), dtype=float).reshape(-1)
        return y if len(y) > 1 else float(y[0])
    return surrogate

# ----- Extract filename details -----
pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$")
eng_pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)_samples(?:_([A-Za-z0-9_]+))?\.csv$")
dbmopp_pattern = re.compile(r"^(DBMOPP\d+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$")

# ----- Extract filename details -----
pattern = re.compile(
    r"^([A-Za-z0-9_]+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$"
)
eng_pattern = re.compile(
    r"^([A-Za-z0-9_]+)_(\d+)_samples(?:_([A-Za-z0-9_]+))?\.csv$"
)
dbmopp_pattern = re.compile(
    r"^(DBMOPP\d+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$"
)

def extract_details(fn):
    # ---------- Engineering ----------
    m = eng_pattern.match(fn)
    if m:
        name = m.group(1)
        samples = int(m.group(2))
        noise_tag = m.group(3) or "none"

        if name not in ENGINEERING_PROBLEMS:
            logging.error(f"[ENGINEERING] Unknown problem '{name}' in file {fn}")
            return None, None, None, None, None

        try:
            inst = ENGINEERING_PROBLEMS[name]()
            nv = len(inst.variables)
            no = len(inst.objectives)
            return name, nv, no, samples, noise_tag
        except Exception as e:
            logging.error(f"[ENGINEERING] Failed to instantiate {name}: {e}")
            return None, None, None, None, None

    # ---------- DBMOPP ----------
    m = dbmopp_pattern.match(fn)
    if m:
        name = m.group(1)
        nv = int(m.group(2))
        no = int(m.group(3))
        samples = int(m.group(4))
        noise_tag = m.group(5) or "none"
        return name, nv, no, samples, noise_tag

    # ---------- DTLZ / WFG ----------
    m = pattern.match(fn)
    if m:
        name = m.group(1)
        nv = int(m.group(2))
        no = int(m.group(3))
        samples = int(m.group(4))
        noise_tag = m.group(5) or "none"
        return name, nv, no, samples, noise_tag

    # ---------- Unknown ----------
    logging.warning(f"Could not parse filename: {fn}")
    return None, None, None, None, None



# ----- Load reference PFs -----
real_path = path.join(base_folder, "modelling_results", "real_paretofronts")
real_fronts = {}
reference_pfs = {}

def get_missing_ea_iteration_pairs(
    existing_pf,
    nm,
    nv,
    no,
    n_samples,
    dataset_file,
    algo_nm,
    noise_tag,
    ea_list,
    n_iterations,
    run_tag
):
    needed = [(ea, i) for i in range(n_iterations) for ea in ea_list]

    if existing_pf is None or existing_pf.empty:
        return needed

    mask = (
        (existing_pf["problem"] == nm) &
        (existing_pf["n_var"] == nv) &
        (existing_pf["n_obj"] == no) &
        (existing_pf["n_samples"] == n_samples) &
        (existing_pf["dataset_file"] == dataset_file) &
        (existing_pf["algo"] == algo_nm) &
        (existing_pf["noise"] == noise_tag) &
        (existing_pf["RUN_TAG"] == run_tag)
    )

    done_df = existing_pf.loc[mask, ["ea", "iteration"]].drop_duplicates()
    done = set(
        (str(r["ea"]), int(r["iteration"]))
        for _, r in done_df.iterrows()
        if pd.notna(r["iteration"])
    )

    return [(ea, i) for (ea, i) in needed if (ea, i) not in done]
    
def combined_row_exists(existing_pf, nm, nv, no, n_samples, dataset_file, algo_nm, noise_tag, iteration_idx, run_tag):
    if existing_pf is None or existing_pf.empty:
        return False

    mask = (
        (existing_pf["problem"] == nm) &
        (existing_pf["n_var"] == nv) &
        (existing_pf["n_obj"] == no) &
        (existing_pf["n_samples"] == n_samples) &
        (existing_pf["dataset_file"] == dataset_file) &
        (existing_pf["ea"] == "Combined") &
        (existing_pf["algo"] == algo_nm) &
        (existing_pf["noise"] == noise_tag) &
        (existing_pf["iteration"] == iteration_idx) &
        (existing_pf["RUN_TAG"] == run_tag)
    )
    return mask.any()
    
def _load_pf(fpath, no):
    try:
        # Load PF
        if fpath.lower().endswith(".txt"):
            pf = np.loadtxt(fpath)
        else:
            pf = pd.read_csv(fpath).values

        # Ensure 2D
        if pf.ndim == 1:
            pf = pf.reshape(1, -1)

        # Validate dimensions
        if pf.shape[1] != no:
            logging.error(f"{fpath} has {pf.shape[1]} objectives, expected {no}. Rejecting.")
            return np.empty((0, no))

        return pf

    except Exception as e:
        logging.error(f"Could not load PF: {fpath}. Error: {e}")
        return np.empty((0, no))

def find_reference_pf(problem_name, no, nv=None):
    """
    Find reference PF using the new folder layout:
      real_paretofronts/
        DTLZ/<problem>/<problem>_<nobj>obj_reference_pf.csv
        WFG/<problem>/<problem>_<nobj>obj_reference_pf.csv
        engineering/<problem>/<problem>_<nobj>obj_reference_pf.csv
        DBMOPP/<problem>/<problem>_<nobj>obj_reference_pf.csv
    """
    base_dir = "/scratch/project_2017216/modelling_results/real_paretofronts"
    pname = str(problem_name)

    if pname.upper().startswith("DTLZ"):
        suite_dir = "DTLZ"
    elif pname.upper().startswith("WFG"):
        suite_dir = "WFG"
    elif pname.upper().startswith("DBMOPP"):
        suite_dir = "DBMOPP"
    else:
        suite_dir = "engineering"

    pf_dir = os.path.join(base_dir, suite_dir, pname)

    if not os.path.exists(pf_dir):
        logging.warning(f"No PF dir exists for {problem_name} @ {pf_dir}")
        return np.empty((0, no))

    expected_name = f"{pname}_{no}obj_reference_pf.csv"
    expected_path = os.path.join(pf_dir, expected_name)

    if os.path.exists(expected_path):
        pf = _load_pf(expected_path, no)
        if pf.shape[1] == no:
            logging.info(f"Loaded reference PF for {problem_name} from {expected_path}")
            return pf
        logging.warning(f"Reference PF had wrong dimension in {expected_path}")

    # fallback: any file with matching n_obj
    files = [f for f in os.listdir(pf_dir) if f.lower().endswith((".csv", ".txt"))]
    matches = [f for f in files if re.search(rf"_{no}obj_", f.lower()) or re.search(rf"{no}obj", f.lower())]

    if not matches:
        logging.warning(f"No PF file with {no} objectives found in {pf_dir}")
        return np.empty((0, no))

    matches.sort(key=lambda f: (not f.lower().endswith(".csv"), f))
    best_file = matches[0]
    best_path = os.path.join(pf_dir, best_file)

    pf = _load_pf(best_path, no)
    if pf.shape[1] == no:
        logging.info(f"Loaded fallback reference PF for {problem_name} from {best_path}")
        return pf

    logging.warning(f"Fallback PF had wrong dimension for {problem_name}: {best_path}")
    return np.empty((0, no))


def run_all_eas(
    nm,
    nv,
    no,
    ml_list,
    lb,
    ub,
    noise_tag,
    eas_to_run=None,
    n_gen=N_GEN,
    save_every=10,
    iteration_idx=0
):
    """
    Run selected EAs on the surrogate problem and return metrics.
    No CSV writing is done here.
    """

    if eas_to_run is None:
        eas_to_run = ["NSGAIII", "RVEA", "IBEA"]

    ea_results = {}

    DBMOPP_PARAMS = {
        "DBMOPP1": dict(n_local_pareto_regions=2, n_dominance_res_regions=0, n_global_pareto_regions=3, pareto_set_type=0, constraint_type=1, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
        "DBMOPP2": dict(n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=3, pareto_set_type=1, constraint_type=3, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
        "DBMOPP3": dict(n_local_pareto_regions=3, n_dominance_res_regions=2, n_global_pareto_regions=4, pareto_set_type=2, constraint_type=5, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
        "DBMOPP4": dict(n_local_pareto_regions=3, n_dominance_res_regions=4, n_global_pareto_regions=5, pareto_set_type=2, constraint_type=8, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.1),
        "DBMOPP5": dict(n_local_pareto_regions=1, n_dominance_res_regions=0, n_global_pareto_regions=3, pareto_set_type=0, constraint_type=1, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
        "DBMOPP6": dict(n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=4, pareto_set_type=1, constraint_type=4, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.2),
        "DBMOPP7": dict(n_local_pareto_regions=2, n_dominance_res_regions=2, n_global_pareto_regions=5, pareto_set_type=2, constraint_type=7, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
    }

    # ----------------------------
    # maximize flags
    # ----------------------------
    maximize_flags = [False] * no
    try:
        inst_fn = ED_MAP.get(nm)
        if inst_fn is not None:
            inst = inst_fn()
            maximize_flags = [
                bool(obj.maximize[0]) if isinstance(obj.maximize, (list, np.ndarray)) else bool(obj.maximize)
                for obj in inst.objectives
            ]
    except Exception:
        maximize_flags = [False] * no

    # ----------------------------
    # reference PF (cached globally)
    # ----------------------------
    pf_key = (nm, no)
    pf_arr = reference_pfs.get(pf_key)
    if pf_arr is None or pf_arr.size == 0:
        pf_arr = find_reference_pf(nm, no, nv)
        reference_pfs[pf_key] = pf_arr

    pf_clean = sanitize_pf(pf_arr, no, problem_name=nm)
    pf_safe = apply_maximize_inversion(pf_clean, maximize_flags)

    # ----------------------------
    # variables / constraints
    # ----------------------------
    vars_ = None
    constraints = None
    try:
        if nm.upper().startswith("DTLZ"):
            vars_ = [
                Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5)
                for i in range(nv)
            ]
            constraints = None
        
        elif nm.upper().startswith("WFG"):
            vars_ = [
                Variable(
                    f"x{i+1}",
                    lower_bound=float(lb[i]),
                    upper_bound=float(ub[i]),
                    initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])),
                )
                for i in range(nv)
            ]
            constraints = None

        elif nm in variable_ranges:
            bounds = variable_ranges[nm]
            if len(bounds) >= nv:
                vars_ = [
                    Variable(
                        f"x{i+1}",
                        lower_bound=bounds[i][0],
                        upper_bound=bounds[i][1],
                        initial_value=bounds[i][0] + 0.5 * (bounds[i][1] - bounds[i][0]),
                    )
                    for i in range(nv)
                ]
            else:
                vars_ = [
                    Variable(
                        f"x{i+1}",
                        lower_bound=float(lb[i]),
                        upper_bound=float(ub[i]),
                        initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])),
                    )
                    for i in range(nv)
                ]

            try:
                inst_fn = ED_MAP.get(nm)
                if inst_fn is not None:
                    inst = inst_fn()
                    constraints = getattr(inst, "constraints", None)
            except Exception:
                constraints = None

        elif nm.upper().startswith("DBMOPP"):
            try:
                params = DBMOPP_PARAMS.get(nm, None)
                if params is not None:
                    dbm = DBMOPP_generator(
                        nlp=params["n_local_pareto_regions"],
                        ndr=params["n_dominance_res_regions"],
                        ngp=params["n_global_pareto_regions"],
                        prop_constraint_checker=params.get("prop_neutral", 0.0),
                        pareto_set_type=params["pareto_set_type"],
                        constraint_type=params["constraint_type"],
                        k=no,
                        n=nv,
                    )
                    constraints = getattr(dbm, "constraints", None)
                else:
                    constraints = None

                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
            except Exception as e:
                logging.warning(f"Could not recreate DBMOPP for {nm}: {e}")
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                constraints = None

        else:
            if lb is not None and ub is not None and len(lb) == nv:
                vars_ = [
                    Variable(
                        f"x{i+1}",
                        lower_bound=float(lb[i]),
                        upper_bound=float(ub[i]),
                        initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])),
                    )
                    for i in range(nv)
                ]
            else:
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
            constraints = None

    except Exception as e:
        logging.warning(f"Error preparing variables/constraints for {nm}: {e}\n{traceback.format_exc()}")
        if vars_ is None:
            vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
        constraints = None

    # ----------------------------
    # surrogate objectives
    # ----------------------------
    objective_funcs = [
        ScalarObjective(
            name=f"f{i+1}",
            evaluator=make_obj_from_model(m),
            maximize=[bool(maximize_flags[i])]
        )
        for i, m in enumerate(ml_list)
    ]

    problem = MOProblem(objectives=objective_funcs, variables=vars_, constraints=constraints)

    logging.info(f"Problem {nm}: nv={nv}, no={no}, variables={len(vars_)}, constraints={constraints is not None}")

    np.random.seed(BASE_SEED + iteration_idx)

    common_params = {"n_iterations": 1, "n_gen_per_iter": n_gen}
    ea_registry = {
        "NSGAIII": (NSGAIII, {}),
        "IBEA": (IBEA, {"population_size": POP_SIZE}),
        "RVEA": (RVEA, {"population_size": POP_SIZE}),
    }

    all_nd_for_combined = []

    for ea_name in eas_to_run:
        if ea_name not in ea_registry:
            logging.warning(f"Unknown EA requested: {ea_name}")
            continue

        ea_class, ea_specific_params = ea_registry[ea_name]
        logging.info(f"Starting {ea_name} on {nm} (iter={iteration_idx})")

        try:
            evolver = ea_class(problem, **{**common_params, **ea_specific_params})
        except Exception as e:
            logging.error(f"{ea_name} init failed on {nm}: {e}")
            ea_results[ea_name] = dict(
                iteration=iteration_idx,
                IGD=np.nan,
                HV=np.nan,
                EpsAdd=np.nan,
                EpsMulti=np.nan,
                nd_arr=np.empty((0, no)),
            )
            continue

        iter_fronts = []

        while True:
            try:
                cont = evolver.continue_evolution()
            except Exception:
                cont = False

            if not cont:
                break

            evolver.iterate()

            pop = getattr(evolver, "population", None)
            if pop is not None and pop.objectives is not None:
                current_objs = np.asarray(pop.objectives, dtype=float)
                nd_objs = nondominated_filter(current_objs)
                if nd_objs.size > 0:
                    iter_fronts.append(nd_objs)

                    if (len(iter_fronts) % save_every) == 0:
                        np.savetxt(
                            f"{ea_name}_intermediate_pf_iter{len(iter_fronts)}.csv",
                            nondominated_filter(np.vstack(iter_fronts)),
                            delimiter=","
                        )

        final_objs = np.empty((0, no))
        try:
            end_ret = evolver.end()
            final_pop = end_ret[1] if isinstance(end_ret, (list, tuple)) and len(end_ret) >= 2 else end_ret

            rows = []
            for ind in final_pop:
                if hasattr(ind, "objectives"):
                    rows.append(np.asarray(ind.objectives, dtype=float))
                elif hasattr(ind, "F"):
                    rows.append(np.asarray(ind.F, dtype=float))

            if rows:
                final_objs = np.vstack(rows)
                iter_fronts.append(final_objs)

        except Exception:
            if iter_fronts:
                final_objs = iter_fronts[-1]

        combined = np.vstack(iter_fronts) if iter_fronts else np.empty((0, no))
        if combined.size > 0:
            combined = combined[np.isfinite(combined).all(axis=1)]

        nd_arr = nondominated_filter(combined) if combined.size > 0 else np.empty((0, no))
        nd_safe = apply_maximize_inversion(nd_arr, maximize_flags)

        metrics = compute_metrics(nd_safe, pf_safe, problem_name=nm)

        ea_results[ea_name] = dict(
            iteration=iteration_idx,
            IGD=metrics["IGD"],
            HV=metrics["HV"],
            EpsAdd=metrics["EpsAdd"],
            EpsMulti=metrics["EpsMulti"],
            nd_arr=nd_safe.copy(),
        )

        if nd_safe.size > 0:
            all_nd_for_combined.append(nd_safe)

        surrogate_algo = get_surrogate_algorithm_name(ml_list)

        plot_and_save_pf(
            approx_pf=nd_safe,
            ref_pf=pf_safe,
            problem_name=nm,
            ea_name=ea_name,
            surrogate=surrogate_algo,
            n_vars=nv,
            n_objs=no,
        )

    # ----------------------------
    # combined across selected EAs
    # ----------------------------
    base_eas = {"IBEA", "NSGAIII", "RVEA"}
    
    if set(eas_to_run) == base_eas:
        if all_nd_for_combined:
            combined_all = np.vstack(all_nd_for_combined)
            combined_nd = nondominated_filter(combined_all)
        else:
            combined_nd = np.empty((0, no))
    
        combined_metrics = compute_metrics(combined_nd, pf_safe, problem_name=nm)
    
        ea_results["Combined"] = dict(
            iteration=iteration_idx,
            IGD=combined_metrics["IGD"],
            HV=combined_metrics["HV"],
            EpsAdd=combined_metrics["EpsAdd"],
            EpsMulti=combined_metrics["EpsMulti"],
            nd_arr=combined_nd.copy(),
        )
    
        surrogate_algo = get_surrogate_algorithm_name(ml_list)
        if not np.isnan(combined_metrics["IGD"]) and combined_metrics["IGD"] <= 0.35:
            plot_and_save_pf(
                approx_pf=combined_nd,
                ref_pf=pf_safe,
                problem_name=nm,
                ea_name="Combined",
                surrogate=surrogate_algo,
                n_vars=nv,
                n_objs=no,
            )

    return ea_results, maximize_flags, pf_safe

def _append_df_to_csv(df_row, fp, cols=None):
    write_header = not os.path.exists(fp)
    if cols is not None:
        df_row = df_row.reindex(columns=cols)
    df_row.to_csv(fp, mode="a", header=write_header, index=False)
    
def build_and_train_surrogates(nm, nv, no, X, Y, algo_nm, algo_fn, noise_tag, dataset_file):
    """
    Train one surrogate per objective using the specified regression algorithm.
    Cross-validation metrics are logged and stored to surrogate_perf.csv.
    """
    ml_list = []

    X = X.copy()
    Y = Y.copy()

    valid_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X.loc[valid_mask], Y.loc[valid_mask]

    n_samples = len(X)
    if n_samples < 5:
        logging.warning(f"Too few samples ({n_samples}) for {nm} - skipping surrogate training.")
        return [None] * no

    n_splits = 5 if n_samples >= 100 else 3 if n_samples >= 30 else 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=BASE_SEED)

    for obj_idx in range(no):
        y = Y.iloc[:, obj_idx].astype(float)

        if y.nunique() <= 1:
            logging.warning(f"Objective f{obj_idx+1} constant for {nm}. Skipping.")
            ml_list.append(None)
            continue

        try:
            model = TransformedTargetRegressor(
                regressor=Pipeline([
                    ("scaler", StandardScaler()),
                    ("regressor", algo_fn())
                ]),
                transformer=StandardScaler(),
                check_inverse=False
            )

            scores = cross_validate(
                model,
                X,
                y,
                cv=kf,
                scoring=("r2", "neg_mean_squared_error"),
                n_jobs=-1,
                error_score="raise"
            )

            model.fit(X, y)

            mean_r2 = float(scores["test_r2"].mean())
            std_r2 = float(scores["test_r2"].std())
            mean_mse = float(-scores["test_neg_mean_squared_error"].mean())

            logging.info(
                f"{nm} | {algo_nm} | f{obj_idx+1}: "
                f"R2={mean_r2:.4f} ± {std_r2:.4f}, "
                f"MSE={mean_mse:.3e}, samples={n_samples}, folds={n_splits}"
            )

            row_df = pd.DataFrame([[
                nm,
                nv,
                no,
                dataset_file,
                algo_nm,
                noise_tag,
                f"f{obj_idx+1}",
                mean_r2,
                std_r2,
                mean_mse,
                n_samples,
                n_splits
            ]], columns=perf_cols)

            _append_df_to_csv(row_df, perf_fp, perf_cols)

        except Exception as e:
            logging.warning(
                f"Training failed for {nm} | {algo_nm} | f{obj_idx+1}: {e}"
            )
            model = None

        ml_list.append(model)

    return ml_list

# ========================= MAIN LOOP =========================
import os
import logging
import pandas as pd
import numpy as np
from os import path

data_root = path.join(base_folder, "Data")
n_iterations = 1
n_gen_per_iter = 50
fallback_warned = set()

# Initialize existing PF/Perf dataframes
existing_pf = None
existing_perf = None
if existing_pf is None or not isinstance(existing_pf, pd.DataFrame):
    existing_pf = pd.DataFrame(columns=pf_cols)
if existing_perf is None or not isinstance(existing_perf, pd.DataFrame):
    existing_perf = pd.DataFrame(columns=perf_cols)

# Global cache for reference PFs
reference_pfs = {}

# Load previously saved results
try:
    existing_pf = load_existing(pf_fp, pf_cols)
except Exception as e:
    logging.error(f"Failed to load existing PF results: {e}")
    existing_pf = pd.DataFrame(columns=pf_cols)

try:
    existing_perf = load_existing(perf_fp, perf_cols)
except Exception as e:
    logging.error(f"Failed to load existing PERF results: {e}")
    existing_perf = pd.DataFrame(columns=perf_cols)

# List of EAs to check
ea_list = ["IBEA", "NSGAIII", "RVEA"]

for suite in ["DTLZ", "WFG", "Engineering", "DBMOPP"]:
    if args.suite != "ALL" and args.suite != suite:
        continue

    suite_dir = path.join(data_root, suite)
    if not path.exists(suite_dir):
        logging.info(f"Suite dir not found: {suite_dir}")
        continue
    logging.info(f"Processing suite {suite} at {suite_dir}")

    for prob_name in sorted(os.listdir(suite_dir)):
        if args.problem is not None and prob_name != args.problem:
            continue

        prob_dir = path.join(suite_dir, prob_name)
        if not path.isdir(prob_dir):
            continue

        files = [f for f in os.listdir(prob_dir) if f.endswith(".csv")]
        logging.info(f"Processing problem {prob_name} with {len(files)} datasets")

        for fn in sorted(files):
            try:
                fpath = path.join(prob_dir, fn)
                nm, nv, no, samples, noise_tag = extract_details(fn)
                if nm is None:
                    logging.info(f"Could not parse file name {fn}, skipping.")
                    continue
                
                if args.n_obj is not None and no != args.n_obj:
                    logging.info(f"Skipping {fn}: n_obj={no} does not match --n_obj={args.n_obj}")
                    continue

                logging.info(f"Loading dataset {fpath}")
                df = pd.read_csv(fpath)
                x_cols = [c for c in df.columns if c.startswith("x")]
                y_cols = [c for c in df.columns if c.startswith("f")]
                X, Y = df[x_cols], df[y_cols]
                lb, ub = X.min(axis=0).values, X.max(axis=0).values

                # Build ML algorithms
                algos = build_algorithms(nv)

                # Loop over ML algorithms
                for algo_nm, algo_fn in algos.items():

                    missing_pairs = get_missing_ea_iteration_pairs(
                        existing_pf=existing_pf,
                        nm=nm,
                        nv=nv,
                        no=no,
                        n_samples=samples,
                        dataset_file=fn,
                        algo_nm=algo_nm,
                        noise_tag=noise_tag,
                        ea_list=ea_list,
                        n_iterations=n_iterations,
                        run_tag=RUN_TAG
                    )

                    if not missing_pairs:
                        logging.info(
                            f"Skipping {nm}, {nv}vars, {no}objs, {algo_nm}, {noise_tag} "
                            f"-> all EA+iteration results already exist for RUN_TAG={RUN_TAG}"
                        )
                        continue

                    logging.info(
                        f"{nm}, {nv}vars, {no}objs, {algo_nm}, {noise_tag} "
                        f"-> missing EA/iteration pairs: {missing_pairs}"
                    )

                    logging.info(f"Training surrogates for {nm} using {algo_nm}")
                    ml_list = build_and_train_surrogates(
                        nm, nv, no, X, Y, algo_nm, algo_fn, noise_tag, fn
                    )

                    if any(m is None for m in ml_list):
                        logging.warning(f"Skipping {nm} with {algo_nm}: some surrogates failed.")
                        continue

                    pf_key = (nm, no)
                    pf_arr = reference_pfs.get(pf_key)
                    if pf_arr is None or pf_arr.size == 0:
                        pf_arr = find_reference_pf(nm, no, nv)
                        if pf_arr.size > 0:
                            reference_pfs[pf_key] = pf_arr
                            logging.info(f"Cached reference PF for {nm}, {no} objs ({pf_arr.shape})")
                        else:
                            logging.warning(f"No reference PF found for {nm}, {no} objs; metrics may be NaN.")
                            pf_arr = np.empty((0, no))

                    # run grouped by iteration
                    for iter_idx in range(n_iterations):
                        eas_this_iter = [ea for (ea, it) in missing_pairs if it == iter_idx]
                        if not eas_this_iter:
                            continue

                        logging.info(
                            f"Iteration {iter_idx + 1}/{n_iterations} for {nm} with {algo_nm}; "
                            f"EAs to run: {eas_this_iter}"
                        )

                        try:
                            ea_results, maximize_flags, pf_safe = run_all_eas(
                                nm=nm,
                                nv=nv,
                                no=no,
                                ml_list=ml_list,
                                lb=lb,
                                ub=ub,
                                noise_tag=noise_tag,
                                eas_to_run=eas_this_iter,
                                n_gen=n_gen_per_iter,
                                iteration_idx=iter_idx
                            )
                        except Exception as e:
                            logging.warning(f"run_all_eas failed on {nm} {algo_nm} iter {iter_idx}: {e}")
                            continue

                        for ea_name, data in ea_results.items():
                            if ea_name == "Combined":
                                already_done = combined_row_exists(
                                    existing_pf=existing_pf,
                                    nm=nm,
                                    nv=nv,
                                    no=no,
                                    n_samples=samples,
                                    dataset_file=fn,
                                    algo_nm=algo_nm,
                                    noise_tag=noise_tag,
                                    iteration_idx=data.get("iteration", iter_idx),
                                    run_tag=RUN_TAG
                                )
                                if already_done:
                                    logging.info(
                                        f"Skipping duplicate Combined row for "
                                        f"{nm}, {nv}vars, {no}objs, {algo_nm}, {noise_tag}, file={fn}"
                                    )
                                    continue
                        
                            row = [
                                nm,
                                nv,
                                no,
                                samples,
                                fn,
                                ea_name,
                                algo_nm,
                                noise_tag,
                                data.get("iteration", iter_idx),
                                data.get("IGD", np.nan),
                                data.get("HV", np.nan),
                                data.get("EpsAdd", np.nan),
                                data.get("EpsMulti", np.nan),
                                RUN_TAG,
                            ]
                        
                            row_df = pd.DataFrame([row], columns=pf_cols)
                            _append_df_to_csv(row_df, pf_fp, pf_cols)
                            existing_pf = pd.concat([existing_pf, row_df], ignore_index=True)
                

            except Exception as e:
                logging.exception(f"Fatal error processing file {fn}: {e}\n{traceback.format_exc()}")

logging.info("========== SCRIPT END ==========")

