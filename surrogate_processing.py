#!/usr/bin/env python3
"""
NSGA-III / RVEA / IBEA surrogate evaluator with stabilized metrics and RunTag versioning.
Supports DTLZ, WFG, Engineering, and DBMOPP problems.

Outputs:
  - modelling_results/surrogate_perf.csv  (CV metrics per objective)
  - modelling_results/surrogate_metrics.csv (EA metrics: IGD_norm, HV, EpsAdd, EpsMulti, IGD_seed_cv, IGD_q25, IGD_q75, RunTag)
"""

import os, re, argparse, logging, warnings, traceback
from os import path, makedirs
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
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

from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_emo.EAs import NSGAIII, RVEA, IBEA
from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
from desdeo_problem.testproblems.EngineeringRealWorld import re21, re22, re23, re24, re25, re31, re32, re33
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
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
base_folder = '/scratch/project_2014748'
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
pf_cols = ['Problem','VarCount','ObjCount','Algorithm','EA','NoiseTag',
           'IGD_norm','HV','EpsAdd','EpsMulti','IGD_seed_cv','IGD_q25','IGD_q75','RunTag']
perf_cols = ['Problem','VarCount','ObjCount','Algorithm','NoiseTag','Objective','R2','MSE']

def load_existing(fp, cols):
    if path.exists(fp):
        df = pd.read_csv(fp)
        for c in cols:
            if c not in df.columns: df[c] = np.nan
        return df[cols].drop_duplicates()
    return pd.DataFrame(columns=cols)

# Try loading any previously saved results at startup
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


# ----- Metrics helpers -----
def euclidean_distance(a, b): return np.linalg.norm(a - b)

def hypervolume(A, ref_point=None):
    if A is None or A.size == 0:
        return np.nan
    A = np.atleast_2d(A).astype(float)
    mask_finite = np.isfinite(A).all(axis=1)
    if not mask_finite.all():
        A = A[mask_finite]
    if A.size == 0:
        return np.nan

    # Normalize and ensure 0 ≤ A ≤ 1
    A = np.clip(A, 0, 1)

    # Safe reference point
    if ref_point is None:
        ref_point = np.ones(A.shape[1]) * 1.1

    try:
        hv = HV(ref_point=ref_point).do(A)
        if not np.isfinite(hv):
            return np.nan
        return float(hv)
    except Exception as e:
        logging.warning(f"HV computation failed: {e}")
        return np.nan

def igd(A, R):
    if A.size == 0 or R.size == 0: return np.nan
    return float(np.mean([np.min([euclidean_distance(r, a) for a in A]) for r in R]))

def epsilon_cleanup(nd: np.ndarray, eps: float) -> np.ndarray:
    if not np.isfinite(eps) or eps <= 0 or nd.size == 0: return nd
    grid = np.floor(nd / eps + 0.5).astype(np.int64)
    _, uniq_idx = np.unique(grid, axis=0, return_index=True)
    return nd[np.sort(uniq_idx)]

def eps_additive(A: np.ndarray, R: np.ndarray) -> float:
    if A.size == 0 or R.size == 0: return np.nan
    d = [np.min(np.max(A - r[np.newaxis, :], axis=1)) for r in R]
    return float(np.max(d))

def eps_multiplicative(A: np.ndarray, R: np.ndarray, tiny=1e-8) -> float:
    if A.size == 0 or R.size == 0:
        return np.nan
    # Clip R to tiny value to avoid huge ratios
    R_safe = np.clip(R, tiny, None)
    d = [np.min(np.max(A / r[np.newaxis, :], axis=1)) for r in R_safe]
    return float(np.max(d))

# ----- Problem instances -----
ED_MAP = {
    're21': re21, 're22': re22, 're23': re23, 're32': re32,
    'river_pollution_problem': river_pollution_problem,
    'vehicle_crashworthiness': vehicle_crashworthiness
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
        "NN": lambda: MLPRegressor(max_iter=1000, tol=1e-4, random_state=rs),
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



# ----- Extract filename details -----
pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$")
eng_pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)_samples(?:_([A-Za-z0-9_]+))?\.csv$")
dbmopp_pattern = re.compile(r"^(DBMOPP\d+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z0-9_]+))?\.csv$")

def extract_details(fn):
    for pat in [pattern, eng_pattern, dbmopp_pattern]:
        m = pat.match(fn)
        if m:
            if pat == eng_pattern:
                name, samples = m.group(1), int(m.group(2))
                noise_tag = m.group(3) or 'none'
                inst = ED_MAP.get(name)
                if inst:
                    return name, len(inst().variables), len(inst().objectives), samples, noise_tag
                return name, None, None, samples, noise_tag
            else:
                name, nv_str, no_str, samples_str = m.group(1), m.group(2), m.group(3), m.group(4)
                noise_tag = m.group(5) or 'none'
                return name, int(nv_str), int(no_str), int(samples_str), noise_tag
    return None, None, None, None, None

# ----- Load reference PFs -----
real_path = path.join(base_folder, "modelling_results", "real_paretofronts")
real_fronts = {}
reference_pfs = {}

def load_pf_files(folder):
    for subdir, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(('.csv','.txt')):
                continue
            fpath = path.join(subdir, fname)
            try:
                arr = pd.read_csv(fpath, header=None).values
            except Exception:
                continue
            if arr.ndim != 2:
                continue
            key = (os.path.basename(subdir), arr.shape[1], arr.shape[0])
            real_fronts[key] = arr

if path.exists(real_path):
    load_pf_files(real_path)
    
    
def find_reference_pf(problem_name, no, nv):
    """Locate and load reference PF, handling different naming patterns."""
    base_dir = "/scratch/project_2014748/modelling_results_all_data/real_paretofronts"
    suite = ("DTLZ" if problem_name.upper().startswith("DTLZ") else
             "WFG" if problem_name.upper().startswith("WFG") else
             "Engineering" if problem_name.lower().startswith(("re", "river", "vehicle")) else
             "DBMOPP" if problem_name.upper().startswith("DBMOPP") else None)
    if suite is None:
        logging.warning(f"Unknown suite for {problem_name}, skipping reference PF.")
        return np.empty((0, no))

    pf_dir = os.path.join(base_dir, suite, problem_name)
    if not os.path.exists(pf_dir):
        logging.warning(f"No PF dir for {problem_name} at {pf_dir}")
        return np.empty((0, no))

    files = [f for f in os.listdir(pf_dir) if f.endswith(".csv")]
    if not files:
        logging.warning(f"No PF files in {pf_dir}")
        return np.empty((0, no))

    # --- Matching rules ---
    candidates = []
    for f in files:
        lower = f.lower()
        if suite == "DTLZ" and f"real_pareto_front_{no}obj" in lower:
            candidates.append(f)
        elif suite == "WFG" and f"pareto_front_{no}obj" in lower:
            candidates.append(f)
        elif suite == "DBMOPP" and f"real_pareto_front_{no}obj" in lower:
            candidates.append(f)
        elif suite == "Engineering" and "_pf_" in lower:
            candidates.append(f)

    if not candidates:
        # fallback: allow mismatched variable count
        for f in files:
            if f"{no}obj" in f:
                candidates.append(f)

    if not candidates:
        logging.warning(f"No matching PF for {problem_name} (no={no}, nv={nv})")
        return np.empty((0, no))

    fsel = sorted(candidates)[0]
    fpath = os.path.join(pf_dir, fsel)
    try:
        pf_arr = pd.read_csv(fpath).values
        logging.info(f"find_reference_pf: loaded reference PF from disk {fpath} -> {pf_arr.shape}")
        return pf_arr
    except Exception as e:
        logging.warning(f"Could not load reference PF {fpath}: {e}")
        return np.empty((0, no))


def run_all_eas(nm, nv, no, ml_list, lb, ub, n_gen=N_GEN, save_every=10):
    results = {}
    results = {ea_name: {} for ea_name in ["NSGAIII", "RVEA", "IBEA"]}
    results["Combined"] = dict(IGD_norm=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan, nd_arr=np.empty((0, no)))

    DBMOPP_PARAMS = {
    "DBMOPP1": dict(n_local_pareto_regions=2, n_dominance_res_regions=0, n_global_pareto_regions=3, pareto_set_type=0, constraint_type=1, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
    "DBMOPP2": dict(n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=3, pareto_set_type=1, constraint_type=3, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
    "DBMOPP3": dict(n_local_pareto_regions=3, n_dominance_res_regions=2, n_global_pareto_regions=4, pareto_set_type=2, constraint_type=5, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
    "DBMOPP4": dict(n_local_pareto_regions=3, n_dominance_res_regions=4, n_global_pareto_regions=5, pareto_set_type=2, constraint_type=8, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.1),
    "DBMOPP5": dict(n_local_pareto_regions=1, n_dominance_res_regions=0, n_global_pareto_regions=3, pareto_set_type=0, constraint_type=1, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
    "DBMOPP6": dict(n_local_pareto_regions=2, n_dominance_res_regions=1, n_global_pareto_regions=4, pareto_set_type=1, constraint_type=4, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.2),
    "DBMOPP7": dict(n_local_pareto_regions=2, n_dominance_res_regions=2, n_global_pareto_regions=5, pareto_set_type=2, constraint_type=7, ndo=0, vary_sol_density=False, vary_objective_scales=False, prop_neutral=0.0),
}

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

    def sanitize_pf(pf, no, floor=1e-8):
        """Ensure pf is finite, 2D, and apply a tiny floor to avoid zero issues."""
        if pf is None or pf.size == 0:
            return np.empty((0, no))
        pf = np.atleast_2d(np.asarray(pf, dtype=float))
        mask = np.isfinite(pf).all(axis=1)
        if not mask.all():
            logging.warning(f"Dropping {np.sum(~mask)} non-finite rows from PF for {nm}")
            pf = pf[mask]
        # apply tiny floor (preserve sign)
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

    def compute_metrics(nd_arr, pf_arr):
        """Compute union-normalized HV, IGD, additive and multiplicative epsilons."""
        if nd_arr is None or pf_arr is None or nd_arr.size == 0 or pf_arr.size == 0:
            return dict(IGD_norm=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan)

        nd = np.atleast_2d(np.asarray(nd_arr, dtype=float))
        pf = np.atleast_2d(np.asarray(pf_arr, dtype=float))

        # union normalization (min..max over both sets)
        union = np.vstack([nd, pf])
        min_vals = np.min(union, axis=0)
        max_vals = np.max(union, axis=0)
        range_vals = max_vals - min_vals
        # avoid zero-range by setting range to 1 for zero-range dims
        zero_mask = range_vals == 0
        if np.any(zero_mask):
            range_vals[zero_mask] = 1.0

        nd_norm = (nd - min_vals) / range_vals
        pf_norm = (pf - min_vals) / range_vals

        # Clip tiny/negative numerical noise (keep >= 0)
        nd_norm = np.clip(nd_norm, 0.0, None)
        pf_norm = np.clip(pf_norm, 0.0, None)

        # compute dynamic ref point for hv slightly above union maxima
        try:
            ref_point = np.max(np.vstack([nd_norm, pf_norm]), axis=0) * 1.05
            # ensure ref_point is >= 1e-6 and > max to be a proper reference
            ref_point = np.where(ref_point <= np.max(np.vstack([nd_norm, pf_norm]), axis=0),
                                 np.max(np.vstack([nd_norm, pf_norm]), axis=0) + 1e-6,
                                 ref_point)
        except Exception:
            ref_point = np.ones(nd_norm.shape[1]) * 1.05

        hv_val = np.nan
        igd_val = np.nan
        eps_a = np.nan
        eps_m = np.nan

        try:
            hv_val = hypervolume(nd_norm, ref_point=ref_point)
        except Exception as e:
            logging.warning(f"Hypervolume computation failed for {nm}: {e}")
            hv_val = np.nan

        try:
            igd_val = igd(nd_norm, pf_norm)
        except Exception as e:
            logging.warning(f"IGD computation failed for {nm}: {e}")
            igd_val = np.nan

        try:
            eps_a = eps_additive(nd_norm, pf_norm)
        except Exception as e:
            logging.warning(f"Additive epsilon computation failed for {nm}: {e}")
            eps_a = np.nan

        try:
            # multiplicative epsilon: avoid division-by-zero with small floor
            eps_m = eps_multiplicative(
                np.clip(nd_norm, 1e-12, None),
                np.clip(pf_norm, 1e-12, None)
            )
        except Exception as e:
            logging.warning(f"Multiplicative epsilon computation failed for {nm}: {e}")
            eps_m = np.nan

        return dict(IGD_norm=igd_val, HV=hv_val, EpsAdd=eps_a, EpsMulti=eps_m)

    def make_obj_from_model(m):
        """Return a callable objective function that evaluates the surrogate m."""
        def obj_fn(x):
            x_arr = np.atleast_2d(np.asarray(x, dtype=float))
            pred = m.predict(x_arr)
            # Ensure scalar float returned for single-point evaluation
            if np.ndim(pred) == 0:
                return float(pred)
            # if model returns array-like, take first row/element
            try:
                return float(np.asarray(pred).reshape(-1)[0])
            except Exception:
                return float(np.asarray(pred).squeeze())
        return obj_fn

    # ---------- Build maximize flags (try to query ED_MAP) ----------
    maximize_flags = [False] * no
    try:
        if nm in ED_MAP:
            try:
                inst = ED_MAP[nm]()
                maximize_flags = [
                    bool(obj.maximize[0]) if isinstance(obj.maximize, (list, np.ndarray)) else bool(obj.maximize)
                    for obj in inst.objectives
                ]
            except Exception:
                maximize_flags = [False] * no
    except Exception:
        maximize_flags = [False] * no

    # ---------- Iterate EAs ----------
    for ea_name, EAcls in [("NSGAIII", NSGAIII), ("RVEA", RVEA), ("IBEA", IBEA)]:
        logging.info(f"Starting {ea_name} on problem={nm} (nv={nv}, no={no})")

        # --- Prepare variables and constraints (same logic, simplified) ---
        vars_ = None
        constraints = None
        try:
            if nm in variable_ranges:
                bounds = variable_ranges[nm]
                if len(bounds) >= nv:
                    vars_ = [
                        Variable(f"x{i+1}", lower_bound=bounds[i][0], upper_bound=bounds[i][1],
                                 initial_value=bounds[i][0] + 0.5 * (bounds[i][1] - bounds[i][0]))
                        for i in range(nv)
                    ]
                else:
                    vars_ = [
                        Variable(f"x{i+1}", lower_bound=float(lb[i]), upper_bound=float(ub[i]),
                                 initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])))
                        for i in range(nv)
                    ]
                # attempt to fetch desdeo constraints
                try:
                    inst_fn = ED_MAP.get(nm)
                    if inst_fn is not None:
                        inst = inst_fn()
                        constraints = getattr(inst, "constraints", None)
                except Exception:
                    logging.debug(f"Could not instantiate ED_MAP[{nm}] to read constraints.")
                    constraints = None

            elif nm.upper().startswith("DBMOPP"):
                # Default variables
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                try:
                    from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
            
                    params = DBMOPP_PARAMS.get(nm, None)
            
                    if params is not None:
                        dbm = DBMOPP_generator(
                            nlp=params["n_local_pareto_regions"],
                            ndr=params["n_dominance_res_regions"],
                            ngp=params["n_global_pareto_regions"],
                            prop_constraint_checker=params.get("prop_neutral", 0.0),
                            pareto_set_type=params["pareto_set_type"],
                            constraint_type=params["constraint_type"],
                            k=no,          # number of objectives
                            n=nv           # number of decision variables
                        )
                        constraints = getattr(dbm, "constraints", None)
                    else:
                        logging.warning(f"No DBMOPP parameters found for {nm}; using empty constraints")
                        constraints = None
            
                except Exception as e:
                    logging.warning(f"Could not recreate DBMOPP for {nm} to extract constraints: {e}")
                    constraints = None

            else:
                # generic: use provided lb/ub if consistent else [0,1]
                if lb is not None and ub is not None and len(lb) == nv:
                    vars_ = [
                        Variable(f"x{i+1}", lower_bound=float(lb[i]), upper_bound=float(ub[i]),
                                 initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])))
                        for i in range(nv)
                    ]
                else:
                    vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                constraints = None
        except Exception as e:
            logging.warning(f"Error while preparing variables/constraints for {nm}: {e}\n{traceback.format_exc()}")
            if vars_ is None:
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
            constraints = None

        # --- Build ScalarObjective list from ml_list (no clipping) ---
        objs = []
        for i, m in enumerate(ml_list):
            if m is None:
                # if model missing, create a dummy objective that returns large values
                def dummy_obj(x, idx=i):
                    return 1e12
                objs.append(ScalarObjective(f"f{i+1}", dummy_obj, maximize=[bool(maximize_flags[i])]))
            else:
                objs.append(ScalarObjective(f"f{i+1}", make_obj_from_model(m), maximize=[bool(maximize_flags[i])]))

        problem = MOProblem(objectives=objs, variables=vars_, constraints=constraints)

        # --- Initialize EA with consistent population size ---
        np.random.seed(BASE_SEED)
        try:
            if ea_name == "NSGAIII":
                evo = NSGAIII(problem, population_size=POP_SIZE)
            elif ea_name == "RVEA":
                evo = RVEA(problem, population_size=POP_SIZE)
            elif ea_name == "IBEA":
                evo = IBEA(problem, population_size=POP_SIZE)
            else:
                logging.warning(f"Unknown EA {ea_name}, skipping.")
                results[ea_name] = dict(IGD_norm=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan, nd_arr=np.empty((0, no)))
                continue
        except Exception as e:
            logging.warning(f"Failed to initialize EA {ea_name} for {nm}: {e}\n{traceback.format_exc()}")
            results[ea_name] = dict(IGD_norm=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan, nd_arr=np.empty((0, no)))
            continue

        # --- Evolution loop: run exactly n_gen iterations (or break earlier if EA signals termination) ---
        iter_count = 0
        all_fronts = []
        try:
            for iter_idx in range(n_gen):
                # check EA continue flag if available
                try:
                    cont = evo.continue_evolution()
                except Exception:
                    cont = True

                if not cont:
                    logging.info(f"{ea_name} signalled termination at iter {iter_count} for {nm}.")
                    break

                # iterate once
                try:
                    evo.iterate()
                except Exception as e:
                    logging.warning(f"{ea_name} iterate failed at iter {iter_count} for {nm}: {e}\n{traceback.format_exc()}")
                    break

                iter_count += 1

                # Save intermediate PFs at requested frequency (and first iter)
                if (iter_count % save_every == 0) or (iter_count == 1):
                    try:
                        # Try extracting objectives from population
                        pop_objs = None
                        try:
                            pop_objs = np.array([ind.objectives for ind in evo.population])
                        except Exception:
                            try:
                                pop_objs = np.array(getattr(evo.population, "objectives", []))
                            except Exception:
                                pop_objs = None

                        if pop_objs is not None and pop_objs.size > 0:
                            pop_objs = np.atleast_2d(pop_objs).astype(float)
                            if pop_objs.shape[1] != no:
                                pop_objs = pop_objs.reshape(-1, no)
                            all_fronts.append(pop_objs)
                            logging.info(f"{ea_name} on {nm}: saved intermediate PF at iter={iter_count} (points={pop_objs.shape[0]})")
                        else:
                            logging.debug(f"{ea_name} on {nm}: could not extract population objectives at iter {iter_count}")
                    except Exception as e:
                        logging.warning(f"Could not extract/populate PF at iter {iter_count} for {ea_name} {nm}: {e}")

        except Exception as e:
            logging.warning(f"{ea_name} iteration loop failed for {nm}: {e}\n{traceback.format_exc()}")

        # --- Extract final solutions robustly ---
        final_arr = np.empty((0, no))
        try:
            # Some EAs provide population attribute, some provide end() return
            try:
                pop = getattr(evo, "population", None)
                if pop is not None and len(pop) > 0:
                    final_arr = np.array([ind.objectives for ind in pop], dtype=float)
                else:
                    # fallback to end()
                    end_res = None
                    try:
                        end_res = evo.end()
                    except Exception:
                        end_res = None
                    if end_res is not None:
                        # try to find a list/array of objectives in end_res
                        if isinstance(end_res, (list, tuple)) and len(end_res) > 1:
                            candidate = end_res[1]
                            try:
                                final_arr = np.vstack([np.asarray(F, dtype=float).reshape(-1) for F in candidate])
                            except Exception:
                                # last fallback: try to treat candidate as already an ndarray
                                final_arr = np.atleast_2d(np.asarray(candidate, dtype=float))
            except Exception:
                final_arr = np.empty((0, no))
        except Exception:
            logging.warning(f"Failed to extract final solutions for {ea_name} {nm}: {traceback.format_exc()}")
            final_arr = np.empty((0, no))

        if final_arr.size > 0:
            all_fronts.append(final_arr)

        # --- Combine fronts into one array and clean ---
        if len(all_fronts) > 0:
            try:
                combined = np.vstack(all_fronts)
            except Exception:
                logging.warning(f"Failed to vstack intermediate fronts for {nm} {ea_name}; falling back to final only.")
                combined = final_arr if final_arr.size > 0 else np.empty((0, no))
        else:
            combined = np.empty((0, no))

        # drop non-finite rows
        if combined.size > 0:
            mask_finite = np.isfinite(combined).all(axis=1)
            if not mask_finite.all():
                logging.warning(f"{ea_name} {nm}: dropping {np.sum(~mask_finite)} non-finite rows from combined fronts")
                combined = combined[mask_finite]

        # compute non-dominated front
        nd_arr = nondominated_filter(combined)

        # --- Get reference PF (cache/disk fallback) ---
        pf_arr = reference_pfs.get(nm, None)
        if pf_arr is None or pf_arr.size == 0:
            try:
                pf_arr = find_reference_pf(nm, no, nv)
                if pf_arr is not None and pf_arr.size > 0:
                    reference_pfs[nm] = pf_arr
                    logging.info(f"Cached reference PF for {nm} ({np.atleast_2d(pf_arr).shape})")
            except Exception as e:
                logging.warning(f"No reference PF found for {nm}: {e}")
                pf_arr = np.empty((0, no))

        # sanitize and apply maximize inversion
        pf_safe = sanitize_pf(pf_arr, no)
        nd_safe = np.atleast_2d(nd_arr) if nd_arr.size > 0 else np.empty((0, no))
        nd_safe = apply_maximize_inversion(nd_safe, maximize_flags)
        pf_safe = apply_maximize_inversion(pf_safe, maximize_flags)

        # debug info
        logging.info(f"Debug {nm} {ea_name}: ND shape={nd_safe.shape}, PF shape={pf_safe.shape}")
        if nd_safe.size > 0:
            logging.info(f"ND min={np.nanmin(nd_safe, axis=0)}, max={np.nanmax(nd_safe, axis=0)}, mean={np.nanmean(nd_safe, axis=0)}")
        if pf_safe.size > 0:
            logging.info(f"PF min={np.nanmin(pf_safe, axis=0)}, max={np.nanmax(pf_safe, axis=0)}, mean={np.nanmean(pf_safe, axis=0)}")

        # compute metrics using union-normalization
        metrics = compute_metrics(nd_safe, pf_safe)

        # store results
        results[ea_name] = dict(
            IGD_norm=metrics["IGD_norm"],
            HV=metrics["HV"],
            EpsAdd=metrics["EpsAdd"],
            EpsMulti=metrics["EpsMulti"],
            nd_arr=nd_safe
        )

    # ---------- Combined across all EAs ----------
    all_nd = [r["nd_arr"] for r in results.values() if r["nd_arr"].size > 0]
    if len(all_nd) > 0:
        combined_all = np.vstack(all_nd)
        combined_nd = nondominated_filter(combined_all)
    else:
        combined_nd = np.empty((0, no))

    # reference PF for combined
    pf_arr = reference_pfs.get(nm, None)
    if pf_arr is None or pf_arr.size == 0:
        try:
            pf_arr = find_reference_pf(nm, no, nv)
            if pf_arr is not None and pf_arr.size > 0:
                reference_pfs[nm] = pf_arr
        except Exception:
            pf_arr = np.empty((0, no))

    pf_safe = sanitize_pf(pf_arr, no)
    combined_nd = apply_maximize_inversion(combined_nd, maximize_flags)
    pf_safe = apply_maximize_inversion(pf_safe, maximize_flags)

    combined_metrics = compute_metrics(combined_nd, pf_safe)

    logging.info(
        f"Combined metrics for {nm}: IGD={combined_metrics['IGD_norm']}, HV={combined_metrics['HV']}, "
        f"EpsAdd={combined_metrics['EpsAdd']}, EpsMulti={combined_metrics['EpsMulti']}, combined_nd_shape={combined_nd.shape}"
    )

    results["Combined"] = dict(
        IGD_norm=combined_metrics["IGD_norm"],
        HV=combined_metrics["HV"],
        EpsAdd=combined_metrics["EpsAdd"],
        EpsMulti=combined_metrics["EpsMulti"],
        nd_arr=combined_nd
    )

    return results

def _append_df_to_csv(df_row, fp, cols):
    """Append a single-row dataframe to CSV safely (create header when file absent)."""
    header = not os.path.exists(fp)
    df_row.to_csv(fp, mode='a', header=header, index=False)

def build_and_train_surrogates(nm, nv, no, X, Y, algo_nm, algo_fn, noise_tag):
    ml_list = []

    # --- Clean and validate data ---
    X = X.copy()
    Y = Y.copy()
    valid_mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X = X.loc[valid_mask]
    Y = Y.loc[valid_mask]

    if len(X) < 5:
        logging.warning(f"Too few samples ({len(X)}) for {nm} - skipping surrogate training.")
        return [None] * no

    # --- Normalize inputs globally to [0, 1] for stability ---
    X_min, X_max = X.min(), X.max()
    X_scaled = (X - X_min) / (X_max - X_min + 1e-12)

    # --- Choose cross-validation folds adaptively ---
    if len(X_scaled) >= 100:
        n_splits = 5
    elif len(X_scaled) >= 30:
        n_splits = 3
    else:
        n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=BASE_SEED)

    # --- Train one surrogate per objective ---
    for obj_idx in range(no):
        best_model = None
        try:
            y = Y.iloc[:, obj_idx].astype(float)
            if y.nunique() <= 1:
                logging.warning(f"Objective f{obj_idx+1} constant for {nm}. Skipping.")
                ml_list.append(None)
                continue

            # Model pipeline: scale X, scale target internally
            model_pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('model', TransformedTargetRegressor(
                    regressor=algo_fn(),
                    check_inverse=False
                ))
            ])

            # Cross-validation evaluation
            scores = cross_validate(
                model_pipe, X_scaled, y, cv=kf,
                scoring=('r2', 'neg_mean_squared_error'),
                n_jobs=-1, error_score='raise'
            )

            # Fit final model on all data
            model_pipe.fit(X_scaled, y)
            best_model = model_pipe

            mean_r2 = float(scores['test_r2'].mean())
            std_r2 = float(scores['test_r2'].std())
            mean_mse = float(-scores['test_neg_mean_squared_error'].mean())

            # Append to performance log
            row = [nm, nv, no, algo_nm, noise_tag, f"f{obj_idx+1}", mean_r2, mean_mse]
            row_df = pd.DataFrame([row], columns=perf_cols)
            _append_df_to_csv(row_df, perf_fp, perf_cols)
            existing_perf.loc[len(existing_perf)] = row

            logging.info(
                f"{nm} | {algo_nm} | f{obj_idx+1}: "
                f"R2={mean_r2:.4f} ± {std_r2:.4f}, MSE={mean_mse:.3e}, "
                f"n_splits={n_splits}, samples={len(X_scaled)}"
            )

        except Exception as e:
            logging.warning(
                f"Surrogate training failed for {nm}, {algo_nm}, f{obj_idx+1}: {e}\n"
                f"{traceback.format_exc()}"
            )
            best_model = None

        ml_list.append(best_model)

    return ml_list

# ========================= MAIN LOOP =========================
data_root = path.join(base_folder, "Data")
n_iterations = 1
n_gen_per_iter = 50
fallback_warned = set()
reference_pfs = {}

if existing_pf is None or not isinstance(existing_pf, pd.DataFrame):
    existing_pf = pd.DataFrame(columns=pf_cols)

if existing_perf is None or not isinstance(existing_perf, pd.DataFrame):
    existing_perf = pd.DataFrame(columns=perf_cols)

# --- Define global cache for loaded reference Pareto fronts ---
reference_pfs = {}

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

                logging.info(f"Loading dataset {fpath}")
                df = pd.read_csv(fpath)
                x_cols = [c for c in df.columns if c.startswith("x")]
                y_cols = [c for c in df.columns if c.startswith("f")]
                X, Y = df[x_cols], df[y_cols]
                lb, ub = X.min(axis=0).values, X.max(axis=0).values

                # Train each surrogate algorithm for all objectives
                algos = build_algorithms(nv)
                for algo_nm, algo_fn in algos.items():
                    logging.info(f"Training surrogates for {nm} using {algo_nm}")
                    ml_list = build_and_train_surrogates(nm, nv, no, X, Y, algo_nm, algo_fn, noise_tag)
                    if any(m is None for m in ml_list):
                        logging.warning(f"Skipping EA for {nm} with {algo_nm}: some surrogates failed to train.")
                        continue

                    # --- Load and cache reference PF (once per problem) ---
                    pf_arr = find_reference_pf(nm, no, nv)
                    if pf_arr.size > 0:
                        reference_pfs[nm] = pf_arr
                        logging.info(f"Cached reference PF for {nm} ({pf_arr.shape})")
                    else:
                        logging.warning(f"No reference PF found for {nm} before EA runs.")

                    # Prepare storage per-EA over multiple iterations
                    all_nd_solutions = {"NSGAIII": [], "RVEA": [], "IBEA": [], "Combined": []}

                    # Run multiple iterations
                    for iter_idx in range(n_iterations):
                        logging.info(f"Iteration {iter_idx + 1}/{n_iterations} for {nm} with {algo_nm}")
                        try:
                            ea_results = run_all_eas(nm, nv, no, ml_list, lb, ub, n_gen=n_gen_per_iter)
                        except Exception as e:
                            logging.warning(f"run_all_eas failed on {nm} {algo_nm} iter {iter_idx}: {e}\n{traceback.format_exc()}")
                            continue

                        # Collect per-EA ND arrays
                        for ea_name, data in ea_results.items():
                            nd = data.get('nd_arr', np.empty((0, no)))
                            shape_info = nd.shape if isinstance(nd, np.ndarray) else 'None'
                            logging.info(f"Iter {iter_idx + 1} - {nm} {algo_nm} {ea_name} ND shape: {shape_info}")
                            if isinstance(nd, np.ndarray) and nd.size > 0:
                                all_nd_solutions[ea_name].append(nd)

                    # --- After iterations, process each EA's aggregated solutions ---
                    for ea_name in ["NSGAIII", "RVEA", "IBEA"]:
                        nd_list = all_nd_solutions.get(ea_name, [])
                        if len(nd_list) > 0:
                            merged = np.vstack(nd_list)
                            # non-dominated filtering
                            npts = merged.shape[0]
                            is_nd = np.ones(npts, dtype=bool)
                            for i in range(npts):
                                if not is_nd[i]:
                                    continue
                                for j in range(npts):
                                    if i == j or not is_nd[j]:
                                        continue
                                    if np.all(merged[j] <= merged[i]) and np.any(merged[j] < merged[i]):
                                        is_nd[i] = False
                                        break
                            final_nd = merged[is_nd]
                        else:
                            final_nd = np.empty((0, no))

                        # --- Compute metrics against real PF ---
                        # Try to retrieve from cache or load it once if missing
                        pf_arr = reference_pfs.get(nm, None)
                        if pf_arr is None or pf_arr.size == 0:
                            pf_arr = find_reference_pf(nm, no, nv)
                            if pf_arr.size > 0:
                                reference_pfs[nm] = pf_arr  # cache it now for future use
                                logging.info(f"Late-cached reference PF for {nm} after EA runs ({pf_arr.shape})")
                            else:
                                logging.warning(f"No reference PF found for {nm} in final aggregation — metrics will be NaN.")
                        
                        final_nd = epsilon_cleanup(final_nd, EPS_CLEAN)
                        
                        hv_val = np.nan
                        if final_nd.size > 0 and pf_arr.size > 0:
                            # remove NaNs and infs
                            if np.any(~np.isfinite(final_nd)) or np.any(~np.isfinite(pf_arr)):
                                logging.warning(f"Non-finite values detected in PF for {nm}, skipping HV normalization.")
                            else:
                                # Union-based normalization
                                combined = np.vstack((pf_arr, final_nd))
                                mins, maxs = combined.min(axis=0), combined.max(axis=0)
                                span = np.where(maxs - mins == 0, 1, maxs - mins)
                                pf_norm = (pf_arr - mins) / span
                                nd_norm = (final_nd - mins) / span
                        
                                try:
                                    ref_point = np.ones(no)  # dominates all [0,1] normalized points
                                    hv_val = hypervolume(nd_norm, ref_point)
                                except Exception as e:
                                    logging.warning(f"HV failed for {nm} {algo_nm} {ea_name}: {e}")
                        
                        igd_val = igd(final_nd, pf_arr)
                        eps_a = eps_additive(final_nd, pf_arr)
                        eps_m = eps_multiplicative(final_nd, pf_arr)
                        
                        logging.info(
                            f"Computed metrics for {nm} {algo_nm} {ea_name} -> IGD={igd_val}, HV={hv_val}, "
                            f"EpsAdd={eps_a}, EpsMulti={eps_m}, final_nd_shape={final_nd.shape}"
                        )

                        # --- Save metrics row ---
                        row = [nm, nv, no, algo_nm, ea_name, noise_tag,
                               igd_val, hv_val, eps_a, eps_m,
                               RUN_TAG]
                        row_df = pd.DataFrame([row], columns=pf_cols)

                        try:
                            _append_df_to_csv(row_df, pf_fp, pf_cols)

                            if 'existing_pf' not in locals() or existing_pf is None:
                                existing_pf = pd.DataFrame(columns=pf_cols)
                            if existing_pf.empty:
                                existing_pf = row_df.copy()
                            else:
                                existing_pf = pd.concat([existing_pf, row_df], ignore_index=True)

                            logging.info(f"Saved metrics row for {nm} {algo_nm} {ea_name}")
                        except Exception as e:
                            logging.exception(f"Failed saving metrics for {nm} {algo_nm} {ea_name}: {e}")

            except Exception as e:
                logging.exception(f"Fatal error processing file {fn}: {e}\n{traceback.format_exc()}")

logging.info("========== SCRIPT END ==========")
