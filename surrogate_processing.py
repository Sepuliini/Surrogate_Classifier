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

try:
    from pymoo.indicators.hv import HV
    def hypervolume(A, ref_point):
        return HV(ref_point=ref_point).do(A)
except ImportError:
    def hypervolume(A, ref_point):
        return np.nan

from desdeo_problem import Variable, ScalarObjective, MOProblem
from desdeo_emo.EAs import NSGAIII, RVEA, IBEA

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
POP_SIZE    = 10
N_GEN       = 10
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

def eps_multiplicative(A: np.ndarray, R: np.ndarray, tiny=1e-12) -> float:
    if A.size == 0 or R.size == 0: return np.nan
    d = [np.min(np.max(A / (r[np.newaxis, :] + tiny), axis=1)) for r in R]
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
    
def find_reference_pf(nm: str, no: int, nv: int):
    nm = nm.lower()
    for k, v in real_fronts.items():
        if k[0].lower() == nm and int(k[1]) == int(no):
            return np.array(v, dtype=float)
    return np.empty((0, no))

def run_all_eas(nm, nv, no, ml_list, lb, ub, n_gen=N_GEN, save_every=10):
    """
    Run NSGAIII, RVEA, IBEA on the surrogate models for up to n_gen iterations (safety-limited).
    Periodically saves intermediate Pareto fronts in memory (every `save_every` iterations),
    then combines them (per EA) into a final non-dominated front and computes metrics.

    Returns a dict with EA names as keys and dicts containing metrics + nd_arr.
    """
    results = {}
    max_iter_guard = max(1, int(n_gen * 3))

    # helper to compute non-dominated filter on a 2D array
    def nondominated_filter(arr):
        if arr.size == 0:
            return np.empty((0, arr.shape[1] if arr.ndim == 2 else 0))
        npts = arr.shape[0]
        is_nd = np.ones(npts, dtype=bool)
        for i in range(npts):
            if not is_nd[i]:
                continue
            for j in range(npts):
                if i == j or not is_nd[j]:
                    continue
                # j dominates i?
                if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                    is_nd[i] = False
                    break
        return arr[is_nd]

    for ea_name, EAcls in [("NSGAIII", NSGAIII), ("RVEA", RVEA), ("IBEA", IBEA)]:
        logging.info(f"Starting {ea_name} on problem={nm} (nv={nv}, no={no})")

        # --- Variables and constraints selection ---
        constraints = None
        vars_ = None

        try:
            # Engineering problems with explicit variable ranges & desdeo instances
            if nm in variable_ranges:
                bounds = variable_ranges[nm]
                # ensure bounds length matches nv (if mismatch, fall back to lb/ub or 0..1)
                if len(bounds) >= nv:
                    vars_ = [
                        Variable(f"x{i+1}",
                                 lower_bound=bounds[i][0],
                                 upper_bound=bounds[i][1],
                                 initial_value=bounds[i][0] + 0.5 * (bounds[i][1] - bounds[i][0]))
                        for i in range(nv)
                    ]
                else:
                    # fallback to numeric lb/ub passed in
                    vars_ = [
                        Variable(f"x{i+1}", lower_bound=float(lb[i]), upper_bound=float(ub[i]),
                                 initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])))
                        for i in range(nv)
                    ]
                # try to fetch desdeo constraints if available
                try:
                    inst_fn = ED_MAP.get(nm)
                    if inst_fn is not None:
                        inst = inst_fn()
                        constraints = getattr(inst, "constraints", None)
                except Exception:
                    logging.debug(f"Could not instantiate ED_MAP[{nm}] to read constraints.")
                    constraints = getattr(ED_MAP.get(nm, lambda: None)(), "constraints", None)

            # DBMOPP problems: recreate a DBMOPP_generator to extract constraints
            elif nm.upper().startswith("DBMOPP"):
                # DBMOPP uses decision variables in [0,1]
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                try:
                    from desdeo_problem.testproblems.DBMOPP.DBMOPP_generator import DBMOPP_generator
                    # attempt to create a DBMOPP instance with minimal params; this will provide constraints attr if generator supports it
                    try:
                        dbm = DBMOPP_generator(k=no, n=nv, nm=1000)
                    except TypeError:
                        # fallback if generator requires different signature
                        dbm = DBMOPP_generator(no, nv)
                    constraints = getattr(dbm, "constraints", None)
                    logging.info(f"DBMOPP recreated for {nm}: constraints loaded={constraints is not None}")
                except Exception as e:
                    constraints = None
                    logging.warning(f"Could not recreate DBMOPP for {nm} to extract constraints: {e}")

            # Benchmarks (DTLZ/WFG) or others: use lb/ub if provided else assume [0,1]
            else:
                # if lb/ub are arrays (from dataset) use them, otherwise default [0,1]
                try:
                    if lb is not None and ub is not None and len(lb) == nv:
                        vars_ = [
                            Variable(f"x{i+1}", lower_bound=float(lb[i]), upper_bound=float(ub[i]),
                                     initial_value=float(lb[i] + 0.5 * (ub[i] - lb[i])))
                            for i in range(nv)
                        ]
                    else:
                        vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                except Exception:
                    vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
                constraints = None

        except Exception as e:
            logging.warning(f"Error while preparing variables/constraints for {nm}: {e}\n{traceback.format_exc()}")
            # ensure we have a vars_ fallback
            if vars_ is None:
                vars_ = [Variable(f"x{i+1}", lower_bound=0.0, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
            constraints = None

        # --- Objective wrappers (surrogates) ---
        def make_obj(m):
            return lambda x: m.predict(np.atleast_2d(x))[0]

        maximize_flags = [False] * no
        try:
            if nm in ED_MAP:
                maximize_flags = [
                    bool(obj.maximize[0]) if isinstance(obj.maximize, (list, np.ndarray))
                    else bool(obj.maximize)
                    for obj in ED_MAP[nm]().objectives
                ]
        except Exception:
            # if ED_MAP item fails, keep default False flags
            maximize_flags = [False] * no

        objs = [ScalarObjective(f"f{i+1}", make_obj(m), maximize=[maximize_flags[i]])
                for i, m in enumerate(ml_list)]

        problem = MOProblem(objectives=objs, variables=vars_, constraints=constraints)

        # --- Initialize EA instance ---
        np.random.seed(BASE_SEED)
        try:
            if ea_name == "NSGAIII":
                evo = NSGAIII(problem)
            elif ea_name == "RVEA":
                evo = RVEA(problem, population_size=POP_SIZE)
            elif ea_name == "IBEA":
                evo = IBEA(problem, population_size=POP_SIZE)
            else:
                logging.warning(f"Unknown EA {ea_name}, skipping.")
                continue
        except Exception as e:
            logging.warning(f"Failed to initialize EA {ea_name} for {nm}: {e}\n{traceback.format_exc()}")
            results[ea_name] = dict(IGD_norm=np.nan, HV=np.nan, EpsAdd=np.nan, EpsMulti=np.nan, nd_arr=np.empty((0, no)))
            continue

        # --- Evolution loop and intermediate PF saving ---
        iter_count = 0
        all_fronts = []  # list of arrays (each saved PF)
        try:
            for _ in range(max_iter_guard):
                try:
                    cont = evo.continue_evolution()
                except Exception:
                    cont = True

                if not cont:
                    logging.info(f"{ea_name} signalled termination at iter {iter_count} for {nm}.")
                    break

                evo.iterate()
                iter_count += 1

                # Save an intermediate PF (population objectives) every save_every iters and at iter 1
                if iter_count % save_every == 0 or iter_count == 1:
                    try:
                        # evo.population may be a list of individuals; extract objectives robustly
                        pop_objs = None
                        try:
                            # try attribute access first (desdeo individuals often have .objectives)
                            pop_objs = np.array([ind.objectives for ind in evo.population])
                        except Exception:
                            try:
                                # maybe evo.population.objectives exists or population is a container
                                pop_objs = np.array(getattr(evo.population, "objectives", []))
                            except Exception:
                                pop_objs = None

                        if pop_objs is None or pop_objs.size == 0:
                            logging.debug(f"{ea_name} on {nm}: could not extract population objectives at iter {iter_count}")
                        else:
                            # ensure shape (n_points, n_objectives)
                            pop_objs = np.atleast_2d(pop_objs).astype(float)
                            if pop_objs.shape[1] != no:
                                # sometimes objectives may be nested differently; try reshape
                                pop_objs = pop_objs.reshape(-1, no)
                            all_fronts.append(pop_objs)
                            logging.info(f"{ea_name} on {nm}: saved intermediate PF at iter={iter_count} (points={pop_objs.shape[0]})")
                    except Exception as e:
                        logging.warning(f"Could not extract/populate PF at iter {iter_count} for {ea_name} {nm}: {e}")

                # stop when we've reached requested n_gen (not guard)
                if iter_count >= n_gen:
                    logging.info(f"{ea_name} reached requested n_gen={n_gen} (iter_count={iter_count}).")
                    break

        except Exception as e:
            logging.warning(f"{ea_name} iteration loop failed for {nm}: {e}\n{traceback.format_exc()}")

        # --- Extract final solutions from the EA to include in saved fronts ---
        try:
            sol = evo.end()[1]
        except Exception:
            sol = None

        if sol is None or len(sol) == 0:
            final_arr = np.empty((0, no))
        else:
            try:
                final_arr = np.vstack([np.asarray(F, dtype=float).reshape(-1) for F in sol])
            except Exception:
                logging.warning(f"Failed to stack final solutions for {nm} {ea_name}: {traceback.format_exc()}")
                final_arr = np.empty((0, no))

        if final_arr.size > 0:
            all_fronts.append(final_arr)

        # --- Combine fronts and filter non-dominated ---
        if len(all_fronts) > 0:
            try:
                combined = np.vstack(all_fronts)
            except Exception:
                logging.warning(f"Failed to vstack intermediate fronts for {nm} {ea_name}; falling back to final only.")
                combined = final_arr if final_arr.size > 0 else np.empty((0, no))
        else:
            combined = np.empty((0, no))

        # --- Basic sanity/logging for combined front before ND filter ---
        try:
            logging.debug(f"{ea_name} {nm}: combined shape before clean = {combined.shape}")
            if combined.size > 0:
                # drop any rows that contain NaN/Inf
                mask_finite = np.isfinite(combined).all(axis=1)
                if not mask_finite.all():
                    logging.warning(f"{ea_name} {nm}: dropping {np.sum(~mask_finite)} non-finite rows from combined fronts")
                    combined = combined[mask_finite]
                logging.debug(f"{ea_name} {nm}: combined stats min={np.nanmin(combined, axis=0)}, max={np.nanmax(combined, axis=0)}, mean={np.nanmean(combined, axis=0)}")
        except Exception as e:
            logging.warning(f"Debugging combined front failed: {e}")

        # Now filter non-dominated (minimization assumed)
        def nondominated_filter_safe(arr):
            if arr.size == 0:
                return np.empty((0, no))
            npts = arr.shape[0]
            is_nd_local = np.ones(npts, dtype=bool)
            for i in range(npts):
                if not is_nd_local[i]:
                    continue
                for j in range(npts):
                    if i == j or not is_nd_local[j]:
                        continue
                    # j dominates i?
                    try:
                        if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                            is_nd_local[i] = False
                            break
                    except Exception:
                        continue
            return arr[is_nd_local]

        nd_arr = nondominated_filter_safe(combined)

        # --- Reference PF selection (cache then disk fallback) ---
        pf_arr = find_reference_pf(nm, no, nv)
        
        # dynamic disk load fallback (keeps your previous paths)
        if pf_arr.size == 0:
            try:
                base = "/scratch/project_2014748/modelling_results_all_data/real_paretofronts"
                fpath = None
                nm_lower = nm.lower()
                if nm_lower.startswith("wfg"):
                    fpath = os.path.join(base, "WFG", nm, f"pareto_front_{no}obj_30vars.csv")
                elif nm_lower.startswith("dtlz"):
                    fpath = os.path.join(base, "DTLZ", nm, f"real_pareto_front_{nm}_{no}obj.txt")
                elif nm_lower.startswith("dbmopp"):
                    fpath = os.path.join(base, "DBMOPP", f"{nm}_real_pareto_front_{no}obj.csv")
                elif nm_lower.startswith("re"):
                    fpath = os.path.join(base, "engineering", f"{nm}_NSGAIII_combinedSeeds.csv")
        
                if fpath and os.path.exists(fpath):
                    pf_arr = np.loadtxt(fpath, delimiter=",")
                    logging.info(f"Loaded reference PF for {nm} from disk ({pf_arr.shape})")
                else:
                    logging.warning(f"No reference PF found for {nm}. Metrics will be NaN.")
            except Exception as e:
                logging.warning(f"Failed to load reference PF for {nm}: {e}")
        
        # --- Sanitize reference PF ---
        if pf_arr.size > 0:
            pf_arr = np.atleast_2d(pf_arr).astype(float)
            pf_mask = np.isfinite(pf_arr).all(axis=1)
            if not pf_mask.all():
                logging.warning(f"{nm}: dropping {np.sum(~pf_mask)} non-finite rows from reference PF")
                pf_arr = pf_arr[pf_mask]
            logging.debug(f"{nm}: ref PF stats min={np.nanmin(pf_arr, axis=0)}, max={np.nanmax(pf_arr, axis=0)}")


        # --- Apply maximize flag inversion consistently (minimization assumed by EAs) ---
        try:
            for i, mf in enumerate(maximize_flags):
                if mf:
                    if pf_arr.size > 0:
                        pf_arr[:, i] = -pf_arr[:, i]
                    if nd_arr.size > 0:
                        nd_arr[:, i] = -nd_arr[:, i]
        except Exception as e:
            logging.warning(f"Failed applying maximize inversion for {nm}: {e}")

        # --- final cleaning of nd_arr before metrics ---
        if nd_arr.size > 0:
            # drop any non-finite after inversion
            nd_mask = np.isfinite(nd_arr).all(axis=1)
            if not nd_mask.all():
                logging.warning(f"{nm}: dropping {np.sum(~nd_mask)} non-finite rows from ND arr")
                nd_arr = nd_arr[nd_mask]

        # --- Protective measures for multiplicative eps and hypervolume ---
        # Avoid division by zero in multiplicative epsilon by enforcing a tiny floor on reference values
        pf_safe = pf_arr.copy() if pf_arr.size > 0 else np.empty((0, no))
        if pf_safe.size > 0:
            # floor reference values (only positive floor); keep sign if there are negatives
            tiny = 1e-12
            pf_safe = np.where(np.abs(pf_safe) < PF_RANGE_FLOOR, np.sign(pf_safe) * PF_RANGE_FLOOR, pf_safe)

        hv_val = np.nan
        igd_val = np.nan
        eps_a = np.nan
        eps_m = np.nan

        # HV: compute only if we have both sets and no degenerate issues
        if nd_arr.size > 0 and pf_safe.size > 0:
            try:
                # make reference that strictly dominates both reference and nd points
                ref_candidate = np.max(np.vstack([pf_safe, nd_arr]), axis=0) * 1.1
                # ensure ref > nd_arr for all dims
                if np.any(ref_candidate <= np.max(nd_arr, axis=0)):
                    ref_candidate = np.max(np.vstack([pf_safe, nd_arr]), axis=0) + 1.0
                hv_val = hypervolume(nd_arr, ref_candidate)
            except Exception as e:
                logging.warning(f"HV computation failed for {nm}, {ea_name}: {e}")
                hv_val = np.nan

            # IGD and epsilons
            try:
                igd_val = igd(nd_arr, pf_safe)
            except Exception as e:
                logging.warning(f"IGD failed for {nm}, {ea_name}: {e}")
                igd_val = np.nan

            try:
                eps_a = eps_additive(nd_arr, pf_safe)
            except Exception as e:
                logging.warning(f"EpsAdd failed for {nm}, {ea_name}: {e}")
                eps_a = np.nan

            try:
                eps_m = eps_multiplicative(nd_arr, pf_safe)
            except Exception as e:
                logging.warning(f"EpsMulti failed for {nm}, {ea_name}: {e}")
                eps_m = np.nan
        else:
            logging.warning(f"Skipping metrics for {nm} {ea_name}: nd_arr={nd_arr.shape}, pf_arr={pf_arr.shape}")

        logging.info(f"Reference PF for {nm}: shape={pf_arr.shape}, ND shape={nd_arr.shape}")
        logging.info(f"{ea_name} done on {nm}: IGD={igd_val}, HV={hv_val}, EpsAdd={eps_a}, EpsMulti={eps_m}, nd_pts={nd_arr.shape[0]}")


        # --- Store results ---
        results[ea_name] = dict(
            IGD_norm=igd_val,
            HV=hv_val,
            EpsAdd=eps_a,
            EpsMulti=eps_m,
            nd_arr=nd_arr
        )


        logging.info(f"{ea_name} done on {nm}: IGD={igd_val}, HV={hv_val}, EpsAdd={eps_a}, EpsMulti={eps_m}, nd_pts={nd_arr.shape[0]}")

    return results




def _append_df_to_csv(df_row, fp, cols):
    """Append a single-row dataframe to CSV safely (create header when file absent)."""
    header = not os.path.exists(fp)
    df_row.to_csv(fp, mode='a', header=header, index=False)


def build_and_train_surrogates(nm, nv, no, X, Y, algo_nm, algo_fn, noise_tag):
    """
    Train surrogates for all objectives for a given algorithm and return trained models.
    Immediately saves R2 and MSE to surrogate_perf.csv (append-mode).
    """
    ml_list = []
    for obj_idx in range(no):
        best_model = None
        try:
            model_pipe = Pipeline([
                ('scale', StandardScaler()),
                ('model', TransformedTargetRegressor(regressor=algo_fn()))
            ])
            kf = KFold(n_splits=3, shuffle=True, random_state=BASE_SEED)
            scores = cross_validate(model_pipe, X, Y.iloc[:, obj_idx], cv=kf,
                                    scoring=('r2','neg_mean_squared_error'))
            model_pipe.fit(X, Y.iloc[:, obj_idx])
            best_model = model_pipe

            # Save surrogate performance immediately (append)
            row = [nm, nv, no, algo_nm, noise_tag, f"f{obj_idx+1}",
                   float(scores['test_r2'].mean()), float(-scores['test_neg_mean_squared_error'].mean())]
            row_df = pd.DataFrame([row], columns=perf_cols)

            # Append to file
            _append_df_to_csv(row_df, perf_fp, perf_cols)

            # Keep in-memory copy to avoid re-saving duplicates in same run
            existing_perf.loc[len(existing_perf)] = row

            logging.info(f"Saved perf for {nm} {algo_nm} f{obj_idx+1}: R2={row[6]:.4f} MSE={row[7]:.4g}")

        except Exception as e:
            logging.warning(f"Surrogate training failed for {nm}, {algo_nm}, f{obj_idx+1}: {e}\n{traceback.format_exc()}")
            best_model = None

        ml_list.append(best_model)

    return ml_list


# ========================= MAIN LOOP =========================
data_root = path.join(base_folder, "Data")
n_iterations = 1
n_gen_per_iter = 50
fallback_warned = set()

# Make sure we’re working with the global objects throughout
# global existing_pf, existing_perf

if existing_pf is None or not isinstance(existing_pf, pd.DataFrame):
    existing_pf = pd.DataFrame(columns=pf_cols)

if existing_perf is None or not isinstance(existing_perf, pd.DataFrame):
    existing_perf = pd.DataFrame(columns=perf_cols)

for suite in ["DTLZ","WFG","Engineering","DBMOPP"]:
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

                    # Prepare storage per-EA over multiple iterations
                    all_nd_solutions = { "NSGAIII": [], "RVEA": [], "IBEA": [] }

                    # Run multiple iterations: call run_all_eas once per iteration (it returns all 3 EAs)
                    for iter_idx in range(n_iterations):
                        logging.info(f"Iteration {iter_idx+1}/{n_iterations} for {nm} with {algo_nm}")
                        try:
                            ea_results = run_all_eas(nm, nv, no, ml_list, lb, ub, n_gen=n_gen_per_iter)
                        except Exception as e:
                            logging.warning(f"run_all_eas failed on {nm} {algo_nm} iter {iter_idx}: {e}\n{traceback.format_exc()}")
                            continue

                        # Log sizes and collect per-ea nd arrays
                        for ea_name, data in ea_results.items():
                            nd = data.get('nd_arr', np.empty((0,no)))
                            shape_info = nd.shape if isinstance(nd, np.ndarray) else 'None'
                            logging.info(f"Iter {iter_idx+1} - {nm} {algo_nm} {ea_name} ND shape: {shape_info}")
                            if isinstance(nd, np.ndarray) and nd.size > 0:
                                all_nd_solutions[ea_name].append(nd)

                    # After iterations, process each EA's aggregated solutions
                    for ea_name in ["NSGAIII", "RVEA", "IBEA"]:
                        nd_list = all_nd_solutions.get(ea_name, [])
                        if len(nd_list) > 0:
                            merged = np.vstack(nd_list)
                            # non-dominated filter
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
                        pf_arr = find_reference_pf(nm, no, nv)
                        final_nd = epsilon_cleanup(final_nd, EPS_CLEAN)

                        hv_val = np.nan
                        if final_nd.size > 0 and pf_arr.size > 0:
                            try:
                                hv_val = hypervolume(final_nd, pf_arr.max(axis=0)+1)
                            except Exception as e:
                                logging.warning(f"HV failed for {nm} {algo_nm} {ea_name}: {e}")

                        igd_val = igd(final_nd, pf_arr)
                        eps_a = eps_additive(final_nd, pf_arr)
                        eps_m = eps_multiplicative(final_nd, pf_arr)

                        logging.info(f"Computed metrics for {nm} {algo_nm} {ea_name} -> IGD={igd_val}, HV={hv_val}, EpsAdd={eps_a}, EpsMulti={eps_m}, final_nd_shape={final_nd.shape}")

                        # --- Save single row for this EA (append-mode) ---
                        row = [nm, nv, no, algo_nm, ea_name, noise_tag,
                               igd_val, hv_val, eps_a, eps_m,
                               np.nan, np.nan, np.nan, RUN_TAG]
                        row_df = pd.DataFrame([row], columns=pf_cols)

                        try:
                            _append_df_to_csv(row_df, pf_fp, pf_cols)

                            # Make sure existing_pf is defined in locals (safety)
                            if 'existing_pf' not in locals() or existing_pf is None:
                                existing_pf = pd.DataFrame(columns=pf_cols)

                            # Safe concat: if existing_pf is empty create copy, else concat
                            if existing_pf.empty:
                                existing_pf = row_df.copy()
                            else:
                                existing_pf = pd.concat([existing_pf, row_df], ignore_index=True)

                            logging.info(f"Saved metrics row for {nm} {algo_nm} {ea_name}")
                        except Exception as e:
                            logging.exception(f"Failed saving metrics for {nm} {algo_nm} {ea_name}: {e}")

            except Exception as e:
                logging.exception(f"Fatal error processing file {fn}: {e}\n{traceback.format_exc()}")
                # continue to next file

logging.info("========== SCRIPT END ==========")
