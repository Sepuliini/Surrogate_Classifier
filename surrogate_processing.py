#!/usr/bin/env python3
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
POP_SIZE    = 50
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

existing_pf = load_existing(pf_fp, pf_cols)
existing_perf = load_existing(perf_fp, perf_cols)

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

def run_all_eas(nm, nv, no, ml_list, lb, ub, n_gen=N_GEN):
    """
    Run NSGAIII, RVEA, IBEA on the surrogate models for up to n_gen iterations (safety-limited).
    Returns a dict with EA names as keys and dicts containing metrics + nd_arr.
    """
    results = {}
    # Safety: allow a small multiplier in case evo uses different internal granularity
    max_iter_guard = max(1, int(n_gen * 3))

    for ea_name, EAcls in [("NSGAIII", NSGAIII), ("RVEA", RVEA), ("IBEA", IBEA)]:

        # --- Variables and bounds ---
        if nm in variable_ranges:
            bounds = variable_ranges[nm]
            vars_ = [Variable(f"x{i+1}", lower_bound=bounds[i][0], upper_bound=bounds[i][1],
                              initial_value=bounds[i][0] + 0.5*(bounds[i][1]-bounds[i][0]))
                     for i in range(nv)]
            constraints = getattr(ED_MAP.get(nm, lambda: None)(), "constraints", None)
        else:
            vars_ = [Variable(f"x{i+1}", lower_bound=lb[i], upper_bound=ub[i],
                              initial_value=lb[i] + 0.5*(ub[i]-lb[i])) for i in range(nv)]
            constraints = None

        # --- Objective functions ---
        def make_obj(m):
            return lambda x: m.predict(np.atleast_2d(x))[0]

        maximize_flags = [False]*no
        if nm in ED_MAP:
            maximize_flags = [bool(obj.maximize[0]) if isinstance(obj.maximize, (list, np.ndarray))
                              else bool(obj.maximize) for obj in ED_MAP[nm]().objectives]

        objs = [ScalarObjective(f"f{i+1}", make_obj(m), maximize=[maximize_flags[i]])
                for i, m in enumerate(ml_list)]

        problem = MOProblem(objectives=objs, variables=vars_, constraints=constraints)

        # --- Initialize EA ---
        np.random.seed(BASE_SEED)
        if ea_name == "NSGAIII":
            evo = NSGAIII(problem)
        elif ea_name == "RVEA":
            evo = RVEA(problem, population_size=POP_SIZE)
        elif ea_name == "IBEA":
            evo = IBEA(problem, population_size=POP_SIZE)
        else:
            continue

        # --- Run for up to n_gen iterations (with safety guard) ---
        iter_count = 0
        try:
            for _ in range(max_iter_guard):
                # If EA provides continue_evolution and says "stop", honor it
                try:
                    cont = evo.continue_evolution()
                except Exception:
                    cont = True  # if it errors, still try iterate but rely on guard

                if not cont:
                    logging.info(f"{ea_name} signalled termination after {iter_count} iterations.")
                    break

                # iterate and track count
                evo.iterate()
                iter_count += 1

                # log occasionally so we can see progress in logs
                if iter_count % 10 == 0 or iter_count == 1:
                    logging.info(f"{ea_name} on {nm}: iteration {iter_count}/{n_gen} (guard={max_iter_guard})")

                # stop if we reached the requested n_gen
                if iter_count >= n_gen:
                    logging.info(f"{ea_name} reached requested n_gen={n_gen}.")
                    break

        except Exception as e:
            logging.warning(f"{ea_name} iteration loop failed for {nm}: {e}\n{traceback.format_exc()}")

        # --- Extract final solutions ---
        try:
            sol = evo.end()[1]
        except Exception:
            sol = None

        if sol is None or len(sol) == 0:
            nd_arr = np.empty((0, no))
        else:
            try:
                arr = np.vstack([np.asarray(F, dtype=float).reshape(-1) for F in sol])
            except Exception:
                logging.warning(f"Failed to stack solutions for {nm} {ea_name}: {traceback.format_exc()}")
                arr = np.empty((0, no))
            # Non-dominated filtering
            if arr.size == 0:
                nd_arr = np.empty((0, no))
            else:
                npts = arr.shape[0]
                is_nd = np.ones(npts, dtype=bool)
                for i in range(npts):
                    if not is_nd[i]:
                        continue
                    for j in range(npts):
                        if i == j or not is_nd[j]:
                            continue
                        if np.all(arr[j] <= arr[i]) and np.any(arr[j] < arr[i]):
                            is_nd[i] = False
                            break
                nd_arr = arr[is_nd]

        # --- Reference PF ---
        pf = real_fronts.get((nm, no, nv)) or real_fronts.get((nm, nv, no))
        if pf is None:
            candidates = [v for k, v in real_fronts.items() if k[1] == no]
            pf_arr = np.array(candidates[0], dtype=float) if candidates else np.empty((0, no))
        else:
            pf_arr = np.array(pf, dtype=float)

        # Invert maximization
        for i, mf in enumerate(maximize_flags):
            if mf and pf_arr.size > 0:
                pf_arr[:, i] = -pf_arr[:, i]
                if nd_arr.size > 0:
                    nd_arr[:, i] = -nd_arr[:, i]

        # --- Clean duplicates and compute metrics ---
        nd_arr = epsilon_cleanup(nd_arr, EPS_CLEAN)
        hv_val = np.nan
        if nd_arr.size > 0 and pf_arr.size > 0:
            try:
                ref = pf_arr.max(axis=0) + 1
                hv_val = hypervolume(nd_arr, ref)
            except Exception as e:
                logging.warning(f"HV computation failed for {nm}, {ea_name}: {e}")
                hv_val = np.nan
        igd_val = igd(nd_arr, pf_arr)
        eps_a = eps_additive(nd_arr, pf_arr)
        eps_m = eps_multiplicative(nd_arr, pf_arr)

        # --- Store results including ND solutions ---
        results[ea_name] = dict(
            IGD_norm=igd_val,
            HV=hv_val,
            EpsAdd=eps_a,
            EpsMulti=eps_m,
            nd_arr=nd_arr
        )

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
n_iterations = 10
n_gen_per_iter = 50
fallback_warned = set()

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
                        pf = real_fronts.get((nm, no, nv)) or real_fronts.get((nm, nv, no))
                        pf_arr = np.array(pf, dtype=float) if pf is not None else np.empty((0, no))
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
                            existing_pf = pd.concat([existing_pf, row_df], ignore_index=True)
                            logging.info(f"Saved metrics row for {nm} {algo_nm} {ea_name}")
                        except Exception as e:
                            logging.exception(f"Failed saving metrics for {nm} {algo_nm} {ea_name}: {e}")

            except Exception as e:
                logging.exception(f"Fatal error processing file {fn}: {e}\n{traceback.format_exc()}")
                # continue to next file

logging.info("========== SCRIPT END ==========")
