#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from os import path, makedirs, walk
from datetime import datetime
import logging
import re
import argparse

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
from desdeo_emo.EAs.NSGAIII import NSGAIII
from desdeo_emo.EAs.IBEA import IBEA
from desdeo_emo.EAs.RVEA import RVEA

from desdeo_problem.testproblems.EngineeringRealWorld import (
    re21, re22, re23, re24, re25, re31, re32, re33
)
from desdeo_problem.testproblems.MultipleClutchBrakes import multiple_clutch_brakes
from desdeo_problem.testproblems.RiverPollution import river_pollution_problem
from desdeo_problem.testproblems.VehicleCrashworthiness import vehicle_crashworthiness
from desdeo_problem.testproblems.CarSideImpact import car_side_impact
from optproblems import wfg

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

# ----- Engineering variable bounds -----
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

# ----- Paths & Logging -----
base_folder = '/scratch/project_2014748'
log_dir = path.join(base_folder, "logs")
makedirs(log_dir, exist_ok=True)

task_id = os.getenv("SLURM_ARRAY_TASK_ID", str(os.getpid()))
log_file = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_task{task_id}.log"
logging.basicConfig(
    filename=path.join(log_dir, log_file),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
logging.getLogger().addHandler(console)
logging.captureWarnings(True)
logging.info("========== SCRIPT START ==========")

# ----- Output Columns -----
pf_cols = [
    'Problem','VarCount','ObjCount','Algorithm','EA','NoiseTag',
    'IGD','AddEps','MultEps','HV',
    'IGD_union','AddEps_union','MultEps_union','HV_union'
]
perf_cols = ['Problem','VarCount','ObjCount','Algorithm','NoiseTag','Objective','R2','MSE']

# ----- Incremental Output Setup (Moved up!) -----
out_dir = path.join(base_folder, 'modelling_results')
makedirs(out_dir, exist_ok=True)
pf_fp = path.join(out_dir, 'surrogate_metrics.csv')
perf_fp = path.join(out_dir, 'surrogate_perf.csv')
if path.exists(pf_fp):
    existing_pf = pd.read_csv(pf_fp)
    existing_pf = existing_pf.drop_duplicates(subset=['Problem','VarCount','ObjCount','Algorithm','EA','NoiseTag'])
else:
    existing_pf = pd.DataFrame(columns=pf_cols)
existing_perf = pd.read_csv(perf_fp) if path.exists(perf_fp) else pd.DataFrame(columns=perf_cols)

# ----- Metrics Definitions -----
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def igd(A, R):
    return np.mean([np.min([euclidean_distance(r, a) for a in A]) for r in R])

# --- FIXED Additive Epsilon (Zitzler et al., for minimization): max over R of min over A of max(r - a)
def additive_epsilon_indicator(R, A):
    eps = -np.inf
    for r in R:
        eps_j = np.inf
        for a in A:
            eps_j = min(eps_j, np.max(r - a))
        eps = max(eps, eps_j)
    return eps

# --- FIXED Multiplicative Epsilon (for minimization): max over R of min over A of max(r/(a+eps))
def multiplicative_epsilon_indicator(R, A):
    eps = -np.inf
    for r in R:
        eps_j = np.inf
        for a in A:
            factors = []
            for ai, ri in zip(a, r):
                if ai == 0:
                    factors.append(np.inf if ri > 0 else 1.0)
                else:
                    factors.append(ri / (ai + 1e-12))
            eps_j = min(eps_j, max(factors))
        eps = max(eps, eps_j)
    return eps

def to_minimization(F, maximize_flags):
    F_min = F.copy()
    for i, maxim in enumerate(maximize_flags):
        if maxim:
            F_min[:, i] = -F_min[:, i]
    return F_min

# ----- Build Algorithms -----
def build_algorithms(nv):
    return {
        "SVM": svm.SVR,
        "NN": lambda: MLPRegressor(max_iter=1000, tol=1e-4),
        "Ada": ensemble.AdaBoostRegressor,
        "GPR": GaussianProcessRegressor,
        "SGD": SGD,
        "KNR": KNR,
        "DTR": DTR,
        "RFR": ensemble.RandomForestRegressor,
        "ExTR": ensemble.ExtraTreesRegressor,
        "GBR": ensemble.GradientBoostingRegressor,
        "XGB": XGBRegressor
    }

# ----- Engineering Problem Map -----
ED_MAP = {
    're21': re21, 're22': re22, 're23': re23, 're32': re32,
    'river_pollution_problem': river_pollution_problem,
    'vehicle_crashworthiness': vehicle_crashworthiness
}

# ----- File Details Extraction -----
pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)var_(\d+)obj_(\d+)samples(?:_([A-Za-z]+))?\.csv$")
eng_pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)_samples(?:_([A-Za-z]+))?\.csv$")

def extract_details(fn):
    m = pattern.match(fn)
    if m:
        name, nv, no, samples = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        noise_tag = m.group(5) or 'none'
        return name, nv, no, samples, noise_tag
    m2 = eng_pattern.match(fn)
    if m2:
        name, samples = m2.group(1), int(m2.group(2))
        noise_tag = m2.group(3) or 'none'
        inst = ED_MAP.get(name)
        if inst:
            prob = inst()
            return name, len(prob.variables), len(prob.objectives), samples, noise_tag
    return None, None, None, None, None

# ----- Real PF Loading -----
real_path = path.join(base_folder, "modelling_results_all_data", "real_paretofronts")
real_fronts = {}
logging.info(f"Loading reference PFs from {real_path}")

def load_pf_files(folder, key_prefix, wfg_special=False):
    for subdir, _, files in os.walk(folder):
        for fname in files:
            if not fname.lower().endswith(('.csv', '.txt')):
                continue
            fpath = path.join(subdir, fname)
            arr = pd.read_csv(fpath, header=None).values
            if arr.ndim != 2:
                continue
            no = arr.shape[1]
            if wfg_special:
                m = re.search(r'_(\d+)obj_(\d+)vars', fname)
                if m:
                    obj = int(m.group(1))
                    vars_ = int(m.group(2))
                    key = (key_prefix, obj, vars_)
                else:
                    key = (key_prefix, no, None)
            else:
                if key_prefix == 'engineering':
                    prob_name = os.path.basename(os.path.dirname(fpath))
                else:
                    prob_name = key_prefix
                key = (prob_name, no)
            if key not in real_fronts or arr.shape[0] > real_fronts[key].shape[0]:
                real_fronts[key] = arr
                logging.info(f"Loaded PF for {key}: {fpath}")

eng_path = path.join(real_path, "engineering")
if path.exists(eng_path):
    load_pf_files(eng_path, 'engineering')

dtlz_path = path.join(real_path, "DTLZ")
if path.exists(dtlz_path):
    for prob_name in os.listdir(dtlz_path):
        prob_dir = path.join(dtlz_path, prob_name)
        if not path.isdir(prob_dir):
            continue
        load_pf_files(prob_dir, prob_name)

wfg_path = path.join(real_path, "WFG")
if path.exists(wfg_path):
    for prob_name in os.listdir(wfg_path):
        prob_dir = path.join(wfg_path, prob_name)
        if not path.isdir(prob_dir):
            continue
        load_pf_files(prob_dir, prob_name, wfg_special=True)

logging.info(f"Available PF keys: {list(real_fronts.keys())}")

# ----- Run EA and PF Quality -----
def run_ea(nm, nv, no, algo_nm, ea_nm, noise_tag, mlist, ea_params, X, lower_bounds, upper_bounds):
    mask = (
        (existing_pf['Problem'] == nm) &
        (existing_pf['VarCount'] == nv) &
        (existing_pf['ObjCount'] == no) &
        (existing_pf['Algorithm'] == algo_nm) &
        (existing_pf['EA'] == ea_nm) &
        (existing_pf['NoiseTag'] == noise_tag)
    )
    if mask.any():
        logging.info(f"Skipping duplicate PF metrics for {nm} {nv} {no} {algo_nm} {ea_nm} {noise_tag}")
        return

    # Set maximize flags
    if nm in ED_MAP:
        maximize_flags = [obj.maximize[0] if isinstance(obj.maximize, (list, np.ndarray)) else obj.maximize
                          for obj in ED_MAP[nm]().objectives]
    else:
        maximize_flags = [False] * no

    noise_str = noise_tag if (noise_tag is not None and str(noise_tag).strip() != "") else "none"

    # Use bounds and constraints
    if nm in variable_ranges:
        bounds = variable_ranges[nm]
        vars_ = [Variable(f"x_{i+1}",
                          lower_bound=bounds[i][0],
                          upper_bound=bounds[i][1],
                          initial_value=bounds[i][0] + 0.5*(bounds[i][1]-bounds[i][0]))
                 for i in range(nv)]
        real_problem = ED_MAP[nm]()
        constraints = getattr(real_problem, "constraints", None)
    else:
        vars_ = [Variable(f"x{i+1}",
                          lower_bound=lower_bounds[i],
                          upper_bound=upper_bounds[i],
                          initial_value=lower_bounds[i] + 0.5*(upper_bounds[i]-lower_bounds[i]))
                 for i in range(nv)]
        constraints = None

    def make_obj(m):
        def obj(x):
            xa = np.atleast_2d(x)      # now xa.shape == (1, nv)
            return m.predict(xa)[0]    # get a scalar back
        return obj

    objs = [
        ScalarObjective(f"f{i+1}", make_obj(m), maximize=[maximize_flags[i]])
        for i, m in enumerate(mlist)
    ]
    
    problem = MOProblem(objectives=objs, variables=vars_, constraints=constraints)

    evo = ea_params.pop('__cls__')(problem, **ea_params)
    while evo.continue_evolution():
        evo.iterate()
    sol = evo.end()[1]
    nd = np.empty((0, no)) if sol is None else np.vstack([
        F for F in sol if not any((G <= F).all() and (G < F).any() for G in sol if not np.array_equal(G, F))
    ])

    # Save approximated PF
    approx_pfs_dir = path.join(base_folder, 'modelling_results', 'approximated_pfs')
    os.makedirs(approx_pfs_dir, exist_ok=True)
    pf_fname = f"PROBLEM_{nm}_ALG_{algo_nm}_EA_{ea_nm}_NOISE_{noise_str}_VARS_{nv}_OBJS_{no}.txt"
    pf_fpath = path.join(approx_pfs_dir, pf_fname)
    if nd.size > 0:
        np.savetxt(pf_fpath, nd, delimiter=",")
        logging.info(f"Saved approximated PF to {pf_fpath}")

    # PF lookup (with closest var count for WFG)
    pf = None
    if nm in ED_MAP:
        pf = real_fronts.get((nm, no))
    elif nm.startswith("WFG"):
        candidates = [k for k in real_fronts if isinstance(k, tuple) and k[0] == nm and k[1] == no and len(k) == 3]
        if candidates:
            best_key = min(candidates, key=lambda k: abs(k[2] - nv) if k[2] is not None else 1e9)
            pf = real_fronts[best_key]
            logging.info(f"WFG: Using PF key {best_key} for {nm} (vars={nv}, objs={no})")
        else:
            possible_pfs = [k for k in real_fronts if k[0] == nm and k[1] == no]
            if possible_pfs:
                pf = real_fronts[possible_pfs[0]]
    else:
        pf = real_fronts.get((nm, no))

    # --- Metrics for both normalizations ---
    if pf is None or nd.size == 0:
        logging.warning(f"Skipping metrics: No PF found or approximated PF empty for {nm} (objs={no})")
        igd_val = eps_add = eps_mul = hv_val = np.nan
        igd_val_union = eps_add_union = eps_mul_union = hv_val_union = np.nan
    elif pf.shape[1] != nd.shape[1]:
        logging.error(
            f"Objective count mismatch for {nm}: real PF has {pf.shape[1]}, "
            f"approx PF has {nd.shape[1]}. Skipping metrics."
        )
        igd_val = eps_add = eps_mul = hv_val = np.nan
        igd_val_union = eps_add_union = eps_mul_union = hv_val_union = np.nan
    else:
        # --- (A) Normalize by real PF min/max ---
        pf_min = pf.min(axis=0)
        pf_max = pf.max(axis=0)
        range_ = np.where(pf_max > pf_min, pf_max - pf_min, 1)
        pf_s = (pf - pf_min) / range_
        nd_s = (nd - pf_min) / range_
        nd_s_clipped = np.clip(nd_s, 0, 1)
        igd_val = igd(nd_s, pf_s)
        eps_add = additive_epsilon_indicator(pf_s, nd_s)
        eps_mul = multiplicative_epsilon_indicator(pf_s, nd_s)
        hv_val = hypervolume(nd_s_clipped, ref_point=np.ones(no))

        # --- (B) Normalize by union min/max (across real + approx PF) ---
        union_min = np.minimum(pf.min(axis=0), nd.min(axis=0))
        union_max = np.maximum(pf.max(axis=0), nd.max(axis=0))
        union_range = np.where(union_max > union_min, union_max - union_min, 1)
        pf_u = (pf - union_min) / union_range
        nd_u = (nd - union_min) / union_range
        nd_u_clipped = np.clip(nd_u, 0, 1)
        igd_val_union = igd(nd_u, pf_u)
        eps_add_union = additive_epsilon_indicator(pf_u, nd_u)
        eps_mul_union = multiplicative_epsilon_indicator(pf_u, nd_u)
        hv_val_union = hypervolume(nd_u_clipped, ref_point=np.ones(no))

    row = {
        'Problem': nm, 'VarCount': nv, 'ObjCount': no,
        'Algorithm': algo_nm, 'EA': ea_nm, 'NoiseTag': noise_str,
        'IGD': igd_val, 'AddEps': eps_add, 'MultEps': eps_mul, 'HV': hv_val,
        'IGD_union': igd_val_union, 'AddEps_union': eps_add_union,
        'MultEps_union': eps_mul_union, 'HV_union': hv_val_union
    }
    pd.DataFrame([row]).to_csv(pf_fp, mode='a', header=not path.exists(pf_fp), index=False)
    existing_pf.loc[len(existing_pf)] = row

# ----- Process File with CV Surrogate Training -----
def process_file(fp):
    try:
        nm, nv, no, samples, noise_tag = extract_details(path.basename(fp))
        if nm is None:
            return
        df = pd.read_csv(fp)
        X, Y = df.iloc[:, :nv], df.iloc[:, -no:]

        if Y.shape[1] != no:
            logging.error(
                f"ERROR: Objective count mismatch in training data for {nm}: Y has {Y.shape[1]}, expected {no}. Skipping."
            )
            return

        if nm.startswith("DTLZ"):
            lower_bounds = np.zeros(nv)
            upper_bounds = np.ones(nv)
        elif nm.startswith("WFG"):
            lower_bounds = np.zeros(nv)
            upper_bounds = np.ones(nv)
        elif nm in variable_ranges:
            lower_bounds = np.array([b[0] for b in variable_ranges[nm]])
            upper_bounds = np.array([b[1] for b in variable_ranges[nm]])
        else:
            lower_bounds = X.min(axis=0).values
            upper_bounds = X.max(axis=0).values

        for i in range(len(lower_bounds)):
            if lower_bounds[i] == upper_bounds[i]:
                logging.warning(f"Variable {i+1} in {nm} has constant value {lower_bounds[i]} in training data.")

        algos = build_algorithms(nv)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        for algo_nm, algo in algos.items():
            for obj_idx in range(no):
                mask = (
                    (existing_perf['Problem']==nm)&(existing_perf['VarCount']==nv)&
                    (existing_perf['ObjCount']==no)&(existing_perf['Algorithm']==algo_nm)&
                    (existing_perf['NoiseTag']==noise_tag)&
                    (existing_perf['Objective']==obj_idx+1)
                )
                if mask.any():
                    continue
                pipe = Pipeline([('scale', StandardScaler()), ('model', algo())])
                reg = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
                scoring = {'r2': 'r2', 'neg_mse': 'neg_mean_squared_error'}
                cv_res = cross_validate(reg, X, Y.iloc[:, obj_idx], cv=kf, scoring=scoring)
                r2_mean = cv_res['test_r2'].mean()
                mse_mean = -cv_res['test_neg_mse'].mean()
                row = {'Problem': nm, 'VarCount': nv, 'ObjCount': no,
                       'Algorithm': algo_nm, 'NoiseTag': noise_tag,
                       'Objective': obj_idx + 1, 'R2': r2_mean, 'MSE': mse_mean}
                existing_perf.loc[len(existing_perf)] = row
                pd.DataFrame([row]).to_csv(perf_fp, mode='a', header=not path.exists(perf_fp), index=False)
        for algo_nm, algo in algos.items():
            trained = []
            for i in range(no):
                pipe = Pipeline([('scale', StandardScaler()), ('model', algo())])
                reg = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())
                reg.fit(X, Y.iloc[:, i])
                trained.append(reg)
            for ea_nm, EAcls in [("NSGAIII", NSGAIII), ("IBEA", IBEA), ("RVEA", RVEA)]:
                params = {"population_size": 100} if ea_nm in ("IBEA", "RVEA") else {}
                run_ea(nm, nv, no, algo_nm, ea_nm, noise_tag, trained, {'__cls__': EAcls, **params}, X, lower_bounds, upper_bounds)
    except Exception as e:
        logging.error(f"Exception processing file {fp}: {str(e)}")

# ----- Main -----
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, choices=['DTLZ', 'WFG', 'Engineering'])
    parser.add_argument('--problem', required=True)
    args = parser.parse_args()
    data_dir = path.join(base_folder, 'Data', args.suite, args.problem)
    for root, _, files in walk(data_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                process_file(path.join(root, f))
    logging.info("All tasks processed.")

    # Deduplicate PF metrics CSV
    if path.exists(pf_fp):
        df_pf = pd.read_csv(pf_fp)
        df_pf = df_pf.drop_duplicates(subset=['Problem', 'VarCount', 'ObjCount', 'Algorithm', 'EA', 'NoiseTag'])
        df_pf.to_csv(pf_fp, index=False)
        logging.info("Deduplicated PF metrics CSV after all runs.")
