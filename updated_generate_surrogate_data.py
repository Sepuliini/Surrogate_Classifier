#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from os import path, makedirs, walk
from datetime import datetime
import logging
import re
import argparse

from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn import ensemble, svm
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

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
from desdeo_problem.testproblems.TestProblems import test_problem_builder
from optproblems import wfg

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*X does not have valid feature names.*")

task_id = os.getenv("SLURM_ARRAY_TASK_ID", str(os.getpid()))
base_folder = '/scratch/project_2012636'
log_dir = path.join(base_folder, "logs")
makedirs(log_dir, exist_ok=True)
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

def build_algorithms():
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

ED_MAP = {
    're21': re21, 're22': re22, 're23': re23, 're24': re24,
    're25': re25, 're31': re31, 're32': re32, 're33': re33,
    'multiple_clutch_brakes': multiple_clutch_brakes,
    'river_pollution_problem': river_pollution_problem,
    'vehicle_crashworthiness': vehicle_crashworthiness,
    'car_side_impact': car_side_impact
}

pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)var_(\d+)obj_(\d+)samples.*\.csv$")
eng_pattern = re.compile(r"^([A-Za-z0-9_]+)_(\d+)_samples(?:_noise)?\.csv$")

def extract_details(fn):
    m = pattern.match(fn)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    m2 = eng_pattern.match(fn)
    if m2:
        name, samples = m2.group(1), int(m2.group(2))
        inst = ED_MAP.get(name)
        if inst:
            prob = inst()
            return name, len(prob.variables), len(prob.objectives), samples
    return None, None, None, None

def extract_noise_tag(filename):
    match = re.search(r"(?:^|_)((?:normal|uniform)(?:_(?:noise|normal|uniform))?|noise)(?=\.csv|_samples)", filename)
    return match.group(1) if match else None

def get_valid_k(D, M):
    base = M - 1
    for k in range(D, 0, -1):
        if k % base == 0 and (D - k) >= 0:
            return k
    return None

def run_ea(nm, algo_nm, ea_nm, EAcls, ea_params, mlist, nv, no, noise_tag=None):
    logging.info(f"→ opt {algo_nm} + {ea_nm}")
    altered = False
    altered_name = None
    cons = None
    if nm.startswith("WFG"):
        k = get_valid_k(nv, no)
        if k is None:
            k = no - 1
            t = max(1, ((nv - k) // 2) if nv >= k else 1)
            nv_new = k + 2 * t
            altered = True
            altered_name = f"{nm}_{algo_nm}_{ea_nm}_{nv}vars_{no}objs_opt_altered_{nv_new}vars_{no}objs.csv"
            logging.warning(f"Adjusted WFG vars for {nm}: {nv}->{nv_new} with k={k}")
            nv = nv_new
        try:
            wfg_class = getattr(wfg, nm)
            _ = wfg_class(num_objectives=no, num_variables=nv, k=k)
        except Exception:
            logging.exception(f"Failed to construct WFG {nm}, n_var={nv}, n_obj={no}, k={k}")
            return False
    elif nm in ED_MAP:
        cons = ED_MAP[nm]().constraints
    elif nm.startswith("DTLZ"):
        cons = test_problem_builder(nm, nv, no).constraints
    vars_ = [Variable(f"x{i+1}", lower_bound=0.1, upper_bound=1.0, initial_value=0.5) for i in range(nv)]
    objs = [ScalarObjective(f"f{i+1}", evaluator=lambda x, m=m: m.predict(np.atleast_2d(x)).flatten()) for i, m in enumerate(mlist)]
    prob = MOProblem(objectives=objs, variables=vars_, constraints=cons)
    evo = EAcls(prob, n_iterations=10, n_gen_per_iter=50, **ea_params)
    while evo.continue_evolution():
        evo.iterate()
    sol = evo.end()[1]
    if sol is None or len(sol) == 0:
        logging.warning(f"no sol {algo_nm}|{ea_nm}")
        return False
    F = np.array(sol)
    nd = [F[i] for i in range(len(F)) if not any((F[j] <= F[i]).all() and (F[j] < F[i]).any() for j in range(len(F)))]
    nd = np.vstack(nd) if nd else np.empty((0, no))
    od = path.join(base_folder, "modelling_results/surrogate_optimization_results_with_noise", nm)
    makedirs(od, exist_ok=True)
    suffix = f"{noise_tag}_opt.csv" if noise_tag else "opt.csv"
    fnout = altered_name if altered else f"{nm}_{algo_nm}_{ea_nm}_{nv}vars_{no}objs_{suffix}"
    pd.DataFrame(nd, columns=[f"f{i+1}" for i in range(no)]).to_csv(path.join(od, fnout), index=False)
    logging.info(f"saved {fnout} {nd.shape[0]}/{F.shape[0]}")
    return True

def process_file(fp):
    fname = path.basename(fp)
    nm, nv, no, _ = extract_details(fname)
    if not nm:
        return False
    noise_tag = extract_noise_tag(fname)

    # Inform early if WFG with invalid k
    if nm.startswith("WFG") and get_valid_k(nv, no) is None:
        logging.info(f"Will adjust WFG problem {nm} due to invalid k for nv={nv}, no={no}")

    df = pd.read_csv(fp)
    X, Y = df.iloc[:, :nv], df.iloc[:, -no:]
    models = {}
    for algo_nm, algo in build_algorithms().items():
        try:
            trained = []
            for i in range(no):
                Xtr, Xte, ytr, yte = tts(X, Y.iloc[:, i], test_size=0.3, random_state=42)
                m = algo(); m.fit(Xtr, ytr)
                trained.append(m)
            models[algo_nm] = (trained, nv, no)
        except Exception:
            logging.exception(f"train fail {fname} {algo_nm}")
    for algo_nm, (mlist, nv, no) in models.items():
        for ea_nm, EAcls, ea_p in [("NSGAIII", NSGAIII, {}), ("IBEA", IBEA, {"population_size":100}), ("RVEA", RVEA, {"population_size":100})]:
            run_ea(nm, algo_nm, ea_nm, EAcls, ea_p, mlist, nv, no, noise_tag)
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--suite', required=True, choices=['DTLZ','WFG','Engineering'])
    parser.add_argument('--problem', required=True)
    args = parser.parse_args()

    data_dir = path.join(base_folder, 'Data', args.suite, args.problem)
    proc_dir = path.join(base_folder, 'modelling_results', 'processed_files_dir')
    makedirs(proc_dir, exist_ok=True)
    plog = path.join(proc_dir, f"processed_{args.suite}_{args.problem}.txt")
    done = set(open(plog).read().splitlines()) if os.path.exists(plog) else set()

    for root, _, files in walk(data_dir):
        for fn in files:
            if not fn.endswith('.csv'):
                continue
            fp = path.join(root, fn)
            nm, nv, no, _ = extract_details(fn)
            if not nm:
                continue

            is_wfg = (args.suite == 'WFG' and nm.startswith('WFG'))

            # Skip non-WFG problems if already done
            if not is_wfg and fp in done:
                continue

            if is_wfg:
                valid_k = get_valid_k(nv, no)

                # WFG problem has valid k – skip (already processed manually)
                if valid_k is not None:
                    logging.info(f"Skipping WFG problem {nm} (k={valid_k}) — already valid.")
                    continue

                # WFG problem already adjusted and processed
                if fp + '_adjusted' in done:
                    logging.info(f"Skipping adjusted WFG problem {nm} — already processed.")
                    continue

            # Run if not skipped
            success = process_file(fp)
            if success:
                entry = fp + ('_adjusted' if is_wfg and get_valid_k(nv, no) is None else '')
                with open(plog, 'a') as f:
                    f.write(entry + '\n')

    logging.info("all done.")
