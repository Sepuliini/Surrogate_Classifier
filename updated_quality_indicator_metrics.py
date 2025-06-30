#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import re
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from numpy.linalg import norm

# ----- SLURM array of engineering problems only -----
engineering_probs = [
    're21','re22','re23','re24','re25',
    're31','re32','re33',
    'multiple_clutch_brakes','river_pollution_problem',
    'vehicle_crashworthiness','car_side_impact'
]
idx = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0')) - 1
if idx < 0 or idx >= len(engineering_probs):
    raise RuntimeError(f"SLURM_ARRAY_TASK_ID {idx+1} out of range (1-{len(engineering_probs)})")
prob = engineering_probs[idx]

# ----- Parallel workers -----
workers = int(os.environ.get('SLURM_CPUS_PER_TASK', '1'))

# ----- Paths -----
base = "/scratch/project_2012636/modelling_results"
real_root = os.path.join(base, "real_paretofronts")
approx_root = os.path.join(base, "surrogate_optimization_results_with_noise")
results_dir = os.path.join(base, "paretofront_comparison_results")
os.makedirs(results_dir, exist_ok=True)
output_csv = os.path.join(results_dir, f"paretofront_quality_indicators_{prob}.csv")

# ----- Resume existing output -----
if os.path.exists(output_csv):
    df_done = pd.read_csv(output_csv)
    if 'RealFile' in df_done.columns:
        done_triples = set(zip(df_done['Problem'], df_done['RealFile'], df_done['File']))
        done_pairs = None
    else:
        done_pairs = set(zip(df_done['Problem'], df_done['File']))
        done_triples = None
    first_write = False
else:
    done_triples = None
    done_pairs = set()
    first_write = True

# ----- Utilities -----
def normalize(A, mn, mx, clip=True):
    normed = (A - mn) / (mx - mn + 1e-12)
    return np.clip(normed,0,1) if clip else normed

def load_csv(path):
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.applymap(lambda x: x.replace(' ', '') if isinstance(x,str) else x)
    return df.apply(pd.to_numeric, errors='coerce').dropna(how='any').values

def igd(A,R): return np.mean([np.min([norm(r-a) for a in A]) for r in R])
def additive_epsilon_indicator(R,A): return max([min([max(r-a) for r in R]) for a in A])
def multiplicative_epsilon_indicator(R,A): return max([min([max(r/(a+1e-12)) for r in R]) for a in A])
try:
    from pymoo.indicators.hv import HV
    def hypervolume(A,ref): return HV(ref_point=ref).do(A)
except ImportError:
    def hypervolume(A,ref): return np.nan

# extract problem prefix more robustly
def parse_approx_filename(fname):
    base = os.path.splitext(fname)[0]
    # problem is all text before first underscore that begins a var/obj or opt
    parts = re.split(r'_(?:\d+var(?:s)?|\d+obj(?:s)?|opt_)', base, flags=re.IGNORECASE)
    problem = parts[0].lower()
    # find numbers
    m_vars = re.search(r"(\d+)var(?:s)?", base, flags=re.IGNORECASE)
    m_objs = re.search(r"(\d+)obj(?:s)?", base, flags=re.IGNORECASE)
    vars_num = int(m_vars.group(1)) if m_vars else None
    objs_num = int(m_objs.group(1)) if m_objs else None
    # data info suffix
    data_info = 'none'
    if '_opt_' in base.lower():
        suf = base.lower().split('_opt_')[-1]
        if 'uniform' in suf: data_info='uniform'
        elif 'noise' in suf: data_info='noise'
        else: data_info=suf
    return problem, vars_num, objs_num, data_info

# no plotting
plot_front_comparison=lambda *a,**k: None

# ----- Gather real fronts -----
# List all real front files for this problem
scenario_list = []
real_dir = os.path.join(real_root, prob)
real_files = []
if os.path.isdir(real_dir):
    real_files = [os.path.join(real_dir, f)
                  for f in os.listdir(real_dir)
                  if f.lower().endswith(('.csv','.txt'))]

for rf in sorted(real_files):
    real_file = os.path.basename(rf)
    R = load_csv(rf)
    if R.size == 0:
        continue
    rmin, rmax = R.min(0), R.max(0)
    Rc = normalize(R, rmin, rmax, clip=True)
    dim = Rc.shape[1]
    # compute union bounds only in this problem's approx dir
    allA = []
    approx_dir = os.path.join(approx_root, prob)
    if os.path.isdir(approx_dir):
        for f in os.listdir(approx_dir):
            if not f.endswith('.csv'):
                continue
            A = load_csv(os.path.join(approx_dir, f))
            if A.size:
                allA.append(A)
    umin = np.minimum(rmin, np.min(np.vstack(allA), axis=0)) if allA else rmin
    umax = np.maximum(rmax, np.max(np.vstack(allA), axis=0)) if allA else rmax
    Ru = normalize(R, umin, umax, clip=False)
    scenario_list.append((real_file, R, rmin, rmax, Rc, umin, umax, Ru, dim))

# ----- Build tasks -----
# Only look in this problem's approx directory
tasks = []
approx_dir = os.path.join(approx_root, prob)
if os.path.isdir(approx_dir):
    for real_file, R, rmin, rmax, Rc, umin, umax, Ru, dim in scenario_list:
        for f in os.listdir(approx_dir):
            if not f.endswith('.csv'):
                continue
            # ensure filename starts with the problem name
            if not f.lower().startswith(prob + '_'):
                continue
            # parse vars and objs
            _, _, o, info = parse_approx_filename(f)
            if o != dim:
                continue
            key_tri = (prob, real_file, f)
            key_pair = (prob, f)
            if done_triples is not None and key_tri in done_triples:
                continue
            if done_triples is None and key_pair in done_pairs:
                continue
            tech = f.split('_')[1] if '_' in f else 'unknown'
            tasks.append((real_file, prob, os.path.join(approx_dir, f), tech, info,
                          rmin, rmax, umin, umax, Rc, Ru, dim))
# ----- Worker -----
def process_task(args):
    real_file,prob,fp,tech,info,rmin,rmax,umin,umax,Rc,Ru,dim=args
    A=load_csv(fp)
    if A.size==0: return None
    Ac=normalize(A,rmin,rmax,clip=True)
    Au=normalize(A,umin,umax,clip=False)
    ref_pt=np.ones(dim)*1.1
    met={
        'IGD_Clipped':igd(Ac,Rc),'IGD_Union':igd(Au,Ru),
        'AddEps_Clipped':additive_epsilon_indicator(Rc,Ac),
        'AddEps_Union':additive_epsilon_indicator(Ru,Au),
        'MultEps_Clipped':multiplicative_epsilon_indicator(Rc,Ac),
        'MultEps_Union':multiplicative_epsilon_indicator(Ru,Au),
        'HV_Clipped':hypervolume(Ac,ref_pt),'HV_Union':hypervolume(Au,ref_pt),
        'HV*_Ref':hypervolume(Rc,ref_pt),'HV*_UnionRef':hypervolume(Ru,ref_pt)
    }
    return {'Problem':prob,'RealFile':real_file,'File':os.path.basename(fp),
            'Objectives':dim,'Technique':tech,'DataType':info,**met}

# ----- Run and write -----
count=0
with ProcessPoolExecutor(max_workers=workers) as ex:
    for fut in as_completed([ex.submit(process_task,t) for t in tasks]):
        res=fut.result()
        if res is None: continue
        pd.DataFrame([res]).to_csv(output_csv,mode='a',header=first_write,index=False)
        first_write=False
        count+=1

print(f"\n✅ Completed {prob}: wrote {count} entries to {output_csv}")
