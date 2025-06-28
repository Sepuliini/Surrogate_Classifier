#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from datetime import datetime
import re
import argparse

# ----- Paths -----
base_path   = "/scratch/project_2012636/modelling_results"
real_path   = os.path.join(base_path, "real_paretofronts")
approx_path = os.path.join(base_path, "surrogate_optimization_results_with_noise")
timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join(base_path, f"paretofront_comparison_results_{timestamp}")
os.makedirs(results_dir, exist_ok=True)
output_csv  = os.path.join(results_dir, "paretofront_quality_indicators.csv")

# ----- CLI Args -----
parser = argparse.ArgumentParser(description="Compute quality indicators (and optional plots) for Pareto front comparisons.")
parser.add_argument('--no-plot', action='store_true', help="Skip plot generation")
args = parser.parse_args()

# ----- Utilities -----
def normalize(obj_set, ref_min, ref_max, clip=True):
    normed = (obj_set - ref_min) / (ref_max - ref_min + 1e-12)
    return np.clip(normed, 0, 1) if clip else normed

def load_numeric_csv(path):
    df = pd.read_csv(path, header=None, dtype=str)
    df = df.applymap(lambda x: x.replace(' ', '') if isinstance(x, str) else x)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.dropna(how='any').values

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def igd(A, R):
    return np.mean([np.min([euclidean_distance(r, a) for a in A]) for r in R])

def additive_epsilon_indicator(R, A):
    return max([min([max(r - a) for r in R]) for a in A])

def multiplicative_epsilon_indicator(R, A):
    return max([min([max(r / (a + 1e-12)) for r in R]) for a in A])

try:
    from pymoo.indicators.hv import HV
    def hypervolume(A, reference_point):
        return HV(ref_point=reference_point).do(A)
except ImportError:
    def hypervolume(A, reference_point):
        return np.nan

# parse file metadata
def parse_approx_filename(fname):
    base = os.path.splitext(fname)[0]
    m_vars = re.search(r"(\d+)vars", base)
    m_objs = re.search(r"(\d+)objs", base)
    vars_num = int(m_vars.group(1)) if m_vars else None
    objs_num = int(m_objs.group(1)) if m_objs else None

    data_info = 'none'
    if '_opt_' in base:
        suffix = base.split('_opt_')[-1].lower()
        if 'uniform' in suffix:
            data_info = 'uniform'
        elif 'noise' in suffix:
            data_info = 'noise'
        else:
            data_info = suffix

    m_prob = re.match(r"([A-Za-z]+\d*)", base)
    problem = m_prob.group(1) if m_prob else base
    return problem, vars_num, objs_num, data_info

# optional plotting
if not args.no_plot:
    def plot_front_comparison(A, R, ref_min, ref_max, title_base, technique, data_info,
                              save_dir, ref_point, suffix, metrics=None):
        obj_dim = A.shape[1]
        fig = plt.figure(figsize=(6,5))
        fig.subplots_adjust(top=0.90, bottom=0.20, right=0.85)
        An = normalize(A, ref_min, ref_max, clip=(suffix == "clipped"))
        Rn = normalize(R, ref_min, ref_max, clip=(suffix == "clipped"))
        if obj_dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(Rn[:,0], Rn[:,1], label="Reference", s=30, c='C0', alpha=0.8)
            ax.scatter(An[:,0], An[:,1], label="Approx",    s=30, c='C1', alpha=0.8)
            ax.set_xlabel("Objective 1"); ax.set_ylabel("Objective 2")
            ax.set_xlim(-0.1, 1.1); ax.set_ylim(-0.1, 1.1)
        elif obj_dim == 3:
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(Rn[:,0], Rn[:,1], Rn[:,2], label="Reference", s=30, c='C0', alpha=0.8)
            ax.scatter(An[:,0], An[:,1], An[:,2], label="Approx",    s=30, c='C1', alpha=0.8)
            ax.set_xlabel("Obj 1"); ax.set_ylabel("Obj 2"); ax.set_zlabel("Obj 3")
        else:
            from pandas.plotting import parallel_coordinates
            cols = [f"Obj{i+1}" for i in range(obj_dim)]
            dfR = pd.DataFrame(Rn, columns=cols); dfR['Type']='Real'
            dfA = pd.DataFrame(An, columns=cols); dfA['Type']='Approx'
            ax = fig.add_subplot(111)
            parallel_coordinates(pd.concat([dfR,dfA],ignore_index=True), 'Type', ax=ax,
                                 color=['C0','C1'], alpha=0.3)
            ax.set_xlabel('Objectives'); ax.set_ylabel('Normalized Value')
        ax.legend(loc='upper left')
        title = f"{title_base} ({technique})"
        if data_info!='none': title += f", {data_info}"
        fig.suptitle(title, fontsize=12, y=0.96)
        if metrics:
            if suffix=="clipped":
                txt = (f"IGD={metrics['IGD_Clipped']:.4f}, "
                       f"ε⁺={metrics['AddEps_Clipped']:.4f}, "
                       f"ε×={metrics['MultEps_Clipped']:.4f}, "
                       f"HV={metrics['HV_Clipped']:.4f}")
            else:
                txt = (f"IGD={metrics['IGD_Union']:.4f}, "
                       f"ε⁺={metrics['AddEps_Union']:.4f}, "
                       f"ε×={metrics['MultEps_Union']:.4f}, "
                       f"HV={metrics['HV_Union']:.4f}")
            fig.text(0.5,0.02,txt,ha='center',va='bottom',fontsize=8)
        fig.tight_layout(rect=[0,0.20,0.85,0.96])
        out_name = f"{title_base}_{technique}_{data_info}_{suffix}.png"
        fig.savefig(os.path.join(save_dir, out_name), bbox_inches='tight')
        plt.close(fig)
else:
    plot_front_comparison = lambda *args, **kwargs: None

# ----- Main Comparison -----
real_problems = {}
for e in os.listdir(real_path):
    p = os.path.join(real_path, e)
    if os.path.isdir(p): real_problems[e] = p
    elif e.endswith(('.csv','.txt')): real_problems[e.split('_')[0]] = real_path

approx_problems = {d: os.path.join(approx_path, d) for d in os.listdir(approx_path)
                   if os.path.isdir(os.path.join(approx_path, d))}

results = []
for prob, rf in real_problems.items():
    real_files = [f for f in os.listdir(rf) if f.endswith(('.csv','.txt'))]
    if not real_files: continue
    R = load_numeric_csv(os.path.join(rf, real_files[0]))
    if R.size==0: continue
    rmin, rmax = R.min(0), R.max(0)
    Rc = normalize(R,rmin,rmax,clip=True)
    dim = Rc.shape[1]
    ref_pt = np.ones(dim)*1.1
    # union bounds
    all_A=[]
    for d,pth in approx_problems.items():
        for f in os.listdir(pth):
            if not f.endswith('.csv'): continue
            pr,v,o,info = parse_approx_filename(f)
            if pr==prob and o==dim:
                all_A.append(load_numeric_csv(os.path.join(pth,f)))
    umin = np.minimum(rmin,np.min(np.vstack(all_A),axis=0)) if all_A else rmin
    umax = np.maximum(rmax,np.max(np.vstack(all_A),axis=0)) if all_A else rmax
    Ru = normalize(R,umin,umax,clip=False)

    for d,pth in approx_problems.items():
        for f in os.listdir(pth):
            if not f.endswith('.csv'): continue
            pr,v,o,info = parse_approx_filename(f)
            if pr!=prob or o!=dim: continue
            af = os.path.join(pth,f)
            A = load_numeric_csv(af)
            if A.size==0: continue
            Ac=normalize(A,rmin,rmax,clip=True)
            Au=normalize(A,umin,umax,clip=False)
            tech = f.split('_')[1] if '_' in f else 'unknown'
            metrics={
                'IGD_Clipped': igd(Ac,Rc),
                'IGD_Union':   igd(Au,Ru),
                'AddEps_Clipped': additive_epsilon_indicator(Rc,Ac),
                'AddEps_Union':   additive_epsilon_indicator(Ru,Au),
                'MultEps_Clipped': multiplicative_epsilon_indicator(Rc,Ac),
                'MultEps_Union':   multiplicative_epsilon_indicator(Ru,Au),
                'HV_Clipped':      hypervolume(Ac,ref_pt),
                'HV_Union':        hypervolume(Au,ref_pt),
                'HV*_Ref':         hypervolume(Rc,ref_pt),
                'HV*_UnionRef':    hypervolume(Ru,ref_pt)
            }
            results.append({
                'Problem':prob,'File':f,'Objectives':dim,'Technique':tech,'DataType':info,
                **metrics})
            title_base=f"{prob}_{os.path.splitext(f)[0]}"
            plot_front_comparison(A,R,rmin,rmax,title_base,tech,info,results_dir,ref_pt,'clipped',metrics)
            plot_front_comparison(A,R,umin,umax,title_base,tech,info,results_dir,ref_pt,'union',metrics)

pd.DataFrame(results).to_csv(output_csv,index=False)
print(f"\n✅ Results saved to {output_csv}")
