#!/usr/bin/env python3
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

USE_R2 = False  # Set False if you do not want R2 in weight schemes

BASE_INDICATORS = ["IGD", "HV", "EpsAdd", "EpsMulti"]
INDICATOR_COLS = BASE_INDICATORS + (["R2"] if USE_R2 else [])

NN_BASELINE_NAME = "NN"


# =========================================================
# HELPERS
# =========================================================

def scheme_uses_r2(s):
    return "R2" in str(s).split("_") or "R2" in str(s).split("+")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def display_model_name(m):
    m = str(m)
    if m in {"nn", "nn_relu", "nn_sigmoid"}:
        return "NN"
    return m


def fmt_mean_std(mean, std, digits=3):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        std = 0.0
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def pretty_weight_scheme(s):
    s = str(s)

    if s.startswith("single_"):
        return s.replace("single_", "")

    if s.startswith("pair_"):
        return s.replace("pair_", "").replace("_", "+")

    if s.startswith("triple_"):
        return s.replace("triple_", "").replace("_", "+")

    if s.startswith("quad_"):
        return s.replace("quad_", "").replace("_", "+")

    if s == "all_equal":
        return "+".join(INDICATOR_COLS)

    return s


def weights_from_scheme(s):
    weights = {c: 0.0 for c in INDICATOR_COLS}
    s = str(s)

    if s.startswith("single_"):
        active = [s.replace("single_", "")]
    elif s.startswith("pair_"):
        active = s.replace("pair_", "").split("_")
    elif s.startswith("triple_"):
        active = s.replace("triple_", "").split("_")
    elif s.startswith("quad_"):
        active = s.replace("quad_", "").split("_")
    elif s == "all_equal":
        active = INDICATOR_COLS.copy()
    else:
        active = s.split("+")

    active = [a for a in active if a in weights]

    if active:
        w = 1.0 / len(active)
        for a in active:
            weights[a] = w

    return weights


def sort_weight_schemes(schemes):
    indicator_order = INDICATOR_COLS

    def key(s):
        s = str(s)

        if s.startswith("single_"):
            parts = [s.replace("single_", "")]
            group = 1
        elif s.startswith("pair_"):
            parts = s.replace("pair_", "").split("_")
            group = 2
        elif s.startswith("triple_"):
            parts = s.replace("triple_", "").split("_")
            group = 3
        elif s.startswith("quad_"):
            parts = s.replace("quad_", "").split("_")
            group = 4
        elif s == "all_equal":
            parts = indicator_order
            group = 5
        else:
            parts = s.split("+")
            group = 6

        idxs = [
            indicator_order.index(p) if p in indicator_order else 99
            for p in parts
        ]
        return group, idxs

    return sorted(schemes, key=key)


# =========================================================
# LOADING
# =========================================================

def load_csvs(base_dir, filename):
    files = sorted(glob.glob(os.path.join(base_dir, "*", filename)))
    if not files:
        raise FileNotFoundError(f"No {filename} files found under {base_dir}")

    dfs = []

    for fp in files:
        scheme = os.path.basename(os.path.dirname(fp))

        # Skip R2-based result folders when USE_R2=False
        if not USE_R2 and scheme_uses_r2(scheme):
            continue

        df = pd.read_csv(fp, low_memory=False)

        if "weight_scheme" not in df.columns:
            df["weight_scheme"] = scheme

        # Also remove any R2 schemes inside the file
        if not USE_R2:
            df = df[~df["weight_scheme"].astype(str).map(scheme_uses_r2)].copy()

        if df.empty:
            continue

        df["weight_scheme_from_path"] = scheme
        df["weight_scheme_pretty"] = df["weight_scheme"].map(pretty_weight_scheme)

        if "model" in df.columns:
            df["model_display"] = df["model"].map(display_model_name)

        dfs.append(df)

    if not dfs:
        raise ValueError(
            "No result files left after filtering. "
            "USE_R2=False removed all R2-based schemes."
        )

    return pd.concat(dfs, ignore_index=True)


def load_combined_dataset(fp):
    df = pd.read_csv(fp, low_memory=False)

    rename_candidates = {
        "problem": ["problem", "problem_merged", "Problem", "problem_key"],
        "n_var": ["n_var", "n_var_merged", "VarCount", "n_var_key"],
        "n_obj": ["n_obj", "n_obj_merged", "ObjCount", "n_obj_key"],
        "num_samples": ["num_samples", "n_samples", "n_samples_merged", "n_samples_key"],
        "noise": ["noise", "noise_merged", "NoiseTag", "noise_key"],
        "surrogate": ["surrogate", "algo", "algo_metrics"],
    }

    for new, cands in rename_candidates.items():
        if new not in df.columns:
            for c in cands:
                if c in df.columns:
                    df[new] = df[c]
                    break

    if USE_R2:
        if "R2" not in df.columns:
            if "perf_mean_r2" in df.columns:
                df["R2"] = df["perf_mean_r2"]
            elif "perf_median_r2" in df.columns:
                df["R2"] = df["perf_median_r2"]
            else:
                df["R2"] = np.nan

    for c in ["n_var", "n_obj", "num_samples"] + INDICATOR_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "noise" in df.columns:
        df["noise"] = df["noise"].fillna("none").astype(str)

    if "surrogate" in df.columns:
        df["surrogate"] = df["surrogate"].astype(str)

    return df


# =========================================================
# TABLES
# =========================================================

def make_result_table(detailed_df, summary_df, eval_split, out_fp):
    sub = detailed_df[detailed_df["eval_split"] == eval_split].copy()
    if sub.empty:
        print(f"[WARN] No rows for {eval_split}")
        return pd.DataFrame()

    sub["model_display"] = sub["model"].map(display_model_name)
    sub["regret"] = pd.to_numeric(sub["regret"], errors="coerce")

    summary_split = eval_split + "_summary"
    sm = summary_df[summary_df["eval_split"] == summary_split].copy()
    sm["model_display"] = sm["model"].map(display_model_name)
    sm["accuracy"] = pd.to_numeric(sm["accuracy"], errors="coerce")

    acc_map = sm.set_index(["model_display", "weight_scheme"])["accuracy"].to_dict()

    rows = []

    for model in sorted(sub["model_display"].dropna().unique()):
        msub = sub[sub["model_display"] == model].copy()

        scheme_scores = (
            msub.groupby("weight_scheme")["regret"]
            .mean()
            .sort_values()
        )

        if scheme_scores.empty:
            continue

        best_scheme = scheme_scores.index[0]
        g = msub[msub["weight_scheme"] == best_scheme]
        w = weights_from_scheme(best_scheme)

        acc = acc_map.get((model, best_scheme), np.nan)

        row = {
            "model": model,
            "IGD": w["IGD"],
            "HV": w["HV"],
            "EpsAdd": w["EpsAdd"],
            "EpsMulti": w["EpsMulti"],
        }

        if USE_R2:
            row["R2"] = w["R2"]

        row.update({
            "weight_scheme": pretty_weight_scheme(best_scheme),
            "accuracy": "" if pd.isna(acc) else f"{acc:.3f}",
            "regret": fmt_mean_std(g["regret"].mean(), g["regret"].std(ddof=0)),
            "sort_regret": g["regret"].mean(),
            "sort_acc": acc,
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        print(f"[WARN] Empty result table for {eval_split}")
        return out

    out = out.sort_values(
        ["sort_regret", "sort_acc"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    out = out.drop(columns=["sort_regret", "sort_acc"])
    out.to_csv(out_fp, index=False)
    return out


def add_nn_baseline_regret(detailed_df, combined_df, out_dir):
    required = [
        "problem", "n_var", "n_obj", "num_samples", "noise",
        "surrogate",
    ] + INDICATOR_COLS

    missing = [c for c in required if c not in combined_df.columns]
    if missing:
        print(f"[WARN] Missing columns for NN baseline: {missing}")
        return detailed_df

    df = combined_df[required].copy()

    for c in ["n_var", "n_obj", "num_samples"] + INDICATOR_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    group_cols = ["problem", "n_var", "n_obj", "num_samples", "noise"]

    df = df.dropna(subset=group_cols + ["surrogate"])
    df = df[df[INDICATOR_COLS].notna().any(axis=1)]

    df = (
        df.groupby(group_cols + ["surrogate"], dropna=False)[INDICATOR_COLS]
        .mean()
        .reset_index()
    )

    baseline_rows = []

    for _, sub in df.groupby(group_cols, dropna=False):
        sub = sub.copy()

        for col in INDICATOR_COLS:
            vals = pd.to_numeric(sub[col], errors="coerce").values.astype(float)

            # Convert all indicators so lower = better.
            if col == "HV" or (USE_R2 and col == "R2"):
                vals = -vals

            if col == "EpsMulti":
                vals = np.clip(vals, 1e-12, None)
                vals = np.log10(vals)

            if np.isfinite(vals).sum() == 0:
                sub[col + "_norm"] = np.nan
                continue

            lo = np.nanmin(vals)
            hi = np.nanmax(vals)
            sub[col + "_norm"] = (vals - lo) / (hi - lo) if hi > lo else 0.0

        norm_cols = [c for c in sub.columns if c.endswith("_norm")]
        if not norm_cols:
            continue

        arr = sub[norm_cols].astype(float).values
        arr = np.where(np.isfinite(arr), arr, 1.0)

        sub["score"] = arr.mean(axis=1)
        best_score = sub["score"].min()

        nn_sub = sub[sub["surrogate"].astype(str) == NN_BASELINE_NAME]
        if nn_sub.empty:
            continue

        nn_score = nn_sub["score"].iloc[0]
        nn_regret = nn_score - best_score

        row = {c: sub[c].iloc[0] for c in group_cols}
        row["nn_baseline_regret"] = nn_regret
        baseline_rows.append(row)

    if not baseline_rows:
        print(f"[WARN] No NN baseline rows found for surrogate name {NN_BASELINE_NAME}")
        print(f"[INFO] Available surrogate names: {sorted(df['surrogate'].astype(str).unique())}")
        return detailed_df

    baseline_df = pd.DataFrame(baseline_rows)

    out = detailed_df.copy()

    for c in ["n_var", "n_obj", "num_samples"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["noise"] = out["noise"].fillna("none").astype(str)
    out["regret"] = pd.to_numeric(out["regret"], errors="coerce")
    out["model_display"] = out["model"].map(display_model_name)

    out = out.merge(baseline_df, on=group_cols, how="left")
    out["regret_vs_NN"] = out["regret"] - out["nn_baseline_regret"]

    out.to_csv(
        os.path.join(out_dir, "selector_detailed_with_nn_baseline.csv"),
        index=False
    )

    return out


def make_nn_best_table(detailed_df, out_fp):
    """
    For each classifier, select the weight scheme with the best mean regret_vs_NN.
    Report:
      - mean ± std regret_vs_NN
      - minimum observed regret_vs_NN
      - win percentage vs NN baseline
    """
    if "regret_vs_NN" not in detailed_df.columns:
        print(f"[WARN] regret_vs_NN missing; cannot create {out_fp}")
        return pd.DataFrame()

    sub = detailed_df[detailed_df["eval_split"] == "nested_cv"].copy()
    sub["model_display"] = sub["model"].map(display_model_name)
    sub["regret_vs_NN"] = pd.to_numeric(sub["regret_vs_NN"], errors="coerce")
    sub = sub.dropna(subset=["regret_vs_NN"])

    rows = []

    for model in sorted(sub["model_display"].dropna().unique()):
        msub = sub[sub["model_display"] == model]

        scheme_scores = (
            msub.groupby("weight_scheme")["regret_vs_NN"]
            .mean()
            .sort_values()
        )

        if scheme_scores.empty:
            continue

        best_scheme = scheme_scores.index[0]
        g = msub[msub["weight_scheme"] == best_scheme]
        w = weights_from_scheme(best_scheme)

        mean_val = g["regret_vs_NN"].mean()
        std_val = g["regret_vs_NN"].std(ddof=0)
        min_val = g["regret_vs_NN"].min()
        win_rate = 100.0 * (g["regret_vs_NN"] < 0).mean()

        row = {
            "model": model,
            "IGD": w["IGD"],
            "HV": w["HV"],
            "EpsAdd": w["EpsAdd"],
            "EpsMulti": w["EpsMulti"],
        }

        if USE_R2:
            row["R2"] = w["R2"]

        row.update({
            "weight_scheme": pretty_weight_scheme(best_scheme),
            "regret_vs_NN": fmt_mean_std(mean_val, std_val),
            "min_regret_vs_NN": f"{min_val:.3f}",
            "win_vs_NN_%": f"{win_rate:.1f}",
            "sort_val": mean_val,
        })

        rows.append(row)

    out = pd.DataFrame(rows)

    if out.empty:
        print("[WARN] NN best table is empty.")
        return out

    out = out.sort_values("sort_val").drop(columns="sort_val").reset_index(drop=True)
    out.to_csv(out_fp, index=False)
    return out


# =========================================================
# PLOTS
# =========================================================

def plot_boxplot_by_model(df, value_col, title, out_fp):
    """
    Boxplots are built from individual rows in selector_detailed.csv.

    Each value corresponds to an individual evaluated case, e.g. one
    classifier prediction for one problem instance under one weight scheme.
    No additional mean points are overlaid.
    """

    if value_col not in df.columns:
        print(f"[WARN] {value_col} missing; skipping {out_fp}")
        return

    data = df[["model_display", value_col]].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=["model_display", value_col])

    if data.empty:
        print(f"[WARN] No data for {title}")
        return

    order = (
        data.groupby("model_display")[value_col]
        .median()
        .sort_values()
        .index
        .tolist()
    )

    vals = [
        data.loc[data["model_display"] == model, value_col].values
        for model in order
    ]

    all_vals = data[value_col].values
    lo = np.nanpercentile(all_vals, 2)
    hi = np.nanpercentile(all_vals, 95)

    lo = min(lo, 0)
    hi = max(hi, 0.05)
    pad = 0.04 * (hi - lo) if hi > lo else 0.01

    plt.figure(figsize=(8, 4.8))

    plt.boxplot(
        vals,
        vert=False,
        tick_labels=order,
        showfliers=False,
        showmeans=True,
        widths=0.55,
        patch_artist=True,
        boxprops=dict(facecolor="#f2f2f2", edgecolor="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=2.0),
        meanprops=dict(
            marker="D",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=4,
        ),
        flierprops=dict(
            marker="o",
            markerfacecolor="red",
            markeredgecolor="red",
            markersize=2.5,
            alpha=0.35,
        ),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
    )

    plt.axvline(0, color="gray", linestyle="--", linewidth=1)
    plt.xlim(lo - pad, hi + pad)

    if value_col == "regret":
        plt.xlabel("Regret")
    elif value_col == "regret_vs_NN":
        plt.xlabel("Regret difference to NN baseline")
    else:
        plt.xlabel(value_col)

    plt.ylabel("Classifier")
    plt.grid(axis="x", linestyle=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()

def plot_combined_accuracy_regret(summary_df, out_fp):
    sub = summary_df[
        summary_df["eval_split"].isin(["nested_cv_summary", "family_holdout_summary"])
    ].copy()

    if sub.empty:
        print("[WARN] No summary rows for combined scatter.")
        return

    sub["accuracy"] = pd.to_numeric(sub["accuracy"], errors="coerce")
    sub["mean_regret"] = pd.to_numeric(sub["mean_regret"], errors="coerce")
    sub["model_display"] = sub["model"].map(display_model_name)

    sub["dataset"] = sub["eval_split"].map({
        "nested_cv_summary": "Full dataset",
        "family_holdout_summary": "Generalization",
    })

    sub = sub.dropna(subset=["accuracy", "mean_regret"])

    plt.figure(figsize=(7, 5))

    colors = {
        "Full dataset": "tab:blue",
        "Generalization": "tab:orange",
    }

    for ds in ["Full dataset", "Generalization"]:
        s = sub[sub["dataset"] == ds]
        if s.empty:
            continue

        plt.scatter(
            s["accuracy"],
            s["mean_regret"],
            label=ds,
            alpha=0.75,
            color=colors[ds],
        )

    plt.xlabel("Accuracy")
    plt.ylabel("Mean regret")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.35)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()


def plot_accuracy_regret_tradeoff(summary_df, eval_split, title, out_fp):
    sub = summary_df[summary_df["eval_split"] == eval_split].copy()

    if sub.empty:
        print(f"[WARN] No rows for {title}")
        return

    sub["accuracy"] = pd.to_numeric(sub["accuracy"], errors="coerce")
    sub["mean_regret"] = pd.to_numeric(sub["mean_regret"], errors="coerce")
    sub["model_display"] = sub["model"].map(display_model_name)
    sub = sub.dropna(subset=["accuracy", "mean_regret"])

    if sub.empty:
        print(f"[WARN] No valid values for {title}")
        return

    plt.figure(figsize=(7, 5))

    for model in sorted(sub["model_display"].dropna().unique()):
        s = sub[sub["model_display"] == model]
        plt.scatter(s["accuracy"], s["mean_regret"], label=model, alpha=0.75)

    plt.xlabel("Accuracy")
    plt.ylabel("Mean regret")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()


def plot_weight_scheme_heatmap(summary_df, eval_split, title, out_fp):
    sub = summary_df[summary_df["eval_split"] == eval_split].copy()

    if sub.empty:
        print(f"[WARN] No data for {title}")
        return

    sub["model_display"] = sub["model"].map(display_model_name)
    sub["mean_regret"] = pd.to_numeric(sub["mean_regret"], errors="coerce")

    table = sub.pivot_table(
        index="model_display",
        columns="weight_scheme",
        values="mean_regret",
        aggfunc="mean"
    )

    if table.empty:
        print(f"[WARN] Empty heatmap table for {title}")
        return

    sorted_cols = sort_weight_schemes(table.columns)
    table = table[sorted_cols]

    pretty_cols = [pretty_weight_scheme(c) for c in table.columns]

    model_order = table.mean(axis=1).sort_values().index
    table = table.loc[model_order]

    plt.figure(figsize=(max(13, 0.45 * len(pretty_cols)), 5))

    im = plt.imshow(table.values, aspect="auto")

    plt.xticks(range(len(pretty_cols)), pretty_cols, rotation=60, ha="right")
    plt.yticks(range(len(table.index)), table.index)

    cbar = plt.colorbar(im)
    cbar.set_label("Mean regret (lower is better)")

    groups = []
    for c in sorted_cols:
        c = str(c)
        if c.startswith("single_"):
            groups.append(1)
        elif c.startswith("pair_"):
            groups.append(2)
        elif c.startswith("triple_"):
            groups.append(3)
        elif c.startswith("quad_"):
            groups.append(4)
        elif c == "all_equal":
            groups.append(5)
        else:
            groups.append(6)

    for i in range(1, len(groups)):
        if groups[i] != groups[i - 1]:
            plt.axvline(i - 0.5, color="black", linewidth=1)

    plt.tight_layout()
    plt.savefig(out_fp, dpi=300)
    plt.close()


# =========================================================
# MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="results/combined_classifier_results")
    parser.add_argument("--combined", default="combined_metrics_features.csv")
    parser.add_argument("--out-dir", default="results/combined_classifier_results_analysis")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    print(f"USE_R2 = {USE_R2}")
    print(f"Indicators used: {INDICATOR_COLS}")

    summary_df = load_csvs(args.base_dir, "selector_summary.csv")
    detailed_df = load_csvs(args.base_dir, "selector_detailed.csv")

    summary_df["model_display"] = summary_df["model"].map(display_model_name)
    detailed_df["model_display"] = detailed_df["model"].map(display_model_name)

    full_table = make_result_table(
        detailed_df,
        summary_df,
        eval_split="nested_cv",
        out_fp=os.path.join(args.out_dir, "table_full_dataset_results.csv"),
    )

    gen_table = make_result_table(
        detailed_df,
        summary_df,
        eval_split="family_holdout",
        out_fp=os.path.join(args.out_dir, "table_generalization_results.csv"),
    )

    print("\nFull dataset results:")
    print(full_table.to_string(index=False))

    print("\nGeneralization results:")
    print(gen_table.to_string(index=False))

    if os.path.exists(args.combined):
        combined_df = load_combined_dataset(args.combined)
        detailed_df = add_nn_baseline_regret(detailed_df, combined_df, args.out_dir)

        nn_best = make_nn_best_table(
            detailed_df,
            os.path.join(args.out_dir, "table_full_dataset_regret_vs_NN.csv"),
        )

        print("\nFull dataset regret vs NN:")
        print(nn_best.to_string(index=False))
    else:
        print(f"[WARN] Combined file not found: {args.combined}")

    cv_df = detailed_df[detailed_df["eval_split"] == "nested_cv"].copy()
    gen_df = detailed_df[detailed_df["eval_split"] == "family_holdout"].copy()

    plot_boxplot_by_model(
        cv_df,
        "regret",
        "Full dataset: regret by classifier",
        os.path.join(args.out_dir, "boxplot_full_dataset_regret_by_model.png"),
    )

    plot_boxplot_by_model(
        gen_df,
        "regret",
        "Generalization: regret by classifier",
        os.path.join(args.out_dir, "boxplot_generalization_regret_by_model.png"),
    )

    plot_boxplot_by_model(
        cv_df,
        "regret_vs_NN",
        "Full dataset: regret difference to NN baseline",
        os.path.join(args.out_dir, "boxplot_full_dataset_regret_vs_NN.png"),
    )

    plot_accuracy_regret_tradeoff(
        summary_df,
        "nested_cv_summary",
        "Full dataset: accuracy vs mean regret",
        os.path.join(args.out_dir, "scatter_full_dataset_accuracy_vs_regret.png"),
    )

    plot_accuracy_regret_tradeoff(
        summary_df,
        "family_holdout_summary",
        "Generalization: accuracy vs mean regret",
        os.path.join(args.out_dir, "scatter_generalization_accuracy_vs_regret.png"),
    )

    plot_combined_accuracy_regret(
        summary_df,
        os.path.join(args.out_dir, "scatter_combined_accuracy_vs_regret.png"),
    )

    plot_weight_scheme_heatmap(
        summary_df,
        "nested_cv_summary",
        "Full dataset: mean regret by weight scheme",
        os.path.join(args.out_dir, "heatmap_full_dataset_weight_schemes.png"),
    )

    plot_weight_scheme_heatmap(
        summary_df,
        "family_holdout_summary",
        "Generalization: mean regret by weight scheme",
        os.path.join(args.out_dir, "heatmap_generalization_weight_schemes.png"),
    )

    print(f"\nSaved thesis-focused outputs to: {args.out_dir}")


if __name__ == "__main__":
    main()
