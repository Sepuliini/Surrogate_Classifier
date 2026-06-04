#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd


def normalize_noise_tag(x):
    if pd.isna(x):
        return "none"
    x = str(x).strip()
    return x if x else "none"


def normalize_problem_name(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def add_common_keys(df, problem_col, n_var_col, n_obj_col, n_samples_col, noise_col):
    out = df.copy()
    out["problem_key"] = out[problem_col].map(normalize_problem_name)
    out["n_var_key"] = pd.to_numeric(out[n_var_col], errors="coerce")
    out["n_obj_key"] = pd.to_numeric(out[n_obj_col], errors="coerce")
    out["n_samples_key"] = pd.to_numeric(out[n_samples_col], errors="coerce")
    out["noise_key"] = out[noise_col].map(normalize_noise_tag)
    return out


def prepare_perf(perf_df):
    """
    surrogate_perf.csv has one row per objective:
      problem,n_var,n_obj,dataset_file,algo,noise,objective,mean_r2,std_r2,mean_mse,n_samples,n_splits

    This aggregates it into one row per:
      problem, n_var, n_obj, n_samples, noise, dataset_file, algo
    """
    perf = add_common_keys(
        perf_df,
        problem_col="problem",
        n_var_col="n_var",
        n_obj_col="n_obj",
        n_samples_col="n_samples",
        noise_col="noise",
    )

    for c in ["mean_r2", "std_r2", "mean_mse", "n_splits"]:
        if c in perf.columns:
            perf[c] = pd.to_numeric(perf[c], errors="coerce")

    group_cols = [
        "problem_key",
        "n_var_key",
        "n_obj_key",
        "n_samples_key",
        "noise_key",
        "dataset_file",
        "algo",
    ]

    agg = (
        perf.groupby(group_cols, dropna=False)
        .agg(
            perf_mean_r2=("mean_r2", "mean"),
            perf_median_r2=("mean_r2", "median"),
            perf_min_r2=("mean_r2", "min"),
            perf_std_r2_across_objectives=("mean_r2", "std"),
            perf_mean_cv_std_r2=("std_r2", "mean"),
            perf_mean_mse=("mean_mse", "mean"),
            perf_median_mse=("mean_mse", "median"),
            perf_max_mse=("mean_mse", "max"),
            perf_std_mse_across_objectives=("mean_mse", "std"),
            perf_n_objectives_reported=("objective", "nunique"),
            perf_n_splits=("n_splits", "max"),
        )
        .reset_index()
    )

    # Avoid NaN std when only one objective was reported.
    for c in ["perf_std_r2_across_objectives", "perf_std_mse_across_objectives"]:
        agg[c] = agg[c].fillna(0.0)

    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Merge surrogate metrics, surrogate predictive performance, and landscape features into one CSV."
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="/scratch/project_2017216/modelling_results/surrogate_metrics.csv",
        help="Path to surrogate_metrics.csv",
    )

    parser.add_argument(
        "--features",
        type=str,
        default="/scratch/project_2017216/modelling_results/features.csv",
        help="Path to features.csv",
    )

    parser.add_argument(
        "--perf",
        type=str,
        default="/scratch/project_2017216/modelling_results/surrogate_perf.csv",
        help="Path to surrogate_perf.csv",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="/scratch/project_2017216/modelling_results/combined_metrics_features.csv",
        help="Output CSV path",
    )

    args = parser.parse_args()

    if not os.path.exists(args.metrics):
        raise FileNotFoundError(f"Metrics file not found: {args.metrics}")

    if not os.path.exists(args.features):
        raise FileNotFoundError(f"Features file not found: {args.features}")

    if not os.path.exists(args.perf):
        raise FileNotFoundError(f"Surrogate performance file not found: {args.perf}")

    print(f"Loading metrics:  {args.metrics}")
    metrics_df = pd.read_csv(args.metrics, low_memory=False)

    print(f"Loading features: {args.features}")
    features_df = pd.read_csv(args.features, low_memory=False)

    print(f"Loading perf:     {args.perf}")
    perf_df = pd.read_csv(args.perf, low_memory=False)

    required_metrics = [
        "problem", "n_var", "n_obj", "n_samples", "noise",
        "dataset_file", "algo",
    ]

    required_features = [
        "Problem", "VarCount", "ObjCount", "num_samples", "NoiseTag",
    ]

    required_perf = [
        "problem", "n_var", "n_obj", "dataset_file", "algo", "noise",
        "objective", "mean_r2", "std_r2", "mean_mse", "n_samples", "n_splits",
    ]

    missing_metrics = [c for c in required_metrics if c not in metrics_df.columns]
    missing_features = [c for c in required_features if c not in features_df.columns]
    missing_perf = [c for c in required_perf if c not in perf_df.columns]

    if missing_metrics:
        raise ValueError(f"Missing required columns in metrics file: {missing_metrics}")

    if missing_features:
        raise ValueError(f"Missing required columns in features file: {missing_features}")

    if missing_perf:
        raise ValueError(f"Missing required columns in perf file: {missing_perf}")

    # -----------------------------
    # Build merge keys
    # -----------------------------
    metrics = add_common_keys(
        metrics_df,
        problem_col="problem",
        n_var_col="n_var",
        n_obj_col="n_obj",
        n_samples_col="n_samples",
        noise_col="noise",
    )

    features = add_common_keys(
        features_df,
        problem_col="Problem",
        n_var_col="VarCount",
        n_obj_col="ObjCount",
        n_samples_col="num_samples",
        noise_col="NoiseTag",
    )

    perf = prepare_perf(perf_df)

    feature_merge_keys = [
        "problem_key",
        "n_var_key",
        "n_obj_key",
        "n_samples_key",
        "noise_key",
    ]

    perf_merge_keys = [
        "problem_key",
        "n_var_key",
        "n_obj_key",
        "n_samples_key",
        "noise_key",
        "dataset_file",
        "algo",
    ]

    # -----------------------------
    # Merge metrics + features
    # -----------------------------
    combined = pd.merge(
        metrics,
        features,
        on=feature_merge_keys,
        how="outer",
        suffixes=("_metrics", "_features"),
    )
    
    # -----------------------------
    # Merge predictive surrogate performance
    # -----------------------------
    combined = pd.merge(
        combined,
        perf,
        on=perf_merge_keys,
        how="outer",
    )

    # -----------------------------
    # Readable merged columns
    # -----------------------------
    combined["problem_merged"] = combined["problem_key"]
    combined["n_var_merged"] = combined["n_var_key"]
    combined["n_obj_merged"] = combined["n_obj_key"]
    combined["n_samples_merged"] = combined["n_samples_key"]
    combined["noise_merged"] = combined["noise_key"]

    # -----------------------------
    # Put important columns first
    # -----------------------------
    preferred_front = [
        "problem_merged",
        "n_var_merged",
        "n_obj_merged",
        "n_samples_merged",
        "noise_merged",
        "dataset_file",
        "ea",
        "algo",
        "iteration",
        "IGD",
        "HV",
        "EpsAdd",
        "EpsMulti",
        "perf_mean_r2",
        "perf_median_r2",
        "perf_min_r2",
        "perf_std_r2_across_objectives",
        "perf_mean_cv_std_r2",
        "perf_mean_mse",
        "perf_median_mse",
        "perf_max_mse",
        "perf_std_mse_across_objectives",
        "perf_n_objectives_reported",
        "perf_n_splits",
        "RUN_TAG",
    ]

    ordered_cols = [c for c in preferred_front if c in combined.columns]
    remaining_cols = [c for c in combined.columns if c not in ordered_cols]
    combined = combined[ordered_cols + remaining_cols]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    combined.to_csv(args.output, index=False)

    print("\nDone.")
    print(f"Metrics rows        : {len(metrics_df)}")
    print(f"Features rows       : {len(features_df)}")
    print(f"Perf raw rows       : {len(perf_df)}")
    print(f"Perf aggregated rows: {len(perf)}")
    print(f"Combined rows       : {len(combined)}")
    print(f"Saved to            : {args.output}")


if __name__ == "__main__":
    main()
