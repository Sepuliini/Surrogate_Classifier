#!/usr/bin/env python3
import os
import warnings
import itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import re
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

N_SPLITS = 3
INNER_CV_SPLITS = 2
N_SEARCH_ITER = 8

TUNING_SCORING = "balanced_accuracy"

# Criteria used to define the oracle-best surrogate.
# Lower is better: IGD, EpsAdd, EpsMulti
# Higher is better: HV, R2
SCORE_COLS = ["IGD", "EpsAdd", "EpsMulti", "HV", "R2"]

GROUP_COLS = ["problem", "n_var", "n_obj", "num_samples", "noise"]

TOP_K = [1, 2, 3]

TRAIN_PROBLEMS = ["DTLZ", "WFG", "DBMOPP"]
TEST_PROBLEMS = ["Engineering"]

EA_MODE = "prefer_combined"

IN_FP = "combined_metrics_features.csv"
BASE_OUT = "results/combined_classifier_results"

USE_HYPERPARAM_TUNING = True

ENABLE_PRE_TUNING = True
REMOVE_TRAIN_OUTLIERS = True
REMOVE_USELESS_COLUMNS = True
REMOVE_TIME_COLUMNS = True

MAX_MISSING_FRAC = 0.95
MIN_UNIQUE_VALUES = 2
MI_ZERO_THRESHOLD = 1e-12

OUTLIER_IQR_MULTIPLIER = 3.0

TIME_COL_PATTERNS = [
    r"time",
    r"runtime",
    r"duration",
    r"elapsed",
    r"seconds?",
    r"minutes?",
    r"hours?",
    r"wallclock",
    r"cpu",
    r"timestamp",
]


# =========================================================
# HELPERS
# =========================================================

def is_time_like_column(col):
    col_l = str(col).lower()
    return any(re.search(p, col_l) for p in TIME_COL_PATTERNS)


def remove_train_outliers_iqr(X_train, y_train, groups_train, numeric_features):
    if not REMOVE_TRAIN_OUTLIERS or len(numeric_features) == 0:
        return X_train, y_train, groups_train

    X_num = X_train[numeric_features].apply(pd.to_numeric, errors="coerce")

    q1 = X_num.quantile(0.25)
    q3 = X_num.quantile(0.75)
    iqr = q3 - q1

    usable = iqr > 0
    if usable.sum() == 0:
        return X_train, y_train, groups_train

    lower = q1[usable] - OUTLIER_IQR_MULTIPLIER * iqr[usable]
    upper = q3[usable] + OUTLIER_IQR_MULTIPLIER * iqr[usable]

    mask = ~((X_num[usable.index[usable]] < lower) | (X_num[usable.index[usable]] > upper)).any(axis=1)

    removed = int((~mask).sum())
    if removed > 0:
        print(f"    removed training outliers: {removed}/{len(mask)}")

    return X_train.loc[mask], y_train.loc[mask], groups_train.loc[mask]


def pre_tune_feature_columns(X_train, y_train, numeric_features, categorical_features):
    if not ENABLE_PRE_TUNING:
        return numeric_features, categorical_features

    selected_num = list(numeric_features)
    selected_cat = list(categorical_features)

    dropped = {
        "time_like": [],
        "too_missing": [],
        "constant": [],
        "zero_mi": [],
    }

    # Remove time-like columns by name.
    if REMOVE_TIME_COLUMNS:
        for c in selected_num + selected_cat:
            if is_time_like_column(c):
                dropped["time_like"].append(c)

        selected_num = [c for c in selected_num if c not in dropped["time_like"]]
        selected_cat = [c for c in selected_cat if c not in dropped["time_like"]]

    if REMOVE_USELESS_COLUMNS:
        # Remove mostly missing columns.
        for c in selected_num + selected_cat:
            miss_frac = X_train[c].isna().mean()
            if miss_frac >= MAX_MISSING_FRAC:
                dropped["too_missing"].append(c)

        selected_num = [c for c in selected_num if c not in dropped["too_missing"]]
        selected_cat = [c for c in selected_cat if c not in dropped["too_missing"]]

        # Remove constant / single-value columns.
        for c in selected_num + selected_cat:
            nunique = X_train[c].nunique(dropna=True)
            if nunique < MIN_UNIQUE_VALUES:
                dropped["constant"].append(c)

        selected_num = [c for c in selected_num if c not in dropped["constant"]]
        selected_cat = [c for c in selected_cat if c not in dropped["constant"]]

        # Remove numeric columns with zero estimated mutual information.
        if len(selected_num) > 0 and y_train.nunique() >= 2:
            X_num = X_train[selected_num].apply(pd.to_numeric, errors="coerce")
            X_num = X_num.replace([np.inf, -np.inf], np.nan)
            X_num = X_num.fillna(X_num.median())

            try:
                mi = mutual_info_classif(
                    X_num,
                    y_train.astype(str),
                    discrete_features=False,
                    random_state=RANDOM_STATE,
                )

                zero_mi_cols = [
                    c for c, v in zip(selected_num, mi)
                    if not np.isfinite(v) or v <= MI_ZERO_THRESHOLD
                ]

                dropped["zero_mi"] = zero_mi_cols
                selected_num = [c for c in selected_num if c not in zero_mi_cols]

            except Exception as e:
                print(f"    [WARN] mutual-info pruning skipped: {e}")

    total_dropped = sum(len(v) for v in dropped.values())
    if total_dropped > 0:
        print("    pre-tuning dropped columns:")
        for reason, cols in dropped.items():
            if cols:
                print(f"      {reason}: {len(cols)}")

    return sorted(selected_num), sorted(selected_cat)

def problem_family(p):
    p = str(p).lower()
    if p.startswith("dtlz"):
        return "DTLZ"
    if p.startswith("wfg"):
        return "WFG"
    if p.startswith("dbmopp"):
        return "DBMOPP"
    return "Engineering"


def normalize_noise_tag(x):
    if pd.isna(x):
        return "none"
    x = str(x).strip()
    return x if x else "none"


def safe_problem_name(x):
    if pd.isna(x):
        return np.nan
    return str(x).strip()


def pick_first_nonnull_rowwise(df, candidates):
    cols = [c for c in candidates if c in df.columns]
    if not cols:
        return pd.Series([np.nan] * len(df), index=df.index)

    out = df[cols[0]].copy()
    for c in cols[1:]:
        out = out.where(out.notna(), df[c])
    return out


# =========================================================
# WEIGHT SCHEMES
# =========================================================

def generate_weight_schemes(score_cols):
    schemes = {}
    n = len(score_cols)
    idx = list(range(n))

    for i in idx:
        w = {col: 0.0 for col in score_cols}
        w[score_cols[i]] = 1.0
        schemes[f"single_{score_cols[i]}"] = w

    for comb in itertools.combinations(idx, 2):
        w = {col: 0.0 for col in score_cols}
        for i in comb:
            w[score_cols[i]] = 1 / 2
        name = "pair_" + "_".join([score_cols[i] for i in comb])
        schemes[name] = w

    for comb in itertools.combinations(idx, 3):
        w = {col: 0.0 for col in score_cols}
        for i in comb:
            w[score_cols[i]] = 1 / 3
        name = "triple_" + "_".join([score_cols[i] for i in comb])
        schemes[name] = w

    for comb in itertools.combinations(idx, 4):
        w = {col: 0.0 for col in score_cols}
        for i in comb:
            w[score_cols[i]] = 1 / 4
        name = "quad_" + "_".join([score_cols[i] for i in comb])
        schemes[name] = w

    w = {col: 1 / n for col in score_cols}
    schemes["all_equal"] = w

    return schemes


ALL_SCHEMES = list(generate_weight_schemes(SCORE_COLS).items())


# =========================================================
# XGB WRAPPER
# =========================================================

class XGBWrapped(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.le = LabelEncoder()

    def fit(self, X, y):
        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.model.fit(X, y_enc)
        return self

    def predict(self, X):
        return self.le.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# =========================================================
# PARAMETER SPACES
# =========================================================

param_spaces = {
    "rf": {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 10, 20, 40],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "et": {
        "clf__n_estimators": [200, 400, 800],
        "clf__max_depth": [None, 10, 20, 40],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__max_features": ["sqrt", "log2", None],
    },
    "nn": {
        "clf__hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "clf__alpha": [1e-5, 1e-4, 1e-3],
        "clf__learning_rate_init": [1e-4, 5e-4, 1e-3],
        "clf__activation": ["relu", "logistic"],
        "clf__solver": ["adam"],
        "clf__max_iter": [800, 1200],
    },
    "svc": {
        "clf__C": [0.01, 0.1, 1, 10, 100],
        "clf__gamma": ["scale", 0.001, 0.01, 0.1, 1.0],
        "clf__kernel": ["rbf"],
    },
    "knn": {
        "clf__n_neighbors": [3, 5, 9, 15, 25],
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2],
    },
    "xgb": {
        "clf__n_estimators": [300, 500, 700],
        "clf__max_depth": [4, 6, 10],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    },
}


# =========================================================
# DATA PREP
# =========================================================

def canonicalize_combined_dataset(df_raw):
    df = df_raw.copy()
    out = pd.DataFrame(index=df.index)

    out["problem"] = pick_first_nonnull_rowwise(
        df, ["problem", "problem_merged", "Problem", "problem_key"]
    ).map(safe_problem_name)

    out["n_var"] = pd.to_numeric(
        pick_first_nonnull_rowwise(df, ["n_var", "n_var_merged", "VarCount", "n_var_key"]),
        errors="coerce",
    )

    out["n_obj"] = pd.to_numeric(
        pick_first_nonnull_rowwise(df, ["n_obj", "n_obj_merged", "ObjCount", "n_obj_key"]),
        errors="coerce",
    )

    out["num_samples"] = pd.to_numeric(
        pick_first_nonnull_rowwise(
            df,
            ["num_samples", "n_samples", "n_samples_merged", "num_samples_key", "n_samples_key"],
        ),
        errors="coerce",
    )

    out["noise"] = pick_first_nonnull_rowwise(
        df, ["noise", "noise_merged", "NoiseTag", "noise_key"]
    ).map(normalize_noise_tag)

    out["surrogate"] = pick_first_nonnull_rowwise(
        df, ["surrogate", "algo", "algo_metrics"]
    ).astype(str)

    for c in ["dataset_file", "ea", "iteration", "RUN_TAG"]:
        out[c] = df[c] if c in df.columns else np.nan

    for c in ["IGD", "HV", "EpsAdd", "EpsMulti"]:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan

    if "perf_mean_r2" in df.columns:
        out["R2"] = pd.to_numeric(df["perf_mean_r2"], errors="coerce")
    elif "perf_median_r2" in df.columns:
        out["R2"] = pd.to_numeric(df["perf_median_r2"], errors="coerce")
    else:
        out["R2"] = np.nan

    out["is_uniform"] = df["is_uniform"] if "is_uniform" in df.columns else np.nan
    out["f_col"] = df["f_col"].astype(str) if "f_col" in df.columns else np.nan

    for c in df.columns:
        if c not in out.columns:
            out[c] = df[c]

    return out


def identify_raw_feature_columns(df_canon):
    reserved = {
        "problem", "n_var", "n_obj", "num_samples", "noise", "surrogate",
        "dataset_file", "ea", "iteration", "RUN_TAG", "f_col", "is_uniform",
        "IGD", "HV", "EpsAdd", "EpsMulti", "R2",
        "problem_merged", "n_var_merged", "n_obj_merged", "n_samples_merged", "noise_merged",
        "problem_key", "n_var_key", "n_obj_key", "n_samples_key", "noise_key",
        "Problem", "VarCount", "ObjCount", "num_samples_key", "NoiseTag",
        "problem_features", "problem_metrics", "algo", "algo_metrics",
        "n_samples",
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
    }

    feature_cols = []
    for c in df_canon.columns:
        if c in reserved:
            continue
        if pd.api.types.is_numeric_dtype(df_canon[c]):
            feature_cols.append(c)

    return sorted(feature_cols)


def aggregate_metrics_table(df_canon):
    needed = GROUP_COLS + ["surrogate", "ea", "iteration"] + SCORE_COLS
    metrics = df_canon[needed].copy()

    metrics = metrics.dropna(
        subset=["problem", "n_var", "n_obj", "num_samples", "noise", "surrogate"]
    )
    metrics = metrics[metrics[SCORE_COLS].notna().any(axis=1)].copy()

    if len(metrics) == 0:
        raise ValueError("No usable performance rows found in combined dataset.")

    key_cols = GROUP_COLS + ["surrogate"]

    if EA_MODE == "prefer_combined":
        combined_rows = metrics[metrics["ea"].astype(str) == "Combined"].copy()
        other_rows = metrics[metrics["ea"].astype(str) != "Combined"].copy()

        if len(combined_rows) > 0:
            combined_rows = (
                combined_rows.groupby(key_cols, dropna=False)[SCORE_COLS]
                .mean()
                .reset_index()
            )

            combined_keys = set(map(tuple, combined_rows[key_cols].values.tolist()))

            if len(other_rows) > 0:
                other_rows["__key__"] = list(map(tuple, other_rows[key_cols].values.tolist()))
                other_rows = other_rows[~other_rows["__key__"].isin(combined_keys)]
                other_rows = other_rows.drop(columns="__key__")

            if len(other_rows) > 0:
                other_rows = (
                    other_rows.groupby(key_cols, dropna=False)[SCORE_COLS]
                    .mean()
                    .reset_index()
                )

            metrics_final = pd.concat([combined_rows, other_rows], ignore_index=True)
        else:
            metrics_final = (
                other_rows.groupby(key_cols, dropna=False)[SCORE_COLS]
                .mean()
                .reset_index()
            )
    else:
        metrics_final = (
            metrics.groupby(key_cols, dropna=False)[SCORE_COLS]
            .mean()
            .reset_index()
        )

    return metrics_final


def build_feature_table(df_canon):
    feature_cols = identify_raw_feature_columns(df_canon)

    base_cols = GROUP_COLS.copy()
    if "is_uniform" in df_canon.columns:
        base_cols.append("is_uniform")
    if "f_col" in df_canon.columns:
        base_cols.append("f_col")

    feat_df = df_canon[base_cols + feature_cols].copy()
    feat_df = feat_df.dropna(subset=["problem", "n_var", "n_obj", "num_samples", "noise"])

    if len(feat_df) == 0:
        raise ValueError("No usable feature rows found in combined dataset.")

    has_f_col = "f_col" in feat_df.columns and feat_df["f_col"].notna().any()

    if has_f_col:
        agg_cols = GROUP_COLS + ["f_col"]

        feat_num = (
            feat_df[agg_cols + feature_cols]
            .groupby(agg_cols, dropna=False)[feature_cols]
            .mean()
            .reset_index()
        )

        feat_uni = None
        if "is_uniform" in feat_df.columns:
            feat_uni = (
                feat_df[GROUP_COLS + ["is_uniform"]]
                .groupby(GROUP_COLS, dropna=False)["is_uniform"]
                .agg(lambda x: x.dropna().iloc[0] if x.dropna().shape[0] > 0 else np.nan)
                .reset_index()
            )

        wide_parts = []
        for fc in feature_cols:
            piv = feat_num.pivot_table(
                index=GROUP_COLS,
                columns="f_col",
                values=fc,
                aggfunc="mean",
            )
            piv.columns = [f"{fc}__{str(col)}" for col in piv.columns]
            wide_parts.append(piv)

        if wide_parts:
            feat_wide = pd.concat(wide_parts, axis=1).reset_index()
        else:
            feat_wide = feat_num[GROUP_COLS].drop_duplicates().reset_index(drop=True)

        if feat_uni is not None:
            feat_wide = feat_wide.merge(feat_uni, on=GROUP_COLS, how="left")

    else:
        agg_dict = {c: "mean" for c in feature_cols}

        if "is_uniform" in feat_df.columns:
            agg_dict["is_uniform"] = lambda x: (
                x.dropna().iloc[0] if x.dropna().shape[0] > 0 else np.nan
            )

        cols = GROUP_COLS + feature_cols
        if "is_uniform" in feat_df.columns:
            cols.append("is_uniform")

        feat_wide = feat_df[cols].groupby(GROUP_COLS, dropna=False).agg(agg_dict).reset_index()

    return feat_wide


def add_objective_aggregates(df):
    base_feats = sorted({c.split("__")[0] for c in df.columns if "__" in c})
    new = {}

    for bf in base_feats:
        cols = [c for c in df.columns if c.startswith(bf + "__")]
        if len(cols) >= 2:
            new[bf + "_mean"] = df[cols].mean(axis=1)
            new[bf + "_std"] = df[cols].std(axis=1)
            new[bf + "_max"] = df[cols].max(axis=1)

    if new:
        return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)

    return df


def build_modeling_table(fp):
    df_raw = pd.read_csv(fp, low_memory=False)
    df_canon = canonicalize_combined_dataset(df_raw)

    metrics_tbl = aggregate_metrics_table(df_canon)
    feat_tbl = build_feature_table(df_canon)
    feat_tbl = add_objective_aggregates(feat_tbl)

    df_model = metrics_tbl.merge(feat_tbl, on=GROUP_COLS, how="left")

    dedup_cols = GROUP_COLS + ["surrogate"]
    if df_model.duplicated(subset=dedup_cols).any():
        num_cols = [
            c for c in df_model.columns
            if c not in dedup_cols and pd.api.types.is_numeric_dtype(df_model[c])
        ]
        cat_cols = [
            c for c in df_model.columns
            if c not in dedup_cols and c not in num_cols
        ]

        agg = {c: "mean" for c in num_cols}
        for c in cat_cols:
            agg[c] = lambda x: x.dropna().iloc[0] if x.dropna().shape[0] > 0 else np.nan

        df_model = df_model.groupby(dedup_cols, dropna=False).agg(agg).reset_index()

    return df_model


# =========================================================
# ORACLE CONSTRUCTION
# =========================================================

def build_multimetric_score(df, weights):
    out = []

    for _, sub in df.groupby(GROUP_COLS, dropna=False):
        sub = sub.copy()

        for col in SCORE_COLS:
            vals = pd.to_numeric(sub[col], errors="coerce").values.astype(float)

            # Convert all criteria to "lower is better".
            if col in ["HV", "R2"]:
                vals = -vals

            if col == "EpsMulti":
                vals = np.clip(vals, 1e-12, None)
                vals = np.log10(vals)

            finite = np.isfinite(vals)
            if finite.sum() == 0:
                sub[col + "_norm"] = np.nan
                continue

            lo = np.nanmin(vals)
            hi = np.nanmax(vals)

            if hi > lo:
                sub[col + "_norm"] = (vals - lo) / (hi - lo)
            else:
                sub[col + "_norm"] = 0.0

        norm_cols = [
            c for c in sub.columns
            if c.endswith("_norm") and c.replace("_norm", "") in weights
        ]

        if len(norm_cols) == 0:
            sub["score"] = np.nan
        else:
            w = np.array([weights[c.replace("_norm", "")] for c in norm_cols], dtype=float)
            w = w / w.sum()

            arr = sub[norm_cols].astype(float).values
            arr = np.where(np.isfinite(arr), arr, 1.0)

            sub["score"] = (arr * w).sum(axis=1)

        out.append(sub)

    return pd.concat(out, ignore_index=True)


def build_oracle_tables(df_scored):
    rank_table = {}
    best_score = {}

    for key, sub in df_scored.groupby(GROUP_COLS, dropna=False):
        sub = sub.copy().sort_values("score", ascending=True)
        rank_table[key] = {s: i + 1 for i, s in enumerate(sub["surrogate"])}
        best_score[key] = sub["score"].iloc[0]

    return rank_table, best_score


def get_score_for_surrogate(df_scored, key, surrogate):
    mask = np.ones(len(df_scored), dtype=bool)

    for i, col in enumerate(GROUP_COLS):
        mask &= df_scored[col] == key[i]

    mask &= df_scored["surrogate"] == surrogate

    sub = df_scored.loc[mask]

    if len(sub) == 0:
        return None

    return sub["score"].iloc[0]


# =========================================================
# MODEL FITTING
# =========================================================

def build_preprocessor(df_cls, numeric_override=None, categorical_override=None):
    exclude_cols = set(
        GROUP_COLS
        + ["problem", "surrogate", "family", "score"]
        + SCORE_COLS
        + [c + "_norm" for c in SCORE_COLS]
        + [
            "dataset_file", "ea", "iteration", "RUN_TAG", "f_col",
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
        ]
    )

    feature_cols = [c for c in df_cls.columns if c not in exclude_cols]

    meta_num = [c for c in ["n_var", "n_obj", "num_samples"] if c in df_cls.columns]
    meta_cat = [c for c in ["noise", "is_uniform"] if c in df_cls.columns]

    candidate_numeric = []
    candidate_categorical = []

    for c in feature_cols:
        if c in meta_num or c in meta_cat:
            continue

        if pd.api.types.is_numeric_dtype(df_cls[c]):
            candidate_numeric.append(c)
        else:
            candidate_categorical.append(c)

    if numeric_override is None:
        numeric_features = sorted(set(candidate_numeric + meta_num))
    else:
        numeric_features = sorted(numeric_override)
    
    if categorical_override is None:
        categorical_features = sorted(set(meta_cat + candidate_categorical))
    else:
        categorical_features = sorted(categorical_override)

    prep = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imp", SimpleImputer(strategy="median")),
                    ("sc", StandardScaler()),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return prep, numeric_features, categorical_features


def fit_model_with_search(pipe, model_name, X_train, y_train, groups_train):
    if not USE_HYPERPARAM_TUNING or model_name not in param_spaces:
        pipe.fit(X_train, y_train)
        return pipe

    n_unique_groups = len(pd.Series(groups_train).astype(str).unique())
    n_splits_inner = min(INNER_CV_SPLITS, n_unique_groups)

    if n_splits_inner < 2:
        pipe.fit(X_train, y_train)
        return pipe

    inner_cv = GroupKFold(n_splits=n_splits_inner)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_spaces[model_name],
        n_iter=N_SEARCH_ITER,
        cv=inner_cv,
        scoring=TUNING_SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        error_score=np.nan,
    )

    search.fit(X_train, y_train, groups=groups_train)

    print(f"{model_name} best params: {search.best_params_}")
    print(f"{model_name} best score: {search.best_score_:.4f}")

    return search.best_estimator_


def build_model_defs():
    model_defs = {
        "rf": RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "et": ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "nn": MLPClassifier(
            max_iter=500,
            early_stopping=False,
            n_iter_no_change=20,
            random_state=RANDOM_STATE,
        ),
        "svc": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(),
    }

    if HAS_XGB:
        model_defs["xgb"] = XGBWrapped(random_state=RANDOM_STATE, n_jobs=4)

    return model_defs


# =========================================================
# EVALUATION HELPERS
# =========================================================

def topk_best_regret(classes, probs_row, y_true, k, key, df_scored, best_score):
    ranked = classes[np.argsort(probs_row)[::-1]]
    topk = ranked[:k]
    hit = y_true in topk

    candidate_regrets = []

    for s in topk:
        score_s = get_score_for_surrogate(df_scored, key, s)
        if score_s is not None:
            candidate_regrets.append(score_s - best_score[key])

    best_topk = np.nan if len(candidate_regrets) == 0 else min(candidate_regrets)

    return hit, best_topk


# =========================================================
# MAIN
# =========================================================

def main():
    os.makedirs(BASE_OUT, exist_ok=True)

    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1")) - 1

    if task_id < 0 or task_id >= len(ALL_SCHEMES):
        raise IndexError(
            f"SLURM_ARRAY_TASK_ID={task_id + 1} out of range for {len(ALL_SCHEMES)} schemes"
        )

    tag, weights = ALL_SCHEMES[task_id]
    outdir = os.path.join(BASE_OUT, tag)
    os.makedirs(outdir, exist_ok=True)

    print(f"\n=== Running weight scheme: {tag} ===")
    print(f"Input file: {IN_FP}")
    print(f"Score columns: {SCORE_COLS}")
    print(f"Weights: {weights}")

    df_all = build_modeling_table(IN_FP)
    df_all["family"] = df_all["problem"].map(problem_family)

    print(f"Modeling rows: {len(df_all)}")
    print(f"Unique groups: {df_all[GROUP_COLS].drop_duplicates().shape[0]}")
    print(f"Surrogates: {sorted(df_all['surrogate'].dropna().astype(str).unique())}")

    df_scored = build_multimetric_score(df_all, weights)
    rank_table, best_score = build_oracle_tables(df_scored)

    idx = [
        sub["score"].idxmin()
        for _, sub in df_scored.groupby(GROUP_COLS, dropna=False)
        if sub["score"].notna().any()
    ]

    df_cls = df_scored.loc[idx].reset_index(drop=True)
    df_cls["family"] = df_cls["problem"].map(problem_family)

    print("\nFamily counts in df_cls:")
    print(df_cls["family"].value_counts(dropna=False))

    prep, numeric_features, categorical_features = build_preprocessor(df_cls)

    print(f"Classification rows: {len(df_cls)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    X = df_cls[numeric_features + categorical_features].copy()
    y = df_cls["surrogate"].astype(str).copy()
    groups = df_cls["problem"].astype(str).copy()

    model_defs = build_model_defs()

    summary = []
    detailed_rows = []

    n_unique_groups = len(groups.unique())
    n_splits_outer = min(N_SPLITS, n_unique_groups)

    if n_splits_outer < 2:
        raise ValueError("Need at least 2 unique groups/problems for outer GroupKFold.")

    outer_cv = GroupKFold(n_splits=n_splits_outer)

    # -----------------------------------------------------
    # Nested GroupKFold CV
    # -----------------------------------------------------
    for name, base in model_defs.items():
        print(f"\n  -> Nested CV model: {name}")

        preds = np.empty(len(y), dtype=object)
        cv_reg = []
        cv_topk_hits = {k: [] for k in TOP_K}
        cv_topk_regret = {k: [] for k in TOP_K}

        for fold_id, (tr, te) in enumerate(outer_cv.split(X, y, groups), start=1):
            X_tr, X_te = X.iloc[tr], X.iloc[te]
            y_tr, y_te = y.iloc[tr], y.iloc[te]
            g_tr = groups.iloc[tr]

            num_fold, cat_fold = pre_tune_feature_columns(
                X_train=X_tr,
                y_train=y_tr,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            
            X_tr_fit = X_tr[num_fold + cat_fold].copy()
            X_te_fit = X_te[num_fold + cat_fold].copy()
            
            X_tr_fit, y_tr_fit, g_tr_fit = remove_train_outliers_iqr(
                X_train=X_tr_fit,
                y_train=y_tr,
                groups_train=g_tr,
                numeric_features=num_fold,
            )
            
            prep_fold, _, _ = build_preprocessor(
                df_cls,
                numeric_override=num_fold,
                categorical_override=cat_fold,
            )
            
            pipe = Pipeline([
                ("prep", prep_fold),
                ("clf", clone(base)),
            ])

            model_fold = fit_model_with_search(
                pipe=pipe,
                model_name=name,
                X_train=X_tr_fit,
                y_train=y_tr_fit,
                groups_train=g_tr_fit,
            )

            preds_fold = model_fold.predict(X_te_fit)
            preds[te] = preds_fold

            if hasattr(model_fold, "predict_proba"):
                probs_fold = model_fold.predict_proba(X_te_fit)
                classes_fold = model_fold.classes_
            else:
                probs_fold = None
                classes_fold = None

            for local_idx, (i, pred) in enumerate(zip(te, preds_fold)):
                row_i = df_cls.iloc[i]
                key = tuple(row_i[c] for c in GROUP_COLS)
                tb = best_score[key]

                pred_score = get_score_for_surrogate(df_scored, key, pred)
                regret = np.nan if pred_score is None else pred_score - tb
                cv_reg.append(regret)

                det = {
                    "row_type": "individual",
                    "weight_scheme": tag,
                    "model": name,
                    "eval_split": "nested_cv",
                    "fold_id": fold_id,
                    "problem": row_i["problem"],
                    "n_var": row_i["n_var"],
                    "n_obj": row_i["n_obj"],
                    "num_samples": row_i["num_samples"],
                    "noise": row_i["noise"],
                    "family": row_i["family"],
                    "true_surrogate": y.iloc[i],
                    "predicted_surrogate": pred,
                    "oracle_best_score": tb,
                    "predicted_score": pred_score,
                    "regret": regret,
                }

                if probs_fold is not None:
                    for k in TOP_K:
                        hit_k, best_topk = topk_best_regret(
                            classes_fold,
                            probs_fold[local_idx],
                            y.iloc[i],
                            k,
                            key,
                            df_scored,
                            best_score,
                        )

                        cv_topk_hits[k].append(hit_k)
                        cv_topk_regret[k].append(best_topk)
                        det[f"top{k}_hit"] = hit_k
                        det[f"top{k}_best_regret"] = best_topk

                detailed_rows.append(det)

        row = {
            "row_type": "summary",
            "weight_scheme": tag,
            "model": name,
            "eval_split": "nested_cv_summary",
            "accuracy": accuracy_score(y, preds),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "mean_regret": np.nanmean(cv_reg),
        }

        for k in TOP_K:
            row[f"top{k}_accuracy"] = np.nanmean(cv_topk_hits[k]) if cv_topk_hits[k] else np.nan
            row[f"top{k}_mean_regret"] = np.nanmean(cv_topk_regret[k]) if cv_topk_regret[k] else np.nan

        summary.append(row)

    # -----------------------------------------------------
    # Family holdout
    # -----------------------------------------------------
    train_mask = df_cls["family"].isin(TRAIN_PROBLEMS)
    test_mask = df_cls["family"].isin(TEST_PROBLEMS)

    print("\nFamily holdout check:")
    print("Train rows:", train_mask.sum())
    print("Test rows:", test_mask.sum())

    if train_mask.sum() > 0 and test_mask.sum() > 0:
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test, y_test = X.loc[test_mask], y.loc[test_mask]
        groups_train = groups.loc[train_mask]
        df_test = df_cls.loc[test_mask].reset_index(drop=True)

        for name, base in model_defs.items():
            print(f"\n  -> Family holdout model: {name}")

            num_hold, cat_hold = pre_tune_feature_columns(
                X_train=X_train,
                y_train=y_train,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )
            
            X_train_fit = X_train[num_hold + cat_hold].copy()
            X_test_fit = X_test[num_hold + cat_hold].copy()
            
            X_train_fit, y_train_fit, groups_train_fit = remove_train_outliers_iqr(
                X_train=X_train_fit,
                y_train=y_train_fit,
                groups_train=groups_train_fit,
            )
            
            prep_hold, _, _ = build_preprocessor(
                df_cls,
                numeric_override=num_hold,
                categorical_override=cat_hold,
            )
            
            pipe = Pipeline([
                ("prep", prep_hold),
                ("clf", clone(base)),
            ])

            final_model = fit_model_with_search(
                pipe=pipe,
                model_name=name,
                X_train=X_train,
                y_train=y_train,
                groups_train=groups_train,
            )

            preds_g = final_model.predict(X_test_fit)

            if hasattr(final_model, "predict_proba"):
                probs_g = final_model.predict_proba(X_test_fit)
                classes_g = final_model.classes_
            else:
                probs_g = None
                classes_g = None

            gen_reg = []
            gen_topk_hits = {k: [] for k in TOP_K}
            gen_topk_regret = {k: [] for k in TOP_K}

            for i, pred in enumerate(preds_g):
                row_i = df_test.iloc[i]
                key = tuple(row_i[c] for c in GROUP_COLS)
                tb = best_score[key]

                pred_score = get_score_for_surrogate(df_scored, key, pred)
                regret = np.nan if pred_score is None else pred_score - tb
                gen_reg.append(regret)

                det = {
                    "row_type": "individual",
                    "weight_scheme": tag,
                    "model": name,
                    "eval_split": "family_holdout",
                    "fold_id": np.nan,
                    "problem": row_i["problem"],
                    "n_var": row_i["n_var"],
                    "n_obj": row_i["n_obj"],
                    "num_samples": row_i["num_samples"],
                    "noise": row_i["noise"],
                    "family": row_i["family"],
                    "true_surrogate": y_test.iloc[i],
                    "predicted_surrogate": pred,
                    "oracle_best_score": tb,
                    "predicted_score": pred_score,
                    "regret": regret,
                }

                if probs_g is not None:
                    for k in TOP_K:
                        hit_k, best_topk = topk_best_regret(
                            classes_g,
                            probs_g[i],
                            y_test.iloc[i],
                            k,
                            key,
                            df_scored,
                            best_score,
                        )

                        gen_topk_hits[k].append(hit_k)
                        gen_topk_regret[k].append(best_topk)
                        det[f"top{k}_hit"] = hit_k
                        det[f"top{k}_best_regret"] = best_topk

                detailed_rows.append(det)

            row = {
                "row_type": "summary",
                "weight_scheme": tag,
                "model": name,
                "eval_split": "family_holdout_summary",
                "accuracy": accuracy_score(y_test, preds_g),
                "balanced_accuracy": balanced_accuracy_score(y_test, preds_g),
                "mean_regret": np.nanmean(gen_reg),
            }

            for k in TOP_K:
                row[f"top{k}_accuracy"] = np.nanmean(gen_topk_hits[k]) if gen_topk_hits[k] else np.nan
                row[f"top{k}_mean_regret"] = np.nanmean(gen_topk_regret[k]) if gen_topk_regret[k] else np.nan

            summary.append(row)
    else:
        print("[WARN] Family holdout skipped because train or test rows are missing.")

    df_summary = pd.DataFrame(summary)
    df_detailed = pd.DataFrame(detailed_rows)
    df_all_out = pd.concat([df_summary, df_detailed], ignore_index=True, sort=False)

    df_summary.to_csv(os.path.join(outdir, "selector_summary.csv"), index=False)
    df_detailed.to_csv(os.path.join(outdir, "selector_detailed.csv"), index=False)
    df_all_out.to_csv(os.path.join(outdir, "selector_metrics.csv"), index=False)

    print(f"\nFinished scheme: {tag}")
    print(f"Saved: {os.path.join(outdir, 'selector_summary.csv')}")
    print(f"Saved: {os.path.join(outdir, 'selector_detailed.csv')}")
    print(f"Saved: {os.path.join(outdir, 'selector_metrics.csv')}")


if __name__ == "__main__":
    main()
